"""Production MTP speculative serving helpers.

Built on top of P118's verify-state-parity math (validated on R6000 in
``benchmarks/p118_mtp_verify_state_parity.py`` — accept and reject paths both
land canonical state within tolerance).

K=1 speculative step:

1. The MTP sidecar proposes one ``draft`` token, conditioned on the current
   ``pending`` token's base-hidden output.
2. The base model **sequentially** decodes ``[pending, draft]`` (two
   ``_decode_layer_fast`` chains, identical math to the single-token loop).
3. Accept rule: if ``draft == argmax_after_pending`` (base's true prediction
   for the position right after pending), commit BOTH tokens; otherwise
   roll back the recurrent/conv state to after-pending and commit ONLY
   pending — the next pending becomes ``argmax_after_pending``.

The KV cache is append-only: rejected positions stay in the slot but
``state.seq_len`` rewinds so they are overwritten by the next write. Only
recurrent + conv state need a true clone-restore snapshot (P118 confirmed
this — ``_commit_verify_python`` does not restore KV either).

**Speedup framing**: this K=1 sequential path runs TWO base forwards per round
(verify pending + decode draft). Per-round token yield is ``1 + accept_rate``.
The throughput math is ``(1 + accept_rate) / (2 + C_mtp / C_base)`` — at 65%
accept and ~3% MTP cost, that is ~0.8× single-token throughput, i.e. a slight
slowdown. The win unlocks only when paired with a **batched K-position decode
path** (single base forward returns logits for K+1 positions, memory-bound so
~free) — see ``M5-M6`` in the task list.

Use this module to ship **token-correctness parity** first, then layer the
batched verify path on top once the dispatcher / generate-loop wiring is in
place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from engine.full_forward import _rms_norm
from engine.inference_state import LynnInferenceState


__all__ = [
    "SpeculativeStepResult",
    "snapshot_recurrent_conv",
    "restore_recurrent_conv",
    "decode_one_to_logits_and_hidden",
    "speculative_step_k1",
]


@dataclass(slots=True)
class SpeculativeStepResult:
    """Output of one K=1 speculative step.

    Attributes:
        committed_tokens: 1 token (reject) or 2 tokens (accept). The caller
            emits these in order and uses ``next_pending_id`` for the next
            step.
        next_pending_id: Token to verify on the next call.
        next_base_hidden: Pre-final-norm hidden state of the last committed
            token. Fed to the MTP sidecar to produce the next draft.
        next_pos: Sequence position of ``next_pending_id``. Equals
            ``state.seq_len - 1`` after this step.
        accepted: Whether the draft matched base's argmax after pending.
        draft_id: The MTP draft proposal (regardless of accept/reject —
            used for accept-rate measurement).
    """

    committed_tokens: list[int]
    next_pending_id: int
    next_base_hidden: torch.Tensor
    next_pos: int
    accepted: bool
    draft_id: int


def snapshot_recurrent_conv(state: LynnInferenceState) -> dict[str, Any]:
    """Lean speculative snapshot — recurrent + conv + seq_len only.

    The full ``_snapshot_state`` in ``resident_runner`` also clones the KV
    cache, which dominates cost at ``max_seq_len`` shapes. For speculative
    serving the KV cache is append-only, so rewinding ``seq_len`` invalidates
    rejected positions naturally — they get overwritten by the next write.
    """
    return {
        "seq_len": int(state.seq_len),
        "recurrent": {i: t.clone() for i, t in state.recurrent_state.items()},
        "conv": {i: t.clone() for i, t in state.conv_state.items()},
    }


def restore_recurrent_conv(state: LynnInferenceState, snap: dict[str, Any]) -> None:
    """Mirror of :func:`snapshot_recurrent_conv` — in-place restore."""
    state.seq_len = int(snap["seq_len"])
    for layer_idx, tensor in snap["recurrent"].items():
        state.recurrent_state[layer_idx].copy_(tensor)
    for layer_idx, tensor in snap["conv"].items():
        state.conv_state[layer_idx].copy_(tensor)


def decode_one_to_logits_and_hidden(
    runner: Any,  # LynnIncrementalRunner — avoid circular import
    state: LynnInferenceState,
    token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run one decoder step.

    Returns ``(hidden_pre_final_norm, logits, argmax_id)``. The hidden tensor
    is what the MTP sidecar will consume on the next draft; ``logits`` is the
    base prediction for the position after ``token_id``.
    """
    token_tensor = torch.tensor([[int(token_id)]], device=runner.device, dtype=torch.long)
    pos_tensor = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
    h = F.embedding(token_tensor, runner.outside["model.language_model.embed_tokens.weight"])
    for layer_idx in range(runner.n_layers):
        h = runner._decode_layer_fast(h, pos_tensor, state, layer_idx)
    state.seq_len += 1
    h_norm = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_norm)
    argmax = int(logits[0].argmax().item())
    return h, logits, argmax


def speculative_step_k1(
    runner: Any,  # LynnIncrementalRunner
    state: LynnInferenceState,
    pending_id: int,
    pending_base_hidden: torch.Tensor,
    pending_pos: int,
) -> SpeculativeStepResult:
    """Execute one K=1 MTP speculative step.

    Args:
        runner: The ``LynnIncrementalRunner`` carrying weights, MTP sidecar,
            and ``_mtp_draft_logits`` / ``_decode_layer_fast`` methods.
        state: Mutable decode state, advanced in-place. On reject, recurrent
            and conv state are restored to the post-pending snapshot.
        pending_id: The token that must be committed at ``state.seq_len``.
        pending_base_hidden: Pre-final-norm hidden for the previous token
            (the one that PRODUCED ``pending_id`` as its argmax). Fed to
            the MTP sidecar.
        pending_pos: Position of the token whose hidden produced
            ``pending_id`` — i.e. ``state.seq_len - 1`` if the previous step
            committed normally.

    Returns:
        :class:`SpeculativeStepResult`. ``committed_tokens`` is 1 or 2.
    """
    # 1. Draft via MTP sidecar (1 attn + 1 MoE layer + lm_head).
    draft_logits = runner._mtp_draft_logits(
        base_hidden=pending_base_hidden,
        current_token_id=pending_id,
        current_pos=pending_pos,
    )
    draft_id = int(draft_logits[0].argmax().item())

    # 2. Base decodes pending (state advances to seq_len + 1).
    h_after_pending, _logits_after_pending, argmax_after_pending = decode_one_to_logits_and_hidden(
        runner, state, pending_id
    )
    # Snapshot AFTER pending so reject path can roll back to here.
    snap_after_pending = snapshot_recurrent_conv(state)

    # 3. Base decodes draft (state advances again).
    h_after_draft, _logits_after_draft, argmax_after_draft = decode_one_to_logits_and_hidden(
        runner, state, draft_id
    )

    # 4. Accept check. P118 commit_count = 2 if draft matches base's argmax-after-pending.
    if draft_id == argmax_after_pending:
        return SpeculativeStepResult(
            committed_tokens=[pending_id, draft_id],
            next_pending_id=argmax_after_draft,
            next_base_hidden=h_after_draft,
            next_pos=state.seq_len - 1,
            accepted=True,
            draft_id=draft_id,
        )

    # REJECT: rewind to post-pending state. KV cache positions after seq_len
    # stay stale but harmless — the next write overwrites them.
    restore_recurrent_conv(state, snap_after_pending)
    return SpeculativeStepResult(
        committed_tokens=[pending_id],
        next_pending_id=argmax_after_pending,
        next_base_hidden=h_after_pending,
        next_pos=state.seq_len - 1,
        accepted=False,
        draft_id=draft_id,
    )
