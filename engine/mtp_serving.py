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

import os
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
    "speculative_step_k1_batched",
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


def speculative_step_k1_batched(
    runner: Any,
    state: LynnInferenceState,
    pending_id: int,
    pending_base_hidden: torch.Tensor,
    pending_pos: int,
) -> SpeculativeStepResult:
    """Batched K=1 MTP speculative step (M5 path).

    Replaces the two sequential T=1 base forwards in
    :func:`speculative_step_k1` with one batched K=2 forward over
    ``[pending, draft]``. The motivation is memory-bound layers
    (MoE weight loads, full-attn KV reads) cost the same for T=1 and T=2 —
    the only per-position cost is linear-attn SSM rollout, which is
    fundamentally sequential and adds ~16% to a K=2 forward on Lynn's
    30/40 hybrid architecture. With OFFSET=2 head and ~65% accept the
    projected speedup is 1.3-1.5× over baseline T=1 decode; without
    OFFSET=2 (Lynn's current head) the wire is correctness-only.

    Reject path requires a recurrent/conv snapshot + a single T=1
    re-decode of pending. KV positions ``[seq_len, seq_len+1]`` get
    overwritten by the next round's K=2 forward, so KV does not need
    restoration.
    """
    if os.environ.get("LYNN_MTP_K2_VERIFY_MODE", "").lower() == "t1_canonical":
        return speculative_step_k1(
            runner=runner,
            state=state,
            pending_id=pending_id,
            pending_base_hidden=pending_base_hidden,
            pending_pos=pending_pos,
        )

    # 1. MTP draft (same as sequential variant — head-internal cost).
    draft_logits = runner._mtp_draft_logits(
        base_hidden=pending_base_hidden,
        current_token_id=pending_id,
        current_pos=pending_pos,
    )
    draft_id = int(draft_logits[0].argmax().item())

    # 2. Snapshot recurrent/conv state pre-batch — required for reject
    # rollback because the K=2 SSM rollout commits state changes for BOTH
    # input positions; only the first position's update should survive a
    # rejected draft.
    snap_pre_batch = snapshot_recurrent_conv(state)

    # 3. Batched K=2 forward over [pending, draft].
    tokens_k2 = torch.tensor(
        [[int(pending_id), int(draft_id)]],
        device=runner.device, dtype=torch.long,
    )
    pos_k2 = torch.tensor(
        [[int(state.seq_len), int(state.seq_len) + 1]],
        device=runner.device, dtype=torch.long,
    )
    h_k2 = F.embedding(tokens_k2, runner.outside["model.language_model.embed_tokens.weight"])
    for layer_idx in range(runner.n_layers):
        h_k2 = runner._decode_layer_k2_fast(h_k2, pos_k2, state, layer_idx)
    state.seq_len += 2
    h_norm_k2 = _rms_norm(h_k2, runner.outside["model.language_model.norm.weight"])
    # MTP K=2 needs logits for both positions. Normal decode/generate keeps the
    # historical final-token-only lm_head contract.
    logits_k2 = runner._lm_head_logits(h_norm_k2, all_positions=True)
    if logits_k2.ndim == 2:
        # Native FP4 lm_head squeezes [B, T, D] → [B*T, V]; un-batch back to [B, T, V].
        logits_k2 = logits_k2.view(h_k2.shape[0], h_k2.shape[1], -1)

    argmax_at_pos0 = int(logits_k2[0, 0].argmax().item())  # base's true prediction for position pending_pos+1
    argmax_at_pos1 = int(logits_k2[0, 1].argmax().item())  # base's prediction for position pending_pos+2

    # 4. Accept rule (P118 K=2 commit math, adapted to batched):
    #    if draft equals base's pos-0 argmax → both tokens valid → commit 2
    #    else → only pending was right at pos 0 → commit 1, rollback
    if draft_id == argmax_at_pos0:
        return SpeculativeStepResult(
            committed_tokens=[pending_id, draft_id],
            next_pending_id=argmax_at_pos1,
            next_base_hidden=h_k2[:, 1:2, :].contiguous(),
            next_pos=state.seq_len - 1,
            accepted=True,
            draft_id=draft_id,
        )

    # REJECT — restore SSM state to pre-batch, then re-run a single T=1
    # decode of pending alone. KV[pending_pos+1] becomes stale but is
    # naturally overwritten on the next K=2 forward.
    restore_recurrent_conv(state, snap_pre_batch)
    h_after_pending, _, argmax_after_pending = decode_one_to_logits_and_hidden(runner, state, pending_id)
    return SpeculativeStepResult(
        committed_tokens=[pending_id],
        # Use the canonical T=1 re-decode result for the next pending token.
        # K=2 pos-0 logits can be close but not bit-identical in long-running
        # state after prior accepts/rejects; using them here caused exact
        # generation drift even though the rejected pending token itself was
        # committed correctly.
        next_pending_id=argmax_after_pending,
        next_base_hidden=h_after_pending,
        next_pos=state.seq_len - 1,
        accepted=False,
        draft_id=draft_id,
    )
