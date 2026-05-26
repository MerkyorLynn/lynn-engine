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
    "decode_block_to_logits_and_hidden",
    "speculative_step_k1",
    "speculative_step_k1_batched",
    "speculative_step_kn_batched",
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
    draft_ids: list[int] | None = None
    accepted_count: int | None = None


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


def decode_block_to_logits_and_hidden(
    runner: Any,
    state: LynnInferenceState,
    token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Run one base-model block verifier over ``token_ids``.

    ``token_ids`` is typically ``[pending, draft_1, ..., draft_k]``. The
    returned logits are all-position logits, where row ``i`` predicts the token
    after ``token_ids[i]``.
    """
    if not token_ids:
        raise ValueError("decode_block_to_logits_and_hidden requires at least one token")
    token_tensor = torch.tensor([[int(x) for x in token_ids]], device=runner.device, dtype=torch.long)
    start = int(state.seq_len)
    pos_tensor = torch.arange(start, start + len(token_ids), device=runner.device, dtype=torch.long).unsqueeze(0)
    h = F.embedding(token_tensor, runner.outside["model.language_model.embed_tokens.weight"])
    for layer_idx in range(runner.n_layers):
        h = runner._decode_layer_block_fast(h, pos_tensor, state, layer_idx)
    state.seq_len += len(token_ids)
    h_norm = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_norm, all_positions=True)
    if logits.ndim == 2:
        logits = logits.view(h.shape[0], h.shape[1], -1)
    argmax_ids = [int(logits[0, idx].argmax().item()) for idx in range(len(token_ids))]
    return h, logits, argmax_ids


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

    # M21 diagnostic switches (default off):
    #   LYNN_MTP_K2_LM_HEAD_MODE=bf16
    #     Force BF16 F.linear for the K=2 verifier's lm_head call even when
    #     native FP4 lm_head is enabled. Bisects the M11 per-row 1D vs 2D
    #     quantize_fp4_m1_native suspicion.
    #   LYNN_MTP_K2_ACCEPT_SOURCE=canonical_t1
    #     Use a shadow sequential T=1 decode of the pending token to derive
    #     the accept decision (argmax_after_pending), instead of K=2 pos-0
    #     logits. K=2 forward still runs and K=2 logits are still computed +
    #     returned; only the accept comparison switches. Bisects accept-
    #     argmax drift from state/next_base_hidden drift.
    #
    # M22 diagnostic switch (default off):
    #   LYNN_MTP_K2_REJECT_ROLLBACK=full_state
    #     On REJECT path, replace the lean ``restore_recurrent_conv``
    #     (recurrent + conv + seq_len only, no KV) with the runner's full
    #     ``_restore_state`` (KV + recurrent + conv + seq_len). M21 found
    #     that even after lean rollback + canonical T=1 re-decode, the
    #     produced argmax differs from an apples-to-apples eager baseline
    #     T=1 decode of the same token from the same prefix; the K=2
    #     forward must be leaving residual state that lean restore does
    #     not undo. Full snapshot/restore tests whether KV+seq_len is the
    #     missing dimension. Accept path is unchanged.
    lm_head_mode = os.environ.get("LYNN_MTP_K2_LM_HEAD_MODE", "").lower()
    accept_source = os.environ.get("LYNN_MTP_K2_ACCEPT_SOURCE", "").lower()
    commit_source = os.environ.get("LYNN_MTP_K2_COMMIT_SOURCE", "").lower()
    reject_rollback = os.environ.get("LYNN_MTP_K2_REJECT_ROLLBACK", "").lower()
    # M24 diagnostic switch (default off):
    #   LYNN_MTP_K2_COMMIT_REPAIR={hidden, next_pending, hidden_next_pending, state, full_canonical}
    #     On ACCEPT path, run a shadow canonical sequential T=1 chain
    #     (pending then draft) from the pre-K=2 state and use selected
    #     fields from it to repair the K=2 commit result:
    #       hidden               — replace next_base_hidden only, keep K=2 state
    #       next_pending         — replace next_pending_id only, keep K=2 state
    #       hidden_next_pending  — replace both, keep K=2 state (cheap-repair test)
    #       state                — replace state via canonical T=1 chain end-state
    #       full_canonical       — alias for commit_source=canonical_t1 (M23)
    #     Reject path is unchanged. Bisects which K=2 commit component
    #     needs canonical replacement. M24 shrinks the M23 full-canonical
    #     slow path into the minimal repair component that restores 6/6
    #     exact at the cheapest cost.
    commit_repair = os.environ.get("LYNN_MTP_K2_COMMIT_REPAIR", "").lower()

    # Full pre-batch snapshot needed by canonical_t1 shadow T=1 chain,
    # full_state reject rollback, and any commit_repair mode (which all
    # need to re-run a canonical T=1 chain from pre-K=2 state). All share
    # the same snapshot.
    snap_pre_full: dict[str, Any] | None = None
    if (
        accept_source == "canonical_t1"
        or commit_source == "canonical_t1"
        or reject_rollback == "full_state"
        or commit_repair in {"hidden", "next_pending", "hidden_next_pending", "state", "full_canonical"}
    ):
        snap_pre_full = runner._snapshot_state(state)

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
    # historical final-token-only lm_head contract. Per-row native FP4 lm_head
    # is the M11 path that M21 suspects of ULP drift; force BF16 when the
    # diagnostic switch asks.
    if lm_head_mode == "bf16":
        # Direct BF16 F.linear over [B, 2, D] -> [B, 2, V]; bypasses native
        # FP4 quantization + per-row loop entirely.
        logits_k2 = F.linear(h_norm_k2, runner.outside["lm_head.weight"])
    else:
        logits_k2 = runner._lm_head_logits(h_norm_k2, all_positions=True)
    if logits_k2.ndim == 2:
        # Native FP4 lm_head squeezes [B, T, D] → [B*T, V]; un-batch back to [B, T, V].
        logits_k2 = logits_k2.view(h_k2.shape[0], h_k2.shape[1], -1)

    argmax_at_pos0 = int(logits_k2[0, 0].argmax().item())  # base's true prediction for position pending_pos+1
    argmax_at_pos1 = int(logits_k2[0, 1].argmax().item())  # base's prediction for position pending_pos+2

    # M21/M23: optionally derive accept signal from canonical sequential T=1
    # shadow decode of pending. K=2 logits stay reported; only the accept
    # comparison id is swapped unless commit_source=canonical_t1 below.
    t1_h_after_pending: torch.Tensor | None = None
    t1_argmax_after_pending: int | None = None
    t1_h_after_draft: torch.Tensor | None = None
    t1_argmax_after_draft: int | None = None
    if accept_source == "canonical_t1":
        assert snap_pre_full is not None
        # Save post-K=2 state so we can return to it after the shadow chain.
        snap_post_k2 = runner._snapshot_state(state)
        runner._restore_state(state, snap_pre_full)
        t1_h_after_pending, _, t1_argmax_after_pending = decode_one_to_logits_and_hidden(
            runner, state, pending_id,
        )
        if commit_source == "canonical_t1" and draft_id == t1_argmax_after_pending:
            t1_h_after_draft, _, t1_argmax_after_draft = decode_one_to_logits_and_hidden(
                runner, state, draft_id,
            )
        runner._restore_state(state, snap_post_k2)
        accept_signal_id = t1_argmax_after_pending
    else:
        accept_signal_id = argmax_at_pos0

    # 4. Accept rule (P118 K=2 commit math, adapted to batched):
    #    if draft equals base's pos-0 argmax → both tokens valid → commit 2
    #    else → only pending was right at pos 0 → commit 1, rollback
    if draft_id == accept_signal_id:
        # M24: diagnostic commit_repair (opt-in). Each mode runs a shadow
        # canonical T=1 chain to compute the would-be-canonical fields,
        # then selectively replaces the K=2 commit outputs. K=2 forward
        # already ran (state.seq_len += 2; KV[pre, pre+1] written; layer
        # recurrent/conv mutated) — for hidden/next_pending/hidden_next_pending
        # we preserve that state; for state/full_canonical we discard it
        # and use the canonical T=1 chain's state as the post-step state.
        if commit_repair == "full_canonical":
            assert snap_pre_full is not None
            runner._restore_state(state, snap_pre_full)
            h_ap, _, am_ap = decode_one_to_logits_and_hidden(runner, state, pending_id)
            if draft_id == am_ap:
                h_ad, _, am_ad = decode_one_to_logits_and_hidden(runner, state, draft_id)
                return SpeculativeStepResult(
                    committed_tokens=[pending_id, draft_id],
                    next_pending_id=am_ad,
                    next_base_hidden=h_ad,
                    next_pos=state.seq_len - 1,
                    accepted=True,
                    draft_id=draft_id,
                )
            # Canonical T=1 disagrees — fall back to T=1 single-commit.
            return SpeculativeStepResult(
                committed_tokens=[pending_id],
                next_pending_id=am_ap,
                next_base_hidden=h_ap,
                next_pos=state.seq_len - 1,
                accepted=False,
                draft_id=draft_id,
            )

        if commit_repair == "state":
            # K=2's accept signal already passed; replace K=2 state with the
            # canonical T=1 chain end-state. K=2 logits are NOT re-checked
            # (per M24 design — that's what canonical_t1 accept_source is for).
            assert snap_pre_full is not None
            runner._restore_state(state, snap_pre_full)
            decode_one_to_logits_and_hidden(runner, state, pending_id)
            h_ad, _, am_ad = decode_one_to_logits_and_hidden(runner, state, draft_id)
            return SpeculativeStepResult(
                committed_tokens=[pending_id, draft_id],
                next_pending_id=am_ad,
                next_base_hidden=h_ad,
                next_pos=state.seq_len - 1,
                accepted=True,
                draft_id=draft_id,
            )

        if commit_repair in {"hidden", "next_pending", "hidden_next_pending"}:
            # Cheap repair: keep K=2 state (KV writes + recurrent/conv) and
            # only replace selected output fields. Tests whether K=2 state
            # is correct and only the next_pending_id / next_base_hidden
            # extraction is off.
            assert snap_pre_full is not None
            snap_post_k2 = runner._snapshot_state(state)
            runner._restore_state(state, snap_pre_full)
            decode_one_to_logits_and_hidden(runner, state, pending_id)
            h_ad_t1, _, am_ad_t1 = decode_one_to_logits_and_hidden(runner, state, draft_id)
            runner._restore_state(state, snap_post_k2)

            next_pending_repaired = (
                am_ad_t1
                if commit_repair in {"next_pending", "hidden_next_pending"}
                else argmax_at_pos1
            )
            next_base_repaired = (
                h_ad_t1
                if commit_repair in {"hidden", "hidden_next_pending"}
                else h_k2[:, 1:2, :].contiguous()
            )
            return SpeculativeStepResult(
                committed_tokens=[pending_id, draft_id],
                next_pending_id=next_pending_repaired,
                next_base_hidden=next_base_repaired,
                next_pos=state.seq_len - 1,
                accepted=True,
                draft_id=draft_id,
            )

        if commit_source == "canonical_t1":
            assert snap_pre_full is not None
            assert t1_h_after_draft is not None
            assert t1_argmax_after_draft is not None
            # M23 diagnostic: commit the exact state produced by the canonical
            # T=1 chain instead of the K=2 verifier state. This is intentionally
            # slower, but it tells us whether the remaining batched K=2 exact
            # drift lives in accept-path state/hidden commit rather than draft
            # accept-rate math.
            runner._restore_state(state, snap_pre_full)
            h_after_pending, _, argmax_after_pending = decode_one_to_logits_and_hidden(
                runner, state, pending_id,
            )
            if draft_id == argmax_after_pending:
                h_after_draft, _, argmax_after_draft = decode_one_to_logits_and_hidden(
                    runner, state, draft_id,
                )
                return SpeculativeStepResult(
                    committed_tokens=[pending_id, draft_id],
                    next_pending_id=argmax_after_draft,
                    next_base_hidden=h_after_draft,
                    next_pos=state.seq_len - 1,
                    accepted=True,
                    draft_id=draft_id,
                )
            return SpeculativeStepResult(
                committed_tokens=[pending_id],
                next_pending_id=argmax_after_pending,
                next_base_hidden=h_after_pending,
                next_pos=state.seq_len - 1,
                accepted=False,
                draft_id=draft_id,
            )
        return SpeculativeStepResult(
            committed_tokens=[pending_id, draft_id],
            next_pending_id=argmax_at_pos1,
            next_base_hidden=h_k2[:, 1:2, :].contiguous(),
            next_pos=state.seq_len - 1,
            accepted=True,
            draft_id=draft_id,
        )

    # REJECT — restore state to pre-batch, then re-run a single T=1
    # decode of pending alone. KV[pending_pos+1] becomes stale on the
    # lean rollback path but is naturally overwritten on the next K=2
    # forward. M22 opt-in full_state rollback restores KV too — used to
    # bisect whether KV residue from the K=2 forward is the source of
    # the post-M21 exact-match drift.
    if reject_rollback == "full_state":
        assert snap_pre_full is not None
        runner._restore_state(state, snap_pre_full)
    else:
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


def speculative_step_kn_batched(
    runner: Any,
    state: LynnInferenceState,
    pending_id: int,
    pending_base_hidden: torch.Tensor,
    pending_pos: int,
    *,
    draft_count: int,
) -> SpeculativeStepResult:
    """Experimental K=N MTP speculative step.

    The current public Lynn sidecar is a one-token MTP head. For ``draft_count``
    greater than one, this chains that head recursively, feeding each MTP hidden
    into the next draft. That is a research path, not a promoted quality path;
    the useful production contract here is the block verifier/rollback skeleton.
    """
    if draft_count <= 1:
        return speculative_step_k1_batched(
            runner=runner,
            state=state,
            pending_id=pending_id,
            pending_base_hidden=pending_base_hidden,
            pending_pos=pending_pos,
        )

    draft_ids: list[int] = []
    draft_hidden = pending_base_hidden
    current_token = int(pending_id)
    current_pos = int(pending_pos)
    for _idx in range(int(draft_count)):
        mtp_hidden, draft_logits = runner._mtp_draft_hidden_logits(
            base_hidden=draft_hidden,
            current_token_id=current_token,
            current_pos=current_pos,
        )
        current_token = int(draft_logits[0].argmax().item())
        draft_ids.append(current_token)
        draft_hidden = mtp_hidden.contiguous()
        current_pos += 1

    tokens = [int(pending_id), *draft_ids]
    snap_pre_batch = snapshot_recurrent_conv(state)
    h_block, _logits_block, argmax_ids = decode_block_to_logits_and_hidden(runner, state, tokens)

    accepted_count = 0
    for idx, draft_id in enumerate(draft_ids):
        if int(draft_id) != int(argmax_ids[idx]):
            break
        accepted_count += 1

    commit_count = 1 + accepted_count
    committed = tokens[:commit_count]

    # Conservative repair path: restore pre-block recurrent/conv state and
    # replay the committed prefix through canonical T=1 decode. Earlier K=2
    # work showed accepted-state drift can survive even when token ids match;
    # K>N promotion should optimize this only after exactness is proven.
    restore_recurrent_conv(state, snap_pre_batch)
    h_last: torch.Tensor | None = None
    next_pending = int(argmax_ids[0])
    for token in committed:
        h_last, _logits, next_pending = decode_one_to_logits_and_hidden(runner, state, int(token))
    assert h_last is not None
    return SpeculativeStepResult(
        committed_tokens=committed,
        next_pending_id=next_pending,
        next_base_hidden=h_last,
        next_pos=state.seq_len - 1,
        accepted=False,
        draft_id=draft_ids[0],
        draft_ids=draft_ids,
        accepted_count=accepted_count,
    )
