#!/usr/bin/env python3
"""M16 bisect probe for Qwen3.6 MTP K=2 batched verifier drift.

M15 found the narrowest repro: event-5 with advance tokens
[71, 248068, 198, 8160, 579, 264, 7047, 1817] and pending/draft [25, 271],
first bad at linear-attention layer 32 position 1, K2-first mode.

M16 bisects:
  1. Advance-token count (0..8) — find the minimum advance that triggers drift.
  2. Layer range — confirm layer 32 is the first bad layer and layer 38
     conv-state is accumulated drift.
  3. Position — confirm pos1 only (pos0 stays exact).

This is a standalone diagnostic probe, not a promotion gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402

# Reuse env and helpers from the existing probe.
import spark_mtp_k2_vs_t1_diff_probe as base_probe  # noqa: E402

_set_env = base_probe._set_env
_restore_env = base_probe._restore_env
BASE_ENV = base_probe.BASE_ENV
_cmp = base_probe._cmp
_argmax_id = base_probe._argmax_id

# Event-5 advance token sequence from M15 trace.
EVENT5_ADVANCE = [71, 248068, 198, 8160, 579, 264, 7047, 1817]
EVENT5_PENDING = 25
EVENT5_DRAFT = 271


def _run_one_advance_bisect(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    prefill_hidden: torch.Tensor,
    prefill_ids: torch.Tensor,
    advance_ids: list[int],
    pending_id: int,
    draft_id: int,
    drift_max_abs: float,
    drift_cos_min: float,
) -> dict[str, Any]:
    """Advance by *advance_ids*, then compare K2-first vs sequential at [pending, draft]."""
    T = int(prefill_ids.shape[1])
    h = prefill_hidden.clone()
    seq_len = T

    # Advance loop — canonical T=1 decode.
    for token_id in advance_ids:
        token = torch.tensor([[int(token_id)]], device=runner.device, dtype=torch.long)
        pos_one = torch.tensor([[seq_len]], device=runner.device, dtype=torch.long)
        h_tok = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        for i in range(runner.n_layers):
            h_tok = runner._decode_layer_fast(h_tok, pos_one, state, i)
        state.seq_len += 1
        seq_len += 1
        h = h_tok

    pre_snap = runner._snapshot_state(state)

    # --- K2-first ---
    runner._restore_state(state, pre_snap)
    tokens_k2 = torch.tensor([[pending_id, draft_id]], device=runner.device, dtype=torch.long)
    pos_k2 = torch.tensor(
        [[int(pre_snap["seq_len"]), int(pre_snap["seq_len"]) + 1]],
        device=runner.device,
        dtype=torch.long,
    )
    hk = F.embedding(tokens_k2, runner.outside["model.language_model.embed_tokens.weight"])
    k2_layers: list[torch.Tensor] = []
    for layer_idx in range(runner.n_layers):
        hk = runner._decode_layer_k2_fast(hk, pos_k2, state, layer_idx)
        k2_layers.append(hk.detach().clone())
    state.seq_len += 2
    k2_logits = runner._lm_head_logits(
        _rms_norm(hk, runner.outside["model.language_model.norm.weight"]),
        all_positions=True,
    )
    k2_snap = runner._snapshot_state(state)

    # --- Sequential ---
    runner._restore_state(state, pre_snap)
    seq_layers_0: list[torch.Tensor] = []
    seq_layers_1: list[torch.Tensor] = []
    # pos0
    t0 = torch.tensor([[pending_id]], device=runner.device, dtype=torch.long)
    p0 = torch.tensor([[int(pre_snap["seq_len"])]], device=runner.device, dtype=torch.long)
    h0 = F.embedding(t0, runner.outside["model.language_model.embed_tokens.weight"])
    for i in range(runner.n_layers):
        h0 = runner._decode_layer_fast(h0, p0, state, i)
        seq_layers_0.append(h0.detach().clone())
    state.seq_len += 1
    # pos1
    t1 = torch.tensor([[draft_id]], device=runner.device, dtype=torch.long)
    p1 = torch.tensor([[int(pre_snap["seq_len"]) + 1]], device=runner.device, dtype=torch.long)
    h1 = F.embedding(t1, runner.outside["model.language_model.embed_tokens.weight"])
    for i in range(runner.n_layers):
        h1 = runner._decode_layer_fast(h1, p1, state, i)
        seq_layers_1.append(h1.detach().clone())
    state.seq_len += 1
    seq_hidden = torch.cat([h0, h1], dim=1)
    seq_logits = runner._lm_head_logits(
        _rms_norm(seq_hidden, runner.outside["model.language_model.norm.weight"]),
        all_positions=True,
    )
    seq_snap = runner._snapshot_state(state)

    # --- Per-layer diff ---
    first_bad: dict[str, Any] | None = None
    layer_rows: list[dict[str, Any]] = []
    for i, hk2_layer in enumerate(k2_layers):
        c0 = _cmp(seq_layers_0[i], hk2_layer[:, 0:1, :])
        c1 = _cmp(seq_layers_1[i], hk2_layer[:, 1:2, :])
        row = {"layer": i, "layer_type": runner.layer_types[i], "pos0": c0, "pos1": c1}
        layer_rows.append(row)
        if first_bad is None:
            bad0 = c0["max_abs"] > drift_max_abs or c0["cosine"] < drift_cos_min
            bad1 = c1["max_abs"] > drift_max_abs or c1["cosine"] < drift_cos_min
            if bad0 or bad1:
                first_bad = {"layer": i, "layer_type": runner.layer_types[i],
                             "pos0_bad": bad0, "pos1_bad": bad1, "pos0": c0, "pos1": c1}

    # State diff
    state_rows: list[dict[str, Any]] = []
    worst_state: dict[str, Any] | None = None
    for kind, snap_a, snap_b in [("recurrent", seq_snap.get("recurrent", {}), k2_snap.get("recurrent", {})),
                                  ("conv", seq_snap.get("conv", {}), k2_snap.get("conv", {}))]:
        for layer, tensor_a in snap_a.items():
            tensor_b = snap_b.get(layer)
            if tensor_b is None:
                continue
            c = _cmp(tensor_a, tensor_b)
            row = {"kind": kind, "layer": int(layer), **c}
            state_rows.append(row)
            if worst_state is None or row["max_abs"] > worst_state["max_abs"]:
                worst_state = row

    return {
        "n_advance": len(advance_ids),
        "advance_ids": advance_ids,
        "pending_id": pending_id,
        "draft_id": draft_id,
        "first_bad_layer": first_bad,
        "worst_state": worst_state,
        "logits_pos0": _cmp(seq_logits[:, 0, :], k2_logits[:, 0, :]),
        "logits_pos1": _cmp(seq_logits[:, 1, :], k2_logits[:, 1, :]),
        "k2_argmax_pos0": _argmax_id(k2_logits[:, 0, :]),
        "k2_argmax_pos1": _argmax_id(k2_logits[:, 1, :]),
        "seq_argmax_pos0": _argmax_id(seq_logits[:, 0, :]),
        "seq_argmax_pos1": _argmax_id(seq_logits[:, 1, :]),
        "state_worst": worst_state,
        "drift_detected": first_bad is not None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.")
    ap.add_argument("--advance-tokens", type=int, nargs="*", default=None,
                     help="Override advance token IDs (default: event-5 sequence).")
    ap.add_argument("--pending-id", type=int, default=EVENT5_PENDING)
    ap.add_argument("--draft-id", type=int, default=EVENT5_DRAFT)
    ap.add_argument("--bisect-steps", type=int, nargs="*", default=None,
                     help="Advance-count steps to test (default: 0,1,2,4,8).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--drift-max-abs", type=float, default=1e-3)
    ap.add_argument("--drift-cos-min", type=float, default=0.99999)
    args = ap.parse_args()

    advance_tokens = args.advance_tokens or EVENT5_ADVANCE
    bisect_steps = args.bisect_steps or [0, 1, 2, 4, len(advance_tokens)]

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    env = dict(BASE_ENV)
    env["LYNN_MTP_SIDECAR"] = args.sidecar
    prev = _set_env(env)
    try:
        t0 = time.time()
        runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=False)
        if not runner.mtp_sidecar_loaded:
            raise SystemExit(f"MTP sidecar did not load: {args.sidecar}")

        tok = runner.tokenizer
        ids = _encode_prompt(tok, args.prompt, runner.device, use_chat_template=False)
        T = int(ids.shape[1])
        state = LynnInferenceState.from_config(
            runner.cfg, batch=1, max_seq_len=runner.max_seq_len,
            device=runner.device, dtype=runner.dtype,
        )

        h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
        pos = torch.arange(T, device=runner.device, dtype=torch.long).unsqueeze(0)
        for i in range(runner.n_layers):
            h = _prefill_layer(h, pos, runner.layer_types[i], runner.layer_weights[i],
                               runner.layer_cfgs[i], state, i)
        state.seq_len = T

        # Bisect: test increasing advance-token counts.
        results: list[dict[str, Any]] = []
        first_drift_n: int | None = None

        for n_adv in sorted(set(bisect_steps)):
            if n_adv > len(advance_tokens):
                continue
            subset = advance_tokens[:n_adv]
            # Fresh state snapshot for each test.
            state_test = LynnInferenceState.from_config(
                runner.cfg, batch=1, max_seq_len=runner.max_seq_len,
                device=runner.device, dtype=runner.dtype,
            )
            h_test = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
            for i in range(runner.n_layers):
                h_test = _prefill_layer(h_test, pos, runner.layer_types[i], runner.layer_weights[i],
                                        runner.layer_cfgs[i], state_test, i)
            state_test.seq_len = T

            result = _run_one_advance_bisect(
                runner, state_test, h_test, ids,
                advance_ids=subset,
                pending_id=args.pending_id,
                draft_id=args.draft_id,
                drift_max_abs=args.drift_max_abs,
                drift_cos_min=args.drift_cos_min,
            )
            results.append(result)
            status = "DRIFT" if result["drift_detected"] else "EXACT"
            fb = result["first_bad_layer"]
            fb_info = f" layer={fb['layer']}({fb['layer_type']})" if fb else ""
            print(f"[m16-bisect] n_advance={n_adv}: {status}{fb_info}")
            if result["drift_detected"] and first_drift_n is None:
                first_drift_n = n_adv

        report = {
            "schema_version": "lynn-mtp-m16-k2-bisect-probe-v1",
            "model": args.model,
            "sidecar": args.sidecar,
            "prompt": args.prompt,
            "prompt_len": T,
            "advance_tokens_full": advance_tokens,
            "pending_id": args.pending_id,
            "draft_id": args.draft_id,
            "bisect_steps": bisect_steps,
            "drift_threshold": {"max_abs": args.drift_max_abs, "cos_min": args.drift_cos_min},
            "first_drift_at_n_advance": first_drift_n,
            "results": results,
            "elapsed_seconds": time.time() - t0,
        }

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[m16-bisect] wrote {out}")
        if first_drift_n is not None:
            print(f"[m16-bisect] first drift at n_advance={first_drift_n} — minimum reproducible advance count")
        else:
            print("[m16-bisect] no drift detected at any bisection point")
        return 0
    finally:
        _restore_env(prev)


if __name__ == "__main__":
    raise SystemExit(main())
