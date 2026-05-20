#!/usr/bin/env python3
"""M18 layer-type sweep probe for Qwen3.6 MTP K=2 batched verifier.

M17 added the t1_canonical correctness oracle (full sequential T=1). User's
next direction: "按层打开 K2 batching，找出 full-attn / linear-attn 哪一段能
安全批量化" — find which layer type can be batched without drift.

This probe sweeps 4 mode combinations on the M16 bisect setup:

  combo                full-attn       linear-attn      meaning
  -----                ---------       -----------      -------
  k2_both              k2 (SDPA)       k2 (internal     current default; M16
                                       per-position)    showed drift at zero advance
  t1_full              t1_loop (2xT=1) k2 (internal)    M13 setup; t1_loop
                                                        opt-in for full-attn only
  t1_linear (NEW)      k2 (SDPA)       t1_loop (2xT=1   new opt-in via
                                       + state.update   LYNN_MTP_K2_LINEAR_ATTN_MODE
                                       interleaved)
  t1_both              t1_loop         t1_loop          per-layer t1_canonical
                                                        equivalent (state is
                                                        updated per layer mid-block)

For each combination, run the M16 bisect with zero advance tokens (smallest
repro from M16) and check first_bad_layer. Combinations with first_bad_layer
== None can be promoted as "safe" for further work.

Standalone diagnostic probe, not a promotion gate.
"""
from __future__ import annotations

import argparse
import json
import os
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

# Reuse helpers from the M16 bisect probe.
import spark_mtp_m16_k2_bisect_probe as bisect_probe  # noqa: E402
import spark_mtp_k2_vs_t1_diff_probe as base_probe  # noqa: E402

_set_env = base_probe._set_env
_restore_env = base_probe._restore_env
BASE_ENV = base_probe.BASE_ENV
_cmp = base_probe._cmp
_run_one_advance_bisect = bisect_probe._run_one_advance_bisect


COMBINATIONS = [
    {
        "name": "k2_both",
        "full_attn": "k2",
        "linear_attn": "k2",
        "description": "Current default. Full-attn decode_full_attn_k2 (SDPA over [B,2,...]); linear-attn decode_linear_attn_k2 (per-position internal).",
    },
    {
        "name": "t1_full_attn_only",
        "full_attn": "t1_loop",
        "linear_attn": "k2",
        "description": "M13 setup. Full-attn falls back to 2x decode_full_attn (per-position); linear-attn keeps K=2 internal per-position.",
    },
    {
        "name": "t1_linear_attn_only",
        "full_attn": "k2",
        "linear_attn": "t1_loop",
        "description": "NEW M18 mode. Full-attn keeps K=2 SDPA; linear-attn falls back to 2x decode_linear_attn with state.update_linear_attn_state interleaved (mirrors sequential exactly).",
    },
    {
        "name": "t1_both",
        "full_attn": "t1_loop",
        "linear_attn": "t1_loop",
        "description": "Per-layer t1_canonical equivalent. Both full-attn and linear-attn use T=1 split with state advance between positions.",
    },
]


def _apply_mode_env(full_attn_mode: str, linear_attn_mode: str) -> dict[str, str | None]:
    """Set the two K=2 layer-type mode env vars; return previous values for restore."""
    return _set_env({
        "LYNN_FULL_ATTN_K2_BACKEND": full_attn_mode,
        "LYNN_MTP_K2_LINEAR_ATTN_MODE": linear_attn_mode,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=bisect_probe.EVENT5_ADVANCE and
                    "Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.")
    ap.add_argument("--pending-id", type=int, default=bisect_probe.EVENT5_PENDING)
    ap.add_argument("--draft-id", type=int, default=bisect_probe.EVENT5_DRAFT)
    ap.add_argument(
        "--advance",
        type=int,
        default=0,
        help="Advance-token count to test (default 0 — M16 minimum reproducible).",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--drift-max-abs", type=float, default=1e-3)
    ap.add_argument("--drift-cos-min", type=float, default=0.99999)
    args = ap.parse_args()

    advance_tokens = bisect_probe.EVENT5_ADVANCE[:args.advance]

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    env = dict(BASE_ENV)
    env["LYNN_MTP_SIDECAR"] = args.sidecar
    prev_env = _set_env(env)
    try:
        t0 = time.time()
        runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=False)
        if not runner.mtp_sidecar_loaded:
            raise SystemExit(f"MTP sidecar did not load: {args.sidecar}")

        tok = runner.tokenizer
        ids = _encode_prompt(tok, args.prompt, runner.device, use_chat_template=False)
        T = int(ids.shape[1])

        # Prefill once into a baseline state we will clone for each combo.
        base_state = LynnInferenceState.from_config(
            runner.cfg, batch=1, max_seq_len=runner.max_seq_len,
            device=runner.device, dtype=runner.dtype,
        )
        h_prefill = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
        pos = torch.arange(T, device=runner.device, dtype=torch.long).unsqueeze(0)
        for i in range(runner.n_layers):
            h_prefill = _prefill_layer(
                h_prefill, pos, runner.layer_types[i], runner.layer_weights[i],
                runner.layer_cfgs[i], base_state, i,
            )
        base_state.seq_len = T

        combo_results: list[dict[str, Any]] = []
        for combo in COMBINATIONS:
            # Each combo gets a fresh state derived from a clean re-prefill — the
            # M16 bisect helper mutates state through advance + K=2 + sequential
            # paths, so isolation across combos requires independent state per
            # combo. (Cheaper than reloading the model.)
            state_combo = LynnInferenceState.from_config(
                runner.cfg, batch=1, max_seq_len=runner.max_seq_len,
                device=runner.device, dtype=runner.dtype,
            )
            h_combo = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
            for i in range(runner.n_layers):
                h_combo = _prefill_layer(
                    h_combo, pos, runner.layer_types[i], runner.layer_weights[i],
                    runner.layer_cfgs[i], state_combo, i,
                )
            state_combo.seq_len = T

            prev_mode = _apply_mode_env(combo["full_attn"], combo["linear_attn"])
            try:
                result = _run_one_advance_bisect(
                    runner, state_combo, h_combo, ids,
                    advance_ids=list(advance_tokens),
                    pending_id=args.pending_id,
                    draft_id=args.draft_id,
                    drift_max_abs=args.drift_max_abs,
                    drift_cos_min=args.drift_cos_min,
                )
            finally:
                _restore_env(prev_mode)

            status = "DRIFT" if result["drift_detected"] else "EXACT"
            fb = result["first_bad_layer"]
            fb_info = f" first_bad=L{fb['layer']}({fb['layer_type']})" if fb else ""
            print(f"[m18-sweep] {combo['name']:24s} full={combo['full_attn']:8s} linear={combo['linear_attn']:8s} → {status}{fb_info}", flush=True)

            combo_results.append({
                "name": combo["name"],
                "full_attn_mode": combo["full_attn"],
                "linear_attn_mode": combo["linear_attn"],
                "description": combo["description"],
                "result": result,
            })

        # Summarize "safe" combos (no drift detected).
        safe = [c["name"] for c in combo_results if not c["result"]["drift_detected"]]

        report = {
            "schema_version": "lynn-mtp-m18-k2-layer-sweep-probe-v1",
            "model": args.model,
            "sidecar": args.sidecar,
            "prompt": args.prompt,
            "advance_token_count": args.advance,
            "advance_tokens": list(advance_tokens),
            "pending_id": args.pending_id,
            "draft_id": args.draft_id,
            "drift_threshold": {"max_abs": args.drift_max_abs, "cos_min": args.drift_cos_min},
            "combinations": combo_results,
            "safe_combinations": safe,
            "elapsed_seconds": time.time() - t0,
        }

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[m18-sweep] wrote {out}")
        print(f"[m18-sweep] safe combinations: {safe if safe else '(none — all drift)'}")
        return 0
    finally:
        _restore_env(prev_env)


if __name__ == "__main__":
    raise SystemExit(main())
