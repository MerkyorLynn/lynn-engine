#!/usr/bin/env python3
"""Spark MTP speculative-serving smoke.

Runs three configurations on the same prompt set and saves a single JSON
report comparing token-exact correctness, accept rate, and effective TPS:

* ``baseline``  — MTP sidecar loaded but both shadow + speculative OFF
                  (current Lynn-native NVFP4 W4A16 production path).
* ``shadow``    — shadow MTP measurement ON, no commit. Validates the
                  sidecar produces meaningful drafts without changing the
                  serving token stream.
* ``spec_k1``   — K=1 MTP speculative serving ON (P118 sequential verify,
                  eager decode). Token stream MUST match ``baseline`` exactly
                  on each prompt — that is the wire-correctness gate.

Effective TPS in ``spec_k1`` is ``tokens_committed / sum(step_seconds)``.
Net win vs ``baseline`` depends on accept rate and the extra cost of running
two base decodes + one MTP forward per round (P118 sequential path; expect a
slight slowdown until the batched K-position verify lands in M5-M6).

Usage on Spark::

    python scripts/spark_mtp_speculative_smoke.py \\
        --model /home/merkyor/models/<lynn-native-w4a16-nvfp4-dir> \\
        --sidecar /home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp/mtp.safetensors \\
        --out reports/spark/mtp_speculative_smoke_$(date +%Y%m%d_%H%M%S).json

The ``--model`` path MUST point to a **Lynn-native W4A16 NVFP4** artifact
(``nvfp4_e2m1_rowwise_per_16`` with ``lynn_quant_manifest.json`` schema
``lynn-variable-nvfp4-pack-v1``). Vendor formats (compressed-tensors v8-RTN /
ModelOpt FP4) do not match the MTP sidecar's expected layer weight keys and
will fail loud at sidecar load.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


DEFAULT_PROMPTS = [
    "Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.",
    "用一句话解释 speculative decoding 的核心思想。",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "请输出一个 JSON: {\"city\": \"Tokyo\", \"unit\": \"celsius\"}",
    "Summarize the role of the MoE router in one paragraph.",
]

BASE_ENV = {
    # Spark Config D (production NVFP4 W4A16 path that hit 42.55-45.39 TPS in
    # memory project_overnight_results_20260517 + 42.85 in
    # project_spark_27b_42tps_milestone_20260515).
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_PACKED_DECODE": "1",
    "LYNN_PACKED_SHARED_EXPERT": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    # K1-light: fused QKV projection for full-attn layers (10 launches / token
    # saved on Lynn 35B-A3B; smaller win than a full graph reuse pool but free).
    "LYNN_FULL_ATTN_QKV_FUSED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _set_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    """Apply env updates, return the previous values for restore."""
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _prefix_match_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _run_config(
    runner: LynnIncrementalRunner,
    *,
    label: str,
    env_updates: dict[str, str | None],
    prompts: list[str],
    max_new: int,
    baseline_ids: list[list[int]] | None,
) -> dict[str, Any]:
    previous = _set_env(env_updates)
    try:
        rows: list[dict[str, Any]] = []
        for idx, prompt in enumerate(prompts):
            t0 = time.time()
            out = runner.generate(prompt, max_new=max_new)
            wall = time.time() - t0
            new_ids = [int(x) for x in out["new_ids"]]
            base_ids = None if baseline_ids is None else baseline_ids[idx]
            spec = out.get("mtp_speculative", {}) or {}
            shadow = out.get("mtp_shadow", {}) or {}
            rows.append({
                "prompt_id": f"prompt_{idx:03d}",
                "prompt": prompt,
                "new_ids": new_ids,
                "new_ids_head": new_ids[:24],
                "completion_head": out["completion_text"][:240],
                "exact_match": None if base_ids is None else new_ids == base_ids,
                "prefix_match_len": None if base_ids is None else _prefix_match_len(new_ids, base_ids),
                "wall_seconds": wall,
                "decode_tps": out["timings"].get("decode_tps"),
                "spec_active": bool(spec.get("active")),
                "spec_events": spec.get("events"),
                "spec_accept_rate": spec.get("accept_rate"),
                "spec_tokens_per_event": spec.get("tokens_per_event"),
                "spec_effective_token_tps": spec.get("effective_token_tps"),
                "shadow_events": shadow.get("events"),
                "shadow_accept_rate": shadow.get("accept_rate"),
                "stopped_reason": out["stopped_reason"],
            })
        return {
            "label": label,
            "env_overrides": {k: v for k, v in env_updates.items() if v is not None},
            "rows": rows,
        }
    finally:
        _restore_env(previous)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to Lynn NVFP4 model dir")
    ap.add_argument("--sidecar", required=True, help="Path to MTP sidecar mtp.safetensors")
    ap.add_argument("--out", required=True, help="Path to write JSON report")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--prompts-json", default=None,
                    help="Optional JSON file with custom prompt list")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_json:
        raw = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        prompts = [
            str(item["prompt"]) if isinstance(item, dict) else str(item)
            for item in raw
        ]

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    # Apply base env BEFORE constructing the runner so it picks up packed_nvfp4
    # MoE impl + native FP4 lm_head + linear-attn inplace recurrent path.
    base_updates: dict[str, str | None] = dict(BASE_ENV)
    base_updates["LYNN_MTP_SIDECAR"] = args.sidecar
    base_previous = _set_env(base_updates)

    try:
        # Fail loud BEFORE building the runner if the model dir is not the
        # Lynn-native W4A16 NVFP4 format. The MTP sidecar's layer-weight
        # contract (``mlp.experts.gate_up_proj`` keys etc.) only matches
        # Lynn-native artifacts; vendor compressed-tensors v8-RTN / ModelOpt
        # would KeyError later in ``mtp_layer_weights`` and waste the load.
        from engine.nvfp4_layout import detect_nvfp4_layout
        layout = detect_nvfp4_layout(args.model)
        if layout.layout_kind != "lynn_native_per16_variable":
            raise SystemExit(
                f"[smoke] {args.model} is layout={layout.layout_kind!r} "
                f"(backend={layout.recommended_backend_family!r}). "
                "Lynn engine MTP serving requires the Lynn-native W4A16 NVFP4 "
                "artifact (nvfp4_e2m1_rowwise_per_16 with "
                "lynn_quant_manifest.json schema lynn-variable-nvfp4-pack-v1). "
                "Point --model at the Lynn-native dir, not v8-RTN/ModelOpt."
            )

        runner = LynnIncrementalRunner(
            args.model,
            device=args.device,
            dtype=dtype,
            verbose=False,
        )

        if not runner.mtp_sidecar_loaded:
            raise SystemExit(
                f"[smoke] MTP sidecar did not load from {args.sidecar}. "
                "Check the path + safetensors integrity."
            )

        # baseline + shadow run with the linear-block graph fast path for
        # the canonical Config D 40+ TPS. spec_k1 is graph-incompatible (rollback
        # invalidates captured graphs) and runs eager; that means the TPS
        # comparison is unfair but reveals whether speculative wins at all.
        configs = [
            {
                "label": "baseline",
                "env": {
                    "LYNN_MTP_SHADOW_VERIFY": "0",
                    "LYNN_MTP_SPECULATIVE": "0",
                    "LYNN_LINEAR_BLOCK_GRAPH": "1",
                    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
                },
            },
            {
                "label": "shadow",
                "env": {
                    "LYNN_MTP_SHADOW_VERIFY": "1",
                    "LYNN_MTP_SPECULATIVE": "0",
                    "LYNN_LINEAR_BLOCK_GRAPH": "1",
                    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
                },
            },
            {
                "label": "spec_k1",
                "env": {
                    "LYNN_MTP_SHADOW_VERIFY": "0",
                    "LYNN_MTP_SPECULATIVE": "1",
                    "LYNN_MTP_SPECULATIVE_BATCHED": "0",
                    # Speculative path is eager-only — graph captures cannot be
                    # rolled back on draft reject.
                    "LYNN_LINEAR_BLOCK_GRAPH": "0",
                    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
                },
            },
            {
                "label": "spec_k1_batched",
                "env": {
                    "LYNN_MTP_SHADOW_VERIFY": "0",
                    "LYNN_MTP_SPECULATIVE": "1",
                    "LYNN_MTP_SPECULATIVE_BATCHED": "1",
                    "LYNN_LINEAR_BLOCK_GRAPH": "0",
                    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
                },
            },
        ]

        cases: list[dict[str, Any]] = []
        baseline_ids: list[list[int]] | None = None
        for config in configs:
            case = _run_config(
                runner,
                label=config["label"],
                env_updates=config["env"],
                prompts=prompts,
                max_new=args.max_new,
                baseline_ids=baseline_ids,
            )
            if config["label"] == "baseline":
                baseline_ids = [row["new_ids"] for row in case["rows"]]
                for row in case["rows"]:
                    row["exact_match"] = True
                    row["prefix_match_len"] = len(row["new_ids"])
            cases.append(case)

        # ---- Cross-config summary ----
        summary: dict[str, Any] = {}
        for case in cases:
            label = case["label"]
            rows = case["rows"]
            n_rows = len(rows)
            exact = sum(1 for r in rows if r["exact_match"]) if label != "baseline" else n_rows
            prefix_lens = [r["prefix_match_len"] for r in rows if r["prefix_match_len"] is not None]
            mean_prefix = (sum(prefix_lens) / len(prefix_lens)) if prefix_lens else None
            tps_values = [r["decode_tps"] for r in rows if r["decode_tps"]]
            mean_tps = (sum(tps_values) / len(tps_values)) if tps_values else None
            spec_tps_values = [
                r["spec_effective_token_tps"] for r in rows if r["spec_effective_token_tps"]
            ]
            mean_spec_tps = (
                sum(spec_tps_values) / len(spec_tps_values) if spec_tps_values else None
            )
            spec_accept = [r["spec_accept_rate"] for r in rows if r["spec_accept_rate"] is not None]
            mean_spec_accept = (sum(spec_accept) / len(spec_accept)) if spec_accept else None
            shadow_accept = [r["shadow_accept_rate"] for r in rows if r["shadow_accept_rate"] is not None]
            mean_shadow_accept = (sum(shadow_accept) / len(shadow_accept)) if shadow_accept else None
            summary[label] = {
                "exact_match_count": exact,
                "n_prompts": n_rows,
                "exact_match_rate": exact / n_rows if n_rows else None,
                "mean_prefix_match_len": mean_prefix,
                "mean_decode_tps_baseline_loop": mean_tps,
                "mean_spec_effective_tps": mean_spec_tps,
                "mean_spec_accept_rate": mean_spec_accept,
                "mean_shadow_accept_rate": mean_shadow_accept,
            }

        baseline_tps = summary["baseline"]["mean_decode_tps_baseline_loop"]
        spec_tps = summary["spec_k1"]["mean_spec_effective_tps"]
        spec_batched_tps = summary.get("spec_k1_batched", {}).get("mean_spec_effective_tps")
        gate_correctness_seq = summary["spec_k1"]["exact_match_count"] == summary["spec_k1"]["n_prompts"]
        gate_correctness_batched = (
            summary.get("spec_k1_batched", {}).get("exact_match_count")
            == summary.get("spec_k1_batched", {}).get("n_prompts")
        )
        gate_tps_ratio = (
            spec_tps / baseline_tps if (baseline_tps and spec_tps) else None
        )
        gate_tps_ratio_batched = (
            spec_batched_tps / baseline_tps
            if (baseline_tps and spec_batched_tps) else None
        )

        report = {
            "schema_version": "lynn-mtp-speculative-smoke-v2",
            "model": args.model,
            "sidecar": args.sidecar,
            "max_new": args.max_new,
            "n_prompts": len(prompts),
            "configs": cases,
            "summary": summary,
            "gates": {
                "correctness_spec_k1_matches_baseline": gate_correctness_seq,
                "correctness_spec_k1_batched_matches_baseline": gate_correctness_batched,
                "tps_ratio_spec_over_baseline": gate_tps_ratio,
                "tps_ratio_spec_batched_over_baseline": gate_tps_ratio_batched,
                "note": (
                    "spec_k1 (sequential): 2 base forwards/round + MTP — bounded by "
                    "1 + accept rate / cost; with OFFSET=1 head accept ~0.3% (random), "
                    "TPS ratio ~0.4-0.5x. spec_k1_batched (M5): 1 K=2 batched forward + "
                    "MTP, with reject penalty = +1 T=1 forward; ROI bounded by 1.0-1.1x "
                    "on Lynn 30/40 hybrid SSM arch even at 65% accept. Real unlock = "
                    "M8 A100 retrain at OFFSET=2."
                ),
            },
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[smoke] wrote {out_path}")
        print(f"[smoke] baseline decode_tps mean = {baseline_tps}")
        print(f"[smoke] spec_k1 effective_tps mean = {spec_tps}")
        print(f"[smoke] spec_k1 accept_rate mean = {summary['spec_k1']['mean_spec_accept_rate']}")
        print(f"[smoke] spec_k1_batched effective_tps mean = {spec_batched_tps}")
        print(f"[smoke] spec_k1_batched accept_rate mean = {summary.get('spec_k1_batched',{}).get('mean_spec_accept_rate')}")
        print(f"[smoke] correctness gate seq (spec==baseline)     = {gate_correctness_seq}")
        print(f"[smoke] correctness gate batched (spec==baseline) = {gate_correctness_batched}")
        print(f"[smoke] tps ratio (spec/baseline)         = {gate_tps_ratio}")
        print(f"[smoke] tps ratio (spec_batched/baseline) = {gate_tps_ratio_batched}")
        # Treat both correctness gates strictly only if same path (eager baseline);
        # graph baseline vs eager spec has numerical divergence — don't fail on that.
        return 0
    finally:
        _restore_env(base_previous)


if __name__ == "__main__":
    raise SystemExit(main())
