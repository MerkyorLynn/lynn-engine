#!/usr/bin/env python3
"""M21 residual-exact probe — bisect the post-M19/M20 2/6 exact gap.

After M18 proved that layer-level K=2 forward is bit-exact equivalent
to two sequential T=1 forwards when ``LYNN_FULL_ATTN_K2_BACKEND=t1_loop``
is set, M19/M20 smoke still showed ``spec_k1_batched`` exact_match 2/6
and effective TPS below baseline. The remaining bug must live in:

  (B) the K=2 lm_head dispatch  → ``LYNN_MTP_K2_LM_HEAD_MODE=bf16`` switch
  (A) the K=2 accept argmax     → ``LYNN_MTP_K2_ACCEPT_SOURCE=canonical_t1`` switch
  (S) state-commit / new_ids emission → neither switch helps

This probe runs the four bisect configs against the canonical smoke
prompt set on a single resident runner (one model load) and captures
per-event divergence trace versus baseline greedy decode. The output
JSON drives an immediate root-cause verdict per the M21 acceptance
rules.

Usage on Spark::

    /home/merkyor/comfyui/ComfyUI/.venv/bin/python -u \
        scripts/spark_mtp_m21_residual_exact_probe.py \
        --model /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
        --sidecar /home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
        --out /tmp/mtp_m21_residual_exact_$(date +%Y%m%d_%H%M%S).json

The script does NOT modify production defaults; it only toggles
opt-in env vars per config inside a single process.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_ENV: dict[str, str] = {
    # Spark Config D production env (matches spark_mtp_speculative_smoke).
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_PACKED_DECODE": "1",
    "LYNN_PACKED_SHARED_EXPERT": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_FULL_ATTN_QKV_FUSED": "1",
    # M18: full-attn K=2 t1_loop fallback already known necessary at layer level.
    "LYNN_FULL_ATTN_K2_BACKEND": "t1_loop",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


DEFAULT_PROMPTS = [
    "Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.",
    "用一句话解释 speculative decoding 的核心思想。",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "请输出一个 JSON: {\"city\": \"Tokyo\", \"unit\": \"celsius\"}",
    "Summarize the role of the MoE router in one paragraph.",
]


# Each config is (label, per-config env overrides applied on top of BASE_ENV).
# Empty dict means "use BASE_ENV alone with batched speculative on" (the
# post-M19/M20 baseline that's known to give 2/6 exact).
CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("baseline_greedy", {
        "LYNN_MTP_SPECULATIVE": "0",
        "LYNN_LINEAR_BLOCK_GRAPH": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
    }),
    ("spec_k1_batched_default", {
        "LYNN_MTP_SPECULATIVE": "1",
        "LYNN_MTP_SPECULATIVE_BATCHED": "1",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
        # No additional knobs — replicates M19 / M20.
    }),
    ("spec_k1_batched_lmhead_bf16", {
        "LYNN_MTP_SPECULATIVE": "1",
        "LYNN_MTP_SPECULATIVE_BATCHED": "1",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
        "LYNN_MTP_K2_LM_HEAD_MODE": "bf16",
    }),
    ("spec_k1_batched_canonical_t1_accept", {
        "LYNN_MTP_SPECULATIVE": "1",
        "LYNN_MTP_SPECULATIVE_BATCHED": "1",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
        "LYNN_MTP_K2_ACCEPT_SOURCE": "canonical_t1",
    }),
    ("spec_k1_batched_both_switches", {
        "LYNN_MTP_SPECULATIVE": "1",
        "LYNN_MTP_SPECULATIVE_BATCHED": "1",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
        "LYNN_MTP_K2_LM_HEAD_MODE": "bf16",
        "LYNN_MTP_K2_ACCEPT_SOURCE": "canonical_t1",
    }),
]


def _set_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    prev: dict[str, str | None] = {}
    for key, value in updates.items():
        prev[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    return prev


def _restore_env(prev: dict[str, str | None]) -> None:
    for key, value in prev.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _prefix_match(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _build_event_trace(
    spec_ids: list[int],
    baseline_ids: list[int],
    drafts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Walk per-event drafts + spec_ids; emit per-event divergence trace.

    drafts entries (from runner.generate) have:
        event, draft_id, draft_text, accepted, committed_count, step_seconds.

    We reconstruct per-event:
        offset_before_event: how many tokens committed before this event
        committed_at_offset: actual tokens emitted in this event (from spec_ids)
        baseline_at_offset:  baseline tokens at the same offset
        diverged_in_this_event: True if any token differs
        first_divergence_in_event_at_token: index within the event where they
                                            first differ (0, 1, or None)
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    first_divergence_event_idx: int | None = None
    first_divergence_after_accept_or_reject: str | None = None
    first_divergence_token_offset: int | None = None

    for ev_idx, ev in enumerate(drafts):
        commit_n = int(ev.get("committed_count", 0))
        accepted = bool(ev.get("accepted", False))

        spec_committed = spec_ids[offset:offset + commit_n]
        baseline_at_offset = baseline_ids[offset:offset + commit_n]

        diverged_here = spec_committed != baseline_at_offset
        first_token_divergence: int | None = None
        if diverged_here:
            for i in range(min(len(spec_committed), len(baseline_at_offset))):
                if spec_committed[i] != baseline_at_offset[i]:
                    first_token_divergence = i
                    break
            if first_token_divergence is None:
                # Length mismatch only (spec ran out / baseline ran out).
                first_token_divergence = min(len(spec_committed), len(baseline_at_offset))

        if first_divergence_event_idx is None and diverged_here:
            first_divergence_event_idx = ev_idx
            first_divergence_after_accept_or_reject = "accept" if accepted else "reject"
            first_divergence_token_offset = offset + (first_token_divergence or 0)

        rows.append({
            "event": ev_idx,
            "accepted": accepted,
            "draft_id": int(ev.get("draft_id", -1)),
            "draft_text": str(ev.get("draft_text", "")),
            "committed_count": commit_n,
            "offset_before_event": offset,
            "offset_after_event": offset + commit_n,
            "spec_committed_at_offset": spec_committed,
            "baseline_at_offset": baseline_at_offset,
            "diverged_in_this_event": diverged_here,
            "first_token_divergence_in_event": first_token_divergence,
            "step_seconds": float(ev.get("step_seconds", 0.0)),
        })

        offset += commit_n

    summary = {
        "n_events": len(drafts),
        "n_accept_events": sum(1 for ev in drafts if ev.get("accepted")),
        "n_reject_events": sum(1 for ev in drafts if not ev.get("accepted")),
        "tokens_committed": offset,
        "first_divergence_event_idx": first_divergence_event_idx,
        "first_divergence_after_accept_or_reject": first_divergence_after_accept_or_reject,
        "first_divergence_token_offset": first_divergence_token_offset,
        "prefix_match_len": _prefix_match(spec_ids, baseline_ids),
        "exact_match": spec_ids == baseline_ids,
    }
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument(
        "--prompts-json",
        default=None,
        help="Optional JSON file with custom prompt list (matching the smoke runner)",
    )
    args = ap.parse_args()

    prompts: list[str] = list(DEFAULT_PROMPTS)
    if args.prompts_json:
        raw = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        prompts = [str(item["prompt"]) if isinstance(item, dict) else str(item) for item in raw]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Apply BASE_ENV BEFORE engine import so module-level env-driven branches pick the production path.
    base_env_with_sidecar = dict(BASE_ENV)
    base_env_with_sidecar["LYNN_MTP_SIDECAR"] = args.sidecar
    base_prev = _set_env(base_env_with_sidecar)

    try:
        # Fail fast if model dir is not Lynn-native NVFP4.
        from engine.nvfp4_layout import detect_nvfp4_layout
        layout = detect_nvfp4_layout(args.model)
        if layout.layout_kind != "lynn_native_per16_variable":
            raise SystemExit(
                f"[m21] {args.model} is layout={layout.layout_kind!r}; "
                "requires lynn_native_per16_variable."
            )

        from engine.resident_runner import LynnIncrementalRunner

        runner = LynnIncrementalRunner(
            args.model,
            device=args.device,
            dtype=dtype,
            verbose=False,
        )
        if not runner.mtp_sidecar_loaded:
            raise SystemExit(f"[m21] MTP sidecar not loaded: {args.sidecar}")

        # ---- Pass 1: baseline greedy per prompt (to capture reference new_ids) ----
        baseline_results: list[dict[str, Any]] = []
        baseline_env = dict(CONFIGS[0][1])
        b_prev = _set_env(baseline_env)
        try:
            for idx, prompt in enumerate(prompts):
                print(f"[m21] baseline_greedy prompt {idx + 1}/{len(prompts)}", flush=True)
                t0 = time.time()
                out = runner.generate(prompt, max_new=args.max_new)
                wall = time.time() - t0
                baseline_results.append({
                    "prompt_idx": idx,
                    "prompt": prompt,
                    "new_ids": [int(x) for x in out["new_ids"]],
                    "decode_tps": out["timings"].get("decode_tps"),
                    "wall": wall,
                })
        finally:
            _restore_env(b_prev)

        # ---- Pass 2..N: spec_k1_batched variants ----
        spec_configs = CONFIGS[1:]
        config_results: dict[str, list[dict[str, Any]]] = {}
        for label, env_overrides in spec_configs:
            print(f"[m21] === config {label} ===", flush=True)
            prev = _set_env(env_overrides)
            try:
                rows: list[dict[str, Any]] = []
                for idx, (prompt, baseline) in enumerate(zip(prompts, baseline_results)):
                    print(f"[m21]   prompt {idx + 1}/{len(prompts)}", flush=True)
                    t0 = time.time()
                    out = runner.generate(prompt, max_new=args.max_new)
                    wall = time.time() - t0
                    spec_ids = [int(x) for x in out["new_ids"]]
                    spec_stats = out.get("mtp_speculative") or {}
                    drafts = spec_stats.get("drafts") or []

                    event_rows, event_summary = _build_event_trace(
                        spec_ids, baseline["new_ids"], drafts,
                    )

                    rows.append({
                        "prompt_idx": idx,
                        "prompt": prompt,
                        "wall": wall,
                        "decode_tps": out["timings"].get("decode_tps"),
                        "spec_active": bool(spec_stats.get("active")),
                        "spec_n_events": int(spec_stats.get("events", 0)),
                        "spec_accepted_events": int(spec_stats.get("accepted_events", 0)),
                        "spec_tokens_committed": int(spec_stats.get("tokens_committed", 0)),
                        "spec_accept_rate": spec_stats.get("accept_rate"),
                        "spec_effective_token_tps": spec_stats.get("effective_token_tps"),
                        "exact_match": spec_ids == baseline["new_ids"],
                        "prefix_match_len": _prefix_match(spec_ids, baseline["new_ids"]),
                        "spec_ids_head": spec_ids[:24],
                        "baseline_ids_head": baseline["new_ids"][:24],
                        "event_summary": event_summary,
                        "events": event_rows,
                    })
                config_results[label] = rows
            finally:
                _restore_env(prev)

        # ---- Cross-config summary ----
        cross_summary: dict[str, Any] = {}
        for label, rows in config_results.items():
            exact_n = sum(1 for r in rows if r["exact_match"])
            prefix_lens = [r["prefix_match_len"] for r in rows]
            accept_rates = [r["spec_accept_rate"] for r in rows if r["spec_accept_rate"] is not None]
            eff_tps = [r["spec_effective_token_tps"] for r in rows if r["spec_effective_token_tps"]]
            cross_summary[label] = {
                "exact_match": f"{exact_n}/{len(rows)}",
                "exact_match_rate": exact_n / len(rows) if rows else None,
                "mean_prefix_match_len": (sum(prefix_lens) / len(prefix_lens)) if prefix_lens else None,
                "mean_spec_accept_rate": (sum(accept_rates) / len(accept_rates)) if accept_rates else None,
                "mean_spec_effective_tps": (sum(eff_tps) / len(eff_tps)) if eff_tps else None,
            }

        # ---- Bisect verdict per M21 acceptance rules ----
        verdict: dict[str, Any] = {
            "default_exact_n": sum(1 for r in config_results.get("spec_k1_batched_default", []) if r["exact_match"]),
            "bf16_lmhead_exact_n": sum(1 for r in config_results.get("spec_k1_batched_lmhead_bf16", []) if r["exact_match"]),
            "canonical_t1_exact_n": sum(1 for r in config_results.get("spec_k1_batched_canonical_t1_accept", []) if r["exact_match"]),
            "both_exact_n": sum(1 for r in config_results.get("spec_k1_batched_both_switches", []) if r["exact_match"]),
            "n_prompts": len(prompts),
        }

        bf16_helped = verdict["bf16_lmhead_exact_n"] > verdict["default_exact_n"]
        canonical_helped = verdict["canonical_t1_exact_n"] > verdict["default_exact_n"]
        both_helped = verdict["both_exact_n"] > verdict["default_exact_n"]
        bf16_full_fix = verdict["bf16_lmhead_exact_n"] == verdict["n_prompts"]
        canonical_full_fix = verdict["canonical_t1_exact_n"] == verdict["n_prompts"]
        both_full_fix = verdict["both_exact_n"] == verdict["n_prompts"]

        if bf16_full_fix and not canonical_full_fix:
            verdict["root_cause"] = "native_fp4_lm_head_per_row_dispatch"
        elif canonical_full_fix and not bf16_full_fix:
            verdict["root_cause"] = "k2_accept_logits_or_argmax"
        elif bf16_full_fix and canonical_full_fix:
            verdict["root_cause"] = "either_path_alone_sufficient_likely_shared_dependency"
        elif both_full_fix:
            verdict["root_cause"] = "only_combined_switches_fix__interaction_between_lm_head_and_accept_argmax"
        else:
            verdict["root_cause"] = "neither_switch_fixes__state_commit_rollback_or_new_ids_emission"

        verdict["bf16_helped_partial"] = bf16_helped
        verdict["canonical_helped_partial"] = canonical_helped
        verdict["both_helped_partial"] = both_helped

        report: dict[str, Any] = {
            "schema_version": "lynn-mtp-m21-residual-exact-v1",
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": args.model,
            "sidecar": args.sidecar,
            "max_new": args.max_new,
            "base_env": BASE_ENV,
            "configs": [{"label": label, "env_overrides": ov} for label, ov in CONFIGS],
            "baseline_results": baseline_results,
            "config_results": config_results,
            "cross_summary": cross_summary,
            "verdict": verdict,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[m21] wrote {out_path}", flush=True)
        print(f"[m21] verdict.root_cause = {verdict['root_cause']}", flush=True)
        for label, row in cross_summary.items():
            print(f"[m21] {label}: exact={row['exact_match']} prefix={row['mean_prefix_match_len']} accept={row['mean_spec_accept_rate']}", flush=True)
        return 0
    finally:
        _restore_env(base_prev)


if __name__ == "__main__":
    raise SystemExit(main())
