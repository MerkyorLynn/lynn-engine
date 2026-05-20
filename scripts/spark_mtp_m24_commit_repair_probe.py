#!/usr/bin/env python3
"""M24 commit-repair bisect — shrink M23 canonical commit into minimal repair.

M23 (commit cf0a187) showed that running a full canonical T=1 commit
chain on accept (commit_source=canonical_t1) restores 6/6 exact at 7.85
effective TPS — well below baseline 25.85 and below sequential 21.26.
M24 bisects which K=2 commit component actually needs canonical
replacement to reach 6/6 exact, with the goal of preserving most of
the K=2 path's speed.

The new ``LYNN_MTP_K2_COMMIT_REPAIR`` env knob has five opt-in modes:

  hidden               — replace next_base_hidden only, keep K=2 state
  next_pending         — replace next_pending_id only, keep K=2 state
  hidden_next_pending  — replace both, keep K=2 state (cheap-repair test)
  state                — discard K=2 state, use canonical T=1 chain state
  full_canonical       — alias for commit_source=canonical_t1 (M23)

This probe runs seven configs (baseline + batched_default + 5 repair
modes) against the canonical 6-prompt smoke set under apples-to-apples
eager (graph off) + shadow off:

Decision tree per M24 acceptance:
  * Any cheap repair (hidden / next_pending / hidden_next_pending) hits
    6/6 with TPS > 7.85 (M23 reference) → M24_CANDIDATE; promote that
    specific repair as the minimal fix.
  * Only `state` or `full_canonical` hits 6/6 → K=2 state itself is
    broken; cheap repair insufficient; M25 must do state-delta probe.
  * `hidden_next_pending` hits 6/6 → K=2 state is fine, only outputs
    drift → cheapest possible repair shape known.

Usage on Spark::

    /home/merkyor/comfyui/ComfyUI/.venv/bin/python -u \
        scripts/spark_mtp_m24_commit_repair_probe.py \
        --model /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
        --sidecar /home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
        --out /tmp/lynn_m24/mtp_m24_commit_repair_$(date +%Y%m%d_%H%M%S).json \
        --max-new 64
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
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_PACKED_DECODE": "1",
    "LYNN_PACKED_SHARED_EXPERT": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_FULL_ATTN_QKV_FUSED": "1",
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


def _spec_common() -> dict[str, str]:
    return {
        "LYNN_MTP_SHADOW_VERIFY": "0",
        "LYNN_MTP_SPECULATIVE": "1",
        "LYNN_MTP_SPECULATIVE_BATCHED": "1",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    }


CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("baseline_greedy", {
        "LYNN_MTP_SHADOW_VERIFY": "0",
        "LYNN_MTP_SPECULATIVE": "0",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    }),
    ("spec_k1_batched_default", _spec_common()),
    ("spec_k1_batched_repair_hidden", {**_spec_common(), "LYNN_MTP_K2_COMMIT_REPAIR": "hidden"}),
    ("spec_k1_batched_repair_next_pending", {**_spec_common(), "LYNN_MTP_K2_COMMIT_REPAIR": "next_pending"}),
    ("spec_k1_batched_repair_hidden_next_pending", {**_spec_common(), "LYNN_MTP_K2_COMMIT_REPAIR": "hidden_next_pending"}),
    ("spec_k1_batched_repair_state", {**_spec_common(), "LYNN_MTP_K2_COMMIT_REPAIR": "state"}),
    ("spec_k1_batched_repair_full_canonical", {**_spec_common(), "LYNN_MTP_K2_COMMIT_REPAIR": "full_canonical"}),
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
    rows: list[dict[str, Any]] = []
    offset = 0
    first_div_event: int | None = None
    first_div_kind: str | None = None
    first_div_token_offset: int | None = None

    for ev_idx, ev in enumerate(drafts):
        commit_n = int(ev.get("committed_count", 0))
        accepted = bool(ev.get("accepted", False))
        spec_committed = spec_ids[offset:offset + commit_n]
        baseline_at_offset = baseline_ids[offset:offset + commit_n]
        diverged = spec_committed != baseline_at_offset
        first_in_event: int | None = None
        if diverged:
            for i in range(min(len(spec_committed), len(baseline_at_offset))):
                if spec_committed[i] != baseline_at_offset[i]:
                    first_in_event = i
                    break
            if first_in_event is None:
                first_in_event = min(len(spec_committed), len(baseline_at_offset))
        if first_div_event is None and diverged:
            first_div_event = ev_idx
            first_div_kind = "accept" if accepted else "reject"
            first_div_token_offset = offset + (first_in_event or 0)
        rows.append({
            "event": ev_idx,
            "accepted": accepted,
            "draft_id": int(ev.get("draft_id", -1)),
            "committed_count": commit_n,
            "offset_before_event": offset,
            "offset_after_event": offset + commit_n,
            "spec_committed_at_offset": spec_committed,
            "baseline_at_offset": baseline_at_offset,
            "diverged_in_this_event": diverged,
            "first_token_divergence_in_event": first_in_event,
            "step_seconds": float(ev.get("step_seconds", 0.0)),
        })
        offset += commit_n

    summary = {
        "n_events": len(drafts),
        "n_accept_events": sum(1 for ev in drafts if ev.get("accepted")),
        "n_reject_events": sum(1 for ev in drafts if not ev.get("accepted")),
        "tokens_committed": offset,
        "first_divergence_event_idx": first_div_event,
        "first_divergence_kind": first_div_kind,
        "first_divergence_token_offset": first_div_token_offset,
        "prefix_match_len": _prefix_match(spec_ids, baseline_ids),
        "exact_match": spec_ids == baseline_ids,
    }
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--prompts-json", default=None)
    args = ap.parse_args()

    prompts: list[str] = list(DEFAULT_PROMPTS)
    if args.prompts_json:
        raw = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        prompts = [str(item["prompt"]) if isinstance(item, dict) else str(item) for item in raw]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    base_env_with_sidecar = dict(BASE_ENV)
    base_env_with_sidecar["LYNN_MTP_SIDECAR"] = args.sidecar
    base_prev = _set_env(base_env_with_sidecar)

    try:
        from engine.nvfp4_layout import detect_nvfp4_layout
        layout = detect_nvfp4_layout(args.model)
        if layout.layout_kind != "lynn_native_per16_variable":
            raise SystemExit(f"[m24] {args.model} is {layout.layout_kind!r}")

        from engine.resident_runner import LynnIncrementalRunner

        runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=False)
        if not runner.mtp_sidecar_loaded:
            raise SystemExit(f"[m24] MTP sidecar not loaded: {args.sidecar}")

        # ---- Baseline pass ----
        baseline_results: list[dict[str, Any]] = []
        baseline_label, baseline_env = CONFIGS[0]
        prev = _set_env(baseline_env)
        try:
            for idx, prompt in enumerate(prompts):
                print(f"[m24] {baseline_label} prompt {idx + 1}/{len(prompts)}", flush=True)
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
            _restore_env(prev)

        # ---- Spec configs ----
        spec_configs = CONFIGS[1:]
        config_results: dict[str, list[dict[str, Any]]] = {}
        for label, env_overrides in spec_configs:
            print(f"[m24] === config {label} ===", flush=True)
            prev = _set_env(env_overrides)
            try:
                rows: list[dict[str, Any]] = []
                for idx, (prompt, baseline) in enumerate(zip(prompts, baseline_results)):
                    print(f"[m24]   prompt {idx + 1}/{len(prompts)}", flush=True)
                    t0 = time.time()
                    out = runner.generate(prompt, max_new=args.max_new)
                    wall = time.time() - t0
                    spec_ids = [int(x) for x in out["new_ids"]]
                    spec_stats = out.get("mtp_speculative") or {}
                    drafts = spec_stats.get("drafts") or []
                    event_rows, event_summary = _build_event_trace(spec_ids, baseline["new_ids"], drafts)
                    rows.append({
                        "prompt_idx": idx,
                        "prompt": prompt,
                        "wall": wall,
                        "decode_tps": out["timings"].get("decode_tps"),
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
        baseline_tps = [r["decode_tps"] for r in baseline_results if r["decode_tps"]]
        baseline_tps_mean = (sum(baseline_tps) / len(baseline_tps)) if baseline_tps else None
        cross_summary: dict[str, Any] = {
            "baseline_greedy": {
                "exact_match": f"{len(baseline_results)}/{len(baseline_results)}",
                "mean_decode_tps": baseline_tps_mean,
            },
        }
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
                "tps_ratio_over_baseline": (
                    (sum(eff_tps) / len(eff_tps)) / baseline_tps_mean
                    if eff_tps and baseline_tps_mean else None
                ),
            }

        # ---- Decision-tree verdict ----
        labels_to_check = [
            "spec_k1_batched_default",
            "spec_k1_batched_repair_hidden",
            "spec_k1_batched_repair_next_pending",
            "spec_k1_batched_repair_hidden_next_pending",
            "spec_k1_batched_repair_state",
            "spec_k1_batched_repair_full_canonical",
        ]
        exact_by_label: dict[str, int] = {}
        tps_by_label: dict[str, float | None] = {}
        for label in labels_to_check:
            rows = config_results.get(label) or []
            exact_by_label[label] = sum(1 for r in rows if r["exact_match"])
            eff_tps = [r["spec_effective_token_tps"] for r in rows if r["spec_effective_token_tps"]]
            tps_by_label[label] = (sum(eff_tps) / len(eff_tps)) if eff_tps else None

        n_prompts = len(prompts)
        # M23 reference TPS — anything cheap repair must beat to be promotable.
        M23_TPS_REFERENCE = 7.85

        candidates: list[dict[str, Any]] = []
        for label in labels_to_check:
            if label == "spec_k1_batched_default":
                continue
            if exact_by_label[label] == n_prompts:
                tps = tps_by_label[label] or 0.0
                candidates.append({
                    "label": label,
                    "exact_n": exact_by_label[label],
                    "effective_tps": tps,
                    "above_m23_reference": tps > M23_TPS_REFERENCE,
                })

        # Pick the M24_CANDIDATE — cheapest 6/6 repair with TPS > M23.
        m24_candidate = None
        repair_order_for_cheapness = [
            "spec_k1_batched_repair_hidden",
            "spec_k1_batched_repair_next_pending",
            "spec_k1_batched_repair_hidden_next_pending",
            "spec_k1_batched_repair_state",
            "spec_k1_batched_repair_full_canonical",
        ]
        for label in repair_order_for_cheapness:
            if exact_by_label[label] == n_prompts and (tps_by_label[label] or 0.0) > M23_TPS_REFERENCE:
                m24_candidate = {
                    "label": label,
                    "effective_tps": tps_by_label[label],
                    "exact_n": exact_by_label[label],
                }
                break

        cheap_6_6 = exact_by_label.get("spec_k1_batched_repair_hidden_next_pending") == n_prompts
        only_heavy_6_6 = (
            exact_by_label.get("spec_k1_batched_repair_state") == n_prompts
            or exact_by_label.get("spec_k1_batched_repair_full_canonical") == n_prompts
        ) and not cheap_6_6

        if m24_candidate is not None:
            verdict_class = "M24_CANDIDATE_FOUND"
        elif cheap_6_6:
            verdict_class = "HIDDEN_NEXT_PENDING_FIXES_BUT_NOT_FASTER_THAN_M23"
        elif only_heavy_6_6:
            verdict_class = "ONLY_STATE_OR_FULL_CANONICAL_FIXES__K2_STATE_BROKEN"
        else:
            verdict_class = "NO_REPAIR_REACHES_6_6__INVESTIGATE_FURTHER"

        verdict: dict[str, Any] = {
            "n_prompts": n_prompts,
            "exact_by_label": exact_by_label,
            "tps_by_label": tps_by_label,
            "m23_tps_reference": M23_TPS_REFERENCE,
            "candidates_6_6": candidates,
            "m24_candidate": m24_candidate,
            "verdict_class": verdict_class,
        }

        # Per-prompt first-divergence cross-table
        first_div_table: list[dict[str, Any]] = []
        for idx in range(n_prompts):
            row_entry: dict[str, Any] = {"prompt_idx": idx}
            for label in labels_to_check:
                rows = config_results.get(label) or []
                if idx >= len(rows):
                    row_entry[label] = None
                    continue
                es = rows[idx].get("event_summary") or {}
                row_entry[label] = {
                    "first_div_event": es.get("first_divergence_event_idx"),
                    "first_div_kind": es.get("first_divergence_kind"),
                    "prefix_match_len": rows[idx].get("prefix_match_len"),
                    "exact_match": rows[idx].get("exact_match"),
                }
            first_div_table.append(row_entry)

        report: dict[str, Any] = {
            "schema_version": "lynn-mtp-m24-commit-repair-v1",
            "generated_at": datetime.now().isoformat(timespec="seconds") + "Z",
            "model": args.model,
            "sidecar": args.sidecar,
            "max_new": args.max_new,
            "base_env": BASE_ENV,
            "configs": [{"label": label, "env_overrides": ov} for label, ov in CONFIGS],
            "baseline_results": baseline_results,
            "config_results": config_results,
            "cross_summary": cross_summary,
            "first_divergence_table": first_div_table,
            "verdict": verdict,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[m24] wrote {out_path}", flush=True)
        print(f"[m24] verdict_class = {verdict['verdict_class']}", flush=True)
        if m24_candidate is not None:
            print(f"[m24] M24_CANDIDATE = {m24_candidate}", flush=True)
        for label, row in cross_summary.items():
            print(f"[m24] {label}: {row}", flush=True)
        return 0
    finally:
        _restore_env(base_prev)


if __name__ == "__main__":
    raise SystemExit(main())
