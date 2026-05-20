#!/usr/bin/env python3
"""M22 reject-rollback full-state probe — does ``runner._restore_state`` close M21's exact gap?

M21 (commit 432a29e) showed that on the K=2 batched REJECT path, even
after the existing lean ``restore_recurrent_conv`` and a canonical T=1
re-decode of the pending token, the produced argmax disagrees with an
apples-to-apples eager baseline's T=1 decode of the same pending from
the same prefix. The K=2 forward leaves residual state that the lean
restore (recurrent + conv + seq_len, no KV) does not undo.

M22 adds an opt-in `LYNN_MTP_K2_REJECT_ROLLBACK=full_state` switch in
``engine/mtp_serving.py::speculative_step_k1_batched``. When enabled,
the REJECT path uses ``runner._snapshot_state`` / ``runner._restore_state``
(KV + recurrent + conv + seq_len) instead of the lean variant.

This probe runs four configs against the canonical 6-prompt smoke set
on a single resident runner under apples-to-apples eager + shadow-off:

  1. baseline_greedy             — eager T=1 reference
  2. spec_k1_sequential          — sequential T=1 verifier (known 6/6)
  3. spec_k1_batched_default     — current main behavior (post-M19)
  4. spec_k1_batched_fullrollback — M22 full-state reject rollback

Output JSON + verdict drives the M22 acceptance:

  * fullrollback exact=6/6   → root cause = lean rollback missing KV;
                               report accept rate + effective TPS +
                               per-event overhead.
  * fullrollback still <6/6  → ``_snapshot_state`` insufficient;
                               residual lives outside captured state
                               (runner.outside / kernel scratch /
                               generation bookkeeping). Next bisect.

Usage on Spark::

    /home/merkyor/comfyui/ComfyUI/.venv/bin/python -u \
        scripts/spark_mtp_m22_reject_rollback_probe.py \
        --model /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
        --sidecar /home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
        --out /tmp/lynn_m22/mtp_m22_reject_rollback_$(date +%Y%m%d_%H%M%S).json \
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
    # Spark Config D production env.
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_PACKED_DECODE": "1",
    "LYNN_PACKED_SHARED_EXPERT": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_FULL_ATTN_QKV_FUSED": "1",
    # M18 layer-level necessary fix.
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


# Each config is (label, env overrides applied on top of BASE_ENV).
# All configs run eager (graph off) and with shadow off so the bisect
# is apples-to-apples — this isolates the speculative commit/rollback
# bug from the known graph-vs-eager numerical drift.
CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("baseline_greedy", {
        "LYNN_MTP_SHADOW_VERIFY": "0",
        "LYNN_MTP_SPECULATIVE": "0",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    }),
    ("spec_k1_sequential", {
        "LYNN_MTP_SHADOW_VERIFY": "0",
        "LYNN_MTP_SPECULATIVE": "1",
        "LYNN_MTP_SPECULATIVE_BATCHED": "0",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    }),
    ("spec_k1_batched_default", {
        "LYNN_MTP_SHADOW_VERIFY": "0",
        "LYNN_MTP_SPECULATIVE": "1",
        "LYNN_MTP_SPECULATIVE_BATCHED": "1",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    }),
    ("spec_k1_batched_fullrollback", {
        "LYNN_MTP_SHADOW_VERIFY": "0",
        "LYNN_MTP_SPECULATIVE": "1",
        "LYNN_MTP_SPECULATIVE_BATCHED": "1",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
        "LYNN_MTP_K2_REJECT_ROLLBACK": "full_state",
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
    """Walk per-event drafts; emit per-event divergence trace.

    Returns (events_rows, summary).
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    first_divergence_event_idx: int | None = None
    first_divergence_kind: str | None = None
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
                first_token_divergence = min(len(spec_committed), len(baseline_at_offset))

        if first_divergence_event_idx is None and diverged_here:
            first_divergence_event_idx = ev_idx
            first_divergence_kind = "accept" if accepted else "reject"
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
        "first_divergence_kind": first_divergence_kind,
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
            raise SystemExit(
                f"[m22] {args.model} is layout={layout.layout_kind!r}; "
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
            raise SystemExit(f"[m22] MTP sidecar not loaded: {args.sidecar}")

        # ---- Pass 1: baseline (reference new_ids) ----
        baseline_results: list[dict[str, Any]] = []
        baseline_label, baseline_env = CONFIGS[0]
        b_prev = _set_env(baseline_env)
        try:
            for idx, prompt in enumerate(prompts):
                print(f"[m22] {baseline_label} prompt {idx + 1}/{len(prompts)}", flush=True)
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

        # ---- Pass 2..4: spec variants ----
        spec_configs = CONFIGS[1:]
        config_results: dict[str, list[dict[str, Any]]] = {}
        for label, env_overrides in spec_configs:
            print(f"[m22] === config {label} ===", flush=True)
            prev = _set_env(env_overrides)
            try:
                rows: list[dict[str, Any]] = []
                for idx, (prompt, baseline) in enumerate(zip(prompts, baseline_results)):
                    print(f"[m22]   prompt {idx + 1}/{len(prompts)}", flush=True)
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
        baseline_tps = [r["decode_tps"] for r in baseline_results if r["decode_tps"]]
        baseline_tps_mean = (sum(baseline_tps) / len(baseline_tps)) if baseline_tps else None
        cross_summary["baseline_greedy"] = {
            "exact_match": f"{len(baseline_results)}/{len(baseline_results)}",
            "mean_decode_tps": baseline_tps_mean,
        }
        for label, rows in config_results.items():
            exact_n = sum(1 for r in rows if r["exact_match"])
            prefix_lens = [r["prefix_match_len"] for r in rows]
            accept_rates = [r["spec_accept_rate"] for r in rows if r["spec_accept_rate"] is not None]
            eff_tps = [r["spec_effective_token_tps"] for r in rows if r["spec_effective_token_tps"]]
            mean_step_seconds_all: list[float] = []
            for r in rows:
                for ev in r["events"]:
                    mean_step_seconds_all.append(float(ev.get("step_seconds", 0.0)))
            cross_summary[label] = {
                "exact_match": f"{exact_n}/{len(rows)}",
                "exact_match_rate": exact_n / len(rows) if rows else None,
                "mean_prefix_match_len": (sum(prefix_lens) / len(prefix_lens)) if prefix_lens else None,
                "mean_spec_accept_rate": (sum(accept_rates) / len(accept_rates)) if accept_rates else None,
                "mean_spec_effective_tps": (sum(eff_tps) / len(eff_tps)) if eff_tps else None,
                "mean_event_step_seconds": (sum(mean_step_seconds_all) / len(mean_step_seconds_all)) if mean_step_seconds_all else None,
                "tps_ratio_over_baseline": (
                    (sum(eff_tps) / len(eff_tps)) / baseline_tps_mean
                    if eff_tps and baseline_tps_mean else None
                ),
            }

        default_label = "spec_k1_batched_default"
        full_label = "spec_k1_batched_fullrollback"
        default_exact_n = sum(1 for r in config_results.get(default_label, []) if r["exact_match"])
        full_exact_n = sum(1 for r in config_results.get(full_label, []) if r["exact_match"])
        n_prompts = len(prompts)

        verdict: dict[str, Any] = {
            "default_exact_n": default_exact_n,
            "fullrollback_exact_n": full_exact_n,
            "n_prompts": n_prompts,
            "fullrollback_helped": full_exact_n > default_exact_n,
            "fullrollback_full_fix": full_exact_n == n_prompts,
        }
        if verdict["fullrollback_full_fix"]:
            verdict["root_cause"] = "lean_rollback_missing_kv_full_state_fixes_exact"
        elif verdict["fullrollback_helped"]:
            verdict["root_cause"] = "lean_rollback_partial__kv_helps_but_additional_residue_remains"
        else:
            verdict["root_cause"] = "full_snapshot_insufficient__residue_outside_captured_state"

        # First-divergence-event comparison default vs fullrollback per prompt.
        first_div_table: list[dict[str, Any]] = []
        for idx in range(n_prompts):
            d_row = config_results.get(default_label, [{}] * n_prompts)[idx] if idx < len(config_results.get(default_label, [])) else {}
            f_row = config_results.get(full_label, [{}] * n_prompts)[idx] if idx < len(config_results.get(full_label, [])) else {}
            first_div_table.append({
                "prompt_idx": idx,
                "default": {
                    "first_div_event": (d_row.get("event_summary") or {}).get("first_divergence_event_idx"),
                    "first_div_kind": (d_row.get("event_summary") or {}).get("first_divergence_kind"),
                    "prefix_match_len": d_row.get("prefix_match_len"),
                    "exact_match": d_row.get("exact_match"),
                },
                "fullrollback": {
                    "first_div_event": (f_row.get("event_summary") or {}).get("first_divergence_event_idx"),
                    "first_div_kind": (f_row.get("event_summary") or {}).get("first_divergence_kind"),
                    "prefix_match_len": f_row.get("prefix_match_len"),
                    "exact_match": f_row.get("exact_match"),
                },
            })

        report: dict[str, Any] = {
            "schema_version": "lynn-mtp-m22-reject-rollback-v1",
            "generated_at": datetime.now().isoformat(timespec="seconds") + "Z",
            "model": args.model,
            "sidecar": args.sidecar,
            "max_new": args.max_new,
            "base_env": BASE_ENV,
            "configs": [{"label": label, "env_overrides": ov} for label, ov in CONFIGS],
            "baseline_results": baseline_results,
            "config_results": config_results,
            "cross_summary": cross_summary,
            "first_divergence_default_vs_fullrollback": first_div_table,
            "verdict": verdict,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[m22] wrote {out_path}", flush=True)
        print(f"[m22] verdict.root_cause = {verdict['root_cause']}", flush=True)
        for label, row in cross_summary.items():
            print(f"[m22] {label}: {row}", flush=True)
        return 0
    finally:
        _restore_env(base_prev)


if __name__ == "__main__":
    raise SystemExit(main())
