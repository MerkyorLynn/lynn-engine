#!/usr/bin/env python3
"""MTP concat-fix diagnostic sweep.

If the embed-first concat fix (493b2da) still yields <5% accept,
this script systematically tests:
  1. pos_offset: current_pos vs current_pos+1 vs current_pos-1
  2. token_embed_source: base embed_tokens vs MTP embed (if separate exists)
  3. hidden_source: pre-norm vs post-norm base hidden state
  4. concat_order: embed|hidden vs hidden|embed (sanity re-check)

Runs a short decode (8 tokens × 3 prompts) for each config, reports
per-config shadow accept rate.

Usage:
  python scripts/mtp_concat_fix_diagnostic_sweep.py \\
    --model /home/merkyor/models/Qwen3.6-35B-A3B-FP8 \\
    --sidecar /home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp_lynn_fused.safetensors \\
    --out /home/merkyor/reports/spark/mtp_diagnostic_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# This script requires the full engine environment (torch, safetensors, etc.)
# It is designed to run on Spark/R6000 with GPU.


SWEEP_CONFIGS = [
    {
        "id": "baseline_embed_first",
        "description": "Current fix: cat([embed_norm, hidden_norm], -1), pos=current_pos",
        "env": {"LYNN_MTP_CONCAT_ORDER": "embed_first", "LYNN_MTP_POS_OFFSET": "0"},
    },
    {
        "id": "hidden_first_revert",
        "description": "Revert to V1: cat([hidden_norm, embed_norm], -1), pos=current_pos",
        "env": {"LYNN_MTP_CONCAT_ORDER": "hidden_first", "LYNN_MTP_POS_OFFSET": "0"},
    },
    {
        "id": "embed_first_pos_plus1",
        "description": "Embed first + pos_offset=+1 (next position)",
        "env": {"LYNN_MTP_CONCAT_ORDER": "embed_first", "LYNN_MTP_POS_OFFSET": "1"},
    },
    {
        "id": "embed_first_pos_minus1",
        "description": "Embed first + pos_offset=-1",
        "env": {"LYNN_MTP_CONCAT_ORDER": "embed_first", "LYNN_MTP_POS_OFFSET": "-1"},
    },
]

PROMPTS = [
    "The capital of France is",
    "Write a Python function that",
    "In quantum mechanics, the Heisenberg uncertainty principle states that",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="MTP concat-fix diagnostic sweep")
    ap.add_argument("--model", required=True, help="Model directory")
    ap.add_argument("--sidecar", required=True, help="MTP fused sidecar path")
    ap.add_argument("--out", required=True, help="Output JSON")
    ap.add_argument("--max-new", type=int, default=8, help="Tokens to generate per prompt")
    ap.add_argument("--dry-run", action="store_true", help="Print configs without running")
    args = ap.parse_args()

    if args.dry_run:
        print("[sweep] DRY_RUN — configs to test:", file=sys.stderr)
        for cfg in SWEEP_CONFIGS:
            print(f"  {cfg['id']}: {cfg['description']}", file=sys.stderr)
            print(f"    env: {cfg['env']}", file=sys.stderr)
        return 0

    # Lazy import — only needed for real runs
    try:
        import torch
        from engine.resident_runner import LynnIncrementalRunner
    except ImportError as e:
        print(f"ERROR: cannot import engine (requires GPU env): {e}", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []

    for cfg in SWEEP_CONFIGS:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[sweep] Config: {cfg['id']}", file=sys.stderr)
        print(f"[sweep] {cfg['description']}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

        # Set env
        old_env = {}
        for k, v in cfg["env"].items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v

        # Also set shadow verify
        os.environ["LYNN_MTP_SHADOW_VERIFY"] = "1"
        os.environ["LYNN_MTP_SIDECAR"] = args.sidecar

        try:
            runner = LynnIncrementalRunner(
                args.model,
                device="cuda",
                dtype=torch.bfloat16,
                max_seq_len=4096,
                verbose=False,
            )

            accepts = []
            for prompt in PROMPTS:
                out = runner.generate(prompt, max_new=args.max_new, use_chat_template=False)
                mtp_stats = out.get("mtp_shadow_stats", {})
                accept_rate = mtp_stats.get("accept_rate", 0.0)
                accepts.append(accept_rate)
                print(f"  prompt={prompt[:40]!r} accept={accept_rate:.3f}", file=sys.stderr)

            mean_accept = sum(accepts) / len(accepts) if accepts else 0.0
            results.append({
                "config_id": cfg["id"],
                "description": cfg["description"],
                "env": cfg["env"],
                "per_prompt_accept": accepts,
                "mean_accept_rate": mean_accept,
            })
            print(f"  → mean_accept={mean_accept:.4f}", file=sys.stderr)

            del runner
            torch.cuda.empty_cache()

        except Exception as e:
            results.append({
                "config_id": cfg["id"],
                "description": cfg["description"],
                "env": cfg["env"],
                "error": str(e),
                "mean_accept_rate": None,
            })
            print(f"  ERROR: {e}", file=sys.stderr)

        finally:
            # Restore env
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # Write report
    report = {
        "schema": "lynn-mtp-concat-diagnostic-sweep-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "sidecar": args.sidecar,
        "max_new": args.max_new,
        "prompts": PROMPTS,
        "results": results,
        "best_config": max(results, key=lambda r: r.get("mean_accept_rate") or 0.0).get("config_id"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[sweep] SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for r in results:
        ar = r.get("mean_accept_rate")
        print(f"  {r['config_id']:30s} accept={ar:.4f}" if ar is not None else f"  {r['config_id']:30s} ERROR", file=sys.stderr)
    print(f"\n[sweep] Best: {report['best_config']}", file=sys.stderr)
    print(f"[sweep] Report: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
