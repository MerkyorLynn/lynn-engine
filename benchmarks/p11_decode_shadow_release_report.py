#!/usr/bin/env python3
"""P11: runtime BF16 shadow release candidate report.

The Lynn-native NVFP4 artifact is ~20 GiB packed, but the current resident
runner still slow-dequants quantized tensors into BF16 for safe prefill. This
gate reports which BF16 tensors have packed decode aliases already attached and
therefore are candidates for a future decode-only / session-scoped release
policy.

It is intentionally read-only. Releasing shadows changes the request lifecycle:
multi-request serving still needs a BF16 prefill path unless we add packed
prefill or reload-on-demand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _gib(n: int) -> float:
    return n / (1024**3)


def _tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _bucket_key(key: str) -> str:
    if key.startswith("linear_attn."):
        if "in_proj" in key:
            return "linear_attn.in_proj"
        if "out_proj" in key:
            return "linear_attn.out_proj"
        return "linear_attn.other"
    if key.startswith("self_attn."):
        if any(name in key for name in ("q_proj", "k_proj", "v_proj")):
            return "full_attn.qkv_proj"
        if "o_proj" in key:
            return "full_attn.o_proj"
        return "full_attn.other"
    if key.startswith("mlp.shared_expert."):
        return "moe.shared_expert"
    if key.startswith("mlp.experts."):
        return "moe.experts"
    return "other"


def _candidate_rows(runner: LynnIncrementalRunner) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_idx, weights in enumerate(runner.layer_weights):
        if "mlp.experts.gate_up_proj" in weights and "mlp.experts._gate_up_packed" in weights:
            rows.append(
                {
                    "layer": layer_idx,
                    "key": "mlp.experts.gate_up_proj",
                    "bucket": "moe.experts.gate_up",
                    "bf16_bytes": _tensor_bytes(weights["mlp.experts.gate_up_proj"]),
                    "packed_alias_type": "grouped_nvfp4",
                }
            )
        if "mlp.experts.down_proj" in weights and "mlp.experts._down_packed" in weights:
            rows.append(
                {
                    "layer": layer_idx,
                    "key": "mlp.experts.down_proj",
                    "bucket": "moe.experts.down",
                    "bf16_bytes": _tensor_bytes(weights["mlp.experts.down_proj"]),
                    "packed_alias_type": "grouped_nvfp4",
                }
            )
        for key, tensor in weights.items():
            if not key.endswith(".weight") or not isinstance(tensor, torch.Tensor):
                continue
            packed_key = key + ".packed"
            if packed_key not in weights:
                continue
            rows.append(
                {
                    "layer": layer_idx,
                    "key": key,
                    "bucket": _bucket_key(key),
                    "bf16_bytes": _tensor_bytes(tensor),
                    "packed_alias_type": type(weights[packed_key]).__name__,
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=torch.bfloat16, verbose=False)
    rows = _candidate_rows(runner)
    by_bucket: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = by_bucket.setdefault(
            row["bucket"],
            {"bucket": row["bucket"], "count": 0, "bf16_bytes": 0, "examples": []},
        )
        bucket["count"] += 1
        bucket["bf16_bytes"] += int(row["bf16_bytes"])
        if len(bucket["examples"]) < 5:
            bucket["examples"].append(f"L{row['layer']:02d}:{row['key']}")

    bucket_rows = list(by_bucket.values())
    bucket_rows.sort(key=lambda x: x["bf16_bytes"], reverse=True)
    for bucket in bucket_rows:
        bucket["bf16_gib"] = round(_gib(bucket["bf16_bytes"]), 4)

    total = sum(int(row["bf16_bytes"]) for row in rows)
    result = {
        "schema_version": "lynn-engine-p11-decode-shadow-release-report-v1",
        "model": args.model,
        "candidate_tensors": len(rows),
        "candidate_bf16_gib": round(_gib(total), 4),
        "by_bucket": bucket_rows,
        "runner_memory_after_load": runner.cuda_memory_after_load,
        "note": (
            "Read-only report. These BF16 tensors have packed decode aliases, "
            "but multi-request serving still needs BF16 prefill unless a later "
            "gate adds packed prefill or a session-scoped release/reload lifecycle."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
