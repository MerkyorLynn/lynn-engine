#!/usr/bin/env python3
"""Qwen3.5-9B compact NVFP4 shrink gate.

Reads a P199 size-audit JSON and emits a release-gate report that defines
four tiers for a future compact NVFP4 artifact (5.3–5.6 GiB target).

Tiers:
  SAFE_NO_CHANGE            – ship current 8.248 GiB artifact as-is.
  COMPACT_EMBED_ONLY        – quantize embed_tokens only.
  COMPACT_LMHEAD_ONLY       – quantize lm_head only.
  COMPACT_EMBED_LMHEAD      – quantize both; target 5.3–5.6 GiB.

No tier beyond SAFE_NO_CHANGE is marked PASS.  All compact tiers are
NEEDS_QUALITY_GATE or NEEDS_FULL_MMLU_GPQA_STRUCTURED_GATE because
quantizing embed_tokens / lm_head from BF16 to NVFP4 is an experimental
change with measurable quality risk.

CPU-only.  No engine / server / GPU dependency.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Schema version ─────────────────────────────────────────────────────────────
SCHEMA = "lynn-compact-nvfp4-shrink-gate-v1"

# ── Size constants (GiB) ──────────────────────────────────────────────────────
Q4KM_TARGET_GIB = 5.3

# When embed_tokens or lm_head is quantized from BF16 to NVFP4, the per-tensor
# size drops from ~1.895 GiB to approximately:
#   packed  = (248320 * 4096) / 2   bytes  = 508 MiB   (uint8, 2 rows per byte)
#   scale   = (248320 * 4096) / 16  bytes = 127 MiB   (uint32 sub-channel scale)
#   gscale  = scalar, negligible
#   total   ≈ 635 MiB ≈ 0.620 GiB
#
# This is an estimate; actual packed layout depends on the packing tool.
# We use 0.62 GiB as the reference estimate and note the uncertainty.
ESTIMATED_TENSOR_NVFP4_GIB = 0.62

# The "other" kept BF16 overhead (norms, conv1d, etc.) is ~0.024 GiB.
# We carry it forward from P199 data rather than hard-coding.

# ── Required quality gates for compact tiers ──────────────────────────────────
COMPACT_REQUIRED_GATES: list[str] = [
    "MMLU_500        — ≥ baseline NVFP4 score (no >1pp regression)",
    "GPQA_Diamond    — ≥ baseline NVFP4 score (no >1pp regression)",
    "Structured_Content  — 10 multi-format prompts, GREEN overall",
    "Context_32K_Smoke   — long-context generation, no crash / garbage",
    "R6000_TPS           — tokens/sec ≥ 90%% of baseline NVFP4",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_p199(path: Path) -> dict[str, Any]:
    """Load and validate P199 JSON."""
    if not path.exists():
        raise FileNotFoundError(f"P199 JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "lynn-nvfp4-size-audit-v1":
        raise ValueError(f"Unexpected schema: {data.get('schema')}")
    return data


def _build_tiers(p199: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the four release-gate tiers."""
    current_gib: float = p199["total_gib"]
    cat = p199.get("category_breakdown", {})
    kept = p199.get("kept_bf16_breakdown", {})

    quantized_total_gib: float = cat.get("quantized_total", {}).get("gib", 0.0)
    kept_bf16_total_gib: float = cat.get("kept_bf16_total", {}).get("gib", 0.0)
    non_tensor_gib: float = cat.get("non_tensor_metadata", {}).get("gib", 0.0)
    embed_gib: float = kept.get("embed_tokens", {}).get("gib", 0.0)
    lmhead_gib: float = kept.get("lm_head", {}).get("gib", 0.0)
    other_bf16_gib = kept_bf16_total_gib - embed_gib - lmhead_gib

    tiers: list[dict[str, Any]] = []

    # ── Tier 1: SAFE_NO_CHANGE ────────────────────────────────────────────────
    tiers.append({
        "tier": "SAFE_NO_CHANGE",
        "verdict": "PASS_STABLE",
        "description": (
            "Ship current artifact as-is.  No embed/lm_head quantization.  "
            "Known-good quality.  Size 8.248 GiB."
        ),
        "estimated_size_gib": round(current_gib, 3),
        "size_delta_gib": 0.0,
        "vs_q4km_delta_gib": round(current_gib - Q4KM_TARGET_GIB, 3),
        "changes": [],
        "quality_risk": "none",
        "required_gates": [],
    })

    # ── Tier 2: COMPACT_EMBED_ONLY ────────────────────────────────────────────
    embed_only_gib = quantized_total_gib + ESTIMATED_TENSOR_NVFP4_GIB + other_bf16_gib + non_tensor_gib + lmhead_gib
    tiers.append({
        "tier": "COMPACT_EMBED_ONLY",
        "verdict": "NEEDS_QUALITY_GATE",
        "description": (
            f"Quantize embed_tokens.weight from BF16 ({embed_gib:.3f} GiB) to "
            f"NVFP4 (~{ESTIMATED_TENSOR_NVFP4_GIB:.2f} GiB).  "
            f"lm_head stays BF16.  Saves ~{embed_gib - ESTIMATED_TENSOR_NVFP4_GIB:.2f} GiB."
        ),
        "estimated_size_gib": round(embed_only_gib, 3),
        "size_delta_gib": round(embed_only_gib - current_gib, 3),
        "vs_q4km_delta_gib": round(embed_only_gib - Q4KM_TARGET_GIB, 3),
        "changes": [
            f"quantize model.language_model.embed_tokens.weight ({embed_gib:.3f} GiB BF16 → ~{ESTIMATED_TENSOR_NVFP4_GIB:.2f} GiB NVFP4)"
        ],
        "quality_risk": "moderate",
        "required_gates": COMPACT_REQUIRED_GATES,
    })

    # ── Tier 3: COMPACT_LMHEAD_ONLY ──────────────────────────────────────────
    lmhead_only_gib = quantized_total_gib + ESTIMATED_TENSOR_NVFP4_GIB + other_bf16_gib + non_tensor_gib + embed_gib
    tiers.append({
        "tier": "COMPACT_LMHEAD_ONLY",
        "verdict": "NEEDS_QUALITY_GATE",
        "description": (
            f"Quantize lm_head.weight from BF16 ({lmhead_gib:.3f} GiB) to "
            f"NVFP4 (~{ESTIMATED_TENSOR_NVFP4_GIB:.2f} GiB).  "
            f"embed_tokens stays BF16.  Saves ~{lmhead_gib - ESTIMATED_TENSOR_NVFP4_GIB:.2f} GiB."
        ),
        "estimated_size_gib": round(lmhead_only_gib, 3),
        "size_delta_gib": round(lmhead_only_gib - current_gib, 3),
        "vs_q4km_delta_gib": round(lmhead_only_gib - Q4KM_TARGET_GIB, 3),
        "changes": [
            f"quantize lm_head.weight ({lmhead_gib:.3f} GiB BF16 → ~{ESTIMATED_TENSOR_NVFP4_GIB:.2f} GiB NVFP4)"
        ],
        "quality_risk": "high",
        "required_gates": COMPACT_REQUIRED_GATES,
    })

    # ── Tier 4: COMPACT_EMBED_LMHEAD ─────────────────────────────────────────
    both_gib = quantized_total_gib + 2 * ESTIMATED_TENSOR_NVFP4_GIB + other_bf16_gib + non_tensor_gib
    tiers.append({
        "tier": "COMPACT_EMBED_LMHEAD",
        "verdict": "NEEDS_FULL_MMLU_GPQA_STRUCTURED_GATE",
        "description": (
            f"Quantize both embed_tokens and lm_head from BF16 to NVFP4.  "
            f"Saves ~{(embed_gib + lmhead_gib) - 2 * ESTIMATED_TENSOR_NVFP4_GIB:.2f} GiB.  "
            f"Target range 5.3–5.6 GiB.  Highest quality risk of all tiers."
        ),
        "estimated_size_gib": round(both_gib, 3),
        "size_delta_gib": round(both_gib - current_gib, 3),
        "vs_q4km_delta_gib": round(both_gib - Q4KM_TARGET_GIB, 3),
        "changes": [
            f"quantize model.language_model.embed_tokens.weight ({embed_gib:.3f} GiB BF16 → ~{ESTIMATED_TENSOR_NVFP4_GIB:.2f} GiB NVFP4)",
            f"quantize lm_head.weight ({lmhead_gib:.3f} GiB BF16 → ~{ESTIMATED_TENSOR_NVFP4_GIB:.2f} GiB NVFP4)",
        ],
        "quality_risk": "very high",
        "required_gates": COMPACT_REQUIRED_GATES,
    })

    return tiers


def _build_report(p199_path: Path, p199: dict[str, Any]) -> dict[str, Any]:
    """Build the full gate report."""
    tiers = _build_tiers(p199)
    return {
        "schema": SCHEMA,
        "p199_source": str(p199_path),
        "source_model_dir": p199.get("model_dir", ""),
        "current_artifact_gib": p199["total_gib"],
        "q4km_target_gib": Q4KM_TARGET_GIB,
        "p199_summary": {
            "quantized_total_gib": p199.get("category_breakdown", {}).get("quantized_total", {}).get("gib"),
            "kept_bf16_total_gib": p199.get("category_breakdown", {}).get("kept_bf16_total", {}).get("gib"),
            "embed_tokens_gib": p199.get("kept_bf16_breakdown", {}).get("embed_tokens", {}).get("gib"),
            "lm_head_gib": p199.get("kept_bf16_breakdown", {}).get("lm_head", {}).get("gib"),
            "non_tensor_metadata_gib": p199.get("category_breakdown", {}).get("non_tensor_metadata", {}).get("gib"),
        },
        "key_facts": [
            "Current stable NVFP4 W4A16 artifact is 8.248 GiB.",
            "embed_tokens (1.895 GiB) and lm_head (1.895 GiB) are the two largest BF16 tensors.",
            "They are NOT tied — cosine similarity 0.0198, must be quantized independently.",
            "Quantizing one saves ~1.27 GiB; quantizing both saves ~2.55 GiB.",
            "Compact NVFP4 is NOT a transparent repack — it changes quantization of output-critical tensors.",
            "lm_head FP4 exact-match gate currently FAILS on 9B (P136b: exact 1/3).",
            "embed_tokens FP4 exact-match gate has NOT been tested on 9B.",
            "Quality impact must be measured before any compact tier can be promoted.",
        ],
        "tiers": tiers,
        "recommended_tier": "SAFE_NO_CHANGE",
        "recommended_reason": (
            "All compact tiers are NEEDS_* — no quality data exists for embed/lm_head "
            "quantization on Qwen3.5-9B.  Ship SAFE_NO_CHANGE now; run quality gates "
            "on COMPACT_EMBED_LMHEAD in parallel.  Promote only after full gate pass."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Compact NVFP4 shrink gate")
    parser.add_argument(
        "--p199-json",
        type=Path,
        default=Path("reports/qwen35_9b/p199_nvfp4_size_audit_20260519_live_size2.json"),
        help="Path to P199 size-audit JSON",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/qwen35_9b/compact_nvfp4_shrink_gate_20260519.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    p199 = _load_p199(args.p199_json)
    report = _build_report(args.p199_json, p199)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[compact-shrink-gate] wrote {args.out_json}")
    print(f"  current:  {report['current_artifact_gib']:.3f} GiB")
    for t in report["tiers"]:
        print(f"  {t['tier']:30s}  {t['estimated_size_gib']:.3f} GiB  {t['verdict']}")


if __name__ == "__main__":
    main()
