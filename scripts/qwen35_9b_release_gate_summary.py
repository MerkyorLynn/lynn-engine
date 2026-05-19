#!/usr/bin/env python3
"""Qwen3.5-9B release gate summary — GPU-free, report-only aggregator.

Reads existing MMLU / GPQA / TPS / structured-content report files under
reports/qwen35_9b/ and emits a single unified JSON with a three-tier
decision: PROMOTE_READY / NEEDS_MORE_DATA / CLOSED.

Missing reports produce null fields, never a crash.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Decision thresholds — edit here, not deep in the logic
# ---------------------------------------------------------------------------
MMLU_MIN: float = 0.74
GPQA_MIN: float = 0.41
GPQA_THINKING32_EX_PARSE_MIN: float = 0.80
GPQA_THINKING32_PARSE_FAIL_MAX: float = 0.25
STRUCTURED_PASS: bool = True       # structured content gate must pass
TPS_DECODE_512_MIN: float = 155.0  # single-request 512-token decode TPS

# ---------------------------------------------------------------------------
# Variant metadata
# ---------------------------------------------------------------------------
VARIANT_DEFS: list[dict[str, Any]] = [
    {
        "variant": "NVFP4",
        "quant": "nvfp4",
        "model_size_gb": 8.3,
        "mmlu_patterns": ["nvfp4*_mmlu*.summary.json"],
        "gpqa_patterns": ["nvfp4*_gpqa*.summary.json"],
        "tps_patterns": ["r6000_qwen35_9b_nvfp4_openai_matrix_full_*.json"],
        "quality_summary_patterns": [],
    },
    {
        "variant": "Q4_K_M",
        "quant": "q4km",
        "model_size_gb": 5.5,
        "mmlu_patterns": ["q4km_*_mmlu*.summary.json"],
        "gpqa_patterns": ["q4km_*_gpqa*.summary.json"],
        "tps_patterns": ["r6000_qwen35_9b_q4km_baseline_*.json",
                         "r6000_qwen35_9b_q4km_cuda_baseline_*.json"],
        "quality_summary_patterns": ["q4km_*_quality_summary.json"],
    },
    {
        "variant": "BF16",
        "quant": "bf16",
        "model_size_gb": 18.0,
        "mmlu_patterns": ["bf16_*_mmlu*.summary.json"],
        "gpqa_patterns": ["bf16_*_gpqa*.summary.json"],
        "tps_patterns": [],
        "quality_summary_patterns": ["bf16_*_quality_summary.json"],
    },
]

THINKING32_GPQA_PATTERNS = ["p201_gpqa_live_summary_*.json"]
STRUCTURED_MD_PATTERNS = ["P196_W4A8_STRUCTURED_CONTENT_GATE_*.md"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _json_load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _latest(reports_dir: Path, patterns: list[str]) -> Path | None:
    """Return the most-recently-modified file matching any of the glob patterns."""
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(reports_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_mmlu(reports_dir: Path, vdef: dict[str, Any]) -> float | None:
    """Extract MMLU accuracy from variant-specific or quality-summary files."""
    path = _latest(reports_dir, vdef["mmlu_patterns"])
    if path is not None:
        data = _json_load(path)
        if data:
            return _safe_float(data.get("accuracy")) or _safe_float(data.get("score"))
    # Fallback: quality summary may contain mmlu sub-object
    path = _latest(reports_dir, vdef.get("quality_summary_patterns", []))
    if path is not None:
        data = _json_load(path)
        if data and "mmlu" in data:
            return _safe_float(data["mmlu"].get("accuracy")) or _safe_float(data["mmlu"].get("score"))
    # Fallback: release matrix
    return _extract_from_matrix(reports_dir, vdef["variant"], "mmlu")


def _extract_gpqa(reports_dir: Path, vdef: dict[str, Any]) -> float | None:
    """Extract GPQA accuracy (non-thinking, standard eval)."""
    path = _latest(reports_dir, vdef["gpqa_patterns"])
    if path is not None:
        data = _json_load(path)
        if data:
            return _safe_float(data.get("accuracy")) or _safe_float(data.get("score"))
    path = _latest(reports_dir, vdef.get("quality_summary_patterns", []))
    if path is not None:
        data = _json_load(path)
        if data and "gpqa" in data:
            return _safe_float(data["gpqa"].get("accuracy")) or _safe_float(data["gpqa"].get("score"))
    return _extract_from_matrix(reports_dir, vdef["variant"], "gpqa")


def _extract_from_matrix(
    reports_dir: Path, variant: str, metric: str
) -> float | None:
    """Last-resort: pull from qwen35_9b_release_matrix.json."""
    path = reports_dir / "qwen35_9b_release_matrix.json"
    data = _json_load(path)
    if not data:
        return None
    for entry in data.get("entries", []):
        if entry.get("variant") == variant:
            return _safe_float(entry.get(metric, {}).get("score"))
    return None


def _extract_tps_decode_512(reports_dir: Path, vdef: dict[str, Any]) -> float | None:
    """Extract single-request 512-token wall TPS from baseline reports."""
    # The release matrix is the canonical cross-variant table for user-facing
    # promotion decisions. Prefer it over ad-hoc probe files because older probe
    # artifacts may use a different server profile or a short smoke path.
    matrix_value = _extract_from_matrix_tps(reports_dir, vdef["variant"], "512")
    if matrix_value is not None:
        return matrix_value

    path = _latest(reports_dir, vdef["tps_patterns"])
    if path is None:
        return None
    data = _json_load(path)
    if not data:
        return None
    # OpenAI matrix format: single.rows[].wall_tps where max_tokens==512
    for row in data.get("single", {}).get("rows", []):
        if row.get("max_tokens") == 512 and row.get("ok"):
            return _safe_float(row.get("wall_tps"))
    # Baseline format: single_tps.512
    stps = data.get("single_tps", {})
    val = stps.get("512") if isinstance(stps, dict) else None
    if val is not None:
        if isinstance(val, dict):
            return _safe_float(val.get("wall_tps")) or _safe_float(val.get("decode_tps"))
        return _safe_float(val)
    return None


def _extract_from_matrix_tps(
    reports_dir: Path, variant: str, key: str
) -> float | None:
    path = reports_dir / "qwen35_9b_release_matrix.json"
    data = _json_load(path)
    if not data:
        return None
    for entry in data.get("entries", []):
        if entry.get("variant") == variant:
            return _safe_float(entry.get("single_tps", {}).get(key))
    return None


def _extract_thinking32_gpqa(reports_dir: Path) -> dict[str, float | None]:
    """Extract GPQA Thinking-32K naive and ex-parse-fail accuracy."""
    path = _latest(reports_dir, THINKING32_GPQA_PATTERNS)
    if path is None:
        return {"naive": None, "ex_parse_fail": None, "parse_fail_rate": None}
    data = _json_load(path)
    if not data:
        return {"naive": None, "ex_parse_fail": None, "parse_fail_rate": None}
    overall = data.get("overall", {})
    return {
        "naive": _safe_float(overall.get("accuracy")),
        "ex_parse_fail": _safe_float(overall.get("accuracy_excluding_parse_fail")),
        "parse_fail_rate": _safe_float(overall.get("parse_fail_rate")),
    }


def _extract_structured_pass(reports_dir: Path) -> bool | None:
    """Parse the P196 structured content gate Markdown for relative regression.

    Returns True if W4A8 shows no regression vs W4A16, False if regression
    detected, None if no P196 report found.
    """
    path = _latest(reports_dir, STRUCTURED_MD_PATTERNS)
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")

    # Look for the decision line indicating regression
    if "REGRESSION" in text and "NO_REGRESSION" not in text:
        return False

    # Parse the table to extract W4A16 and W4A8 full pass rates
    w4a16_rate: float | None = None
    w4a8_full_rate: float | None = None
    row_pattern = re.compile(r"^\|\s*(.*?)\s*\|", re.IGNORECASE)
    for line in text.splitlines():
        line_lower = line.lower().strip()
        if not line_lower.startswith("|") or "pass rate" in line_lower or "---" in line_lower:
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 4:
            continue
        label = cols[1].lower()
        rate_match = re.search(r"(\d+(?:\.\d+)?)\s*%", cols[3] if len(cols) > 3 else "")
        if rate_match is None:
            continue
        rate = float(rate_match.group(1)) / 100.0
        if "w4a16" in label and "reference" in label:
            w4a16_rate = rate
        elif "w4a8" in label and "full" in label:
            w4a8_full_rate = rate

    if w4a16_rate is None or w4a8_full_rate is None:
        # Can't determine — treat as unknown
        return None

    # No regression if W4A8 full rate >= W4A16 rate - 2% absolute
    return w4a8_full_rate >= w4a16_rate - 0.02


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _decide(
    *,
    mmlu: float | None,
    gpqa: float | None,
    gpqa_thinking32_ex_parse_fail: float | None,
    parse_fail_rate: float | None,
    structured_pass: bool | None,
    tps_decode_512: float | None,
) -> tuple[str, list[str]]:
    """Return (decision, reasons).

    PROMOTE_READY  — all present metrics meet thresholds, none missing.
    NEEDS_MORE_DATA — at least one required metric is missing, no hard fail.
    CLOSED         — at least one present metric is below threshold.
    """
    reasons: list[str] = []
    closed = False
    missing = False

    # MMLU
    if mmlu is None:
        missing = True
        reasons.append("MMLU data missing")
    elif mmlu < MMLU_MIN:
        closed = True
        reasons.append(f"MMLU {mmlu:.4f} < {MMLU_MIN}")
    else:
        reasons.append(f"MMLU {mmlu:.4f} >= {MMLU_MIN} OK")

    # GPQA. For the 9B release, thinking-on GPQA is a first-class signal. A
    # low reasoning-off GPQA score is not a hard close if the 32K thinking-on
    # run clears the ex-parse-fail floor with a bounded parse-fail rate.
    thinking32_ok = (
        gpqa_thinking32_ex_parse_fail is not None
        and gpqa_thinking32_ex_parse_fail >= GPQA_THINKING32_EX_PARSE_MIN
        and (parse_fail_rate is None or parse_fail_rate <= GPQA_THINKING32_PARSE_FAIL_MAX)
    )
    if gpqa is not None and gpqa >= GPQA_MIN:
        reasons.append(f"GPQA {gpqa:.4f} >= {GPQA_MIN} OK")
    elif thinking32_ok:
        pf_text = "unknown" if parse_fail_rate is None else f"{parse_fail_rate:.4f}"
        reasons.append(
            "GPQA reasoning-off below floor but thinking32 clears gate: "
            f"ex_parse_fail={gpqa_thinking32_ex_parse_fail:.4f} >= "
            f"{GPQA_THINKING32_EX_PARSE_MIN}, parse_fail={pf_text}"
        )
    elif gpqa is None and gpqa_thinking32_ex_parse_fail is None:
        missing = True
        reasons.append("GPQA data missing")
    elif gpqa is not None and gpqa < GPQA_MIN and gpqa_thinking32_ex_parse_fail is None:
        missing = True
        reasons.append(f"GPQA {gpqa:.4f} < {GPQA_MIN}; thinking32 data missing")
    else:
        closed = True
        reasons.append(
            f"GPQA {gpqa} below floor and thinking32 gate not cleared "
            f"(ex_parse_fail={gpqa_thinking32_ex_parse_fail}, parse_fail={parse_fail_rate})"
        )

    # Structured pass
    if structured_pass is None:
        missing = True
        reasons.append("Structured content gate data missing")
    elif not structured_pass:
        closed = True
        reasons.append("Structured content gate FAILED (regression vs W4A16)")
    else:
        reasons.append("Structured content gate PASS")

    # TPS decode 512
    if tps_decode_512 is None:
        missing = True
        reasons.append("TPS decode 512 data missing")
    elif tps_decode_512 < TPS_DECODE_512_MIN:
        closed = True
        reasons.append(f"TPS decode 512 {tps_decode_512:.1f} < {TPS_DECODE_512_MIN}")
    else:
        reasons.append(f"TPS decode 512 {tps_decode_512:.1f} >= {TPS_DECODE_512_MIN} OK")

    if closed:
        return "CLOSED", reasons
    if missing:
        return "NEEDS_MORE_DATA", reasons
    return "PROMOTE_READY", reasons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_summary(reports_dir: Path) -> dict[str, Any]:
    thinking32 = _extract_thinking32_gpqa(reports_dir)
    structured = _extract_structured_pass(reports_dir)

    variants: list[dict[str, Any]] = []
    for vdef in VARIANT_DEFS:
        mmlu = _extract_mmlu(reports_dir, vdef)
        gpqa = _extract_gpqa(reports_dir, vdef)
        tps = _extract_tps_decode_512(reports_dir, vdef)

        decision, reasons = _decide(
            mmlu=mmlu,
            gpqa=gpqa,
            gpqa_thinking32_ex_parse_fail=thinking32["ex_parse_fail"],
            parse_fail_rate=thinking32["parse_fail_rate"],
            structured_pass=structured,
            tps_decode_512=tps,
        )

        variants.append({
            "schema": "lynn-qwen35-9b-release-gate-summary-v1",
            "created": _iso_now(),
            "reports_dir": str(reports_dir),
            "variant": vdef["variant"],
            "quant": vdef["quant"],
            "model_size_gb": vdef["model_size_gb"],
            "mmlu": mmlu,
            "gpqa": gpqa,
            "gpqa_thinking32_naive": thinking32["naive"],
            "gpqa_thinking32_ex_parse_fail": thinking32["ex_parse_fail"],
            "parse_fail_rate": thinking32["parse_fail_rate"],
            "tps_decode_512": tps,
            "structured_pass": structured,
            "decision": decision,
            "decision_reasons": reasons,
        })

    return {
        "schema": "lynn-qwen35-9b-release-gate-summary-collection-v1",
        "created": _iso_now(),
        "reports_dir": str(reports_dir),
        "thresholds": {
            "mmlu_min": MMLU_MIN,
            "gpqa_min": GPQA_MIN,
            "gpqa_thinking32_ex_parse_min": GPQA_THINKING32_EX_PARSE_MIN,
            "gpqa_thinking32_parse_fail_max": GPQA_THINKING32_PARSE_FAIL_MAX,
            "structured_pass": STRUCTURED_PASS,
            "tps_decode_512_min": TPS_DECODE_512_MIN,
        },
        "variants": variants,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/qwen35_9b"),
        help="Directory containing Qwen3.5-9B report files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/qwen35_9b/qwen35_9b_release_gate_summary_latest.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports_dir = args.reports_dir.resolve()
    if not reports_dir.is_dir():
        print(f"[ERROR] reports dir not found: {reports_dir}", flush=True)
        return 1

    summary = build_summary(reports_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Print compact status to stdout
    for v in summary["variants"]:
        print(
            f"[{v['variant']}] decision={v['decision']}  "
            f"MMLU={v['mmlu']}  GPQA={v['gpqa']}  "
            f"TPS512={v['tps_decode_512']}  structured={v['structured_pass']}"
        )
    print(f"[summary] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
