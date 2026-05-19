#!/usr/bin/env python3
"""P193 · Qwen3.5-9B native packed boundary admission gate.

Unifies upstream quality/speed/capability gates into a single decision
before any native FP4×FP8 packed boundary reaches resident serving.

Consumed reports (all optional; absent = skipped):
  - P160  dense FFN fixture contract  (cosine, max_abs, exact)
  - P185  dense W4A8 fixture gate     (cosine, rel_l2, speedup)
  - P189  FP4×FP8 capability probe    (torch._scaled_mm vs CuTe)
  - P191  CuTe PoC kernel report      (scalar ref correctness)
  - P192  offline repack manifest     (repack contract pass/fail)

Decision levels (strictest → loosest):
  CLOSED_NUMERIC       cosine/rel_l2 outside safe envelope or P160 RED
  AMBER_FIXTURE_FAST   fixture quality marginal but speed improves
  GREEN_FIXTURE        all available gates pass, numerics solid
  PROMOTE_BLOCKED      numerics OK but capability/contract gates block

No engine/*, csrc/*, server/*, or Triton kernel edits.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA = "lynn-qwen35-9b-p193-native-boundary-admission-v1"

# ── Thresholds (CLI-overridable) ──────────────────────────────────────────────
DEFAULT_THRESHOLDS: dict[str, float] = {
    "cosine_closed": 0.995,       # below → CLOSED_NUMERIC
    "cosine_green": 0.999,        # above → eligible for GREEN
    "rel_l2_closed": 0.10,        # above → CLOSED_NUMERIC
    "rel_l2_green": 0.05,         # below → eligible for GREEN
    "speedup_amber": 1.00,        # speedup >= this + amber cosine → AMBER_FIXTURE_FAST
}


# ── Report loaders ────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest(report_dir: Path, prefix: str) -> Path | None:
    """Return the most-recent file matching prefix*.json, or None."""
    candidates = sorted(report_dir.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_p160(report_dir: Path, explicit: Path | None) -> dict[str, Any] | None:
    path = explicit or _find_latest(report_dir, "p160_")
    return _load_json(path) if path else None


def load_p185(report_dir: Path, explicit: Path | None) -> dict[str, Any] | None:
    path = explicit or _find_latest(report_dir, "p185_")
    return _load_json(path) if path else None


def load_p189(report_dir: Path, explicit: Path | None) -> dict[str, Any] | None:
    path = explicit or _find_latest(report_dir, "p189_")
    return _load_json(path) if path else None


def load_p191(report_dir: Path, explicit: Path | None) -> dict[str, Any] | None:
    path = explicit or _find_latest(report_dir, "p191_")
    return _load_json(path) if path else None


def load_p192(report_dir: Path, explicit: Path | None) -> dict[str, Any] | None:
    if explicit:
        return _load_json(explicit)
    # Try p192b_ (contract) first, then p192_ (manifest)
    for prefix in ("p192b_", "p192_"):
        path = _find_latest(report_dir, prefix)
        if path:
            return _load_json(path)
    return None


# ── Gate extractors ───────────────────────────────────────────────────────────

def _extract_p160(data: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from P160 fixture contract report."""
    return {
        "gate": "P160",
        "decision": data.get("decision", "UNKNOWN"),
        "cosine_min": data.get("cosine_min"),
        "max_abs_max": data.get("max_abs_max"),
        "passed": data.get("passed"),
        "total": data.get("total"),
        "exact": data.get("exact"),
    }


def _extract_p185(data: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from P185 W4A8 fixture gate report."""
    summaries = data.get("summaries", [])
    full = next((s for s in summaries if s.get("mode") == "full"), {})
    gateup = next((s for s in summaries if s.get("mode") == "gateup"), {})
    best = full if full.get("cosine_min") is not None else gateup
    return {
        "gate": "P185",
        "decision": data.get("decision", "UNKNOWN"),
        "cosine_min": best.get("cosine_min"),
        "rel_l2_max": best.get("rel_l2_max"),
        "speedup_vs_ref_mean": best.get("speedup_vs_ref_mean"),
        "fp8_format": data.get("fp8_format"),
        "granularity": data.get("granularity"),
    }


def _extract_p189(data: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from P189 capability probe report."""
    return {
        "gate": "P189",
        "decision": data.get("decision", "UNKNOWN"),
        "cute_sm120_e4m3_e2m1_header": data.get("cute_sm120_e4m3_e2m1_header"),
        "device": data.get("device"),
    }


def _extract_p191(data: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from P191 CuTe PoC kernel report.

    Original P191 schema (p191_dense_fp4xfp8_poc):
      mma_compiled: bool
      scalar_reference_available: bool
      results[]: layer_id, max_abs_vs_bf16_ref, rel_l2_vs_bf16_ref,
                 cosine_vs_bf16_ref, scalar_ms, mma_available

    Real-MMA P191 schema (p191_dense_fp4xfp8_mma_real):
      results[].scalar_reference.{max_abs_vs_bf16_ref, rel_l2_vs_bf16_ref,
                                  cosine_vs_bf16_ref, scalar_ms}
      results[].mma_kernel.{available, real_compute, mma_vs_scalar_max_abs,
                            mma_vs_scalar_cosine, mma_ms, error}
    """
    results = data.get("results", [])

    def _finite(value: Any) -> float | None:
        if value is None:
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    scalar_cosines: list[float] = []
    scalar_rel_l2s: list[float] = []
    scalar_max_abs: list[float] = []
    scalar_ms: list[float] = []
    mma_cosines: list[float] = []
    mma_max_abs: list[float] = []
    mma_ms: list[float] = []
    mma_real_flags: list[bool] = []
    mma_available_flags: list[bool] = []
    scaled_cosines: list[float] = []
    scaled_rel_l2s: list[float] = []
    scaled_max_abs: list[float] = []
    scaled_ms: list[float] = []
    scaled_real_flags: list[bool] = []

    for row in results:
        scalar = row.get("scalar_reference") if isinstance(row.get("scalar_reference"), dict) else row
        mma = row.get("mma_kernel") if isinstance(row.get("mma_kernel"), dict) else row

        for target, key in (
            (scalar_cosines, "cosine_vs_bf16_ref"),
            (scalar_rel_l2s, "rel_l2_vs_bf16_ref"),
            (scalar_max_abs, "max_abs_vs_bf16_ref"),
            (scalar_ms, "scalar_ms"),
        ):
            value = _finite(scalar.get(key))
            if value is not None:
                target.append(value)

        value = _finite(mma.get("mma_vs_scalar_cosine"))
        if value is not None:
            mma_cosines.append(value)
        value = _finite(mma.get("mma_vs_scalar_max_abs"))
        if value is not None:
            mma_max_abs.append(value)
        value = _finite(mma.get("mma_ms"))
        if value is not None:
            mma_ms.append(value)
        if "real_compute" in mma:
            mma_real_flags.append(bool(mma.get("real_compute")))
        if "available" in mma:
            mma_available_flags.append(bool(mma.get("available")))
        elif "mma_available" in row:
            mma_available_flags.append(bool(row.get("mma_available")))
        scaled = row.get("scaled_mma_kernel")
        if isinstance(scaled, dict):
            value = _finite(scaled.get("scaled_vs_scalar_cosine"))
            if value is not None:
                scaled_cosines.append(value)
            value = _finite(scaled.get("scaled_vs_scalar_rel_l2"))
            if value is not None:
                scaled_rel_l2s.append(value)
            value = _finite(scaled.get("scaled_vs_scalar_max_abs"))
            if value is not None:
                scaled_max_abs.append(value)
            value = _finite(scaled.get("scaled_ms"))
            if value is not None:
                scaled_ms.append(value)
            if "real_compute" in scaled:
                scaled_real_flags.append(bool(scaled.get("real_compute")))

    scalar_ms_mean = sum(scalar_ms) / len(scalar_ms) if scalar_ms else None
    mma_ms_mean = sum(mma_ms) / len(mma_ms) if mma_ms else None
    return {
        "gate": "P191",
        "mma_compiled": data.get("mma_compiled"),
        "scalar_reference_available": data.get("scalar_reference_available"),
        "layers_tested": len(results),
        "cosine_min": min(scalar_cosines) if scalar_cosines else None,
        "rel_l2_max": max(scalar_rel_l2s) if scalar_rel_l2s else None,
        "max_abs_max": max(scalar_max_abs) if scalar_max_abs else None,
        "scalar_ms_mean": scalar_ms_mean,
        "mma_available_all": all(mma_available_flags) if mma_available_flags else None,
        "mma_real_compute_all": all(mma_real_flags) if mma_real_flags else None,
        "mma_vs_scalar_cosine_min": min(mma_cosines) if mma_cosines else None,
        "mma_vs_scalar_max_abs_max": max(mma_max_abs) if mma_max_abs else None,
        "mma_ms_mean": mma_ms_mean,
        "mma_speedup_vs_scalar": (
            scalar_ms_mean / mma_ms_mean
            if scalar_ms_mean is not None and mma_ms_mean is not None and mma_ms_mean > 0
            else None
        ),
        "scaled_mma_real_compute_all": all(scaled_real_flags) if scaled_real_flags else None,
        "scaled_vs_scalar_cosine_min": min(scaled_cosines) if scaled_cosines else None,
        "scaled_vs_scalar_rel_l2_max": max(scaled_rel_l2s) if scaled_rel_l2s else None,
        "scaled_vs_scalar_max_abs_max": max(scaled_max_abs) if scaled_max_abs else None,
        "scaled_ms_mean": (sum(scaled_ms) / len(scaled_ms)) if scaled_ms else None,
    }


def _extract_p192(data: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from P192 repack contract.

    Supports two schemas:
      p192_  (repack manifest): schema_version, layers{}, overall
      p192b_ (contract):        schema=...repack-contract-v1, results[], overall
    """
    schema = data.get("schema", data.get("schema_version", ""))
    overall = data.get("overall", "UNKNOWN")

    # P192b contract: results[].ok per layer
    results = data.get("results", [])
    if results:
        all_ok = all(r.get("ok", False) for r in results)
        failed_layers = [r["layer"] for r in results if not r.get("ok", False)]
        return {
            "gate": "P192",
            "schema": schema,
            "overall": "GREEN" if all_ok else "RED",
            "layers_checked": len(results),
            "failed_layers": failed_layers,
        }

    # P192 repack manifest: layers{} per layer
    layers = data.get("layers", {})
    if layers:
        failed = [int(k) for k, v in layers.items() if "error" in v or v.get("ok") is False]
        return {
            "gate": "P192",
            "schema": schema,
            "overall": overall,
            "layers_checked": len(layers),
            "failed_layers": failed,
        }

    return {
        "gate": "P192",
        "schema": schema,
        "overall": overall,
    }


# ── Decision engine ───────────────────────────────────────────────────────────

def _decide(
    p160: dict[str, Any] | None,
    p185: dict[str, Any] | None,
    p189: dict[str, Any] | None,
    p191: dict[str, Any] | None,
    p192: dict[str, Any] | None,
    thresholds: dict[str, float],
) -> tuple[str, list[str]]:
    """Return (decision, reasons)."""
    reasons: list[str] = []

    # ── Numeric envelope check ────────────────────────────────────────────
    numeric_ok = True
    numeric_green = True

    if p160 is not None:
        if p160["decision"] == "DENSE_FFN_FIXTURE_RED":
            reasons.append("P160 decision=DENSE_FFN_FIXTURE_RED")
            return "CLOSED_NUMERIC", reasons
        cos = p160.get("cosine_min")
        if cos is not None and cos < thresholds["cosine_closed"]:
            reasons.append(f"P160 cosine_min={cos:.6f} < {thresholds['cosine_closed']}")
            return "CLOSED_NUMERIC", reasons
        if cos is not None and cos < thresholds["cosine_green"]:
            numeric_green = False
            reasons.append(f"P160 cosine_min={cos:.6f} < green threshold {thresholds['cosine_green']}")
    else:
        reasons.append("P160 report absent -- numeric baseline missing")
        numeric_green = False

    if p185 is not None:
        p185_decision = p185["decision"]
        if p185_decision == "DENSE_W4A8_FIXTURE_RED":
            reasons.append("P185 decision=DENSE_W4A8_FIXTURE_RED")
            return "CLOSED_NUMERIC", reasons
        cos = p185.get("cosine_min")
        if cos is not None and cos < thresholds["cosine_closed"]:
            reasons.append(f"P185 cosine_min={cos:.6f} < {thresholds['cosine_closed']}")
            return "CLOSED_NUMERIC", reasons
        r2 = p185.get("rel_l2_max")
        if r2 is not None and r2 > thresholds["rel_l2_closed"]:
            reasons.append(f"P185 rel_l2_max={r2:.6f} > {thresholds['rel_l2_closed']}")
            return "CLOSED_NUMERIC", reasons
        if cos is not None and cos < thresholds["cosine_green"]:
            numeric_green = False
        if r2 is not None and r2 > thresholds["rel_l2_green"]:
            numeric_green = False
    else:
        reasons.append("P185 report absent -- W4A8 quality not verified")

    # ── P191 kernel numeric check ─────────────────────────────────────────
    if p191 is not None:
        cos = p191.get("cosine_min")
        r2 = p191.get("rel_l2_max")
        mma_real = p191.get("mma_real_compute_all")
        mma_cos = p191.get("mma_vs_scalar_cosine_min")
        if cos is not None and cos < thresholds["cosine_closed"]:
            reasons.append(f"P191 cosine_min={cos:.6f} < {thresholds['cosine_closed']}")
            return "CLOSED_NUMERIC", reasons
        if r2 is not None and r2 > thresholds["rel_l2_closed"]:
            reasons.append(f"P191 rel_l2_max={r2:.6f} > {thresholds['rel_l2_closed']}")
            return "CLOSED_NUMERIC", reasons
        if mma_real:
            if mma_cos is None or mma_cos < thresholds["cosine_green"]:
                reasons.append(
                    "P191 real MMA fragment layout incorrect: "
                    f"mma_vs_scalar_cosine_min={mma_cos}"
                )
                return "CLOSED_NUMERIC", reasons
        scaled_real = p191.get("scaled_mma_real_compute_all")
        scaled_cos = p191.get("scaled_vs_scalar_cosine_min")
        scaled_r2 = p191.get("scaled_vs_scalar_rel_l2_max")
        if scaled_real:
            if scaled_cos is None or scaled_cos < thresholds["cosine_green"]:
                reasons.append(
                    "P191 scaled MMA does not match scalar reference: "
                    f"scaled_vs_scalar_cosine_min={scaled_cos}"
                )
                return "CLOSED_NUMERIC", reasons
            if scaled_r2 is not None and scaled_r2 > thresholds["rel_l2_green"]:
                reasons.append(
                    "P191 scaled MMA rel_l2 outside green envelope: "
                    f"scaled_vs_scalar_rel_l2_max={scaled_r2:.6f}"
                )
                return "CLOSED_NUMERIC", reasons
        if cos is not None and cos < thresholds["cosine_green"]:
            numeric_green = False
            reasons.append(f"P191 cosine_min={cos:.6f} < green {thresholds['cosine_green']}")
        if r2 is not None and r2 > thresholds["rel_l2_green"]:
            numeric_green = False

    # ── P192 repack contract check ────────────────────────────────────────
    if p192 is not None:
        p192_overall = p192.get("overall", "UNKNOWN")
        if p192_overall == "RED":
            failed = p192.get("failed_layers", [])
            reasons.append(f"P192 overall=RED, failed layers: {failed}")
            return "CLOSED_NUMERIC", reasons

    # ── Speed check (AMBER_FIXTURE_FAST) ──────────────────────────────────
    speedup = None
    if p185 is not None:
        speedup = p185.get("speedup_vs_ref_mean")

    if not numeric_green and speedup is not None and speedup >= thresholds["speedup_amber"]:
        reasons.append(f"numeric below green but speedup={speedup:.3f}x >= {thresholds['speedup_amber']}")
        return "AMBER_FIXTURE_FAST", reasons

    if not numeric_green:
        reasons.append("numeric below green threshold and no compensating speedup")
        return "CLOSED_NUMERIC", reasons

    # ── Capability / contract gates ────────────────────────────────────────
    capability_blocked = False

    if p189 is not None:
        p189_decision = p189["decision"]
        if p189_decision == "TORCH_MIXED_FP4XFP8_UNAVAILABLE_CUTE_REQUIRED":
            if p191 is not None and p191.get("mma_real_compute_all", False):
                reasons.append("P189: torch._scaled_mm lacks FP8×FP4, covered by P191 CuTe real MMA")
            else:
                reasons.append("P189: torch._scaled_mm cannot do FP8×FP4; CuTe kernel required")
                capability_blocked = True
        elif p189_decision == "FP4_FP8_TOOLCHAIN_INCOMPLETE":
            reasons.append("P189: FP4×FP8 toolchain incomplete")
            capability_blocked = True

    if p191 is not None:
        # If MMA didn't compile, the kernel path is blocked
        if not p191.get("mma_compiled", False):
            reasons.append("P191: MMA instruction did not compile")
            capability_blocked = True
        if not p191.get("scalar_reference_available", False):
            reasons.append("P191: scalar reference unavailable")
            capability_blocked = True
        if not p191.get("mma_real_compute_all", False):
            reasons.append("P191: real MMA compute not available yet")
            capability_blocked = True
        if p191.get("scaled_mma_real_compute_all") is False:
            reasons.append("P191: scaled MMA compute not available yet")
            capability_blocked = True

    if p192 is not None:
        if p192.get("overall") == "RED":
            reasons.append(f"P192 repack contract RED, failed: {p192.get('failed_layers', [])}")
            capability_blocked = True

    if capability_blocked:
        return "PROMOTE_BLOCKED", reasons

    # ── All clear ─────────────────────────────────────────────────────────
    reasons.append("all available gates pass")
    return "GREEN_FIXTURE", reasons


# ── Main ──────────────────────────────────────────────────────────────────────

def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = Path(args.report_dir)

    p160_data = load_p160(report_dir, Path(args.p160) if args.p160 else None)
    p185_data = load_p185(report_dir, Path(args.p185) if args.p185 else None)
    p189_data = load_p189(report_dir, Path(args.p189) if args.p189 else None)
    p191_data = load_p191(report_dir, Path(args.p191) if args.p191 else None)
    p192_data = load_p192(report_dir, Path(args.p192) if args.p192 else None)

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.cosine_closed is not None:
        thresholds["cosine_closed"] = args.cosine_closed
    if args.cosine_green is not None:
        thresholds["cosine_green"] = args.cosine_green
    if args.rel_l2_closed is not None:
        thresholds["rel_l2_closed"] = args.rel_l2_closed
    if args.rel_l2_green is not None:
        thresholds["rel_l2_green"] = args.rel_l2_green
    if args.speedup_amber is not None:
        thresholds["speedup_amber"] = args.speedup_amber

    sources: dict[str, Any] = {}
    if p160_data is not None:
        sources["p160"] = _extract_p160(p160_data)
    if p185_data is not None:
        sources["p185"] = _extract_p185(p185_data)
    if p189_data is not None:
        sources["p189"] = _extract_p189(p189_data)
    if p191_data is not None:
        sources["p191"] = _extract_p191(p191_data)
    if p192_data is not None:
        sources["p192"] = _extract_p192(p192_data)

    decision, reasons = _decide(
        sources.get("p160"),
        sources.get("p185"),
        sources.get("p189"),
        sources.get("p191"),
        sources.get("p192"),
        thresholds,
    )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "thresholds": thresholds,
        "sources": sources,
        "decision": decision,
        "reasons": reasons,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"P193 decision: {decision}")
    for r in reasons:
        print(f"  - {r}")
    print(f"Report: {out_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", default=str(ROOT / "reports" / "qwen35_9b"),
                     help="Directory to scan for P160/P185/P189/P191/P192 reports")
    ap.add_argument("--p160", default=None, help="Explicit path to P160 report JSON")
    ap.add_argument("--p185", default=None, help="Explicit path to P185 report JSON")
    ap.add_argument("--p189", default=None, help="Explicit path to P189 report JSON")
    ap.add_argument("--p191", default=None, help="Explicit path to P191 report JSON")
    ap.add_argument("--p192", default=None, help="Explicit path to P192 report JSON")
    ap.add_argument("--cosine-closed", type=float, default=None)
    ap.add_argument("--cosine-green", type=float, default=None)
    ap.add_argument("--rel-l2-closed", type=float, default=None)
    ap.add_argument("--rel-l2-green", type=float, default=None)
    ap.add_argument("--speedup-amber", type=float, default=None)
    ap.add_argument("--out", default=None, help="Output report path")
    args = ap.parse_args()

    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = str(Path(args.report_dir) / f"p193_native_boundary_admission_{ts}.json")

    run_gate(args)


if __name__ == "__main__":
    main()
