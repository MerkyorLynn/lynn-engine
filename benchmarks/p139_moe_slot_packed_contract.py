#!/usr/bin/env python3
"""P139 · MoE slot packed NVFP4 decode contract.

Purpose: Load p138-packed fixtures, dequantize the NVFP4 slot weights back
to BF16, and verify they match the corresponding p135 BF16 slot weights
exactly (max_abs == 0).

Contract metrics (per tensor pair):
  - max_abs:  max|p135_bf16 - p138_dequant_bf16|
  - mean_abs: mean|p135_bf16 - p138_dequant_bf16|
  - rel_l2:   ||diff||_2 / ||ref||_2
  - cosine:   cos(ref, dequant)
  - exact:    1 if max_abs == 0 else 0
  - load_ms:  time to load the packed fixture
  - unpack_ms: time to dequantize both gate_up and down
  - pass/fail: max_abs == 0 (exact match required)

Usage:
  python benchmarks/p139_moe_slot_packed_contract.py \
    --p138-fixtures reports/qwen36_35b/p138_packed_slot_fixtures \
    --p135-fixtures reports/qwen36_35b/p135_repacked_fixtures_official_w4a16 \
    --out reports/qwen36_35b/p139_slot_packed_contract_report.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.dequant import unpack_fp4_e2m1_from_uint8  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Contract metrics
# ─────────────────────────────────────────────────────────────

@dataclass
class ContractResult:
    fixture_file: str
    layer_id: int
    prompt_id: int
    max_abs_gate_up: float
    max_abs_down: float
    max_abs_overall: float
    mean_abs_gate_up: float
    mean_abs_down: float
    rel_l2_gate_up: float
    rel_l2_down: float
    cosine_gate_up: float
    cosine_down: float
    exact: int
    load_ms: float
    unpack_ms: float
    passed: bool
    fail_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_metrics(ref: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    rf = ref.float().flatten()
    cf = candidate.float().flatten()
    diff = rf - cf
    ref_norm = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    diff_norm = torch.linalg.vector_norm(diff)
    cosine = torch.dot(rf, cf) / (
        torch.linalg.vector_norm(rf).clamp_min(1e-12)
        * torch.linalg.vector_norm(cf).clamp_min(1e-12)
    )
    return {
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "rel_l2": float(diff_norm / ref_norm),
        "cosine": float(cosine),
    }


# ─────────────────────────────────────────────────────────────
# NVFP4 dequant helpers (clean-room, mirrors engine.loader)
# ─────────────────────────────────────────────────────────────

def _dequantize_slot_packed(
    packed_3d: torch.Tensor,
    scale_3d: torch.Tensor,
    global_scale: torch.Tensor,
    out_features: int,
    in_features: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a 3D slot-packed NVFP4 tensor to BF16.

    Args:
        packed_3d: [K, out_features, in_features // 2] uint8
        scale_3d:  [K, out_features, in_features // 16] fp16
        global_scale: scalar fp16
        out_features: number of output rows
        in_features: number of input columns
    Returns:
        [K, out_features, in_features] output_dtype
    """
    K = packed_3d.shape[0]
    # Flatten to 2D for unpack helper
    packed_2d = packed_3d.reshape(K * out_features, in_features // 2)
    scale_2d = scale_3d.reshape(K * out_features, in_features // 16).float()
    gs = global_scale.float()

    # Unpack FP4 -> float32
    unpacked = unpack_fp4_e2m1_from_uint8(packed_2d, dtype=torch.float32)  # [K*out, in]

    # Broadcast per-16 scale
    scale_full = scale_2d.repeat_interleave(16, dim=-1) / gs
    dequant = (unpacked * scale_full).reshape(K, out_features, in_features)
    return dequant.to(output_dtype)


# ─────────────────────────────────────────────────────────────
# Load helper (handles .safetensors and .safetensors.gz)
# ─────────────────────────────────────────────────────────────

def _load_fixture(path: Path, device: str = "cuda") -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file, load as load_buffer

    if len(path.suffixes) >= 2 and path.suffixes[-2:] == [".safetensors", ".gz"]:
        with gzip.open(str(path), "rb") as f:
            raw = f.read()
        return load_buffer(raw)
    else:
        return load_file(str(path), device=device)


# ─────────────────────────────────────────────────────────────
# Main contract runner
# ─────────────────────────────────────────────────────────────

def run_contract(
    p138_fixtures_dir: str,
    p135_fixtures_dir: str,
    device: str = "cuda",
    max_abs_threshold: float = 0.0,
) -> list[ContractResult]:
    """Run the packed slot decode contract test suite."""
    p138_path = Path(p138_fixtures_dir)
    p135_path = Path(p135_fixtures_dir)

    p138_manifest_path = p138_path / "manifest.json"
    if not p138_manifest_path.exists():
        print(f"[p139] ERROR: p138 manifest not found: {p138_manifest_path}", flush=True)
        return []

    with open(p138_manifest_path) as f:
        p138_manifest = json.load(f)

    print(f"[p139] p138 fixtures: {p138_path}")
    print(f"[p139] p135 fixtures: {p135_path}")
    print(f"[p139] Fixtures count: {p138_manifest['num_fixtures']}")

    results: list[ContractResult] = []

    print(f"\n{'='*90}")
    print(
        f"{'FIXTURE':<40} {'GUP_MAX':>10} {'DOWN_MAX':>10} "
        f"{'OVERALL':>10} {'COS_GUP':>8} {'COS_DOWN':>8} {'LOAD_MS':>8} {'UNPACK_MS':>8} {'PASS':>5}"
    )
    print(f"{'='*90}")

    for entry in p138_manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = entry["layer_id"]
        prompt_id = entry["prompt_id"]

        # Load packed fixture
        t_load = time.time()
        p138_data = _load_fixture(p138_path / fixture_file, device=device)
        load_ms = (time.time() - t_load) * 1000.0

        # Dequantize
        t_unpack = time.time()
        slot_gate_up_dequant = _dequantize_slot_packed(
            p138_data["slot_gate_up_packed"],
            p138_data["slot_gate_up_scale"],
            p138_data["slot_gate_up_global_scale"],
            out_features=1024,
            in_features=2048,
        )
        slot_down_dequant = _dequantize_slot_packed(
            p138_data["slot_down_packed"],
            p138_data["slot_down_scale"],
            p138_data["slot_down_global_scale"],
            out_features=2048,
            in_features=512,
        )
        unpack_ms = (time.time() - t_unpack) * 1000.0

        # Load p135 BF16 reference
        p135_file = f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_slots.safetensors"
        p135_data = _load_fixture(p135_path / p135_file, device=device)
        ref_gate_up = p135_data["slot_gate_up_weight"].to(torch.bfloat16)
        ref_down = p135_data["slot_down_weight"].to(torch.bfloat16)

        # Metrics
        gu_metrics = _compute_metrics(ref_gate_up, slot_gate_up_dequant)
        down_metrics = _compute_metrics(ref_down, slot_down_dequant)
        overall_max = max(gu_metrics["max_abs"], down_metrics["max_abs"])
        exact = 1 if overall_max == 0.0 else 0

        fail_reasons = []
        if overall_max > max_abs_threshold:
            fail_reasons.append(f"max_abs {overall_max:.6e} > {max_abs_threshold}")

        passed = len(fail_reasons) == 0

        result = ContractResult(
            fixture_file=fixture_file,
            layer_id=layer_id,
            prompt_id=prompt_id,
            max_abs_gate_up=gu_metrics["max_abs"],
            max_abs_down=down_metrics["max_abs"],
            max_abs_overall=overall_max,
            mean_abs_gate_up=gu_metrics["mean_abs"],
            mean_abs_down=down_metrics["mean_abs"],
            rel_l2_gate_up=gu_metrics["rel_l2"],
            rel_l2_down=down_metrics["rel_l2"],
            cosine_gate_up=gu_metrics["cosine"],
            cosine_down=down_metrics["cosine"],
            exact=exact,
            load_ms=load_ms,
            unpack_ms=unpack_ms,
            passed=passed,
            fail_reasons=fail_reasons,
        )
        results.append(result)

        status = "GREEN" if passed else "RED"
        fixture_label = f"L{layer_id:02d}/P{prompt_id:02d}"
        print(
            f"{fixture_label:<40} "
            f"{gu_metrics['max_abs']:>10.2e} "
            f"{down_metrics['max_abs']:>10.2e} "
            f"{overall_max:>10.2e} "
            f"{gu_metrics['cosine']:>8.6f} "
            f"{down_metrics['cosine']:>8.6f} "
            f"{load_ms:>8.3f} "
            f"{unpack_ms:>8.3f} "
            f"{status:>5}",
            flush=True,
        )

    # Summary
    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    max_abs_max = max(r.max_abs_overall for r in results) if results else 0.0
    load_ms_mean = sum(r.load_ms for r in results) / total if total else 0.0
    unpack_ms_mean = sum(r.unpack_ms for r in results) / total if total else 0.0

    print(f"\n{'='*90}")
    print("CONTRACT SUMMARY")
    print(f"  Threshold: max_abs == {max_abs_threshold}")
    print(f"  Total:     {total}")
    print(f"  Passed:    {passed_count} GREEN")
    print(f"  Failed:    {total - passed_count} RED")
    print(f"  max_abs_max: {max_abs_max:.6e}")
    print(f"  load_ms_mean: {load_ms_mean:.3f} ms")
    print(f"  unpack_ms_mean: {unpack_ms_mean:.3f} ms")
    if passed_count == total:
        print(f"\n  VERDICT: ALL {total} fixtures exact GREEN.")
        print(f"  Packed NVFP4 round-trip is bit-exact with BF16 reference.")
    else:
        print(f"\n  VERDICT: {total - passed_count} fixture(s) FAILED.")
        for r in results:
            if not r.passed:
                print(f"    {r.fixture_file}: {'; '.join(r.fail_reasons)}")
    print(f"{'='*90}")

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="MoE slot packed NVFP4 decode contract test.")
    ap.add_argument("--p138-fixtures", required=True, help="Path to p138 packed fixture directory.")
    ap.add_argument("--p135-fixtures", required=True, help="Path to p135 BF16 slot fixture directory.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-abs-threshold", type=float, default=0.0, help="Exact match required.")
    ap.add_argument("--out", default=None, help="Output JSON report path.")
    args = ap.parse_args()

    results = run_contract(
        p138_fixtures_dir=args.p138_fixtures,
        p135_fixtures_dir=args.p135_fixtures,
        device=args.device,
        max_abs_threshold=args.max_abs_threshold,
    )

    if not results:
        return 1

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    max_abs_max = max(r.max_abs_overall for r in results)
    load_ms_mean = sum(r.load_ms for r in results) / total
    unpack_ms_mean = sum(r.unpack_ms for r in results) / total

    out_path = args.out or str(Path(args.p138_fixtures) / "p139_slot_packed_contract_report.json")
    report = {
        "schema": "lynn-moe-slot-packed-contract-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "p138_fixtures_dir": args.p138_fixtures,
        "p135_fixtures_dir": args.p135_fixtures,
        "max_abs_threshold": args.max_abs_threshold,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "verdict": "GREEN" if passed == total else "RED",
        "max_abs_max": max_abs_max,
        "load_ms_mean": load_ms_mean,
        "unpack_ms_mean": unpack_ms_mean,
        "results": [r.to_dict() for r in results],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[p139] Report written: {out_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
