#!/usr/bin/env python3
"""Summarize Stage 6 P4 real-runtime bridge preflight artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4-runtime-bridge-preflight-v1"
PACKED_KEYS = {
    "mlp.experts._gate_up_packed": ((None, 1024, 1024), "torch.uint8"),
    "mlp.experts._gate_up_scale": ((None, 1024, 128), "torch.float32"),
    "mlp.experts._gate_up_global_scale": ((1,), "torch.float32"),
    "mlp.experts._down_packed": ((None, 2048, 256), "torch.uint8"),
    "mlp.experts._down_scale": ((None, 2048, 32), "torch.float32"),
    "mlp.experts._down_global_scale": ((1,), "torch.float32"),
}
SCRATCH_KEYS = {
    "mlp.experts._active_inter_scratch": ("torch.bfloat16", 2),
    "mlp.experts._active_out_scratch": ("torch.bfloat16", 1),
}


def _shape_matches(actual: list[int], expected: tuple[int | None, ...]) -> bool:
    return len(actual) == len(expected) and all(exp is None or got == exp for got, exp in zip(actual, expected))


def _packed_manifest_ok(manifest: dict[str, Any]) -> bool:
    if set(manifest) != set(PACKED_KEYS):
        return False
    for key, meta in manifest.items():
        shape, dtype = PACKED_KEYS[key]
        if meta.get("dtype") != dtype or not meta.get("contiguous"):
            return False
        if not _shape_matches(list(meta.get("shape") or []), shape):
            return False
    return True


def _scratch_manifest_ok(manifest: dict[str, Any]) -> bool:
    if set(manifest) != set(SCRATCH_KEYS):
        return False
    for key, meta in manifest.items():
        dtype, dims = SCRATCH_KEYS[key]
        shape = list(meta.get("shape") or [])
        if meta.get("dtype") != dtype or not meta.get("contiguous") or len(shape) != dims:
            return False
        if key.endswith("_active_inter_scratch") and shape[-1] != 512:
            return False
        if key.endswith("_active_out_scratch") and shape != [2048]:
            return False
    return True


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    baseline = data.get("baseline") or {}
    packed = data.get("packed_manifest_before_candidate") or {}
    scratch = data.get("active_scratch_manifest") or {}
    candidate_error = data.get("candidate_error") or {}
    candidate = data.get("candidate") or {}
    native_call_count = data.get("native_backend_call_count") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel promotion boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    if data.get("banked_runtime_bridge_preflight") is not True:
        return "FAIL", "runtime bridge preflight was not banked"
    for gate in (
        "baseline_triton_nonzero",
        "native_layer_selected",
        "native_backend_called",
        "baseline_shape_dtype",
        "packed_tensors_present",
        "active_scratch_present",
        "active_shadows_removed",
        "candidate_output_returned",
        "candidate_shape_dtype",
        "candidate_numeric_vs_triton",
        "fused_kernel_unbanked",
        "default_promotion_closed",
        "all",
    ):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if data.get("decision") != "PASS_TWO_STAGE_RUNTIME_BRIDGE":
        return "FAIL", "top-level decision is not PASS_TWO_STAGE_RUNTIME_BRIDGE"
    if baseline.get("output_shape") != [1, 1, 2048] or baseline.get("output_dtype") != "torch.bfloat16":
        return "FAIL", "baseline shape/dtype mismatch"
    if data.get("native_layer_selected_for_candidate") is not True:
        return "FAIL", "native layer selection was not proven"
    if native_call_count.get("delta") != 1:
        return "FAIL", "native backend call count did not advance exactly once"
    norm = baseline.get("norm")
    if not isinstance(norm, (int, float)) or not math.isfinite(float(norm)) or float(norm) <= 0.0:
        return "FAIL", "baseline norm is not finite positive"
    if not _packed_manifest_ok(packed):
        return "FAIL", "packed manifest mismatch"
    if not _scratch_manifest_ok(scratch):
        return "FAIL", "active scratch manifest mismatch"
    if data.get("active_shadow_keys_present_after_delete"):
        return "FAIL", "explicit BF16 active shadow keys remain"
    if data.get("bf16_active_shadow_aliases_after_delete"):
        return "FAIL", "BF16 active shadow aliases remain"
    if candidate_error:
        return "FAIL", "candidate raised instead of returning output"
    if candidate.get("output_shape") != baseline.get("output_shape") or candidate.get("output_dtype") != "torch.bfloat16":
        return "FAIL", "candidate shape/dtype mismatch"
    if candidate.get("finite") is not True:
        return "FAIL", "candidate output is not finite"
    if not isinstance(candidate.get("rel_l2_vs_baseline"), (int, float)):
        return "FAIL", "missing candidate rel_l2 metric"
    return "PASS", "runtime bridge returns two-stage P4 output on real runner path"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    baseline = data.get("baseline") or {}
    candidate = data.get("candidate") or {}
    removed = data.get("removed_active_shadows") or {}
    packed = data.get("packed_manifest_before_candidate") or {}
    scratch = data.get("active_scratch_manifest") or {}
    native_call_count = data.get("native_backend_call_count") or {}
    candidate_error = data.get("candidate_error") or data.get("runner_error") or {}
    lines = [
        "# Stage 6 P4 Runtime Bridge Preflight Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Layer | `{data.get('layer')}` |",
        f"| Expected backend | `{data.get('expected_backend')}` |",
        f"| Native layer selected | `{data.get('native_layer_selected_for_candidate')}` |",
        f"| Native backend call delta | `{native_call_count.get('delta')}` |",
        f"| Banked runtime bridge preflight | `{data.get('banked_runtime_bridge_preflight')}` |",
        f"| Banked fused kernel | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Baseline norm | `{baseline.get('norm')}` |",
        f"| Candidate norm | `{candidate.get('norm')}` |",
        f"| Candidate rel L2 vs baseline | `{candidate.get('rel_l2_vs_baseline')}` |",
        f"| Candidate max abs diff vs baseline | `{candidate.get('max_abs_diff_vs_baseline')}` |",
        f"| Packed tensors present | `{passes.get('packed_tensors_present')}` |",
        f"| Active scratch present | `{passes.get('active_scratch_present')}` |",
        f"| Active shadows removed | `{passes.get('active_shadows_removed')}` |",
        f"| Candidate output returned | `{passes.get('candidate_output_returned')}` |",
        f"| Candidate numeric vs Triton | `{passes.get('candidate_numeric_vs_triton')}` |",
        f"| Elapsed seconds | `{data.get('elapsed_s')}` |",
        "",
        "## Removed Active Shadows",
        "",
        "| Key | Shape | DType | Bytes |",
        "|---|---|---|---:|",
    ]
    for key, meta in removed.items():
        lines.append(f"| `{key}` | `{meta.get('shape')}` | `{meta.get('dtype')}` | `{meta.get('bytes')}` |")
    lines.extend(["", "## Active Scratch", "", "| Key | Shape | DType | Bytes |", "|---|---|---|---:|"])
    for key, meta in scratch.items():
        lines.append(f"| `{key}` | `{meta.get('shape')}` | `{meta.get('dtype')}` | `{meta.get('bytes')}` |")
    lines.extend(["", "## Packed Tensor Inputs", "", "| Key | Shape | DType | Bytes |", "|---|---|---|---:|"])
    for key, meta in packed.items():
        lines.append(f"| `{key}` | `{meta.get('shape')}` | `{meta.get('dtype')}` | `{meta.get('bytes')}` |")
    if candidate_error:
        lines.extend(["", "## Error Tail", "", "```text", str(candidate_error.get("message", candidate_error))[-1200:], "```"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="Path to P4 runtime bridge result.json")
    ap.add_argument("--markdown-out", default="", help="Optional Markdown output path")
    ap.add_argument("--strict-exit", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.result_json).read_text())
    md = summarize(data)
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    verdict, _ = _verdict(data)
    return 0 if (verdict == "PASS" or not args.strict_exit) else 2


if __name__ == "__main__":
    raise SystemExit(main())
