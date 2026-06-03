#!/usr/bin/env python3
"""Summarize Stage 6 P4B runtime bridge fail-loud preflight artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_stage6_p4_runtime_bridge_preflight import _packed_manifest_ok


SCHEMA = "lynn-stage6-p4b-runtime-bridge-preflight-v1"
EXPECTED_BACKEND = "fused_zero_shadow_single_kernel_contract"
NATIVE_CALL_COUNT_KEY = "_p4b_fused_zero_shadow_single_kernel_contract_call_count"
FAIL_LOUD_NEEDLES = (
    "P4B single-kernel fused zero-shadow contract is not implemented yet",
    "do not bank fused-kernel speed or promote this backend",
)


def _active_out_scratch_ok(manifest: dict[str, Any]) -> bool:
    meta = manifest.get("mlp.experts._active_out_scratch") or {}
    return (
        meta.get("shape") == [2048]
        and meta.get("dtype") == "torch.bfloat16"
        and meta.get("contiguous") is True
    )


def _last_shapes_out_only(last_shapes: Any) -> bool:
    if not isinstance(last_shapes, dict):
        return False
    if "inter_scratch" in last_shapes or "inter_out" in last_shapes:
        return False
    hidden = last_shapes.get("hidden")
    expert_ids = last_shapes.get("expert_ids")
    out = last_shapes.get("out")
    return hidden == [1, 2048] and isinstance(expert_ids, list) and expert_ids[:1] == [1] and out == [1, 2048]


def _error_fail_loud(error: dict[str, Any]) -> bool:
    message = str(error.get("message", ""))
    return all(needle in message for needle in FAIL_LOUD_NEEDLES)


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    baseline = data.get("baseline") or {}
    packed = data.get("packed_manifest_before_candidate") or {}
    scratch = data.get("active_scratch_manifest") or {}
    candidate_error = data.get("candidate_error") or {}
    native_call_count = data.get("native_backend_call_count") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("expected_backend") != EXPECTED_BACKEND:
        return "FAIL", "expected backend mismatch"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel promotion boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    if data.get("banked_p4b_runtime_bridge_preflight") is not True:
        return "FAIL", "P4B runtime bridge preflight was not banked"
    for gate in (
        "baseline_triton_nonzero",
        "native_layer_selected",
        "native_backend_called",
        "baseline_shape_dtype",
        "packed_tensors_present",
        "active_out_scratch_present",
        "active_shadows_removed",
        "candidate_returned_false",
        "p4b_fail_loud_not_implemented",
        "p4b_last_shapes_out_only",
        "fused_kernel_unbanked",
        "default_promotion_closed",
        "all",
    ):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if data.get("decision") != "PASS_P4B_RUNTIME_BRIDGE_FAILLOUD":
        return "FAIL", "top-level decision is not PASS_P4B_RUNTIME_BRIDGE_FAILLOUD"
    if baseline.get("output_shape") != [1, 1, 2048] or baseline.get("output_dtype") != "torch.bfloat16":
        return "FAIL", "baseline shape/dtype mismatch"
    norm = baseline.get("norm")
    if not isinstance(norm, (int, float)) or not math.isfinite(float(norm)) or float(norm) <= 0.0:
        return "FAIL", "baseline norm is not finite positive"
    if data.get("native_layer_selected_for_candidate") is not True:
        return "FAIL", "native layer selection was not proven"
    if native_call_count.get("key") != NATIVE_CALL_COUNT_KEY or native_call_count.get("delta") != 1:
        return "FAIL", "P4B native backend call count did not advance exactly once"
    if not _last_shapes_out_only(native_call_count.get("last_shapes")):
        return "FAIL", "P4B last_shapes did not prove out-only ABI"
    if not _packed_manifest_ok(packed):
        return "FAIL", "packed manifest mismatch"
    if not _active_out_scratch_ok(scratch):
        return "FAIL", "active out scratch manifest mismatch"
    if data.get("active_shadow_keys_present_after_delete"):
        return "FAIL", "explicit BF16 active shadow keys remain"
    if data.get("bf16_active_shadow_aliases_after_delete"):
        return "FAIL", "BF16 active shadow aliases remain"
    if data.get("candidate_returned") is not False:
        return "FAIL", "candidate returned output instead of failing loud"
    if not _error_fail_loud(candidate_error):
        return "FAIL", "expected P4B fail-loud error missing"
    return "PASS", "real runtime bridge reaches P4B fail-loud symbol; fused kernel still unbanked"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    baseline = data.get("baseline") or {}
    removed = data.get("removed_active_shadows") or {}
    packed = data.get("packed_manifest_before_candidate") or {}
    scratch = data.get("active_scratch_manifest") or {}
    native_call_count = data.get("native_backend_call_count") or {}
    candidate_error = data.get("candidate_error") or data.get("runner_error") or {}
    lines = [
        "# Stage 6 P4B Runtime Bridge Fail-Loud Preflight Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Layer | `{data.get('layer')}` |",
        f"| Expected backend | `{data.get('expected_backend')}` |",
        f"| Native layer selected | `{data.get('native_layer_selected_for_candidate')}` |",
        f"| P4B native call delta | `{native_call_count.get('delta')}` |",
        f"| P4B last shapes | `{native_call_count.get('last_shapes')}` |",
        f"| Banked P4B runtime bridge preflight | `{data.get('banked_p4b_runtime_bridge_preflight')}` |",
        f"| Banked fused kernel | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Baseline norm | `{baseline.get('norm')}` |",
        f"| Candidate returned | `{data.get('candidate_returned')}` |",
        f"| P4B fail-loud not implemented | `{passes.get('p4b_fail_loud_not_implemented')}` |",
        f"| P4B last shapes out-only | `{passes.get('p4b_last_shapes_out_only')}` |",
        f"| Packed tensors present | `{passes.get('packed_tensors_present')}` |",
        f"| Active out scratch present | `{passes.get('active_out_scratch_present')}` |",
        f"| Active shadows removed | `{passes.get('active_shadows_removed')}` |",
        f"| Elapsed seconds | `{data.get('elapsed_s')}` |",
        "",
        "## Removed Active Shadows",
        "",
        "| Key | Shape | DType | Bytes |",
        "|---|---|---|---:|",
    ]
    for key, meta in removed.items():
        lines.append(f"| `{key}` | `{meta.get('shape')}` | `{meta.get('dtype')}` | `{meta.get('bytes')}` |")
    lines.extend(["", "## Active Scratch Manifest", "", "| Key | Shape | DType | Bytes |", "|---|---|---|---:|"])
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
    ap.add_argument("result_json", help="Path to P4B runtime bridge result.json")
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
