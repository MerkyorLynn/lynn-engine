#!/usr/bin/env python3
"""Summarize Stage 6 P4B single-kernel fail-loud preflight artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4b-single-kernel-preflight-v1"
EXPECTED_TENSORS = {
    "hidden": ("torch.bfloat16", 2),
    "expert_ids": ("torch.int32", 2),
    "routing_weights": ("torch.float32", 2),
    "gate_up_packed": ("torch.uint8", 3),
    "gate_up_scale": ("torch.float32", 3),
    "gate_up_global_scale": ("torch.float32", 1),
    "down_packed": ("torch.uint8", 3),
    "down_scale": ("torch.float32", 3),
    "down_global_scale": ("torch.float32", 1),
    "out": ("torch.bfloat16", 2),
}
FAIL_LOUD_NEEDLES = (
    "P4B single-kernel fused zero-shadow contract is not implemented yet",
    "do not bank fused-kernel speed or promote this backend",
)


def _tensor_manifest_ok(manifest: dict[str, Any]) -> bool:
    if set(manifest) != set(EXPECTED_TENSORS):
        return False
    for key, meta in manifest.items():
        dtype, dims = EXPECTED_TENSORS[key]
        if meta.get("dtype") != dtype or not meta.get("contiguous"):
            return False
        shape = list(meta.get("shape") or [])
        if len(shape) != dims:
            return False
    return True


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    tensor_manifest = data.get("tensor_manifest") or {}
    byte_budget = data.get("byte_budget") or {}
    call_error = str(data.get("call_error_tail") or "")
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel promotion boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    if data.get("banked_single_kernel_contract_preflight") is not True:
        return "FAIL", "single-kernel contract preflight was not banked"
    if data.get("decision") != "PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT":
        return "FAIL", "top-level decision is not PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT"
    for gate in (
        "extension_loaded",
        "symbol_present",
        "call_returned_false",
        "fail_loud_not_implemented",
        "zero_shadow_abi",
        "packed_byte_budget",
        "no_inter_scratch_abi",
        "all",
    ):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if data.get("call_returned") is not False:
        return "FAIL", "contract call returned instead of failing loud"
    if not all(needle in call_error for needle in FAIL_LOUD_NEEDLES):
        return "FAIL", "expected fail-loud error text missing"
    if not _tensor_manifest_ok(tensor_manifest):
        return "FAIL", "tensor ABI manifest mismatch"
    if byte_budget.get("no_inter_scratch_abi") is not True:
        return "FAIL", "byte budget admits inter_scratch"
    if "inter_scratch" in tensor_manifest:
        return "FAIL", "tensor manifest contains inter_scratch"
    return "PASS", "single-kernel fail-loud contract preflight passed; fused kernel still unbanked"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    tensor_manifest = data.get("tensor_manifest") or {}
    byte_budget = data.get("byte_budget") or {}
    lines = [
        "# Stage 6 P4B Single-Kernel Fail-Loud Preflight Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Symbol | `{data.get('symbol')}` |",
        f"| Device | `{data.get('device_name', 'unknown')}` |",
        f"| Capability | `{data.get('capability', 'unknown')}` |",
        f"| Torch/CUDA | `{data.get('torch_version')}` / `{data.get('torch_cuda')}` |",
        f"| Build dir | `{data.get('build_dir', 'unknown')}` |",
        f"| Banked contract preflight | `{data.get('banked_single_kernel_contract_preflight')}` |",
        f"| Banked fused kernel | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Extension loaded | `{passes.get('extension_loaded')}` |",
        f"| Symbol present | `{passes.get('symbol_present')}` |",
        f"| Call returned false | `{passes.get('call_returned_false')}` |",
        f"| Fail-loud not implemented | `{passes.get('fail_loud_not_implemented')}` |",
        f"| Zero-shadow ABI | `{passes.get('zero_shadow_abi')}` |",
        f"| Packed byte budget | `{passes.get('packed_byte_budget')}` |",
        f"| No inter_scratch ABI | `{passes.get('no_inter_scratch_abi')}` |",
        f"| Packed weight bytes | `{byte_budget.get('packed_weight_bytes')}` |",
        f"| BF16 shadow-equivalent bytes | `{byte_budget.get('bf16_shadow_equivalent_bytes')}` |",
        f"| Packed/BF16 ratio | `{byte_budget.get('packed_vs_bf16_shadow_ratio')}` |",
        f"| Elapsed seconds | `{data.get('elapsed_s')}` |",
        "",
        "## Tensor ABI",
        "",
        "| Tensor | Shape | DType | Bytes | Contiguous |",
        "|---|---|---|---:|---|",
    ]
    for name, meta in tensor_manifest.items():
        lines.append(
            f"| `{name}` | `{meta.get('shape')}` | `{meta.get('dtype')}` | "
            f"`{meta.get('bytes')}` | `{meta.get('contiguous')}` |"
        )
    error_tail = data.get("call_error_tail") or data.get("load_error_tail") or data.get("probe_error")
    if error_tail:
        lines.extend(["", "## Error Tail", "", "```text", str(error_tail)[-1200:], "```"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="Path to P4B preflight result.json")
    ap.add_argument("--markdown-out", default="", help="Optional Markdown output path")
    ap.add_argument("--strict-exit", action="store_true", help="Exit non-zero unless verdict is PASS")
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
