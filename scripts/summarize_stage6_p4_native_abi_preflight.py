#!/usr/bin/env python3
"""Summarize Stage 6 P4 native fused-MoE ABI preflight artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4-native-fused-moe-abi-preflight-v1"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel promotion boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    if data.get("banked_native_abi_preflight") is not True:
        return "FAIL", "ABI preflight was not banked"
    for gate in ("extension_loaded", "symbol_present", "fail_loud_boundary", "all"):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if data.get("decision") != "PASS_ABI_CONTRACT":
        return "FAIL", "top-level decision is not PASS_ABI_CONTRACT"
    return "PASS", "native ABI preflight passed; fused kernel still unbanked"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    tensor_manifest = data.get("tensor_manifest") or {}
    lines = [
        "# Stage 6 P4 Native Fused-MoE ABI Preflight Summary",
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
        f"| Banked ABI preflight | `{data.get('banked_native_abi_preflight')}` |",
        f"| Banked fused kernel | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Extension loaded | `{passes.get('extension_loaded')}` |",
        f"| Symbol present | `{passes.get('symbol_present')}` |",
        f"| Fail-loud boundary | `{passes.get('fail_loud_boundary')}` |",
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
    ap.add_argument("result_json", help="Path to P4 preflight result.json")
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
