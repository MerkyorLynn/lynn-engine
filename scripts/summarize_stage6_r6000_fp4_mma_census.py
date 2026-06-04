#!/usr/bin/env python3
"""Summarize Stage 6 R6000 FP4-MMA bring-up/census artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-r6000-fp4-mma-census-v1"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    passes = data.get("passes") or {}
    if passes.get("all") is True and data.get("decision") == "PASS_R6000_FP4_MMA_BRINGUP":
        return "PASS", "R6000 FP4-MMA bring-up and public-kernel census passed"
    if passes.get("contract_suite_recorded") is not True:
        return "READY", "tooling/census exists but contract suite was not run"
    return "FAIL", "R6000 FP4-MMA bring-up did not pass all gates"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _torch(data: dict[str, Any]) -> dict[str, Any]:
    return ((data.get("torch") or {}).get("json") or {})


def _disk_free(data: dict[str, Any]) -> str:
    text = (((data.get("system") or {}).get("disk_workspace") or {}).get("stdout_tail") or "").strip()
    return text.splitlines()[-1] if text else "unknown"


def _public_summary(data: dict[str, Any]) -> tuple[str, str]:
    public = ((data.get("public_kernel_census") or {}).get("json") or {})
    packages = public.get("packages") or {}
    explicit = public.get("explicit_imports") or {}
    package_line = ", ".join(
        f"{name}:{'yes' if (row or {}).get('importable') else 'no'}"
        for name, row in packages.items()
    )
    explicit_yes = [name for name, row in explicit.items() if isinstance(row, dict) and row.get("importable")]
    return package_line or "unknown", ", ".join(explicit_yes) or "none"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    torch_info = _torch(data)
    passes = data.get("passes") or {}
    contract_passes = data.get("contract_passes") or {}
    package_line, explicit_line = _public_summary(data)
    lines = [
        "# Stage 6 R6000 FP4-MMA Bring-Up Census",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Device | `{torch_info.get('device_name', 'unknown')}` |",
        f"| Capability | `{torch_info.get('capability', 'unknown')}` |",
        f"| Memory GiB | `{_fmt(torch_info.get('total_memory_gib'))}` |",
        f"| Disk workspace | `{_disk_free(data)}` |",
        f"| Public packages | `{package_line}` |",
        f"| Explicit NVFP4 imports | `{explicit_line}` |",
        f"| Contract passes | `{contract_passes}` |",
        f"| Promotion boundary | `{data.get('promotion_boundary')}` |",
        "",
        "## Pass Gates",
        "",
        "| Gate | Value |",
        "|---|---|",
    ]
    for key, value in passes.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend([
        "",
        "## Boundary",
        "",
        "- This banks machine/toolchain/public-kernel census only.",
        "- It does not promote any Lynn kernel or runtime default.",
        "- If PASS, the next step is a Lynn NVFP4 grouped-MoE FP4-MMA POC using CUTLASS/CuTe plus the public Marlin/Machete census.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument("--strict-exit", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
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
