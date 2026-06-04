#!/usr/bin/env python3
"""Summarize Stage 6 R5-C1 CUTLASS numeric-smoke artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(data: dict[str, Any], result_path: Path) -> str:
    passes = data.get("passes") or {}
    git = data.get("git") or {}
    head = ((git.get("head") or {}).get("stdout_tail") or "unknown").strip()
    branch = ((git.get("branch") or {}).get("stdout_tail") or "unknown").strip()
    shape = data.get("shape") or {}
    parse = data.get("run_parse") or {}
    build = data.get("build_result") or {}
    patch = (build.get("atomic_scope_patch") or {}) if isinstance(build, dict) else {}
    lines = [
        "# Stage 6 R5-C1 CUTLASS Numeric Smoke Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
        f"| CUTLASS dir | `{data.get('cutlass_dir')}` |",
        f"| CUTLASS git | `{head}` (`{branch}`) |",
        f"| Example | `{data.get('example')}` |",
        f"| Shape | `M={shape.get('m')} N={shape.get('n')} K={shape.get('k')} groups={shape.get('groups')} iterations={shape.get('iterations')}` |",
        f"| Build invoked | `{passes.get('build_invoked')}` |",
        f"| Build succeeded | `{passes.get('build_succeeded')}` |",
        f"| Temporary CUDA atomic patch applied/restored | `{patch.get('applied')}` / `{patch.get('restored')}` |",
        f"| Numeric smoke banked | `{passes.get('banked_numeric_smoke')}` |",
        f"| Grouped-MoE FP4-MMA POC banked | `{passes.get('banked_grouped_moe_fp4_mma_poc')}` |",
        f"| Kernel speed banked | `{passes.get('banked_kernel_speed')}` |",
        f"| Default promotion banked | `{passes.get('banked_default_promotion')}` |",
        "",
        "## Run Gates",
        "",
        "| Gate | Value |",
        "|---|---:|",
        f"| Cooperative schedule passed | `{passes.get('cooperative_passed')}` |",
        f"| Pingpong schedule passed | `{passes.get('pingpong_passed')}` |",
        f"| Host reference seen | `{passes.get('host_reference_seen')}` |",
        f"| Disposition passed count >= 2 | `{passes.get('dispositions_passed_count_ge_2')}` |",
        f"| No no-op device gate | `{passes.get('no_noop_device_gate')}` |",
        f"| Avg runtime ms | `{parse.get('avg_runtime_ms')}` |",
        f"| TFLOPS | `{parse.get('tflops')}` |",
        "",
        "## Boundary",
        "",
        "- This R5-C1 artifact banks only `banked_numeric_smoke=true` for CUTLASS native NVF4 + UE4M3.",
        "- It does not bank a Lynn grouped-MoE FP4-MMA kernel, speed claim, or default runtime promotion.",
        "- The next gate is selected expert gate/up numeric smoke, not a grouped-MoE speed headline.",
        "",
    ]
    return "\n".join(lines)


def _strict_ok(data: dict[str, Any]) -> bool:
    passes = data.get("passes") or {}
    return bool(
        data.get("decision") == "PASS_R5C1_CUTLASS_NVF4_UE4M3_NUMERIC_SMOKE"
        and passes.get("banked_numeric_smoke") is True
        and passes.get("banked_grouped_moe_fp4_mma_poc") is False
        and passes.get("banked_kernel_speed") is False
        and passes.get("banked_default_promotion") is False
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_json")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument("--strict-exit", action="store_true")
    args = ap.parse_args()
    result_path = Path(args.result_json)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    md = summarize(data, result_path)
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
    print(md)
    return 2 if args.strict_exit and not _strict_ok(data) else 0


if __name__ == "__main__":
    raise SystemExit(main())
