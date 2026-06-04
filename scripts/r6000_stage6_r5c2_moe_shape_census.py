#!/usr/bin/env python3
"""Stage 6 R5-C2 MoE-shape source census.

This gate decides the implementation substrate for R5-C2 selected-expert
gate/up numeric smoke. It compares:

- CUTLASS 79d: SM120 native NVF4+UE4M3 generic grouped GEMM.
- CUTLASS 92: MoEProblemShape + tokens_per_expert semantics, but Sm100 schedule.

The expected PASS is not a kernel PASS. It banks only the source-census verdict
that R5-C2 needs a new minimal harness combining 92-style MoE shape semantics
with 79d-style SM120 NVF4+UE4M3 execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


EX79 = "examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu"
EX92 = "examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_fp4_grouped.cu"


def _run(cmd: list[str], cwd: Path, timeout_s: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "ok": False,
            "returncode": None,
            "timeout": True,
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or "").strip(),
        }


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _hits(path: Path, tokens: list[str]) -> list[dict[str, Any]]:
    text = _read(path)
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(token in line for token in tokens):
            rows.append({"path": str(path), "line": lineno, "text": line.strip()})
    return rows


def _contains(path: Path, token: str) -> bool:
    return token in _read(path)


def run(cutlass_dir: Path) -> dict[str, Any]:
    ex79 = cutlass_dir / EX79
    ex92 = cutlass_dir / EX92
    checks = {
        "ex79_exists": ex79.exists(),
        "ex92_exists": ex92.exists(),
        "ex79_sm120": _contains(ex79, "cutlass::arch::Sm120"),
        "ex79_group_problem_shape": _contains(ex79, "GroupProblemShape<Shape<int,int,int>>"),
        "ex79_nvfp4_ue4m3": all(_contains(ex79, token) for token in ["nv_float4_t", "float_e2m1_t", "float_ue4m3_t"]),
        "ex79_lacks_moe_problem_shape": not _contains(ex79, "MoEProblemShape"),
        "ex79_lacks_tokens_per_expert": not _contains(ex79, "tokens_per_expert"),
        "ex92_moe_problem_shape": _contains(ex92, "MoEProblemShape<Shape<int,int,int>>"),
        "ex92_tokens_per_expert": _contains(ex92, "tokens_per_expert"),
        "ex92_nvfp4_ue4m3": all(_contains(ex92, token) for token in ["nv_float4_t", "float_e2m1_t", "float_ue4m3_t"]),
        "ex92_sm100_schedule": _contains(ex92, "BlockScaledSm100") and _contains(ex92, "cutlass::arch::Sm100"),
        "ex92_lacks_sm120_schedule": "Sm120" not in _read(ex92),
    }
    banked = all(checks.values())
    passes = {
        **checks,
        "requires_new_minimal_harness": banked,
        "banked_moe_shape_census": banked,
        "banked_selected_expert_gate_up_smoke": False,
        "banked_grouped_moe_fp4_mma_poc": False,
        "banked_kernel_speed": False,
        "banked_default_promotion": False,
        "all": banked,
    }
    return {
        "schema": "lynn-stage6-r5c2-moe-shape-census-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cutlass_dir": str(cutlass_dir),
        "git": {
            "head": _run(["git", "rev-parse", "HEAD"], cutlass_dir),
            "branch": _run(["git", "branch", "--show-current"], cutlass_dir),
            "status": _run(["git", "status", "--short"], cutlass_dir),
        } if cutlass_dir.exists() else {},
        "examples": {
            "sm120_grouped_gemm": str(ex79),
            "sm100_moe_gemm": str(ex92),
        },
        "evidence_hits": {
            "ex79": _hits(ex79, ["Sm120", "GroupProblemShape", "nv_float4_t", "float_ue4m3_t", "problem_sizes_host"]),
            "ex92": _hits(ex92, ["Sm100", "MoEProblemShape", "tokens_per_expert", "nv_float4_t", "float_ue4m3_t"]),
        },
        "passes": passes,
        "decision": (
            "PASS_R5C2_MOE_SHAPE_CENSUS_NEW_HARNESS_REQUIRED"
            if banked
            else "FAIL_R5C2_MOE_SHAPE_CENSUS"
        ),
        "next_contract": {
            "r5c2_selected_expert_gate_up_numeric_smoke": [
                "combine 92-style MoEProblemShape/tokens_per_expert semantics with 79d-style SM120 NVF4+UE4M3 schedule",
                "preserve expert IDs/top-k token assignment",
                "verify per-expert gate/up outputs against host reference",
                "bank only selected-expert numeric smoke; no speed/default promotion",
            ]
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutlass-dir", default="/root/autodl-tmp/src/cutlass")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = run(Path(args.cutlass_dir).expanduser().resolve())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": data["decision"], "passes": data["passes"]}, indent=2))
    return 0 if data["passes"]["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
