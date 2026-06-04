#!/usr/bin/env python3
"""Stage 6 R5-C CUTLASS/CuTe native NVF4 + UE4M3 ABI census.

Run on the RTX PRO 6000 lane after R5-B closes simple e8m0 repack. This
does not compile or promote a Lynn kernel. It only proves whether the local
CUTLASS checkout exposes an sm120 native mxf4nvf4 block16 + UE4M3 route that
can be used as the next POC base.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "ok": False,
            "timeout": True,
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or "").strip(),
        }


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _hits(path: Path, needles: list[str], max_hits: int = 12) -> list[dict[str, Any]]:
    text = _read(path)
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(needle in line for needle in needles):
            rows.append({"path": str(path), "line": lineno, "text": line.strip()})
        if len(rows) >= max_hits:
            break
    return rows


def _contains(path: Path, needle: str) -> bool:
    return needle in _read(path)


def _find_files(root: Path, patterns: list[str], tokens: list[str], limit: int = 80) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        low = rel.lower()
        if not any(pattern.lower() in low for pattern in patterns):
            continue
        text = _read(path)
        matched = [token for token in tokens if token in text or token.lower() in low]
        if matched:
            rows.append({"path": rel, "matched": matched[:8]})
        if len(rows) >= limit:
            break
    return rows


def run(cutlass_dir: Path) -> dict[str, Any]:
    arch_config = cutlass_dir / "include" / "cute" / "arch" / "config.hpp"
    sm100_desc = cutlass_dir / "include" / "cute" / "arch" / "mma_sm100_desc.hpp"
    sm120_mma = cutlass_dir / "include" / "cute" / "arch" / "mma_sm120.hpp"
    sm100_umma = cutlass_dir / "include" / "cute" / "arch" / "mma_sm100_umma.hpp"

    key_files = {
        "arch_config": arch_config,
        "sm100_desc": sm100_desc,
        "sm120_mma": sm120_mma,
        "sm100_umma": sm100_umma,
    }
    file_exists = {name: path.exists() for name, path in key_files.items()}

    required = {
        "sm120_ue4m3_macro": (
            arch_config,
            "CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED",
        ),
        "scale_format_ue4m3": (
            sm100_desc,
            "ScaleFormat::UE4M3",
        ),
        "scale_type_ue4m3": (
            sm100_desc,
            "float_ue4m3_t",
        ),
        "mxf4_e2m1_format": (
            sm100_desc,
            "MXF4Format::E2M1",
        ),
        "sm120_e2m1_ue4m3_specialization": (
            sm120_mma,
            "SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t",
        ),
        "sm120_mxf4nvf4_ue4m3_asm": (
            sm120_mma,
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
        ),
    }
    token_passes = {name: _contains(path, needle) for name, (path, needle) in required.items()}

    examples = _find_files(
        cutlass_dir / "examples",
        patterns=["nvfp4", "fp4", "moe", "grouped"],
        tokens=["float_ue4m3_t", "nvfp4", "fp4", "grouped", "moe"],
    ) if (cutlass_dir / "examples").exists() else []
    sm120_tests = _find_files(
        cutlass_dir / "test" / "unit" / "gemm" / "device" / "sm120_blockscaled_tensorop_gemm",
        patterns=["nvf4", "fp4", "blockscaled", "group"],
        tokens=["float_ue4m3_t", "nvf4", "blockscaled", "group"],
    ) if (cutlass_dir / "test" / "unit" / "gemm" / "device" / "sm120_blockscaled_tensorop_gemm").exists() else []

    expected_example_names = [
        "examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu",
        "examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_fp4_grouped.cu",
    ]
    expected_test_substr = "sm120_blockscaled_tensorop_gemm"
    expected_examples_seen = all((cutlass_dir / rel).exists() for rel in expected_example_names)
    sm120_tests_seen = any(expected_test_substr in row["path"] or row["path"] for row in sm120_tests)

    passes = {
        "cutlass_dir_exists": cutlass_dir.exists(),
        "cutlass_git_recorded": False,
        "key_files_present": all(file_exists.values()),
        "sm120_ue4m3_macro_seen": token_passes["sm120_ue4m3_macro"],
        "scale_format_ue4m3_seen": token_passes["scale_format_ue4m3"],
        "scale_type_ue4m3_seen": token_passes["scale_type_ue4m3"],
        "mxf4_e2m1_format_seen": token_passes["mxf4_e2m1_format"],
        "sm120_e2m1_ue4m3_specialization_seen": token_passes["sm120_e2m1_ue4m3_specialization"],
        "sm120_mxf4nvf4_ue4m3_asm_seen": token_passes["sm120_mxf4nvf4_ue4m3_asm"],
        "expected_examples_seen": expected_examples_seen,
        "sm120_tests_seen": sm120_tests_seen,
        "banked_cutlass_abi": False,
        "banked_grouped_moe_fp4_mma_poc": False,
        "banked_kernel_speed": False,
        "banked_default_promotion": False,
    }

    git = {
        "head": _run(["git", "rev-parse", "HEAD"], cutlass_dir),
        "branch": _run(["git", "branch", "--show-current"], cutlass_dir),
        "status": _run(["git", "status", "--short"], cutlass_dir),
    } if cutlass_dir.exists() else {}
    passes["cutlass_git_recorded"] = bool((git.get("head") or {}).get("ok"))

    abi_ok = all(
        passes[key] is True
        for key in [
            "cutlass_dir_exists",
            "cutlass_git_recorded",
            "key_files_present",
            "sm120_ue4m3_macro_seen",
            "scale_format_ue4m3_seen",
            "scale_type_ue4m3_seen",
            "mxf4_e2m1_format_seen",
            "sm120_e2m1_ue4m3_specialization_seen",
            "sm120_mxf4nvf4_ue4m3_asm_seen",
            "expected_examples_seen",
            "sm120_tests_seen",
        ]
    )
    passes["banked_cutlass_abi"] = abi_ok
    passes["all"] = abi_ok

    return {
        "schema": "lynn-stage6-r5c-cutlass-ue4m3-census-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cutlass_dir": str(cutlass_dir),
        "git": git,
        "file_exists": {name: str(path) if exists else None for name, path in key_files.items() for exists in [path.exists()]},
        "token_passes": token_passes,
        "token_hits": {
            "config": _hits(arch_config, ["CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED"]),
            "scale_desc": _hits(sm100_desc, ["ScaleFormat::UE4M3", "float_ue4m3_t", "MXF4Format::E2M1"]),
            "sm120_mma": _hits(sm120_mma, ["SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t", "mxf4nvf4.block_scale.scale_vec::4X"]),
        },
        "expected_example_paths": expected_example_names,
        "examples": examples,
        "sm120_tests": sm120_tests,
        "passes": passes,
        "decision": "PASS_R5C_NVF4_UE4M3_CUTLASS_ABI" if abi_ok else "FAIL_R5C_NVF4_UE4M3_CUTLASS_ABI",
        "promotion_boundary": {
            "grouped_moe_fp4_mma_poc": False,
            "kernel_speed": False,
            "default_runtime": False,
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
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": data["decision"], "passes": data["passes"]}, indent=2))
    return 0 if data["passes"]["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
