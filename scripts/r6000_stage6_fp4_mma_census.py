#!/usr/bin/env python3
"""Stage 6 R6000 FP4-MMA bring-up and public-kernel census.

Run this on the rented RTX PRO 6000 Blackwell host. It answers three questions:

1. Is the machine actually usable for Lynn FP4-MMA work?
2. Do CUTLASS/CuTe + native SM120 FP4 contracts pass?
3. Is there an installed public kernel route to inspect first (vLLM
   Marlin/Machete/CUTLASS NVFP4), before writing new CUDA from scratch?

This script does not promote a kernel. It only banks bring-up evidence.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pkgutil
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _tail(text: str, max_lines: int = 24) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[-max_lines:])


def _run(cmd: list[str], *, cwd: Path = ROOT, timeout_s: int = 300) -> dict[str, Any]:
    t0 = time.time()
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
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "seconds": time.time() - t0,
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "ok": False,
            "returncode": None,
            "seconds": time.time() - t0,
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
            "timeout": True,
        }


def _json_from_python(code: str) -> dict[str, Any]:
    proc = _run([sys.executable, "-c", code], timeout_s=90)
    out: dict[str, Any] = {"process": proc}
    if proc["ok"]:
        try:
            out["json"] = json.loads(proc["stdout_tail"])
        except json.JSONDecodeError as exc:
            out["json_error"] = repr(exc)
    return out


def _system_inventory(workspace: Path) -> dict[str, Any]:
    return {
        "nvidia_smi_query": _run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ], timeout_s=30),
        "nvidia_smi_full": _run(["nvidia-smi"], timeout_s=30),
        "nvcc_version": _run(["bash", "-lc", "command -v nvcc && nvcc --version"], timeout_s=30),
        "disk_workspace": _run(["df", "-BG", str(workspace)], timeout_s=30),
        "python": _run([sys.executable, "--version"], timeout_s=10),
        "git_head": _run(["git", "rev-parse", "HEAD"], timeout_s=10),
        "git_status": _run(["git", "status", "--short"], timeout_s=10),
    }


def _torch_inventory() -> dict[str, Any]:
    return _json_from_python(
        r"""
import json
try:
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gib": props.total_memory / (1024**3),
        }
    else:
        payload = {"cuda_available": False}
except Exception as exc:
    payload = {"error": repr(exc)}
print(json.dumps(payload))
"""
    )


def _public_kernel_census() -> dict[str, Any]:
    code = r"""
import importlib
import importlib.util
import json
import pkgutil

payload = {
    "packages": {},
    "module_candidates": {},
}
for name in ["vllm", "cutlass", "triton", "flashinfer", "tensorrt_llm"]:
    spec = importlib.util.find_spec(name)
    row = {"importable": spec is not None, "origin": getattr(spec, "origin", None)}
    if spec is not None:
        try:
            mod = importlib.import_module(name)
            row["version"] = getattr(mod, "__version__", None)
            row["path"] = getattr(mod, "__path__", None) and [str(p) for p in mod.__path__]
        except Exception as exc:
            row["import_error"] = repr(exc)
    payload["packages"][name] = row

vllm_spec = importlib.util.find_spec("vllm")
if vllm_spec is not None:
    try:
        import vllm
        names = []
        for mod in pkgutil.walk_packages(vllm.__path__, prefix="vllm."):
            low = mod.name.lower()
            if any(k in low for k in ["nvfp4", "machete", "marlin", "cutlass"]):
                names.append(mod.name)
            if len(names) >= 200:
                break
        payload["module_candidates"]["vllm"] = names
    except Exception as exc:
        payload["module_candidates"]["vllm_error"] = repr(exc)

explicit = [
    "vllm.model_executor.kernels.linear.nvfp4.marlin",
    "vllm.model_executor.kernels.linear.nvfp4.machete",
    "vllm.model_executor.layers.quantization.nvfp4",
    "vllm.model_executor.layers.fused_moe.cutlass_moe",
]
payload["explicit_imports"] = {}
for name in explicit:
    spec = importlib.util.find_spec(name)
    payload["explicit_imports"][name] = {
        "importable": spec is not None,
        "origin": getattr(spec, "origin", None),
    }
print(json.dumps(payload))
"""
    return _json_from_python(code)


def _parse_memory_gib(torch_info: dict[str, Any]) -> float | None:
    data = torch_info.get("json") or {}
    value = data.get("total_memory_gib")
    return float(value) if isinstance(value, (int, float)) else None


def _parse_capability(torch_info: dict[str, Any]) -> list[int] | None:
    data = torch_info.get("json") or {}
    cap = data.get("capability")
    if isinstance(cap, list) and len(cap) == 2 and all(isinstance(x, int) for x in cap):
        return cap
    return None


def _disk_free_gib(system: dict[str, Any]) -> float | None:
    text = ((system.get("disk_workspace") or {}).get("stdout_tail") or "").strip()
    if not text:
        return None
    lines = text.splitlines()
    if len(lines) < 2:
        return None
    parts = lines[-1].split()
    if len(parts) < 4:
        return None
    raw = parts[3].rstrip("G")
    try:
        return float(raw)
    except ValueError:
        return None


def _run_contracts(out_dir: Path, timeout_s: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    contracts = {
        "p76_cutlass_cute_toolchain": [
            py,
            "benchmarks/p76_cutlass_cute_toolchain_probe.py",
            "--out",
            str(out_dir / "p76_cutlass_cute_toolchain.json"),
        ],
        "p79_nvcc_fp4_mma_target_matrix": [
            py,
            "benchmarks/p79_nvcc_fp4_mma_target_matrix.py",
            "--out",
            str(out_dir / "p79_nvcc_fp4_mma_target_matrix.json"),
        ],
        "p85_blockscaled_fp4_mma_contract": [
            py,
            "benchmarks/p85_sm120a_blockscaled_fp4_mma_contract.py",
            "--out",
            str(out_dir / "p85_blockscaled_fp4_mma_contract.json"),
            "--blocks",
            "256",
            "--repeats",
            "3",
            "--warmup",
            "1",
        ],
        "p87_layout_tile_contract": [
            py,
            "benchmarks/p87_sm120a_fp4_layout_tile_contract.py",
            "--out",
            str(out_dir / "p87_layout_tile_contract.json"),
            "--blocks",
            "256",
            "--repeats",
            "3",
            "--warmup",
            "1",
        ],
        "p103_fp8_activation_fp4_weight_mma": [
            py,
            "benchmarks/p103_sm120a_fp8_activation_fp4_weight_mma_probe.py",
            "--out",
            str(out_dir / "p103_fp8_activation_fp4_weight_mma.json"),
            "--timeout-s",
            "45",
        ],
    }
    results: dict[str, Any] = {}
    for name, cmd in contracts.items():
        proc = _run(cmd, timeout_s=timeout_s)
        result_path = Path(cmd[cmd.index("--out") + 1])
        row = dict(proc)
        if result_path.exists():
            try:
                row["result"] = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                row["result_json_error"] = repr(exc)
        results[name] = row
    return results


def _contract_passes(contracts: dict[str, Any]) -> dict[str, bool]:
    if not contracts:
        return {}
    out: dict[str, bool] = {}
    out["p76_cutlass_cute_toolchain"] = bool(((contracts.get("p76_cutlass_cute_toolchain") or {}).get("result") or {}).get("compile_ok"))
    out["p79_nvcc_fp4_mma_target_matrix"] = bool(((contracts.get("p79_nvcc_fp4_mma_target_matrix") or {}).get("result") or {}).get("any_fp4_mma_target_ok"))
    out["p85_blockscaled_fp4_mma_contract"] = bool(((contracts.get("p85_blockscaled_fp4_mma_contract") or {}).get("result") or {}).get("all_exact"))
    out["p87_layout_tile_contract"] = bool(((contracts.get("p87_layout_tile_contract") or {}).get("result") or {}).get("all_exact"))
    p103 = ((contracts.get("p103_fp8_activation_fp4_weight_mma") or {}).get("result") or {})
    out["p103_fp8_activation_fp4_weight_mma"] = bool(p103.get("controls_ok") and p103.get("fp8_activation_fp4_weight_supported"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="R6000 FP4-MMA bring-up and public-kernel census.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifact-dir", default="")
    ap.add_argument("--run-contracts", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--min-disk-gib", type=float, default=500.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else out_path.parent / "contract_outputs"

    system = _system_inventory(ROOT)
    torch_info = _torch_inventory()
    public = _public_kernel_census()
    contracts = _run_contracts(artifact_dir, args.timeout_s) if args.run_contracts else {}
    contract_passes = _contract_passes(contracts)

    cap = _parse_capability(torch_info)
    mem_gib = _parse_memory_gib(torch_info)
    disk_free = _disk_free_gib(system)
    public_json = public.get("json") or {}
    vllm_modules = ((public_json.get("module_candidates") or {}).get("vllm") or [])
    explicit = public_json.get("explicit_imports") or {}

    passes = {
        "torch_cuda_recorded": cap is not None,
        "blackwell_capability": bool(cap and cap[0] == 12),
        "r6000_class_memory": bool(mem_gib and mem_gib >= 85.0),
        "disk_headroom": bool(disk_free is not None and disk_free >= args.min_disk_gib),
        "public_kernel_census_recorded": "packages" in public_json,
        "vllm_nvfp4_or_marlin_seen": bool(
            vllm_modules
            or any((row or {}).get("importable") for row in explicit.values() if isinstance(row, dict))
        ),
        "contract_suite_recorded": bool(contracts),
        "contract_suite_all_pass": bool(contract_passes and all(contract_passes.values())),
    }
    passes["all"] = all(passes.values())

    decision = "PASS_R6000_FP4_MMA_BRINGUP"
    if not passes["all"]:
        decision = "READY_BUT_NOT_BANKED"
        if contracts:
            decision = "FAIL_R6000_FP4_MMA_BRINGUP"

    result = {
        "schema": "lynn-stage6-r6000-fp4-mma-census-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workspace": str(ROOT),
        "min_disk_gib": args.min_disk_gib,
        "system": system,
        "torch": torch_info,
        "public_kernel_census": public,
        "contracts": contracts,
        "contract_passes": contract_passes,
        "passes": passes,
        "decision": decision,
        "promotion_boundary": {
            "kernel_promoted": False,
            "default_runtime_changed": False,
            "speed_claim": False,
        },
        "next_step": (
            "If PASS, start Lynn NVFP4 grouped-MoE FP4-MMA kernel POC from CUTLASS/CuTe and public Marlin/Machete census. "
            "If FAIL, fix machine/toolchain before writing kernels."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "passes": passes,
        "capability": cap,
        "memory_gib": mem_gib,
        "disk_free_gib": disk_free,
        "out": str(out_path),
    }, ensure_ascii=False, indent=2))
    return 0 if passes["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
