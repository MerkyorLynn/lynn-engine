#!/usr/bin/env python3
"""Stage 6 P4 real-runtime bridge preflight.

This gate is narrower than a fused-kernel benchmark: it loads a real resident
runner, obtains a real layer MoE input, deletes active-expert BF16 shadows, then
checks that the opt-in P4 backend reaches the native zero-shadow fail-loud
boundary through ``moe_forward_decode_packed_nvfp4``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import _prefill_to_layer_input  # noqa: E402
from benchmarks.p37_moe_config_generate_gate import BASE_ENV  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


EXPECTED_BACKEND = "fused_zero_shadow_out_contract"
EXPECTED_ERROR = "P4 fused 4-bit zero-shadow CUDA kernel is not implemented yet"
ACTIVE_SHADOW_KEYS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
PACKED_KEYS = (
    "mlp.experts._gate_up_packed",
    "mlp.experts._gate_up_scale",
    "mlp.experts._gate_up_global_scale",
    "mlp.experts._down_packed",
    "mlp.experts._down_scale",
    "mlp.experts._down_global_scale",
)


def _set_env(updates: dict[str, str]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _tensor_meta(t: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "bytes": _tensor_bytes(t),
        "contiguous": bool(t.is_contiguous()),
    }


def _remove_active_shadows(w: dict[str, Any]) -> dict[str, Any]:
    removed: dict[str, Any] = {}
    for key in ACTIVE_SHADOW_KEYS:
        t = w.pop(key, None)
        if isinstance(t, torch.Tensor):
            removed[key] = _tensor_meta(t)
    return removed


def _packed_manifest(w: dict[str, Any]) -> dict[str, Any]:
    return {key: _tensor_meta(w[key]) for key in PACKED_KEYS if isinstance(w.get(key), torch.Tensor)}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "lynn-stage6-p4-runtime-bridge-preflight-v1",
        "model": args.model,
        "layer": args.layer,
        "prompt": args.prompt,
        "expected_backend": EXPECTED_BACKEND,
        "expected_error": EXPECTED_ERROR,
        "banked_runtime_bridge_preflight": False,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "passes": {"all": False},
    }
    if not torch.cuda.is_available():
        result["decision"] = "BLOCKED_NO_CUDA"
        result["passes"]["cuda_available"] = False
        return result

    env = dict(BASE_ENV)
    env.update({
        "LYNN_MOE_FAST_FIXED": "0",
        "LYNN_NATIVE_ACTIVE_MOE_BACKEND": "triton",
        "LYNN_NATIVE_ACTIVE_MOE_LAYERS": str(args.layer),
        "LYNN_NATIVE_GATEUP_BACKEND": "triton_fast_decode",
        "LYNN_NATIVE_DOWN_BACKEND": "triton",
        "LYNN_NATIVE_CUDA_BUILD_DIR": os.environ.get(
            "LYNN_NATIVE_CUDA_BUILD_DIR",
            f"/tmp/lynn_engine_native_build/p4_runtime_bridge_{int(time.time())}",
        ),
    })
    old = _set_env(env)
    try:
        runner = LynnIncrementalRunner(
            args.model,
            device="cuda",
            dtype=torch.bfloat16,
            max_seq_len=args.max_seq_len,
            verbose=args.verbose,
        )
        h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
        w = runner.layer_weights[args.layer]
        cfg = runner.layer_cfgs[args.layer]
        h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
        result["packed_manifest_before_candidate"] = _packed_manifest(w)

        baseline = moe_forward_decode_packed_nvfp4(h_moe, w, cfg)
        baseline_norm = float(baseline.float().norm().item())
        result["baseline"] = {
            "backend": "triton",
            "output_shape": list(baseline.shape),
            "output_dtype": str(baseline.dtype),
            "norm": baseline_norm,
        }

        removed = _remove_active_shadows(w)
        result["removed_active_shadows"] = removed
        result["active_shadow_keys_present_after_delete"] = [key for key in ACTIVE_SHADOW_KEYS if key in w]

        candidate_error: dict[str, str] | None = None
        old_backend = _set_env({"LYNN_NATIVE_ACTIVE_MOE_BACKEND": EXPECTED_BACKEND})
        try:
            try:
                _ = moe_forward_decode_packed_nvfp4(h_moe, w, cfg)
                result["candidate_unexpected_output"] = True
            except Exception as exc:
                candidate_error = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            _restore_env(old_backend)

        result["candidate_error"] = candidate_error
        packed_present = all(key in result["packed_manifest_before_candidate"] for key in PACKED_KEYS)
        shadow_absent = not result["active_shadow_keys_present_after_delete"]
        fail_loud = bool(candidate_error and EXPECTED_ERROR in candidate_error.get("message", ""))
        baseline_ok = baseline_norm > 0.0
        result["passes"] = {
            "cuda_available": True,
            "baseline_triton_nonzero": baseline_ok,
            "packed_tensors_present": packed_present,
            "active_shadows_removed": bool(removed) and shadow_absent,
            "candidate_fail_loud": fail_loud,
            "fused_kernel_unbanked": result["banked_fused_kernel"] is False,
            "default_promotion_closed": result["banked_default_promotion"] is False,
            "all": bool(baseline_ok and packed_present and removed and shadow_absent and fail_loud),
        }
        result["banked_runtime_bridge_preflight"] = bool(result["passes"]["all"])
        result["decision"] = "PASS_RUNTIME_BRIDGE_CONTRACT" if result["passes"]["all"] else "FAIL_RUNTIME_BRIDGE_CONTRACT"
        return result
    except Exception as exc:
        result["decision"] = "BLOCKED_RUNTIME_EXCEPTION"
        result["runner_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return result
    finally:
        _restore_env(old)


def main() -> int:
    ap = argparse.ArgumentParser(description="P4 real-runtime bridge preflight.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prompt", default="Explain MoE active parameters in one sentence.")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    started = time.time()
    result = run_preflight(args)
    result["elapsed_s"] = round(time.time() - started, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[p4-runtime-bridge] decision={result.get('decision')}")
    print(f"[p4-runtime-bridge] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
