#!/usr/bin/env python3
"""Stage 6 P4 real-runtime bridge preflight.

This gate is narrower than a fused-kernel benchmark: it loads a real resident
runner, obtains a real layer MoE input, deletes active-expert BF16 shadows, then
checks that the opt-in P4 backend reaches the native zero-shadow two-stage
reference path through ``moe_forward_decode_packed_nvfp4``.
"""
from __future__ import annotations

import argparse
import json
import math
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
from engine.moe_packed_nvfp4 import _layer_selected_for_native_cuda, moe_forward_decode_packed_nvfp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


EXPECTED_BACKEND = "fused_zero_shadow_out_contract"
NATIVE_CALL_COUNT_KEY = "_p4_fused_zero_shadow_out_contract_call_count"
NATIVE_LAST_SHAPES_KEY = "_p4_fused_zero_shadow_out_contract_last_shapes"
ACTIVE_SHADOW_KEYS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
PACKED_KEYS = (
    "mlp.experts._gate_up_packed",
    "mlp.experts._gate_up_scale",
    "mlp.experts._gate_up_global_scale",
    "mlp.experts._down_packed",
    "mlp.experts._down_scale",
    "mlp.experts._down_global_scale",
)
ACTIVE_SCRATCH_KEYS = ("mlp.experts._active_inter_scratch", "mlp.experts._active_out_scratch")
EXPECTED_PACKED_SHAPES = {
    "mlp.experts._gate_up_packed": (None, 1024, 1024),
    "mlp.experts._gate_up_scale": (None, 1024, 128),
    "mlp.experts._gate_up_global_scale": (1,),
    "mlp.experts._down_packed": (None, 2048, 256),
    "mlp.experts._down_scale": (None, 2048, 32),
    "mlp.experts._down_global_scale": (1,),
}
EXPECTED_PACKED_DTYPES = {
    "mlp.experts._gate_up_packed": "torch.uint8",
    "mlp.experts._gate_up_scale": "torch.float32",
    "mlp.experts._gate_up_global_scale": "torch.float32",
    "mlp.experts._down_packed": "torch.uint8",
    "mlp.experts._down_scale": "torch.float32",
    "mlp.experts._down_global_scale": "torch.float32",
}


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


def _shape_matches(actual: list[int], expected: tuple[int | None, ...]) -> bool:
    return len(actual) == len(expected) and all(exp is None or got == exp for got, exp in zip(actual, expected))


def _packed_manifest_ok(manifest: dict[str, Any]) -> bool:
    if set(manifest) != set(PACKED_KEYS):
        return False
    for key, meta in manifest.items():
        if meta.get("dtype") != EXPECTED_PACKED_DTYPES[key]:
            return False
        if not meta.get("contiguous"):
            return False
        if not _shape_matches(list(meta.get("shape") or []), EXPECTED_PACKED_SHAPES[key]):
            return False
    return True


def _active_scratch_ok(manifest: dict[str, Any], top_k: int) -> bool:
    expected = {
        "mlp.experts._active_inter_scratch": ([top_k, 512], "torch.bfloat16"),
        "mlp.experts._active_out_scratch": ([2048], "torch.bfloat16"),
    }
    if set(manifest) != set(expected):
        return False
    for key, (shape, dtype) in expected.items():
        meta = manifest[key]
        if meta.get("shape") != shape or meta.get("dtype") != dtype or not meta.get("contiguous"):
            return False
    return True


def _bf16_active_shadow_aliases(w: dict[str, Any]) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for key, value in w.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16:
            continue
        if "mlp.experts" not in key:
            continue
        if key.startswith("mlp.experts._active_"):
            continue
        if "gate_up" in key or "down" in key:
            aliases[key] = _tensor_meta(value)
    return aliases


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "lynn-stage6-p4-runtime-bridge-preflight-v1",
        "model": args.model,
        "layer": args.layer,
        "prompt": args.prompt,
        "expected_backend": EXPECTED_BACKEND,
        "expected_reference": "caller-owned two-stage packed-NVFP4 active-MoE reference",
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
        "LYNN_MOE_ACTIVE_SCRATCH": "1",
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
        result["native_layer_selected_for_candidate"] = bool(_layer_selected_for_native_cuda(cfg))
        result["packed_manifest_before_candidate"] = _packed_manifest(w)
        result["active_scratch_manifest"] = {
            key: _tensor_meta(w[key]) for key in ACTIVE_SCRATCH_KEYS if isinstance(w.get(key), torch.Tensor)
        }

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
        result["bf16_active_shadow_aliases_after_delete"] = _bf16_active_shadow_aliases(w)

        candidate_error: dict[str, str] | None = None
        candidate = None
        native_call_count_before = int(w.get(NATIVE_CALL_COUNT_KEY, 0))
        old_backend = _set_env({"LYNN_NATIVE_ACTIVE_MOE_BACKEND": EXPECTED_BACKEND})
        try:
            try:
                candidate = moe_forward_decode_packed_nvfp4(h_moe, w, cfg)
            except Exception as exc:
                candidate_error = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            _restore_env(old_backend)
        native_call_count_after = int(w.get(NATIVE_CALL_COUNT_KEY, 0))
        result["native_backend_call_count"] = {
            "key": NATIVE_CALL_COUNT_KEY,
            "before": native_call_count_before,
            "after": native_call_count_after,
            "delta": native_call_count_after - native_call_count_before,
            "last_shapes": w.get(NATIVE_LAST_SHAPES_KEY),
        }

        result["candidate_error"] = candidate_error
        if candidate is not None:
            diff = (candidate.float() - baseline.float()).abs()
            baseline_norm_for_rel = baseline.float().norm().clamp_min(1e-20)
            result["candidate"] = {
                "backend": EXPECTED_BACKEND,
                "output_shape": list(candidate.shape),
                "output_dtype": str(candidate.dtype),
                "norm": float(candidate.float().norm().item()),
                "finite": bool(torch.isfinite(candidate.float()).all().item()),
                "max_abs_diff_vs_baseline": float(diff.max().item()),
                "mean_abs_diff_vs_baseline": float(diff.mean().item()),
                "rel_l2_vs_baseline": float((candidate.float() - baseline.float()).norm().item() / baseline_norm_for_rel.item()),
            }
        baseline_shape_ok = result["baseline"]["output_shape"] == list(h_moe.shape)
        baseline_dtype_ok = result["baseline"]["output_dtype"] == "torch.bfloat16"
        native_layer_selected = result.get("native_layer_selected_for_candidate") is True
        native_backend_called = (result.get("native_backend_call_count") or {}).get("delta") == 1
        packed_manifest_ok = _packed_manifest_ok(result["packed_manifest_before_candidate"])
        active_scratch_ok = _active_scratch_ok(result["active_scratch_manifest"], int(cfg["num_experts_per_tok"]))
        shadow_absent = not result["active_shadow_keys_present_after_delete"]
        no_bf16_aliases = not result["bf16_active_shadow_aliases_after_delete"]
        candidate_out = result.get("candidate") or {}
        candidate_shape_ok = candidate_out.get("output_shape") == result["baseline"]["output_shape"]
        candidate_dtype_ok = candidate_out.get("output_dtype") == "torch.bfloat16"
        candidate_finite = candidate_out.get("finite") is True
        candidate_numeric_ok = (
            candidate_error is None
            and candidate_shape_ok
            and candidate_dtype_ok
            and candidate_finite
            and float(candidate_out.get("rel_l2_vs_baseline", 1.0)) <= args.rel_l2_threshold
            and float(candidate_out.get("max_abs_diff_vs_baseline", 1e9)) <= args.max_abs_threshold
        )
        baseline_ok = baseline_norm > 0.0 and math.isfinite(baseline_norm) and baseline_shape_ok and baseline_dtype_ok
        result["passes"] = {
            "cuda_available": True,
            "native_layer_selected": native_layer_selected,
            "native_backend_called": native_backend_called,
            "baseline_triton_nonzero": baseline_ok,
            "baseline_shape_dtype": bool(baseline_shape_ok and baseline_dtype_ok),
            "packed_tensors_present": packed_manifest_ok,
            "active_scratch_present": active_scratch_ok,
            "active_shadows_removed": bool(removed) and shadow_absent and no_bf16_aliases,
            "candidate_output_returned": candidate_error is None and candidate is not None,
            "candidate_shape_dtype": bool(candidate_shape_ok and candidate_dtype_ok),
            "candidate_numeric_vs_triton": bool(candidate_numeric_ok),
            "fused_kernel_unbanked": result["banked_fused_kernel"] is False,
            "default_promotion_closed": result["banked_default_promotion"] is False,
            "all": bool(native_layer_selected and native_backend_called and baseline_ok and packed_manifest_ok and active_scratch_ok and removed and shadow_absent and no_bf16_aliases and candidate_numeric_ok),
        }
        result["banked_runtime_bridge_preflight"] = bool(result["passes"]["all"])
        result["decision"] = "PASS_TWO_STAGE_RUNTIME_BRIDGE" if result["passes"]["all"] else "FAIL_RUNTIME_BRIDGE_CONTRACT"
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
    ap.add_argument("--rel-l2-threshold", type=float, default=0.02)
    ap.add_argument("--max-abs-threshold", type=float, default=1.0)
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
