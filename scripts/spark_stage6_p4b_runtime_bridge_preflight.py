#!/usr/bin/env python3
"""Stage 6 P4B real-runtime bridge fail-loud preflight.

This gate loads a real resident runner and proves the opt-in P4B backend reaches
the P4B native symbol after BF16 active-expert shadows are removed. P4B is still
unimplemented, so the expected behavior is a fail-loud not-implemented error
with the P4B call counter advanced exactly once. This banks route integrity
only; it never banks fused-kernel speed or default promotion.
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
from scripts.spark_stage6_p4_runtime_bridge_preflight import (  # noqa: E402
    ACTIVE_SHADOW_KEYS,
    PACKED_KEYS,
    _bf16_active_shadow_aliases,
    _packed_manifest,
    _packed_manifest_ok,
    _remove_active_shadows,
    _restore_env,
    _set_env,
    _tensor_meta,
)


EXPECTED_BACKEND = "fused_zero_shadow_single_kernel_contract"
NATIVE_CALL_COUNT_KEY = "_p4b_fused_zero_shadow_single_kernel_contract_call_count"
NATIVE_LAST_SHAPES_KEY = "_p4b_fused_zero_shadow_single_kernel_contract_last_shapes"
ACTIVE_OUT_SCRATCH_KEY = "mlp.experts._active_out_scratch"
ACTIVE_INTER_SCRATCH_KEY = "mlp.experts._active_inter_scratch"
FAIL_LOUD_NEEDLES = (
    "P4B single-kernel fused zero-shadow contract is not implemented yet",
    "do not bank fused-kernel speed or promote this backend",
)


def _active_out_scratch_ok(manifest: dict[str, Any]) -> bool:
    meta = manifest.get(ACTIVE_OUT_SCRATCH_KEY) or {}
    return (
        meta.get("shape") == [2048]
        and meta.get("dtype") == "torch.bfloat16"
        and meta.get("contiguous") is True
    )


def _last_shapes_out_only(last_shapes: Any, *, top_k: int) -> bool:
    if not isinstance(last_shapes, dict):
        return False
    if "inter_scratch" in last_shapes or "inter_out" in last_shapes:
        return False
    return (
        last_shapes.get("hidden") == (1, 2048)
        or last_shapes.get("hidden") == [1, 2048]
    ) and (
        last_shapes.get("expert_ids") == (1, top_k)
        or last_shapes.get("expert_ids") == [1, top_k]
    ) and (
        last_shapes.get("out") == (1, 2048)
        or last_shapes.get("out") == [1, 2048]
    )


def _fail_loud_error(candidate_error: dict[str, str] | None) -> bool:
    if not candidate_error:
        return False
    message = candidate_error.get("message", "")
    return all(needle in message for needle in FAIL_LOUD_NEEDLES)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "lynn-stage6-p4b-runtime-bridge-preflight-v1",
        "model": args.model,
        "layer": args.layer,
        "prompt": args.prompt,
        "expected_backend": EXPECTED_BACKEND,
        "expected_behavior": "real runner reaches P4B native symbol and fails loud because fused implementation is absent",
        "banked_p4b_runtime_bridge_preflight": False,
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
            f"/tmp/lynn_engine_native_build/p4b_runtime_bridge_{int(time.time())}",
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
        top_k = int(cfg["num_experts_per_tok"])
        h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
        result["native_layer_selected_for_candidate"] = bool(_layer_selected_for_native_cuda(cfg))
        result["packed_manifest_before_candidate"] = _packed_manifest(w)
        result["active_scratch_manifest"] = {
            key: _tensor_meta(w[key])
            for key in (ACTIVE_INTER_SCRATCH_KEY, ACTIVE_OUT_SCRATCH_KEY)
            if isinstance(w.get(key), torch.Tensor)
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
        candidate_returned = False
        native_call_count_before = int(w.get(NATIVE_CALL_COUNT_KEY, 0))
        old_backend = _set_env({"LYNN_NATIVE_ACTIVE_MOE_BACKEND": EXPECTED_BACKEND})
        try:
            try:
                _ = moe_forward_decode_packed_nvfp4(h_moe, w, cfg)
                candidate_returned = True
            except Exception as exc:
                candidate_error = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            _restore_env(old_backend)
        native_call_count_after = int(w.get(NATIVE_CALL_COUNT_KEY, 0))
        native_last_shapes = w.get(NATIVE_LAST_SHAPES_KEY)
        result["native_backend_call_count"] = {
            "key": NATIVE_CALL_COUNT_KEY,
            "before": native_call_count_before,
            "after": native_call_count_after,
            "delta": native_call_count_after - native_call_count_before,
            "last_shapes": native_last_shapes,
        }
        result["candidate_returned"] = candidate_returned
        result["candidate_error"] = candidate_error
        result["fail_loud_needles"] = list(FAIL_LOUD_NEEDLES)

        baseline_shape_ok = result["baseline"]["output_shape"] == list(h_moe.shape)
        baseline_dtype_ok = result["baseline"]["output_dtype"] == "torch.bfloat16"
        baseline_ok = baseline_norm > 0.0 and math.isfinite(baseline_norm) and baseline_shape_ok and baseline_dtype_ok
        native_layer_selected = result.get("native_layer_selected_for_candidate") is True
        native_backend_called = (result.get("native_backend_call_count") or {}).get("delta") == 1
        packed_manifest_ok = _packed_manifest_ok(result["packed_manifest_before_candidate"])
        active_out_scratch_ok = _active_out_scratch_ok(result["active_scratch_manifest"])
        shadow_absent = not result["active_shadow_keys_present_after_delete"]
        no_bf16_aliases = not result["bf16_active_shadow_aliases_after_delete"]
        fail_loud = _fail_loud_error(candidate_error)
        last_shapes_out_only = _last_shapes_out_only(native_last_shapes, top_k=top_k)
        result["passes"] = {
            "cuda_available": True,
            "native_layer_selected": native_layer_selected,
            "native_backend_called": native_backend_called,
            "baseline_triton_nonzero": baseline_ok,
            "baseline_shape_dtype": bool(baseline_shape_ok and baseline_dtype_ok),
            "packed_tensors_present": packed_manifest_ok,
            "active_out_scratch_present": active_out_scratch_ok,
            "active_shadows_removed": bool(removed) and shadow_absent and no_bf16_aliases,
            "candidate_returned_false": not candidate_returned,
            "p4b_fail_loud_not_implemented": fail_loud,
            "p4b_last_shapes_out_only": last_shapes_out_only,
            "fused_kernel_unbanked": result["banked_fused_kernel"] is False,
            "default_promotion_closed": result["banked_default_promotion"] is False,
            "all": bool(
                native_layer_selected
                and native_backend_called
                and baseline_ok
                and packed_manifest_ok
                and active_out_scratch_ok
                and removed
                and shadow_absent
                and no_bf16_aliases
                and not candidate_returned
                and fail_loud
                and last_shapes_out_only
            ),
        }
        result["banked_p4b_runtime_bridge_preflight"] = bool(result["passes"]["all"])
        result["decision"] = "PASS_P4B_RUNTIME_BRIDGE_FAILLOUD" if result["passes"]["all"] else "FAIL_P4B_RUNTIME_BRIDGE"
        return result
    except Exception as exc:
        result["decision"] = "BLOCKED_RUNTIME_EXCEPTION"
        result["runner_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return result
    finally:
        _restore_env(old)


def main() -> int:
    ap = argparse.ArgumentParser(description="P4B real-runtime bridge fail-loud preflight.")
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
    print(f"[p4b-runtime-bridge] decision={result.get('decision')}")
    print(f"[p4b-runtime-bridge] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
