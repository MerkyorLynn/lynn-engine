#!/usr/bin/env python3
"""P57: inventory the official/vendor NVFP4 route for Lynn 27B.

This is intentionally fail-loud and read-only. It answers:

* is ModelOpt / llmcompressor available in the current environment?
* does compressed-tensors only provide a converter or a BF16 quantizer?
* is the model HF-vanilla-compatible, or variable-expert Lynn-native only?
* what concrete route is possible without corrupting the active engine env?
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


def _module_status(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return {"available": False}
        mod = __import__(name)
        return {
            "available": True,
            "file": getattr(mod, "__file__", None),
            "version": getattr(mod, "__version__", None),
        }
    except Exception as exc:  # pragma: no cover - inventory only
        return {"available": False, "error": repr(exc)}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    model = args.model
    cfg = _load_json(model / "config.json") or {}
    spec = _load_json(model / "lynn_engine_variable_expert_spec.json")
    modules = {
        name: _module_status(name)
        for name in [
            "modelopt",
            "llmcompressor",
            "compressed_tensors",
            "transformers",
            "accelerate",
            "triton",
            "torch",
        ]
    }
    converter_available = _module_status("compressed_tensors.entrypoints.convert.converters.modelopt_nvfp4")
    modules["compressed_tensors.modelopt_nvfp4_converter"] = converter_available

    text_cfg = cfg.get("text_config", cfg)
    variable_expert = bool(spec and spec.get("hf_vanilla_compatible") is False)
    remaining_counts = None
    if spec and "remaining_experts_by_layer" in spec:
        remaining_counts = sorted(set(int(v) for v in spec["remaining_experts_by_layer"].values()))

    can_quantize_bf16_with_vendor_here = bool(
        modules["modelopt"]["available"] or modules["llmcompressor"]["available"]
    )
    has_converter_only = bool(converter_available.get("available") and not can_quantize_bf16_with_vendor_here)

    blockers = []
    if variable_expert:
        blockers.append("model is physically variable-expert and HF-vanilla-incompatible")
    if not can_quantize_bf16_with_vendor_here:
        blockers.append("current environment lacks ModelOpt/llmcompressor BF16->NVFP4 quantizer")
    if has_converter_only:
        blockers.append("compressed-tensors modelopt_nvfp4 is a converter for already-quantized ModelOpt artifacts")

    recommended_routes = []
    recommended_routes.append(
        {
            "route": "lynn_native_per16_engine",
            "status": "current_mainline",
            "why": "works with physical variable-expert 27B and existing per-16 FP32 scale artifact",
        }
    )
    recommended_routes.append(
        {
            "route": "vendor_friendly_nvfp4_v2_from_bf16",
            "status": "possible_but_requires_new_artifact",
            "requirements": [
                "separate ModelOpt/llmcompressor environment",
                "BF16 final input",
                "either pad/mask variable experts back to HF-vanilla 256 experts or add variable-expert support",
                "full V8/V9/tool/no-think/longctx retention gates",
            ],
        }
    )
    recommended_routes.append(
        {
            "route": "posthoc_convert_current_lynn_native_to_vendor_layout",
            "status": "rejected_by_p54",
            "why": "e8m0/group32 scale-search upper bound failed 0.995 inter-cosine safety gate",
        }
    )

    result = {
        "schema_version": "lynn-engine-p57-vendor-route-inventory-v1",
        "python": sys.version,
        "model": str(model),
        "model_type": text_cfg.get("model_type") or cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "num_hidden_layers": text_cfg.get("num_hidden_layers"),
        "num_experts_config": text_cfg.get("num_experts"),
        "num_experts_per_tok": text_cfg.get("num_experts_per_tok"),
        "variable_expert": variable_expert,
        "remaining_expert_counts": remaining_counts,
        "modules": modules,
        "blockers": blockers,
        "can_quantize_bf16_with_vendor_here": can_quantize_bf16_with_vendor_here,
        "has_converter_only": has_converter_only,
        "recommended_routes": recommended_routes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
