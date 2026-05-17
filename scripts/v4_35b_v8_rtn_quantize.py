#!/usr/bin/env python3
"""Quantize one Lynn V4 35B-A3B BF16 checkpoint to compressed-tensors NVFP4 v8-RTN.

This is the compatibility/public-serving artifact line used by Spark/SGLang
quality evals.  It is separate from Lynn-native per16 NVFP4 used by the custom
R6000 runner.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import time
from typing import Any


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(
            f"missing Python module {name!r}; install the quantization env "
            "with llmcompressor, modelopt, transformers, accelerate, and safetensors"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_tokenizer_sidecars(src: Path, out: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        src_path = src / name
        if src_path.exists() and not (out / name).exists():
            shutil.copy2(src_path, out / name)


def _patch_config(src: Path, out: Path) -> dict[str, Any]:
    src_cfg_path = src / "config.json"
    out_cfg_path = out / "config.json"
    if not src_cfg_path.exists() or not out_cfg_path.exists():
        return {"patched": False, "reason": "missing config"}
    src_cfg = _read_json(src_cfg_path)
    out_cfg = _read_json(out_cfg_path)
    copied: list[str] = []
    for key in (
        "architectures",
        "auto_map",
        "model_type",
        "vision_config",
        "image_token_id",
        "video_token_id",
        "mm_tokens_per_image",
        "mm_tokens_per_video",
    ):
        if key in src_cfg:
            out_cfg[key] = src_cfg[key]
            copied.append(key)
    _write_json(out_cfg_path, out_cfg)
    return {"patched": True, "copied_keys": copied}


def _du_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _load_model(model_path: str, loader: str, dtype: Any) -> tuple[Any, str]:
    import torch
    from transformers import AutoModelForCausalLM

    loaders: list[tuple[str, Any]] = []
    if loader in {"auto", "image_text"}:
        try:
            from transformers import AutoModelForImageTextToText

            loaders.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
        except Exception:
            if loader == "image_text":
                raise
    if loader in {"auto", "causal"}:
        loaders.append(("AutoModelForCausalLM", AutoModelForCausalLM))

    errors: list[str] = []
    for name, cls in loaders:
        kwargs = {
            "device_map": "auto",
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }
        try:
            return cls.from_pretrained(model_path, dtype=dtype, **kwargs), name
        except TypeError:
            try:
                return cls.from_pretrained(model_path, torch_dtype=dtype, **kwargs), name
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raise SystemExit("failed to load model:\n" + "\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-model", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--loader", default="auto", choices=["auto", "image_text", "causal"])
    parser.add_argument("--ignore-lm-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _require_module("torch")
    _require_module("transformers")
    _require_module("llmcompressor")

    import torch
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.transformers import oneshot
    from transformers import AutoTokenizer

    src = Path(args.src_model).resolve()
    out = Path(args.out_model).resolve()
    report_path = Path(args.report).resolve()
    if not (src / "config.json").exists():
        raise SystemExit(f"source model is incomplete: {src}")
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {out}")
        if out == src:
            raise SystemExit("refusing to overwrite source model")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=False)
    model, loader_name = _load_model(str(src), args.loader, torch.bfloat16)
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=["lm_head"] if args.ignore_lm_head else None,
    )
    oneshot(
        model=model,
        recipe=recipe,
        output_dir=str(out),
        save_compressed=True,
    )
    tokenizer.save_pretrained(str(out))
    _copy_tokenizer_sidecars(src, out)
    patch_report = _patch_config(src, out) if args.patch_config else {"patched": False, "reason": "disabled"}

    elapsed = time.time() - started
    result = {
        "schema_version": "lynn-v4-35b-v8-rtn-quantize-v1",
        "src_model": str(src),
        "out_model": str(out),
        "loader": loader_name,
        "ignore_lm_head": args.ignore_lm_head,
        "patch_config": patch_report,
        "elapsed_seconds": elapsed,
        "source_bytes": _du_bytes(src),
        "output_bytes": _du_bytes(out),
        "decision": (
            "GREEN: compressed-tensors NVFP4 v8-RTN artifact written."
            if (out / "model.safetensors.index.json").exists() or any(out.glob("*.safetensors"))
            else "RED: quantization finished but no safetensors output found."
        ),
    }
    _write_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["decision"].startswith("GREEN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
