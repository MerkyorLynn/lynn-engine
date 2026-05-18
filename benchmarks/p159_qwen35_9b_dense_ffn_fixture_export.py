#!/usr/bin/env python3
"""P159 · Export Qwen3.5-9B dense-FFN fixtures.

The 9B dense line needs the same fast kernel-development target that p133/p134
gave the 35B MoE line: prompt-derived hidden states plus a strict reference
output, without running the whole model for every candidate iteration.

Each fixture stores the post-attention RMSNorm input to a dense FFN and the
reference dense FFN output:

    ffn_out = down_proj(silu(gate_proj(ffn_in)) * up_proj(ffn_in))

By default the large layer weights are not duplicated into the fixture files.
p160 can reload the needed layer weights from the model artifact.  Pass
``--export-weights`` if a standalone per-layer weight shard is desired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.incremental_decode import prefill_full_attn, prefill_linear_attn  # noqa: E402
from engine.inference_state import LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


DEFAULT_PROMPTS = [
    "用一句话解释 NVFP4 量化为什么能提升推理吞吐。",
    "Return a compact JSON object with keys model, quant, and target for Qwen3.5-9B.",
]

DEFAULT_LAYERS = "0,8,16,-1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_layers(spec: str, n_layers: int) -> list[int]:
    layers: list[int] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        idx = int(raw)
        if idx < 0:
            idx = n_layers + idx
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"layer index {raw!r} resolved to {idx}, outside 0..{n_layers - 1}")
        if idx not in layers:
            layers.append(idx)
    if not layers:
        raise ValueError("no layers selected")
    return layers


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().cpu()


def _dense_prefill_layer_capture(
    *,
    h: torch.Tensor,
    position_ids: torch.Tensor,
    layer_type: str,
    w: dict[str, Any],
    cfg: dict[str, Any],
    state: LynnInferenceState,
    layer_idx: int,
    export_intermediates: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one prefill layer and capture the last-token dense FFN contract."""
    if cfg.get("is_moe", int(cfg.get("num_experts", 0) or 0) > 0):
        raise RuntimeError(f"P159 expected dense layer, but layer {layer_idx} is MoE")

    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])
    if layer_type == "linear_attention":
        attn_out, last_state, last_conv = prefill_linear_attn(h_norm, w)
        state.update_linear_attn_state(layer_idx, last_state, last_conv)
    elif layer_type == "full_attention":
        attn_out, k_cache, v_cache = prefill_full_attn(h_norm, position_ids, w, cfg)
        state.update_full_attn_kv(layer_idx, k_cache, v_cache, position_start=0)
    else:
        raise ValueError(f"unknown layer type {layer_type!r}")

    h_attn = residual + attn_out
    ffn_in_all = _rms_norm(h_attn, w["post_attention_layernorm.weight"])
    ffn_in = ffn_in_all[:, -1, :].contiguous()

    gate = F.linear(ffn_in, w["mlp.gate_proj.weight"])
    up = F.linear(ffn_in, w["mlp.up_proj.weight"])
    inter = F.silu(gate) * up
    ffn_out = F.linear(inter, w["mlp.down_proj.weight"])

    # Continue the real prefill path for downstream layer captures.
    gate_all = F.linear(ffn_in_all, w["mlp.gate_proj.weight"])
    up_all = F.linear(ffn_in_all, w["mlp.up_proj.weight"])
    ffn_all = F.linear(F.silu(gate_all) * up_all, w["mlp.down_proj.weight"])
    h_next = h_attn + ffn_all

    captured = {
        "ffn_in": _cpu(ffn_in),
        "ffn_output": _cpu(ffn_out),
    }
    if export_intermediates:
        captured.update(
            {
                "gate_output": _cpu(gate),
                "up_output": _cpu(up),
                "intermediate": _cpu(inter),
            }
        )
    return h_next, captured


def _weight_tensors(w: dict[str, Any]) -> dict[str, torch.Tensor]:
    keys = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
    missing = [k for k in keys if k not in w or not isinstance(w[k], torch.Tensor)]
    if missing:
        raise KeyError(f"dense FFN weights missing or non-tensor: {missing}")
    return {k: _cpu(w[k]) for k in keys}


def export_fixtures(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    runner = LynnIncrementalRunner(
        args.model,
        device=args.device,
        dtype=dtype,
        max_seq_len=args.max_seq_len,
        verbose=bool(args.verbose),
    )
    if any(cfg.get("is_moe", int(cfg.get("num_experts", 0) or 0) > 0) for cfg in runner.layer_cfgs):
        raise RuntimeError("P159 expected dense Qwen3.5-9B; loaded config contains MoE layers")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = time.time() - t0

    layers = _parse_layers(args.layers, runner.n_layers)
    prompts = args.prompt or DEFAULT_PROMPTS
    manifest: dict[str, Any] = {
        "schema": "lynn-qwen35-9b-dense-ffn-fixture-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "n_layers": runner.n_layers,
        "selected_layers": layers,
        "prompt_count": len(prompts),
        "load_seconds": load_seconds,
        "export_weights": bool(args.export_weights),
        "fixtures": [],
        "weights": {},
    }

    if args.export_weights:
        for layer_idx in layers:
            weights_path = out_dir / f"layer_{layer_idx:02d}_dense_ffn_weights.safetensors"
            save_file(_weight_tensors(runner.layer_weights[layer_idx]), str(weights_path))
            manifest["weights"][str(layer_idx)] = {
                "file": weights_path.name,
                "sha256": _sha256(weights_path),
            }

    for prompt_id, prompt in enumerate(prompts):
        ids = _encode_prompt(runner.tokenizer, prompt, args.device, use_chat_template=args.use_chat_template)
        if ids.shape[1] > args.max_seq_len:
            raise ValueError(f"prompt {prompt_id} has {ids.shape[1]} tokens > max_seq_len={args.max_seq_len}")
        state = LynnInferenceState.from_config(
            runner.cfg,
            batch=1,
            max_seq_len=args.max_seq_len,
            device=args.device,
            dtype=dtype,
        )
        h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
        pos = torch.arange(ids.shape[1], device=args.device, dtype=torch.long).unsqueeze(0)

        for layer_idx in range(runner.n_layers):
            if layer_idx in layers:
                h, tensors = _dense_prefill_layer_capture(
                    h=h,
                    position_ids=pos,
                    layer_type=runner.layer_types[layer_idx],
                    w=runner.layer_weights[layer_idx],
                    cfg=runner.layer_cfgs[layer_idx],
                    state=state,
                    layer_idx=layer_idx,
                    export_intermediates=args.export_intermediates,
                )
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                fixture_name = f"layer_{layer_idx:02d}_prompt_{prompt_id:02d}.safetensors"
                fixture_path = out_dir / fixture_name
                save_file(tensors, str(fixture_path))
                manifest["fixtures"].append(
                    {
                        "file": fixture_name,
                        "sha256": _sha256(fixture_path),
                        "layer_id": layer_idx,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "prompt_tokens": int(ids.shape[1]),
                        "layer_type": runner.layer_types[layer_idx],
                        "shapes": {k: list(v.shape) for k, v in tensors.items()},
                        "dtypes": {k: str(v.dtype) for k, v in tensors.items()},
                    }
                )
            else:
                h = _prefill_layer(
                    h,
                    pos,
                    runner.layer_types[layer_idx],
                    runner.layer_weights[layer_idx],
                    runner.layer_cfgs[layer_idx],
                    state,
                    layer_idx,
                )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
        state.seq_len = int(ids.shape[1])

    manifest["total_fixtures"] = len(manifest["fixtures"])
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Qwen3.5-9B dense FFN fixtures.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--export-intermediates", action="store_true")
    ap.add_argument("--export-weights", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    manifest = export_fixtures(args)
    print(json.dumps({
        "out": args.out,
        "total_fixtures": manifest["total_fixtures"],
        "selected_layers": manifest["selected_layers"],
        "export_weights": manifest["export_weights"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
