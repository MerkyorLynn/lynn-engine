#!/usr/bin/env python3
"""P133 · Export active MoE fixtures for native kernel contract testing.

Purpose: stream the official Qwen3.6-35B W4A16 model layer-by-layer, then
capture real prompt-derived MoE activations for specific layers/prompts so that
downstream kernel development (Stream A native grouped GEMM) can validate
correctness without re-loading the full 35B model.

Exported per (layer, prompt) pair:
  - hidden_in:        [1, 2048] BF16 — MoE input hidden state
  - expert_ids:       [top_k=8] int32 — selected expert indices
  - routing_weights:  [top_k=8] float32 — softmax routing weights
  - moe_output:       [1, 2048] BF16 — ground truth MoE output (Triton reference)
  - metadata JSON with layer_id, prompt_id, topk, model_dir, sidecar path, etc.

Sidecar contract:
  When the model uses NVFP4 v8-RTN fused weights, each layer fixture records the
  path to the packed expert weights so that p134 can load the folded-scale sidecar
  for native kernel testing without a full model reload.

Output layout:
  <out_dir>/
    manifest.json
    layer_<L>_prompt_<P>.safetensors

Usage:
  python benchmarks/p133_export_active_moe_fixtures.py \
    --model-dir /root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0 \
    --sidecar-dir /root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-folded-scale-v0 \
    --layers 0,4,8,16,20,28,32,36,39 \
    --prompts "Hello" "The capital of France is" \
    --out reports/qwen36_35b/p133_fixtures \
    --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.loader import load_qwen36_layer  # noqa: E402
from engine.full_forward import _moe_forward, _rms_norm, load_outside_weights  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402


# ─────────────────────────────────────────────────────────────
# MoE capture hook: intercept hidden + routing before/after MoE
# ─────────────────────────────────────────────────────────────

def _capture_moe_activations(
    h_flat: torch.Tensor,
    layer_weights: dict[str, Any],
    cfg: dict[str, Any],
    export_intermediates: bool = False,
) -> dict[str, torch.Tensor]:
    """Run MoE router + reference forward, capturing all intermediates.

    This re-implements the MoE forward path from moe_optimized.py with explicit
    capture points. Does NOT modify the model.
    """
    K = cfg["num_experts_per_tok"]

    # Router
    router_logits = F.linear(h_flat, layer_weights["mlp.gate.weight"])  # [N, E]
    topk_logits, expert_indices = torch.topk(router_logits, K, dim=-1)  # [N, K]
    routing_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32)

    # Active-expert decode forward (reference path)
    active_experts = torch.unique(expert_indices).tolist()
    moe_out = torch.zeros_like(h_flat)

    for e in active_experts:
        mask = (expert_indices == e)
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]

        # FFN using fused layout if available
        if "mlp.experts.gate_up_proj" in layer_weights and "mlp.experts.down_proj" in layer_weights:
            gate_up = F.linear(x_e, layer_weights["mlp.experts.gate_up_proj"][e])
            gate, up = gate_up.chunk(2, dim=-1)
            ffn_e = F.linear(F.silu(gate) * up, layer_weights["mlp.experts.down_proj"][e])
        else:
            gate = F.linear(x_e, layer_weights[f"mlp.experts.{e}.gate_proj.weight"])
            up = F.linear(x_e, layer_weights[f"mlp.experts.{e}.up_proj.weight"])
            ffn_e = F.linear(F.silu(gate) * up, layer_weights[f"mlp.experts.{e}.down_proj.weight"])

        weight_e = routing_weights[token_idx, slot_idx].unsqueeze(-1).to(h_flat.dtype)
        moe_out.index_add_(0, token_idx, ffn_e * weight_e)

    routed_out = moe_out.detach().clone()

    # Shared expert
    if "mlp.shared_expert.gate_proj.weight" in layer_weights:
        gate_s = F.linear(h_flat, layer_weights["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, layer_weights["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, layer_weights["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in layer_weights:
            shared_gate = torch.sigmoid(
                F.linear(h_flat, layer_weights["mlp.shared_expert_gate.weight"])
            )
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    captured = {
        "hidden_in": h_flat.detach().clone(),
        "expert_ids": expert_indices[0].to(torch.int32).detach().clone(),
        "routing_weights": routing_weights[0].detach().clone(),
        "routed_output": routed_out,
        "moe_output": moe_out.detach().clone(),
    }

    if export_intermediates:
        slot_intermediates = []
        slot_down_outputs = []
        slot_weighted_outputs = []
        for slot_idx in range(K):
            e = int(expert_indices[0, slot_idx].item())
            x_e = h_flat[:1]
            if "mlp.experts.gate_up_proj" in layer_weights and "mlp.experts.down_proj" in layer_weights:
                gate_up = F.linear(x_e, layer_weights["mlp.experts.gate_up_proj"][e])
                gate, up = gate_up.chunk(2, dim=-1)
                inter = F.silu(gate) * up
                down = F.linear(inter, layer_weights["mlp.experts.down_proj"][e])
            else:
                gate = F.linear(x_e, layer_weights[f"mlp.experts.{e}.gate_proj.weight"])
                up = F.linear(x_e, layer_weights[f"mlp.experts.{e}.up_proj.weight"])
                inter = F.silu(gate) * up
                down = F.linear(inter, layer_weights[f"mlp.experts.{e}.down_proj.weight"])
            weighted = down * routing_weights[0, slot_idx].to(h_flat.dtype)
            slot_intermediates.append(inter.squeeze(0).detach())
            slot_down_outputs.append(down.squeeze(0).detach())
            slot_weighted_outputs.append(weighted.squeeze(0).detach())

        captured.update(
            {
                "router_logits": router_logits[0].detach().clone(),
                "topk_logits": topk_logits[0].detach().clone(),
                "slot_intermediate": torch.stack(slot_intermediates, dim=0),
                "slot_down_output": torch.stack(slot_down_outputs, dim=0),
                "slot_weighted_output": torch.stack(slot_weighted_outputs, dim=0),
            }
        )

    return captured


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_shapes(tensors: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {k: list(v.shape) for k, v in tensors.items()}


def _tensor_dtypes(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    return {k: str(v.dtype) for k, v in tensors.items()}


def _prefill_layer_and_moe_input(
    h: torch.Tensor,
    position_ids: torch.Tensor,
    layer_type: str,
    w: dict[str, Any],
    cfg: dict[str, Any],
    state: LynnInferenceState,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one prefill layer while exposing the exact MoE sublayer input.

    This mirrors `engine.full_forward._prefill_layer` but splits the block at
    the attention/MoE boundary:

      h_attn = residual + attention(input_layernorm(h))
      moe_in = post_attention_layernorm(h_attn)
      h_out  = h_attn + moe(moe_in)

    p133 fixtures need `moe_in` for the last token.  Keeping the same operation
    order as `_prefill_layer` makes the fixture a real prompt-derived target
    instead of a synthetic hidden-state proxy.
    """
    from engine.incremental_decode import prefill_full_attn, prefill_linear_attn

    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])
    if layer_type == "linear_attention":
        attn_out, last_state, last_conv = prefill_linear_attn(h_norm, w)
        state.update_linear_attn_state(layer_idx, last_state, last_conv)
    elif layer_type == "full_attention":
        attn_out, K, V = prefill_full_attn(h_norm, position_ids, w, cfg)
        state.update_full_attn_kv(layer_idx, K, V, position_start=0)
    else:
        raise ValueError(f"unknown layer type {layer_type!r}")

    h_attn = residual + attn_out
    moe_input = _rms_norm(h_attn, w["post_attention_layernorm.weight"])
    moe_out = _moe_forward(moe_input, w, cfg)
    return h_attn + moe_out, moe_input


# ─────────────────────────────────────────────────────────────
# Main export pipeline
# ─────────────────────────────────────────────────────────────

def export_fixtures(
    model_dir: str,
    layers: list[int],
    prompts: list[str],
    out_dir: str,
    sidecar_dir: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    export_intermediates: bool = False,
) -> dict[str, Any]:
    """Export MoE fixtures for given layers and prompts."""
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_dir_path = Path(model_dir)
    print(f"[p133] Model: {model_dir}", flush=True)
    print(f"[p133] Layers: {layers}", flush=True)
    print(f"[p133] Prompts: {prompts}", flush=True)
    print(f"[p133] Output: {out_path}", flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load config
    with open(model_dir_path / "config.json") as f:
        full_cfg = json.load(f)
    tc = full_cfg["text_config"]
    rope_p = tc.get("rope_parameters", {})
    cfg = {
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "num_experts": tc["num_experts"],
        "num_experts_per_tok": tc["num_experts_per_tok"],
        "rope_theta": rope_p.get("rope_theta", tc.get("rope_theta", 1e6)),
        "partial_rotary_factor": rope_p.get("partial_rotary_factor", 1.0),
    }
    n_layers = tc["num_hidden_layers"]
    max_target_layer = max(layers)

    # Load outside weights (embeddings, final norm, lm_head)
    print(f"[p133] Loading outside weights...", flush=True)
    outside = load_outside_weights(model_dir, device, dtype)

    layers_needed = max_target_layer + 1
    layer_load_seconds = 0.0
    print(
        f"[p133] Will stream-load {layers_needed} layers per prompt "
        f"(up to layer {max_target_layer}) to keep memory bounded.",
        flush=True,
    )

    # Detect sidecar info
    quant_cfg = full_cfg.get("quantization_config", {})
    sidecar_info = {
        "quant_method": quant_cfg.get("quant_method"),
        "quant_format": quant_cfg.get("format"),
        "model_safetensors": str(model_dir_path / "model.safetensors"),
        "sidecar_dir": sidecar_dir,
        "folded_scale_model_dir": sidecar_dir,
        "uses_folded_scale": bool(sidecar_dir),
    }

    # Export fixtures
    manifest_entries = []
    export_t0 = time.time()

    for prompt_idx, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        T = ids.shape[1]
        print(f"\n[p133] Prompt {prompt_idx}: {prompt!r} (T={T})", flush=True)

        # Run prefill through all layers, capturing MoE inputs at target layers
        state = LynnInferenceState(batch=1, max_seq_len=T + 64, device=device, dtype=dtype)
        h = F.embedding(ids, outside["model.language_model.embed_tokens.weight"])
        pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)

        for i in range(layers_needed):
            t_layer = time.time()
            w, _ = load_qwen36_layer(
                model_dir,
                i,
                num_experts=cfg["num_experts"],
                device=device,
                dequant_dtype=dtype,
            )
            layer_load_seconds += time.time() - t_layer

            h, moe_h = _prefill_layer_and_moe_input(
                h,
                pos,
                LAYER_TYPES[i],
                w,
                cfg,
                state,
                i,
            )
            if i in layers:
                moe_h_last = moe_h[:, -1:, :].view(1, cfg["hidden_size"])  # [1, 2048]

                # Capture MoE activations
                captured = _capture_moe_activations(
                    moe_h_last,
                    w,
                    cfg,
                    export_intermediates=export_intermediates,
                )

                # Save to safetensors
                fixture_name = f"layer_{i:02d}_prompt_{prompt_idx:02d}"
                fixture_path = out_path / f"{fixture_name}.safetensors"
                fixture_tensors = {
                    "hidden_in": captured["hidden_in"].contiguous().cpu(),
                    "expert_ids": captured["expert_ids"].contiguous().cpu(),
                    "routing_weights": captured["routing_weights"].contiguous().cpu(),
                    "routed_output": captured["routed_output"].contiguous().cpu(),
                    "moe_output": captured["moe_output"].contiguous().cpu(),
                }
                if export_intermediates:
                    for key in (
                        "router_logits",
                        "topk_logits",
                        "slot_intermediate",
                        "slot_down_output",
                        "slot_weighted_output",
                    ):
                        fixture_tensors[key] = captured[key].contiguous().cpu()
                save_file(fixture_tensors, str(fixture_path))
                fixture_sha256 = _file_sha256(fixture_path)

                # Record metadata
                entry = {
                    "fixture_file": fixture_name + ".safetensors",
                    "fixture_sha256": fixture_sha256,
                    "layer_id": i,
                    "prompt_id": prompt_idx,
                    "prompt_text": prompt,
                    "prompt_tokens": T,
                    "token_pos": T - 1,
                    "top_k": cfg["num_experts_per_tok"],
                    "hidden_size": cfg["hidden_size"],
                    "num_experts": cfg["num_experts"],
                    "tensor_keys": sorted(fixture_tensors.keys()),
                    "tensor_shapes": _tensor_shapes(fixture_tensors),
                    "tensor_dtypes": _tensor_dtypes(fixture_tensors),
                    "expert_ids": captured["expert_ids"].tolist(),
                    "routing_weights": [round(float(x), 6) for x in captured["routing_weights"].tolist()],
                    "hidden_in_norm": float(captured["hidden_in"].float().norm()),
                    "routed_output_norm": float(captured["routed_output"].float().norm()),
                    "moe_output_norm": float(captured["moe_output"].float().norm()),
                    "export_intermediates": export_intermediates,
                    "sidecar": {
                        **sidecar_info,
                        "layer_prefix": f"model.language_model.layers.{i}.",
                    },
                }
                manifest_entries.append(entry)
                print(
                    f"  [p133] L{i:02d} exported: experts={captured['expert_ids'].tolist()}, "
                    f"h_norm={entry['hidden_in_norm']:.4f}, out_norm={entry['moe_output_norm']:.4f}",
                    flush=True,
                )
            del w, moe_h
            if device.startswith("cuda") and i % 4 == 3:
                torch.cuda.empty_cache()
    export_time = time.time() - export_t0

    # Write manifest
    manifest = {
        "schema": "lynn-moe-fixture-v2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_dir": str(model_dir),
        "device": device,
        "dtype": str(dtype),
        "export_intermediates": export_intermediates,
        "layers": layers,
        "prompts": prompts,
        "top_k": cfg["num_experts_per_tok"],
        "hidden_size": cfg["hidden_size"],
        "num_experts": cfg["num_experts"],
        "num_fixtures": len(manifest_entries),
        "layer_load_seconds": layer_load_seconds,
        "export_seconds": export_time,
        "sidecar": sidecar_info,
        "fixtures": manifest_entries,
    }
    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[p133] Manifest written: {manifest_path}", flush=True)
    print(f"[p133] {len(manifest_entries)} fixtures exported in {export_time:.1f}s", flush=True)

    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export active MoE fixtures for native kernel contract testing."
    )
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Path to Qwen3.6-35B W4A16 model directory (NVFP4 v8-RTN or BF16).",
    )
    ap.add_argument(
        "--layers",
        type=str,
        default="0,4,8,16,20,28,32,36,39",
        help="Comma-separated layer indices to export.",
    )
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "Hello",
            "The capital of France is",
        ],
        help="Prompts to use for fixture generation.",
    )
    ap.add_argument(
        "--out",
        default="reports/qwen36_35b/p133_fixtures",
        help="Output directory for fixtures.",
    )
    ap.add_argument(
        "--sidecar-dir",
        default=None,
        help="Optional folded MoE sidecar path to record in fixture metadata.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument(
        "--export-intermediates",
        action="store_true",
        help="Also export router logits and slot-level routed FFN debug tensors.",
    )

    args = ap.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    manifest = export_fixtures(
        model_dir=args.model_dir,
        layers=layers,
        prompts=args.prompts,
        out_dir=args.out,
        sidecar_dir=args.sidecar_dir,
        device=args.device,
        dtype=dtype,
        export_intermediates=args.export_intermediates,
    )

    print(f"\n{'='*60}")
    print(f"P133 DONE: {manifest['num_fixtures']} fixtures exported")
    print(f"  Layer loads: {manifest['layer_load_seconds']:.1f}s")
    print(f"  Export:     {manifest['export_seconds']:.1f}s")
    print(f"  Output:     {args.out}/")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
