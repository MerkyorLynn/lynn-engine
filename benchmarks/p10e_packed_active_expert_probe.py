#!/usr/bin/env python3
"""P10-E: packed NVFP4 active-expert MoE probe.

Compare the current BF16/Triton active-expert path with a packed NVFP4
gate_up/down path for the same router top-k experts. This is intentionally an
isolation probe: it excludes the shared expert so we can decide whether packed
active experts are a viable speed path before touching production MoE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.nvfp4_runtime import _read_weight_map, _load_tensor_by_key  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.nvfp4_linear import nvfp4_matvec_packed  # noqa: E402


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _prefill_to_layer_input(runner: LynnIncrementalRunner, layer: int, prompt: str):
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(layer):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    # Use the last token as a decode-shaped layer input.
    return h[:, -1:, :].contiguous(), state


def _manifest_record(model_dir: Path, key: str):
    manifest = json.loads((model_dir / "lynn_quant_manifest.json").read_text())
    return manifest["quantized_tensors"][key]


def _load_grouped(model_dir: Path, base_key: str, device: str):
    weight_map = _read_weight_map(model_dir)
    rec = _manifest_record(model_dir, base_key)
    packed = _load_tensor_by_key(model_dir, weight_map, rec["packed_key"], device=device).contiguous()
    scale = _load_tensor_by_key(model_dir, weight_map, rec["scale_key"], device=device).float().contiguous()
    global_scale = _load_tensor_by_key(model_dir, weight_map, rec["global_scale_key"], device=device).float().contiguous()
    original_shape = rec["original_shape"]
    if len(original_shape) == 3 and packed.ndim == 2:
        experts, out_features, in_features = map(int, original_shape)
        packed = packed.reshape(experts, out_features, in_features // 2).contiguous()
        scale = scale.reshape(experts, out_features, in_features // 16).contiguous()
    return packed, scale, global_scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.view(-1, h_moe.shape[-1])
    hidden = h_flat[0]
    k_top = int(cfg["num_experts_per_tok"])

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, k_top, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    expert_ids = expert_indices[0].to(torch.long)

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )
    down_packed, down_scale, down_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.down_proj",
        runner.device,
    )

    def bf16_active_experts() -> torch.Tensor:
        gate_up = w["mlp.experts.gate_up_proj"][expert_ids]
        gate_w, up_w = gate_up.chunk(2, dim=1)
        down_w = w["mlp.experts.down_proj"][expert_ids]
        hidden_f = hidden.float()
        gate_out = torch.einsum("d,kid->ki", hidden_f, gate_w.float())
        up_out = torch.einsum("d,kid->ki", hidden_f, up_w.float())
        inter = F.silu(gate_out) * up_out
        out = torch.einsum("ki,kdi->kd", inter, down_w.float())
        return (out * routing_weights[:, None]).sum(dim=0).to(torch.bfloat16)

    def packed_active_experts() -> torch.Tensor:
        outs = []
        for slot, expert in enumerate(expert_ids.tolist()):
            gate_up_out = nvfp4_matvec_packed(
                hidden,
                gate_up_packed[expert],
                gate_up_scale[expert],
                gate_up_global,
            )
            gate_out, up_out = gate_up_out.chunk(2, dim=0)
            inter = (F.silu(gate_out) * up_out).to(torch.bfloat16)
            down_out = nvfp4_matvec_packed(
                inter,
                down_packed[expert],
                down_scale[expert],
                down_global,
            )
            outs.append(down_out * routing_weights[slot])
        return torch.stack(outs, dim=0).sum(dim=0).to(torch.bfloat16)

    ref = bf16_active_experts()
    out = packed_active_experts()
    diff = (out.float() - ref.float()).abs()
    timing = {
        "bf16_active_experts_ms": _bench(bf16_active_experts, args.warmup, args.iters),
        "packed_nvfp4_active_experts_ms": _bench(packed_active_experts, args.warmup, args.iters),
    }
    timing["packed_vs_bf16_ratio"] = timing["bf16_active_experts_ms"] / timing["packed_nvfp4_active_experts_ms"]
    result = {
        "schema_version": "lynn-engine-p10e-packed-active-expert-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "routing_weights": [float(x) for x in routing_weights.tolist()],
        "diff": {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(out.float() - ref.float()).item() / torch.linalg.vector_norm(ref.float()).item()),
            "cosine": float(F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()),
        },
        "timing_ms": timing,
        "pass": bool(F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item() > 0.98),
        "notes": [
            "Active expert only; shared expert and router timing excluded.",
            "Packed path loops over top-k experts and uses scalar bridge matvec kernels.",
        ],
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
