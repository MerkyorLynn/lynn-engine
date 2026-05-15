#!/usr/bin/env python3
"""P9-P: latency breakdown for after-prefill 4-layer graph slots.

P9-O establishes a strict-correct after-prefill 4-layer graph path at ~73 TPS.
This probe breaks the token into:
  - per 4-layer group graph replay
  - final RMSNorm
  - lm_head

The goal is to locate the next real 100 TPS bottleneck instead of guessing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _decode_layer, _prefill_layer, _rms_norm  # noqa: E402
from engine.nvfp4_runtime import _compact_scale_to_swizzled_fp8  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native  # noqa: E402


_E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _quantize_weight_to_fp4(weight: torch.Tensor, *, chunk_rows: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    table = _E2M1_TABLE.to(device=weight.device)
    v, k = weight.shape
    groups = k // 16
    packed = torch.empty((v, k // 2), device=weight.device, dtype=torch.uint8)
    scale = torch.empty((v, groups), device=weight.device, dtype=torch.float32)
    for start in range(0, v, chunk_rows):
        end = min(start + chunk_rows, v)
        wg = weight[start:end].float().reshape(end - start, groups, 16)
        scale_chunk = (wg.abs().amax(dim=-1) / float(table[-1])).clamp_min(1e-8)
        normalized = wg.abs() / scale_chunk.unsqueeze(-1)
        mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
        sign = (wg < 0).to(torch.uint8) * 8
        codes = (mag.to(torch.uint8) | sign).reshape(end - start, k)
        packed[start:end] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        scale[start:end] = scale_chunk
    return packed, scale.contiguous()


def _snapshot_state(state: LynnInferenceState):
    return {
        "seq_len": state.seq_len,
        "kv": {i: (kv[0].clone(), kv[1].clone()) for i, kv in state.kv_cache.items()},
        "recurrent": {i: t.clone() for i, t in state.recurrent_state.items()},
        "conv": {i: t.clone() for i, t in state.conv_state.items()},
    }


def _restore_state(state: LynnInferenceState, snap) -> None:
    state.seq_len = int(snap["seq_len"])
    for i, (k, v) in snap["kv"].items():
        state.kv_cache[i][0].copy_(k)
        state.kv_cache[i][1].copy_(v)
    for i, t in snap["recurrent"].items():
        state.recurrent_state[i].copy_(t)
    for i, t in snap["conv"].items():
        state.conv_state[i].copy_(t)


def _bench(fn, warmup: int, iters: int) -> float:
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


def _prefill(runner: LynnIncrementalRunner, state: LynnInferenceState, prompt: str):
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = ids.shape[1]
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    return int(logits[0].argmax().item()), int(ids.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--native-fp4-lm-head", action="store_true")
    ap.add_argument("--quant-chunk-rows", type=int, default=4096)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    native_lm_head = None
    if args.native_fp4_lm_head:
        packed_lm_head, lm_head_scale = _quantize_weight_to_fp4(
            runner.outside["lm_head.weight"].contiguous(),
            chunk_rows=args.quant_chunk_rows,
        )
        lm_head_scale_b = _compact_scale_to_swizzled_fp8(
            lm_head_scale,
            outer_dim=runner.outside["lm_head.weight"].shape[0],
            k=runner.outside["lm_head.weight"].shape[1],
        )

        def native_lm_head(x: torch.Tensor) -> torch.Tensor:
            act_packed, scale_a = quantize_fp4_m1_native(x)
            return torch._scaled_mm(
                act_packed.view(torch.float4_e2m1fn_x2),
                packed_lm_head.view(torch.float4_e2m1fn_x2).t(),
                scale_a=scale_a,
                scale_b=lm_head_scale_b,
                out_dtype=torch.float16,
            )

    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    next_id, decode_position = _prefill(runner, state, args.prompt)
    pos_tensor = torch.tensor([[decode_position]], device=runner.device, dtype=torch.long)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    base = _snapshot_state(state)

    slots = []
    for start_layer in range(0, runner.n_layers, args.group_size):
        layers = list(range(start_layer, min(start_layer + args.group_size, runner.n_layers)))
        input_buf = torch.empty_like(h_seed)
        output_buf = torch.empty_like(h_seed)

        def body(layers=layers, input_buf=input_buf, output_buf=output_buf):
            state.seq_len = decode_position
            h = input_buf
            for i in layers:
                h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
            output_buf.copy_(h)

        input_buf.copy_(h_seed)
        body()
        _restore_state(state, base)
        input_buf.copy_(h_seed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        torch.cuda.synchronize()
        _restore_state(state, base)
        slots.append({"start_layer": start_layer, "input": input_buf, "output": output_buf, "graph": graph})

    def run_groups_only():
        _restore_state(state, base)
        h = h_seed
        for slot in slots:
            slot["input"].copy_(h)
            slot["graph"].replay()
            h = slot["output"]
        return h

    # Capture representative h after groups for final-op microbenchmarks.
    h_after = run_groups_only().clone()
    norm_buf = torch.empty_like(h_after)
    logits_buf = torch.empty((1, runner.outside["lm_head.weight"].shape[0]), device=runner.device, dtype=runner.dtype)

    def final_norm_only():
        norm_buf.copy_(_rms_norm(h_after, runner.outside["model.language_model.norm.weight"]))

    final_norm_only()
    h_norm = norm_buf.clone()

    def lm_head_only():
        if native_lm_head is None:
            logits_buf.copy_(F.linear(h_norm[:, -1, :], runner.outside["lm_head.weight"]))
        else:
            logits_buf.copy_(native_lm_head(h_norm[:, -1, :]).to(logits_buf.dtype))

    def final_norm_lm_head():
        h_norm_local = _rms_norm(h_after, runner.outside["model.language_model.norm.weight"])
        if native_lm_head is None:
            logits_buf.copy_(F.linear(h_norm_local[:, -1, :], runner.outside["lm_head.weight"]))
        else:
            logits_buf.copy_(native_lm_head(h_norm_local[:, -1, :]).to(logits_buf.dtype))

    def full_graph_path():
        h = run_groups_only()
        h_norm_local = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        if native_lm_head is None:
            logits_buf.copy_(F.linear(h_norm_local[:, -1, :], runner.outside["lm_head.weight"]))
        else:
            logits_buf.copy_(native_lm_head(h_norm_local[:, -1, :]).to(logits_buf.dtype))

    full_input = torch.empty_like(h_seed)
    full_logits = torch.empty_like(logits_buf)

    def full_decode_final_body():
        state.seq_len = decode_position
        h = full_input
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        h_norm_local = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        if native_lm_head is None:
            full_logits.copy_(F.linear(h_norm_local[:, -1, :], runner.outside["lm_head.weight"]))
        else:
            full_logits.copy_(native_lm_head(h_norm_local[:, -1, :]).to(full_logits.dtype))

    _restore_state(state, base)
    full_input.copy_(h_seed)
    full_decode_final_body()
    torch.cuda.synchronize()
    _restore_state(state, base)
    full_input.copy_(h_seed)
    full_decode_final_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(full_decode_final_graph):
        full_decode_final_body()
    torch.cuda.synchronize()
    _restore_state(state, base)

    def full_decode_final_graph_path():
        _restore_state(state, base)
        full_input.copy_(h_seed)
        full_decode_final_graph.replay()
        return full_logits

    def full_decode_final_graph_replay_only():
        full_decode_final_graph.replay()
        return full_logits

    group_rows = []
    for slot in slots:
        def one_slot(slot=slot):
            slot["input"].copy_(h_seed)
            slot["graph"].replay()
        ms = _bench(one_slot, args.warmup, args.iters)
        group_rows.append({"start_layer": slot["start_layer"], "ms": ms, "tps_equiv": 1000.0 / ms})

    groups_ms = _bench(run_groups_only, args.warmup, args.iters)
    norm_ms = _bench(final_norm_only, args.warmup, args.iters)
    lm_head_ms = _bench(lm_head_only, args.warmup, args.iters)
    final_ms = _bench(final_norm_lm_head, args.warmup, args.iters)
    full_ms = _bench(full_graph_path, args.warmup, args.iters)
    full_decode_final_graph_ms = _bench(full_decode_final_graph_path, args.warmup, args.iters)
    # This omits benchmark-only snapshot restoration. It is the closest proxy
    # for a resident serving loop where the CUDA graph mutates the live cache
    # state from token to token instead of restoring the same synthetic state.
    full_decode_final_graph_replay_only_ms = _bench(
        full_decode_final_graph_replay_only,
        args.warmup,
        args.iters,
    )

    result = {
        "schema_version": "lynn-engine-p9p-hybrid-group-latency-breakdown-v1",
        "model": args.model,
        "decode_position": decode_position,
        "group_size": args.group_size,
        "native_fp4_lm_head": args.native_fp4_lm_head,
        "group_count": len(slots),
        "groups_ms": groups_ms,
        "groups_tps_equiv": 1000.0 / groups_ms,
        "final_norm_ms": norm_ms,
        "lm_head_ms": lm_head_ms,
        "final_norm_lm_head_ms": final_ms,
        "full_graph_path_ms": full_ms,
        "full_graph_path_tps": 1000.0 / full_ms,
        "full_decode_final_graph_ms": full_decode_final_graph_ms,
        "full_decode_final_graph_tps": 1000.0 / full_decode_final_graph_ms,
        "full_decode_final_graph_replay_only_ms": full_decode_final_graph_replay_only_ms,
        "full_decode_final_graph_replay_only_tps": 1000.0 / full_decode_final_graph_replay_only_ms,
        "sum_groups_plus_final_ms": groups_ms + final_ms,
        "group_rows": group_rows,
        "top_groups": sorted(group_rows, key=lambda r: r["ms"], reverse=True)[:5],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
