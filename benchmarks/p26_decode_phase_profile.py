#!/usr/bin/env python3
"""P26: phase profile for the R6000 resident decode step.

P25 pins the service number near 100 tok/s, while ablations show the linear
block CUDA graph is the main runtime pillar. P26 opens one decode token and
measures where the remaining per-token wall time goes: graph replay, eager
full-attention layers, lm_head/argmax, and host/runtime overhead.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


DEFAULT_PROMPT = (
    "请连续输出一段关于 MoE 推理优化、NVFP4、CUDA graph 和工具调用服务化的中文技术说明。"
    "要求持续展开，不要提前结束。"
)


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _summarize_steps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "wall_ms",
        "embed_ms",
        "linear_blocks_ms",
        "full_layers_ms",
        "norm_lm_head_ms",
        "argmax_ms",
        "accounted_cuda_ms",
        "host_gap_ms",
    ]
    out: dict[str, Any] = {"steps": len(rows)}
    for key in keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        out[key] = {
            "mean": _mean(vals),
            "median": _median(vals),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }
    wall = [float(r["wall_ms"]) for r in rows if r.get("wall_ms") is not None]
    if wall:
        out["decode_tps_from_wall"] = 1000.0 / statistics.mean(wall)
    return out


def _env_snapshot() -> dict[str, str | None]:
    names = [
        "LYNN_MOE_IMPL",
        "LYNN_MOE_FAST_FIXED",
        "LYNN_LINEAR_BLOCK_GRAPH",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE",
        "LYNN_LINEAR_BLOCK_GRAPH_PREWARM",
        "LYNN_FULL_TOKEN_GRAPH_SLOT",
        "LYNN_NATIVE_FP4_LM_HEAD",
        "LYNN_NATIVE_DOWN_BACKEND",
        "LYNN_NATIVE_GATEUP_BACKEND",
        "LYNN_LINEAR_ATTN_RECURRENT_BACKEND",
        "LYNN_LINEAR_ATTN_RECURRENT_INPLACE",
        "LYNN_LINEAR_ATTN_GQA_RECURRENT",
        "LYNN_LINEAR_ATTN_CONV_BACKEND",
        "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4",
        "LYNN_QK_NORM_ROPE_BACKEND",
        "LYNN_RMSNORM_GATED_BACKEND",
        "LYNN_SHARED_EXPERT_GATE_BACKEND",
        "LYNN_DECODE_FAST_DISPATCH",
        "LYNN_PACKED_DECODE",
    ]
    return {name: os.environ.get(name) for name in names}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--skip-steps", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.max_new < 2:
        raise ValueError("--max-new must be at least 2")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    runner = LynnIncrementalRunner(
        args.model,
        device=args.device,
        dtype=dtype,
        max_seq_len=args.max_seq_len,
        verbose=True,
    )
    tok = runner.tokenizer
    ids = _encode_prompt(tok, args.prompt, args.device, use_chat_template=args.use_chat_template)
    state = LynnInferenceState(
        batch=1,
        max_seq_len=args.max_seq_len,
        device=args.device,
        dtype=dtype,
    )

    prefill_t0 = time.time()
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=args.device, dtype=torch.long).unsqueeze(0)
    for layer_idx in range(runner.n_layers):
        h = _prefill_layer(
            h,
            pos,
            LAYER_TYPES[layer_idx],
            runner.layer_weights[layer_idx],
            runner.layer_cfgs[layer_idx],
            state,
            layer_idx,
        )
    state.seq_len = int(ids.shape[1])
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_final)
    next_id = int(logits[0].argmax().item())
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    prefill_seconds = time.time() - prefill_t0

    new_token_tensor = torch.empty((1, 1), device=args.device, dtype=torch.long)
    pos_tensor = torch.empty((1, 1), device=args.device, dtype=torch.long)
    linear_block_graphs = None
    graph_capture_seconds = None
    graph_reused = None
    if os.environ.get("LYNN_LINEAR_BLOCK_GRAPH", "0") == "1" and args.device.startswith("cuda"):
        new_token_tensor.fill_(next_id)
        h_seed = F.embedding(new_token_tensor, runner.outside["model.language_model.embed_tokens.weight"])
        pos_tensor.fill_(state.seq_len)
        if os.environ.get("LYNN_LINEAR_BLOCK_GRAPH_REUSE", "0") == "1":
            linear_block_graphs, graph_capture_seconds, graph_created = runner._get_reusable_linear_block_graphs(
                state,
                h_seed,
                pos_tensor,
            )
            graph_reused = not graph_created
        else:
            linear_block_graphs, graph_capture_seconds = runner._capture_linear_block_graphs(
                state,
                h_seed,
                pos_tensor,
            )
            graph_reused = False

    step_rows: list[dict[str, Any]] = []
    topk_trace: list[dict[str, Any]] = []
    new_ids = [next_id]
    for step in range(1, args.max_new):
        if next_id in runner.stop_token_ids:
            break
        wall_t0 = time.time()
        new_token_tensor.fill_(next_id)
        pos_tensor.fill_(state.seq_len)
        event_rows: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        cpu_fallback_ms: dict[str, float] = {}

        def measure(name: str, fn: Callable[[], Any]) -> Any:
            if not args.device.startswith("cuda"):
                t0 = time.time()
                out = fn()
                cpu_fallback_ms[name] = cpu_fallback_ms.get(name, 0.0) + (time.time() - t0) * 1000.0
                return out
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            event_rows.append((name, start, end))
            return out

        h = measure(
            "embed_ms",
            lambda: F.embedding(new_token_tensor, runner.outside["model.language_model.embed_tokens.weight"]),
        )
        if linear_block_graphs is None:
            for layer_idx in range(runner.n_layers):
                name = "linear_blocks_ms" if LAYER_TYPES[layer_idx] == "linear_attention" else "full_layers_ms"
                h = measure(
                    name,
                    lambda h=h, layer_idx=layer_idx: runner._decode_layer_fast(h, pos_tensor, state, layer_idx),
                )
        else:
            for bi, block in enumerate(linear_block_graphs):
                def replay_block(block=block, h=h):
                    block["input"].copy_(h)
                    block["graph"].replay()
                    return block["output"]

                h = measure("linear_blocks_ms", replay_block)
                full_layer = bi * 4 + 3
                h = measure(
                    "full_layers_ms",
                    lambda h=h, full_layer=full_layer: runner._decode_layer_fast(
                        h,
                        pos_tensor,
                        state,
                        full_layer,
                    ),
                )

        state.seq_len += 1
        logits = measure(
            "norm_lm_head_ms",
            lambda h=h: runner._lm_head_logits(
                _rms_norm(h, runner.outside["model.language_model.norm.weight"])
            ),
        )
        raw_next_id_tensor = measure("argmax_ms", lambda logits=logits: logits[0].argmax())
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            measured_ms: dict[str, float] = {}
            for name, start, end in event_rows:
                measured_ms[name] = measured_ms.get(name, 0.0) + float(start.elapsed_time(end))
        else:
            measured_ms = cpu_fallback_ms
        next_id = int(raw_next_id_tensor.item())
        wall_ms = (time.time() - wall_t0) * 1000.0
        embed_ms = measured_ms.get("embed_ms", 0.0)
        linear_ms = measured_ms.get("linear_blocks_ms", 0.0)
        full_ms = measured_ms.get("full_layers_ms", 0.0)
        norm_lm_head_ms = measured_ms.get("norm_lm_head_ms", 0.0)
        argmax_ms = measured_ms.get("argmax_ms", 0.0)
        accounted = embed_ms + linear_ms + full_ms + norm_lm_head_ms + argmax_ms
        row = {
            "step": step,
            "token_id": next_id,
            "token_text": tok.decode([next_id]),
            "wall_ms": wall_ms,
            "embed_ms": embed_ms,
            "linear_blocks_ms": linear_ms,
            "full_layers_ms": full_ms,
            "norm_lm_head_ms": norm_lm_head_ms,
            "argmax_ms": argmax_ms,
            "accounted_cuda_ms": accounted,
            "host_gap_ms": wall_ms - accounted,
        }
        step_rows.append(row)
        new_ids.append(next_id)
        if args.top_k > 0:
            values, indices = torch.topk(logits[0].float(), k=args.top_k)
            topk_trace.append({
                "step": step,
                "ids": [int(x) for x in indices.tolist()],
                "values": [float(x) for x in values.tolist()],
            })

    analyzed_rows = step_rows[int(args.skip_steps) :]
    report = {
        "schema_version": "lynn-engine-p26-decode-phase-profile-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "env": _env_snapshot(),
        "prompt_chars": len(args.prompt),
        "prompt_tokens": int(ids.shape[1]),
        "max_new": args.max_new,
        "skip_steps": args.skip_steps,
        "prefill_seconds": prefill_seconds,
        "first_token_id": int(new_ids[0]),
        "first_token_text": tok.decode([int(new_ids[0])]),
        "graph_capture_seconds": graph_capture_seconds,
        "linear_block_graph_reused": graph_reused,
        "summary_all": _summarize_steps(step_rows),
        "summary_after_skip": _summarize_steps(analyzed_rows),
        "top_steps_by_host_gap": sorted(
            step_rows,
            key=lambda row: float(row["host_gap_ms"]),
            reverse=True,
        )[:10],
        "steps": step_rows,
        "completion_preview": tok.decode(new_ids[:64], skip_special_tokens=True),
    }
    if topk_trace:
        report["topk_trace"] = topk_trace
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary_after_skip": report["summary_after_skip"],
        "graph_capture_seconds": graph_capture_seconds,
        "linear_block_graph_reused": graph_reused,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
