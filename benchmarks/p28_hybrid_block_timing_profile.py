#!/usr/bin/env python3
"""P28: layerwise timing for graph-linear + eager-full hybrid decode.

P26 showed that the promoted Qwen3.6 W4A16 profile is GPU-bound:
linear-attention graph blocks plus eager full-attention layers dominate each
token. P28 keeps the same execution shape, but records timing per 3-layer
linear block and per following full-attention layer so the next kernel island is
chosen from measured hot spots.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

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


def _summ(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
    }


def _env_snapshot() -> dict[str, str | None]:
    names = [
        "LYNN_MOE_IMPL",
        "LYNN_LINEAR_BLOCK_GRAPH",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE",
        "LYNN_LINEAR_BLOCK_GRAPH_PREWARM",
        "LYNN_LINEAR_STATE_UPDATE",
        "LYNN_NATIVE_FP4_LM_HEAD",
        "LYNN_NATIVE_DOWN_BACKEND",
        "LYNN_NATIVE_GATEUP_BACKEND",
        "LYNN_LINEAR_ATTN_RECURRENT_BACKEND",
        "LYNN_LINEAR_ATTN_RECURRENT_INPLACE",
        "LYNN_LINEAR_ATTN_GQA_RECURRENT",
        "LYNN_LINEAR_ATTN_CONV_BACKEND",
        "LYNN_FULL_ATTN_QKV_FUSED",
        "LYNN_QK_NORM_ROPE_BACKEND",
        "LYNN_RMSNORM_GATED_BACKEND",
        "LYNN_SHARED_EXPERT_GATE_BACKEND",
        "LYNN_DECODE_FAST_DISPATCH",
        "LYNN_MOE_FAST_FIXED",
        "LYNN_NATIVE_ACTIVE_MOE",
    ]
    return {name: os.environ.get(name) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument("--skip-steps", type=int, default=10)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    runner = LynnIncrementalRunner(
        args.model,
        device=args.device,
        dtype=dtype,
        max_seq_len=args.max_seq_len,
        verbose=True,
    )
    ids = _encode_prompt(
        runner.tokenizer,
        args.prompt,
        args.device,
        use_chat_template=args.use_chat_template,
    )
    state = LynnInferenceState(batch=1, max_seq_len=args.max_seq_len, device=args.device, dtype=dtype)
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

    token_buf = torch.empty((1, 1), device=args.device, dtype=torch.long)
    pos_tensor = torch.empty((1, 1), device=args.device, dtype=torch.long)
    token_buf.fill_(next_id)
    pos_tensor.fill_(state.seq_len)
    h_seed = F.embedding(token_buf, runner.outside["model.language_model.embed_tokens.weight"])
    blocks, capture_seconds, graph_created = runner._get_reusable_linear_block_graphs(
        state,
        h_seed,
        pos_tensor,
    )
    graph_state = runner._linear_block_graph_slot["state"]
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    step_rows: list[dict[str, Any]] = []
    new_ids = [next_id]
    for step in range(1, args.max_new):
        if next_id in runner.stop_token_ids:
            break
        token_buf.fill_(next_id)
        pos_tensor.fill_(state.seq_len)
        h = F.embedding(token_buf, runner.outside["model.language_model.embed_tokens.weight"])
        event_rows: list[tuple[str, int, torch.cuda.Event, torch.cuda.Event]] = []

        def timed(kind: str, layer: int, fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            event_rows.append((kind, layer, start, end))
            return out

        for bi, block in enumerate(blocks):
            def replay_block(block=block, h=h):
                block["input"].copy_(h)
                block["graph"].replay()
                return block["output"]

            h = timed("linear_block", int(block["start_layer"]), replay_block)
            full_layer = bi * 4 + 3
            h = timed(
                "full_layer",
                full_layer,
                lambda h=h, full_layer=full_layer: runner._decode_layer_fast(
                    h,
                    pos_tensor,
                    state,
                    full_layer,
                ),
            )

        state.seq_len += 1
        graph_state.seq_len = int(state.seq_len)
        logits = timed(
            "norm_lm_head",
            -1,
            lambda h=h: runner._lm_head_logits(
                _rms_norm(h, runner.outside["model.language_model.norm.weight"])
            ),
        )
        next_id_tensor = timed("argmax", -1, lambda logits=logits: logits[0].argmax())
        torch.cuda.synchronize()

        timings: list[dict[str, Any]] = []
        for kind, layer, start, end in event_rows:
            timings.append({"kind": kind, "layer": layer, "ms": float(start.elapsed_time(end))})
        next_id = int(next_id_tensor.item())
        new_ids.append(next_id)
        step_rows.append({"step": step, "token_id": next_id, "timings": timings})

    analyzed = step_rows[int(args.skip_steps) :]
    by_key: dict[str, list[float]] = {}
    for row in analyzed:
        for item in row["timings"]:
            key = f"{item['kind']}:{item['layer']}"
            by_key.setdefault(key, []).append(float(item["ms"]))
    rows = []
    for key, values in by_key.items():
        kind, layer_s = key.split(":", 1)
        rows.append({"kind": kind, "layer": int(layer_s), **_summ(values)})
    rows.sort(key=lambda item: float(item["mean"] or 0.0), reverse=True)

    linear_rows = [row for row in rows if row["kind"] == "linear_block"]
    full_rows = [row for row in rows if row["kind"] == "full_layer"]
    report = {
        "schema_version": "lynn-engine-p28-hybrid-block-timing-profile-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "env": _env_snapshot(),
        "prompt_tokens": int(ids.shape[1]),
        "max_new": args.max_new,
        "skip_steps": args.skip_steps,
        "steps": len(step_rows),
        "analyzed_steps": len(analyzed),
        "linear_graph_capture_seconds": capture_seconds,
        "linear_graph_created": graph_created,
        "linear_blocks": linear_rows,
        "full_layers": full_rows,
        "top_hotspots": rows[:16],
        "completion_preview": runner.tokenizer.decode(new_ids[:64], skip_special_tokens=True),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "top_hotspots": report["top_hotspots"],
        "linear_blocks": linear_rows,
        "full_layers": full_rows,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
