#!/usr/bin/env python3
"""P155 · Qwen3.5-9B NVFP4 dense FFN phase profile.

This probe opens the current Lynn resident decode path at token granularity and
times the dense FFN projections separately:

  gate_proj, up_proj, activation/multiply, down_proj

It also records linear-attention/SSM decode, full-attention decode, lm_head, and
the residual host/runtime gap.  The script intentionally does not set runtime
environment knobs; the R6000 wrapper leaves model defaults untouched unless the
caller explicitly exports overrides before launch.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.incremental_decode import decode_full_attn, decode_linear_attn  # noqa: E402
from engine.inference_state import LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


DEFAULT_PROMPT = (
    "请连续输出一段关于 Qwen3.5-9B、NVFP4、dense FFN、linear attention 和 "
    "CUDA runtime profiling 的中文技术说明。要求持续展开，不要提前结束。"
)

QUALITY_ARTIFACTS = {
    "mmlu_summary": "reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_mmlu_n500.summary.json",
    "mmlu_jsonl": "reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_mmlu_n500.jsonl",
    "gpqa_summary": "reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_gpqa.summary.json",
    "gpqa_jsonl": "reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_gpqa.jsonl",
}

RELEASE_ARTIFACTS = {
    "release_matrix_json": "reports/qwen35_9b/qwen35_9b_release_matrix.json",
    "release_matrix_md": "reports/qwen35_9b/qwen35_9b_release_matrix.md",
    "linear_graph_summary": "reports/qwen35_9b/p151_qwen35_9b_nvfp4_linear_graph_matrix_summary_20260519_0418.json",
}

ENV_KEYS = [
    "LYNN_LINEAR_STATE_UPDATE",
    "LYNN_LINEAR_BLOCK_GRAPH",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE",
    "LYNN_LINEAR_BLOCK_GRAPH_PREWARM",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE",
    "LYNN_LINEAR_ATTN_GQA_RECURRENT",
    "LYNN_LINEAR_ATTN_CONV_BACKEND",
    "LYNN_QK_NORM_ROPE_BACKEND",
    "LYNN_RMSNORM_GATED_BACKEND",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4",
    "LYNN_NATIVE_FP4_LM_HEAD",
    "LYNN_PACKED_DECODE",
    "LYNN_PACKED_DECODE_PREPARE_NATIVE",
    "LYNN_DECODE_FAST_DISPATCH",
]

PHASE_KEYS = [
    "wall_ms",
    "embed_ms",
    "input_rmsnorm_ms",
    "linear_ssm_ms",
    "full_attention_ms",
    "attention_residual_ms",
    "post_rmsnorm_ms",
    "dense_gate_ms",
    "dense_up_ms",
    "dense_act_mul_ms",
    "dense_down_ms",
    "dense_ffn_total_ms",
    "ffn_residual_ms",
    "norm_ms",
    "lm_head_ms",
    "argmax_ms",
    "accounted_cuda_ms",
    "host_gap_ms",
]


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"steps": len(rows)}
    for key in PHASE_KEYS:
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = {
            "mean": _mean(vals),
            "median": _median(vals),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }
    wall = [float(row["wall_ms"]) for row in rows if row.get("wall_ms") is not None]
    if wall:
        out["decode_tps_from_wall"] = 1000.0 / statistics.mean(wall)
    return out


def _env_snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in ENV_KEYS}


def _load_json_if_present(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _release_metrics(repo_root: Path) -> dict[str, Any]:
    matrix = _load_json_if_present(repo_root / RELEASE_ARTIFACTS["release_matrix_json"])
    p151 = _load_json_if_present(repo_root / RELEASE_ARTIFACTS["linear_graph_summary"])
    nvfp4 = None
    if matrix:
        for entry in matrix.get("entries", []):
            if entry.get("variant") == "NVFP4":
                nvfp4 = entry
                break
    return {
        "release_matrix_nvfp4": nvfp4,
        "linear_graph_candidate": p151,
    }


def _make_measure(
    *,
    device: str,
    event_rows: list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
    cpu_fallback_ms: dict[str, float],
) -> Callable[[str, Callable[[], Any]], Any]:
    def measure(name: str, fn: Callable[[], Any]) -> Any:
        if not device.startswith("cuda"):
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

    return measure


def _decode_layer_profiled(
    *,
    runner: LynnIncrementalRunner,
    h: torch.Tensor,
    pos_tensor: torch.Tensor,
    state: LynnInferenceState,
    layer_idx: int,
    measure: Callable[[str, Callable[[], Any]], Any],
) -> torch.Tensor:
    layer_type = runner.layer_types[layer_idx]
    w = runner.layer_weights[layer_idx]
    cfg = runner.layer_cfgs[layer_idx]

    residual = h
    h_norm = measure("input_rmsnorm_ms", lambda: _rms_norm(h, w["input_layernorm.weight"]))
    if layer_type == "linear_attention":
        def run_linear() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return decode_linear_attn(
                h_norm,
                w,
                state.recurrent_state[layer_idx],
                state.conv_state[layer_idx],
                recurrent_backend=runner.decode_recurrent_backend,
            )

        attn_out, new_state, new_conv = measure("linear_ssm_ms", run_linear)
        if runner.decode_linear_state_update == "inplace":
            recurrent_target = state.recurrent_state[layer_idx]
            if recurrent_target.data_ptr() != new_state.data_ptr():
                recurrent_target.copy_(new_state)
            conv_target = state.conv_state[layer_idx]
            if conv_target.data_ptr() != new_conv.data_ptr():
                conv_target.copy_(new_conv)
        else:
            state.update_linear_attn_state(layer_idx, new_state, new_conv)
    elif layer_type == "full_attention":
        def run_full() -> torch.Tensor:
            k, v = state.kv_cache[layer_idx]
            return decode_full_attn(
                h_norm,
                pos_tensor,
                w,
                cfg,
                k,
                v,
                cached_seq_len=state.seq_len,
            )

        attn_out = measure("full_attention_ms", run_full)
    else:
        raise ValueError(f"unknown layer type at layer {layer_idx}: {layer_type!r}")

    h = measure("attention_residual_ms", lambda: residual + attn_out)
    residual = h
    h_norm = measure("post_rmsnorm_ms", lambda: _rms_norm(h, w["post_attention_layernorm.weight"]))

    if cfg.get("is_moe", int(cfg.get("num_experts", 0) or 0) > 0):
        raise RuntimeError("P155 is dense-FFN only; loaded layer config is MoE")

    gate = measure("dense_gate_ms", lambda: F.linear(h_norm, w["mlp.gate_proj.weight"]))
    up = measure("dense_up_ms", lambda: F.linear(h_norm, w["mlp.up_proj.weight"]))
    inter = measure("dense_act_mul_ms", lambda: F.silu(gate) * up)
    ffn_out = measure("dense_down_ms", lambda: F.linear(inter, w["mlp.down_proj.weight"]))
    return measure("ffn_residual_ms", lambda: residual + ffn_out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5-9B NVFP4 dense FFN phase profiler.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--skip-steps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.max_new < 2:
        raise ValueError("--max-new must be at least 2")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    t_load0 = time.time()
    runner = LynnIncrementalRunner(
        args.model,
        device=args.device,
        dtype=dtype,
        max_seq_len=args.max_seq_len,
        verbose=True,
    )
    if any(cfg.get("is_moe", int(cfg.get("num_experts", 0) or 0) > 0) for cfg in runner.layer_cfgs):
        raise RuntimeError("P155 expected dense Qwen3.5-9B layers, but MoE config was loaded")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = time.time() - t_load0

    tok = runner.tokenizer
    ids = _encode_prompt(tok, args.prompt, args.device, use_chat_template=args.use_chat_template)
    state = LynnInferenceState.from_config(
        runner.cfg,
        batch=1,
        max_seq_len=args.max_seq_len,
        device=args.device,
        dtype=dtype,
    )

    t_prefill0 = time.time()
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=args.device, dtype=torch.long).unsqueeze(0)
    for layer_idx in range(runner.n_layers):
        h = _prefill_layer(
            h,
            pos,
            runner.layer_types[layer_idx],
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
    prefill_seconds = time.time() - t_prefill0

    new_token_tensor = torch.empty((1, 1), device=args.device, dtype=torch.long)
    pos_tensor = torch.empty((1, 1), device=args.device, dtype=torch.long)
    new_ids = [next_id]
    step_rows: list[dict[str, Any]] = []
    topk_trace: list[dict[str, Any]] = []

    for step in range(1, args.max_new):
        if next_id in runner.stop_token_ids:
            break

        wall_t0 = time.time()
        new_token_tensor.fill_(next_id)
        pos_tensor.fill_(state.seq_len)
        event_rows: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        cpu_fallback_ms: dict[str, float] = {}
        measure = _make_measure(
            device=args.device,
            event_rows=event_rows,
            cpu_fallback_ms=cpu_fallback_ms,
        )

        h = measure(
            "embed_ms",
            lambda: F.embedding(new_token_tensor, runner.outside["model.language_model.embed_tokens.weight"]),
        )
        for layer_idx in range(runner.n_layers):
            h = _decode_layer_profiled(
                runner=runner,
                h=h,
                pos_tensor=pos_tensor,
                state=state,
                layer_idx=layer_idx,
                measure=measure,
            )
        state.seq_len += 1
        h_final = measure("norm_ms", lambda h=h: _rms_norm(h, runner.outside["model.language_model.norm.weight"]))
        logits = measure("lm_head_ms", lambda h_final=h_final: runner._lm_head_logits(h_final))
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
        dense_total = (
            measured_ms.get("dense_gate_ms", 0.0)
            + measured_ms.get("dense_up_ms", 0.0)
            + measured_ms.get("dense_act_mul_ms", 0.0)
            + measured_ms.get("dense_down_ms", 0.0)
        )
        accounted = (
            measured_ms.get("embed_ms", 0.0)
            + measured_ms.get("input_rmsnorm_ms", 0.0)
            + measured_ms.get("linear_ssm_ms", 0.0)
            + measured_ms.get("full_attention_ms", 0.0)
            + measured_ms.get("attention_residual_ms", 0.0)
            + measured_ms.get("post_rmsnorm_ms", 0.0)
            + dense_total
            + measured_ms.get("ffn_residual_ms", 0.0)
            + measured_ms.get("norm_ms", 0.0)
            + measured_ms.get("lm_head_ms", 0.0)
            + measured_ms.get("argmax_ms", 0.0)
        )
        row = {
            "step": step,
            "token_id": next_id,
            "token_text": tok.decode([next_id]),
            "wall_ms": wall_ms,
            "embed_ms": measured_ms.get("embed_ms", 0.0),
            "input_rmsnorm_ms": measured_ms.get("input_rmsnorm_ms", 0.0),
            "linear_ssm_ms": measured_ms.get("linear_ssm_ms", 0.0),
            "full_attention_ms": measured_ms.get("full_attention_ms", 0.0),
            "attention_residual_ms": measured_ms.get("attention_residual_ms", 0.0),
            "post_rmsnorm_ms": measured_ms.get("post_rmsnorm_ms", 0.0),
            "dense_gate_ms": measured_ms.get("dense_gate_ms", 0.0),
            "dense_up_ms": measured_ms.get("dense_up_ms", 0.0),
            "dense_act_mul_ms": measured_ms.get("dense_act_mul_ms", 0.0),
            "dense_down_ms": measured_ms.get("dense_down_ms", 0.0),
            "dense_ffn_total_ms": dense_total,
            "ffn_residual_ms": measured_ms.get("ffn_residual_ms", 0.0),
            "norm_ms": measured_ms.get("norm_ms", 0.0),
            "lm_head_ms": measured_ms.get("lm_head_ms", 0.0),
            "argmax_ms": measured_ms.get("argmax_ms", 0.0),
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
        "schema": "lynn-qwen35-9b-dense-ffn-phase-profile-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "dtype": args.dtype,
        "env": _env_snapshot(),
        "quality_artifact_paths": QUALITY_ARTIFACTS,
        "release_artifact_paths": RELEASE_ARTIFACTS,
        "preserved_release_metrics": _release_metrics(ROOT),
        "prompt_chars": len(args.prompt),
        "prompt_tokens": int(ids.shape[1]),
        "max_new": args.max_new,
        "skip_steps": args.skip_steps,
        "load_seconds": load_seconds,
        "prefill_seconds": prefill_seconds,
        "first_token_id": int(new_ids[0]),
        "first_token_text": tok.decode([int(new_ids[0])]),
        "summary_all": _stats(step_rows),
        "summary_after_skip": _stats(analyzed_rows),
        "top_steps_by_host_gap": sorted(step_rows, key=lambda row: float(row["host_gap_ms"]), reverse=True)[:10],
        "steps": step_rows,
        "completion_preview": tok.decode(new_ids[:96], skip_special_tokens=True),
    }
    if topk_trace:
        report["topk_trace"] = topk_trace

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "summary_after_skip": report["summary_after_skip"],
        "quality_artifact_paths": QUALITY_ARTIFACTS,
        "preserved_release_metrics": report["preserved_release_metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
