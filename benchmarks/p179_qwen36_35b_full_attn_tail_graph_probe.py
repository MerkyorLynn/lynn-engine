#!/usr/bin/env python3
"""P179: Qwen3.6-35B full-attention MoE-tail CUDA graph probe.

P174 proved the same post-attention tail boundary on Qwen3.5-9B dense layers.
This probe keeps the boundary but allows Qwen3.6-35B MoE layers:

    post_attention_layernorm -> exact resident FFN/MoE -> residual add

The goal is to test a larger boundary that preserves the current Triton MoE
numerical authority instead of swapping in approximate native MoE kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT if (ROOT / "engine").is_dir() else ROOT / "lynn-engine"
for candidate in (ENGINE_ROOT, ROOT):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmarks.p174_qwen35_9b_full_attn_tail_graph_probe import (  # noqa: E402
    _bench_cuda,
    _cmp,
    _extract_after_attention,
    _jsonable_error,
    _layer_types_from_runner,
    _make_state,
    _prefill,
    _restore_state,
    _run_suffix_logits,
    _snapshot_state,
    _tail,
    _write_result,
)
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0"
DEFAULT_PROMPT = "用一句话解释 MoE 推理里为什么要先保证数值严格再谈速度。"


def _tail_available(w: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str]:
    if "post_attention_layernorm.weight" not in w:
        return False, "missing post_attention_layernorm.weight"
    dense = all(k in w for k in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"))
    moe = bool(cfg.get("is_moe", cfg.get("num_experts", 0) > 0))
    if dense or moe:
        return True, "dense" if dense else "moe"
    return False, "layer has neither dense FFN keys nor MoE config"


def _probe_layer(
    runner: LynnIncrementalRunner,
    layer_types: list[str],
    base_state: Any,
    token_id: int,
    layer_idx: int,
    warmup: int,
    iters: int,
    compare_logits: bool,
) -> dict[str, Any]:
    try:
        state = _make_state(runner, layer_types)
        _restore_state(state, _snapshot_state(base_state))
        h_after_attn, position_id, extraction = _extract_after_attention(
            runner, layer_types, state, token_id, layer_idx
        )
        snap_after_attention = _snapshot_state(state)

        w = runner.layer_weights[layer_idx]
        cfg = runner.layer_cfgs[layer_idx]
        available, tail_kind = _tail_available(w, cfg)
        if not available:
            return {
                "layer": layer_idx,
                "layer_type": layer_types[layer_idx],
                "status": "blocked",
                "capture_isolated": False,
                "reason": tail_kind,
                "extraction": extraction,
            }

        input_buf = torch.empty_like(h_after_attn)
        output_buf = torch.empty_like(h_after_attn)

        def eager_tail_input_buf() -> torch.Tensor:
            return _tail(input_buf, w, cfg, runner)

        def graph_body() -> None:
            output_buf.copy_(eager_tail_input_buf())

        input_buf.copy_(h_after_attn)
        eager_base = _tail(h_after_attn, w, cfg, runner)
        eager_ms = _bench_cuda(lambda: eager_tail_input_buf(), warmup, max(1, iters // 4))

        for _ in range(max(3, warmup)):
            graph_body()
        torch.cuda.synchronize()
        capture_t0 = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_body()
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - capture_t0) * 1000.0

        input_buf.copy_(h_after_attn)
        graph.replay()
        torch.cuda.synchronize()
        graph_base = output_buf.clone()

        alt = (h_after_attn.float() * 1.0009765625).to(h_after_attn.dtype)
        input_buf.copy_(alt)
        eager_alt = _tail(alt, w, cfg, runner)
        graph.replay()
        torch.cuda.synchronize()
        graph_alt = output_buf.clone()

        graph_replay_ms = _bench_cuda(lambda: graph.replay(), warmup, iters)
        graph_with_copy_ms = _bench_cuda(
            lambda: (input_buf.copy_(h_after_attn), graph.replay()),
            warmup,
            iters,
        )

        result: dict[str, Any] = {
            "layer": layer_idx,
            "layer_type": layer_types[layer_idx],
            "tail_kind": tail_kind,
            "status": "ok",
            "capture_isolated": True,
            "extraction": extraction,
            "timing_ms": {
                "eager_tail": eager_ms,
                "cuda_graph_capture_wall": capture_ms,
                "cuda_graph_replay": graph_replay_ms,
                "cuda_graph_replay_with_input_copy": graph_with_copy_ms,
                "speedup_replay_vs_eager": eager_ms / graph_replay_ms if graph_replay_ms > 0 else None,
                "speedup_with_copy_vs_eager": eager_ms / graph_with_copy_ms if graph_with_copy_ms > 0 else None,
            },
            "exactness": {
                "tail_output_base": _cmp(graph_base, eager_base),
                "tail_output_alt_input": _cmp(graph_alt, eager_alt),
            },
        }

        if compare_logits:
            _restore_state(state, snap_after_attention)
            logits_eager = _run_suffix_logits(
                runner, layer_types, state, position_id, layer_idx + 1, eager_base
            )
            eager_id = int(logits_eager[0].argmax().item())
            _restore_state(state, snap_after_attention)
            logits_graph = _run_suffix_logits(
                runner, layer_types, state, position_id, layer_idx + 1, graph_base
            )
            graph_id = int(logits_graph[0].argmax().item())
            result["exactness"]["suffix_logits"] = _cmp(logits_graph, logits_eager)
            result["exactness"]["greedy_local"] = {
                "eager_token_id": eager_id,
                "graph_token_id": graph_id,
                "exact": eager_id == graph_id,
            }

        tail_exact = result["exactness"]["tail_output_base"]["max_abs"] == 0.0
        alt_exact = result["exactness"]["tail_output_alt_input"]["max_abs"] == 0.0
        greedy_exact = result["exactness"].get("greedy_local", {}).get("exact", True)
        result["verdict"] = "pass" if tail_exact and alt_exact and greedy_exact else "investigate"
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "layer": layer_idx,
            "layer_type": layer_types[layer_idx],
            "status": "error",
            "capture_isolated": False,
            "error": _jsonable_error(exc),
            "verdict": "error",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--layer", type=int, default=-1, help="-1 means first discoverable full_attention layer")
    parser.add_argument("--max-layers", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--no-logit-compare", action="store_true")
    parser.add_argument("--fail-on-investigate", action="store_true")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p179-qwen36-35b-full-attn-tail-graph-probe-v1",
        "benchmark": "p179_qwen36_35b_full_attn_tail_graph_probe",
        "model": args.model,
        "prompt": args.prompt,
        "device_arg": args.device,
        "dtype": args.dtype,
        "max_seq_len": args.max_seq_len,
        "env_observed": {
            key: os.environ.get(key)
            for key in (
                "LYNN_MOE_FAST_FIXED",
                "LYNN_NATIVE_ACTIVE_MOE_BACKEND",
                "LYNN_MOE_ACTIVE_SCRATCH",
                "LYNN_FULL_ATTN_ROPE_CACHE",
                "LYNN_QK_NORM_ROPE_BACKEND",
            )
            if os.environ.get(key) is not None
        },
        "layers": [],
    }
    started = time.time()
    try:
        if not torch.cuda.is_available() and args.device.startswith("cuda"):
            raise RuntimeError("CUDA is not available")
        runner = LynnIncrementalRunner(
            args.model,
            device=args.device,
            dtype=dtype,
            max_seq_len=args.max_seq_len,
            verbose=False,
        )
        layer_types = _layer_types_from_runner(runner)
        full_layers = [idx for idx, kind in enumerate(layer_types) if kind == "full_attention"]
        if not full_layers:
            raise RuntimeError("no full_attention layers discoverable from runner/config")
        if args.layer >= 0:
            if args.layer not in full_layers:
                raise ValueError(f"requested layer {args.layer} is not full_attention; full_layers={full_layers}")
            probe_layers = [args.layer]
        else:
            probe_layers = full_layers[: max(1, args.max_layers)]

        token_id, ids, state = _prefill(runner, layer_types, args.prompt, args.use_chat_template)
        torch.cuda.synchronize()
        result.update(
            {
                "status": "ok",
                "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
                "n_layers": int(runner.n_layers),
                "layer_types": {
                    "linear_attention": sum(1 for kind in layer_types if kind == "linear_attention"),
                    "full_attention": len(full_layers),
                },
                "full_attention_layers": full_layers,
                "probed_layers": probe_layers,
                "prompt_tokens": int(ids.shape[1]),
                "prefill_next_token_id": int(token_id),
            }
        )
        base_snap = _snapshot_state(state)
        for layer_idx in probe_layers:
            result["layers"].append(
                _probe_layer(
                    runner,
                    layer_types,
                    state,
                    token_id,
                    layer_idx,
                    args.warmup,
                    args.iters,
                    compare_logits=not args.no_logit_compare,
                )
            )
            _restore_state(state, base_snap)

        verdicts = [layer.get("verdict", layer.get("status")) for layer in result["layers"]]
        if all(verdict == "pass" for verdict in verdicts):
            result["verdict"] = "pass"
            result["summary"] = "Qwen3.6 full-attention MoE tail graph replay is exact for probed layers."
        elif any(layer.get("capture_isolated") for layer in result["layers"]):
            result["verdict"] = "investigate"
            result["summary"] = "At least one MoE tail was captured, but exactness or greedy parity needs investigation."
        else:
            result["verdict"] = "blocked"
            result["summary"] = "No probed full-attention MoE tail could be captured."
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "error", "verdict": "error", "error": _jsonable_error(exc)})
    finally:
        result["elapsed_sec"] = round(time.time() - started, 3)
        _write_result(args.out, result)

    if args.fail_on_investigate and result.get("verdict") != "pass":
        return 1
    return 0 if result.get("status") != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
