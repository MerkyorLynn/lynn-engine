#!/usr/bin/env python3
"""P174: Qwen3.5-9B full-attention dense-tail CUDA graph probe.

This is intentionally a probe, not a resident-runner feature. It loads a
Qwen3.5-9B Lynn-native NVFP4 model through LynnIncrementalRunner, pre-fills one
prompt, extracts the tensor immediately after a selected full-attention layer's
attention/residual add, and compares:

  post_attention_layernorm -> dense FFN -> residual add

in eager mode versus a CUDA graph replay over a fixed-shape input buffer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT if (ROOT / "engine").is_dir() else ROOT / "lynn-engine"
for candidate in (ENGINE_ROOT, ROOT):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from engine.full_forward import _moe_forward, _rms_norm  # noqa: E402
from engine.incremental_decode import decode_full_attn, decode_linear_attn  # noqa: E402
from engine.incremental_decode import prefill_full_attn, prefill_linear_attn  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0"
DEFAULT_PROMPT = "用一句话解释 CUDA graph 为什么适合固定形状的推理尾部。"


def _jsonable_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-8:],
    }


def _write_result(path: str | Path, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text, end="")


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    diff = af - bf
    denom = torch.linalg.vector_norm(bf).clamp_min(1e-12)
    cos = torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    )
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(diff).item() / denom.item()),
        "cosine": float(cos.item()),
    }


def _bench_cuda(fn: Callable[[], Any], warmup: int, iters: int) -> float:
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
    return float(start.elapsed_time(end) / max(1, iters))


def _layer_types_from_runner(runner: LynnIncrementalRunner) -> list[str]:
    cfg_types = getattr(runner, "cfg", {}).get("layer_types")
    if cfg_types:
        return [str(x) for x in cfg_types]
    if len(LAYER_TYPES) >= int(runner.n_layers):
        return [str(x) for x in LAYER_TYPES[: int(runner.n_layers)]]
    raise RuntimeError(
        "Cannot discover layer types from runner.cfg['layer_types'] or engine.inference_state.LAYER_TYPES"
    )


def _make_state(runner: LynnIncrementalRunner, layer_types: list[str]) -> LynnInferenceState:
    if hasattr(LynnInferenceState, "from_config"):
        return LynnInferenceState.from_config(  # type: ignore[attr-defined]
            runner.cfg,
            batch=1,
            max_seq_len=runner.max_seq_len,
            device=runner.device,
            dtype=runner.dtype,
        )
    return LynnInferenceState(
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )


def _dense_ffn(h: torch.Tensor, w: dict[str, Any]) -> torch.Tensor:
    gate = F.linear(h, w["mlp.gate_proj.weight"])
    up = F.linear(h, w["mlp.up_proj.weight"])
    return F.linear(F.silu(gate) * up, w["mlp.down_proj.weight"])


def _ffn_forward(h_norm: torch.Tensor, w: dict[str, Any], cfg: dict[str, Any], runner: LynnIncrementalRunner) -> torch.Tensor:
    if all(k in w for k in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")):
        return _dense_ffn(h_norm, w)
    if cfg.get("is_moe", cfg.get("num_experts", 0) > 0):
        moe_fn = getattr(runner, "decode_moe_fn", None)
        if moe_fn is not None and h_norm.shape[1] == 1:
            return moe_fn(h_norm, w, cfg)
        return _moe_forward(h_norm, w, cfg)
    raise KeyError(
        "layer has neither dense FFN keys nor MoE config; cannot isolate post-attention tail"
    )


def _tail(h_after_attn: torch.Tensor, w: dict[str, Any], cfg: dict[str, Any], runner: LynnIncrementalRunner) -> torch.Tensor:
    residual = h_after_attn
    h_norm = _rms_norm(h_after_attn, w["post_attention_layernorm.weight"])
    return residual + _ffn_forward(h_norm, w, cfg, runner)


def _prefill_layer_probe(
    h: torch.Tensor,
    position_ids: torch.Tensor,
    layer_type: str,
    w: dict[str, Any],
    cfg: dict[str, Any],
    state: LynnInferenceState,
    layer_idx: int,
    runner: LynnIncrementalRunner,
) -> torch.Tensor:
    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])
    if layer_type == "linear_attention":
        attn_out, last_state, last_conv = prefill_linear_attn(h_norm, w)
        state.update_linear_attn_state(layer_idx, last_state, last_conv)
    elif layer_type == "full_attention":
        attn_out, k, v = prefill_full_attn(h_norm, position_ids, w, cfg)
        state.update_full_attn_kv(layer_idx, k, v, position_start=0)
    else:
        raise ValueError(f"unknown layer_type={layer_type!r}")
    return _tail(residual + attn_out, w, cfg, runner)


def _decode_layer_probe(
    h: torch.Tensor,
    position_id: torch.Tensor,
    layer_type: str,
    w: dict[str, Any],
    cfg: dict[str, Any],
    state: LynnInferenceState,
    layer_idx: int,
    runner: LynnIncrementalRunner,
) -> torch.Tensor:
    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])
    if layer_type == "linear_attention":
        attn_out, new_state, new_conv = decode_linear_attn(
            h_norm,
            w,
            state.recurrent_state[layer_idx],
            state.conv_state[layer_idx],
            recurrent_backend=getattr(runner, "decode_recurrent_backend", None)
            or os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch"),
        )
        if getattr(runner, "decode_linear_state_update", os.environ.get("LYNN_LINEAR_STATE_UPDATE", "assign")) == "inplace":
            state.recurrent_state[layer_idx].copy_(new_state)
            state.conv_state[layer_idx].copy_(new_conv)
        else:
            state.update_linear_attn_state(layer_idx, new_state, new_conv)
    elif layer_type == "full_attention":
        k_cache, v_cache = state.kv_cache[layer_idx]
        attn_out = decode_full_attn(
            h_norm,
            position_id,
            w,
            cfg,
            k_cache,
            v_cache,
            cached_seq_len=int(state.seq_len),
        )
    else:
        raise ValueError(f"unknown layer_type={layer_type!r}")
    return _tail(residual + attn_out, w, cfg, runner)


def _prefill(
    runner: LynnIncrementalRunner,
    layer_types: list[str],
    prompt: str,
    use_chat_template: bool,
) -> tuple[int, torch.Tensor, LynnInferenceState]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=use_chat_template)
    state = _make_state(runner, layer_types)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i, layer_type in enumerate(layer_types):
        h = _prefill_layer_probe(
            h,
            pos,
            layer_type,
            runner.layer_weights[i],
            runner.layer_cfgs[i],
            state,
            i,
            runner,
        )
    state.seq_len = int(ids.shape[1])
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    return int(logits[0].argmax().item()), ids, state


def _snapshot_state(state: LynnInferenceState) -> dict[str, Any]:
    return {
        "seq_len": int(state.seq_len),
        "kv_cache": {i: (k.clone(), v.clone()) for i, (k, v) in state.kv_cache.items()},
        "recurrent_state": {i: s.clone() for i, s in state.recurrent_state.items()},
        "conv_state": {i: c.clone() for i, c in state.conv_state.items()},
    }


def _restore_state(state: LynnInferenceState, snap: dict[str, Any]) -> None:
    state.seq_len = int(snap["seq_len"])
    for i, (k_src, v_src) in snap["kv_cache"].items():
        if i in state.kv_cache:
            k_dst, v_dst = state.kv_cache[i]
            k_dst.copy_(k_src)
            v_dst.copy_(v_src)
        else:
            state.kv_cache[i] = (k_src.clone(), v_src.clone())
    for i, src in snap["recurrent_state"].items():
        if i in state.recurrent_state:
            state.recurrent_state[i].copy_(src)
        else:
            state.recurrent_state[i] = src.clone()
    for i, src in snap["conv_state"].items():
        if i in state.conv_state:
            state.conv_state[i].copy_(src)
        else:
            state.conv_state[i] = src.clone()


def _extract_after_attention(
    runner: LynnIncrementalRunner,
    layer_types: list[str],
    state: LynnInferenceState,
    token_id: int,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    position_id = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
    token = torch.tensor([[int(token_id)]], device=runner.device, dtype=torch.long)
    h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])

    for i in range(layer_idx):
        h = _decode_layer_probe(
            h,
            position_id,
            layer_types[i],
            runner.layer_weights[i],
            runner.layer_cfgs[i],
            state,
            i,
            runner,
        )

    w = runner.layer_weights[layer_idx]
    cfg = runner.layer_cfgs[layer_idx]
    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])
    k_cache, v_cache = state.kv_cache[layer_idx]
    attn_out = decode_full_attn(
        h_norm,
        position_id,
        w,
        cfg,
        k_cache,
        v_cache,
        cached_seq_len=int(state.seq_len),
    )
    extracted = {
        "position": int(state.seq_len),
        "input_shape": list(h.shape),
        "after_attention_shape": list((residual + attn_out).shape),
        "layer_keys_present": {
            "post_attention_layernorm.weight": "post_attention_layernorm.weight" in w,
            "mlp.gate_proj.weight": "mlp.gate_proj.weight" in w,
            "mlp.up_proj.weight": "mlp.up_proj.weight" in w,
            "mlp.down_proj.weight": "mlp.down_proj.weight" in w,
        },
    }
    return residual + attn_out, position_id, extracted


def _run_suffix_logits(
    runner: LynnIncrementalRunner,
    layer_types: list[str],
    state: LynnInferenceState,
    position_id: torch.Tensor,
    start_layer_exclusive: int,
    h: torch.Tensor,
) -> torch.Tensor:
    for i in range(start_layer_exclusive, len(layer_types)):
        h = _decode_layer_probe(
            h,
            position_id,
            layer_types[i],
            runner.layer_weights[i],
            runner.layer_cfgs[i],
            state,
            i,
            runner,
        )
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    return F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])


def _probe_layer(
    runner: LynnIncrementalRunner,
    layer_types: list[str],
    base_state: LynnInferenceState,
    token_id: int,
    layer_idx: int,
    warmup: int,
    iters: int,
    compare_logits: bool,
) -> dict[str, Any]:
    state = _make_state(runner, layer_types)
    _restore_state(state, _snapshot_state(base_state))
    h_after_attn, position_id, extraction = _extract_after_attention(
        runner, layer_types, state, token_id, layer_idx
    )
    snap_after_attention = _snapshot_state(state)

    w = runner.layer_weights[layer_idx]
    cfg = runner.layer_cfgs[layer_idx]
    if not all(k in w for k in ("post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")):
        return {
            "layer": layer_idx,
            "layer_type": layer_types[layer_idx],
            "status": "blocked",
            "capture_isolated": False,
            "reason": "dense tail keys are not all present; extracted tensor is reported but tail capture is not isolated",
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--layer", type=int, default=-1, help="-1 means first discoverable full_attention layer")
    ap.add_argument("--max-layers", type=int, default=1, help="number of full_attention layers to probe from --layer/all list")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--no-logit-compare", action="store_true")
    ap.add_argument("--fail-on-investigate", action="store_true")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    started = time.time()
    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p174-qwen35-9b-full-attn-tail-graph-probe-v1",
        "benchmark": "p174_qwen35_9b_full_attn_tail_graph_probe",
        "model": args.model,
        "prompt": args.prompt,
        "device_arg": args.device,
        "dtype": args.dtype,
        "max_seq_len": args.max_seq_len,
        "env_observed": {
            key: os.environ.get(key)
            for key in (
                "LYNN_MOE_IMPL",
                "LYNN_PACKED_DECODE",
                "LYNN_PACKED_DECODE_FULL_ATTN",
                "LYNN_FULL_ATTN_QKV_FUSED",
                "LYNN_FULL_ATTN_ROPE_CACHE",
                "LYNN_FULL_ATTN_DECODE_BACKEND",
                "LYNN_QK_NORM_ROPE_BACKEND",
            )
            if os.environ.get(key) is not None
        },
        "layers": [],
    }

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
        full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
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
                    "linear_attention": sum(1 for t in layer_types if t == "linear_attention"),
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
            layer_result = _probe_layer(
                runner,
                layer_types,
                state,
                token_id,
                layer_idx,
                args.warmup,
                args.iters,
                compare_logits=not args.no_logit_compare,
            )
            result["layers"].append(layer_result)
            _restore_state(state, base_snap)

        layer_verdicts = [x.get("verdict", x.get("status")) for x in result["layers"]]
        if all(v == "pass" for v in layer_verdicts):
            result["verdict"] = "pass"
            result["summary"] = "Dense full-attention tail graph replay is exact for probed layer(s)."
        elif any(x.get("capture_isolated") for x in result["layers"]):
            result["verdict"] = "investigate"
            result["summary"] = "At least one tail was captured, but exactness or greedy parity needs investigation."
        else:
            result["verdict"] = "blocked"
            result["summary"] = "Post-attention tensors were extracted, but dense tail capture could not be isolated."
    except BaseException as exc:
        result.update(
            {
                "status": "error",
                "verdict": "blocked",
                "summary": "Probe failed before a tail graph comparison could complete.",
                "error": _jsonable_error(exc),
            }
        )

    result["elapsed_s"] = time.time() - started
    _write_result(args.out, result)
    if result["verdict"] == "pass":
        return 0
    return 1 if args.fail_on_investigate else 0


if __name__ == "__main__":
    raise SystemExit(main())
