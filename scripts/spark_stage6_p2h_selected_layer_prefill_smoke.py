#!/usr/bin/env python3
"""Stage 6 Phase 2-H: selected-layer full transformer prefill smoke.

P2-G chained several MoE blocks only. P2-H runs the engine's full prefill layer
path over selected real transformer layers: RMSNorm, linear/full attention cache
population, residuals, and MoE FFN. The active routed expert BF16 shadows are
deleted before packed modes run, so this verifies the P2E path inside the
prefill chain rather than as a standalone MoE microbench.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _with_inferred_layer_config  # noqa: E402
from engine.inference_state import LynnInferenceState, infer_layer_types  # noqa: E402
from engine.loader import load_qwen36_layer  # noqa: E402
from scripts.spark_stage6_p1_dense_projection_poc import (  # noqa: E402
    _bench_cuda,
    _diff_stats,
    _nbytes,
)
from scripts.spark_stage6_p2_grouped_moe_prefill_census import _attach_packed_moe  # noqa: E402


DEFAULT_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"
GIB = 1024**3


def _parse_layers(text: str) -> list[int]:
    out: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" in raw:
            lo, hi = raw.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(raw))
    return sorted(dict.fromkeys(out))


def _parse_seqs(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _model_cfg(model_dir: Path) -> dict[str, Any]:
    cfg = json.loads((model_dir / "config.json").read_text())
    tc = dict(cfg.get("text_config", cfg))
    rope_p = tc.get("rope_parameters", {})
    num_experts = int(tc.get("num_experts", 0) or 0)
    # Match engine.full_forward's runtime cfg construction for full-attn
    # prefill fields while preserving layer_types/linear-attn fields for state.
    tc.update({
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "num_experts": num_experts,
        "num_experts_per_tok": int(tc.get("num_experts_per_tok", 0) or 0),
        "is_moe": num_experts > 0,
        "rope_theta": rope_p.get("rope_theta", tc.get("rope_theta", 1e6)),
        "partial_rotary_factor": rope_p.get("partial_rotary_factor", 1.0),
    })
    return tc


def _tensor_bytes(items: list[torch.Tensor | None]) -> int:
    return sum(_nbytes(t) for t in items if isinstance(t, torch.Tensor))


def _cuda_mem_gib() -> float:
    return float(torch.cuda.memory_allocated() / GIB)


def _peak_once(fn: Callable[[], torch.Tensor]) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = _cuda_mem_gib()
    y = fn()
    torch.cuda.synchronize()
    after = _cuda_mem_gib()
    peak = float(torch.cuda.max_memory_allocated() / GIB)
    del y
    torch.cuda.empty_cache()
    return {"before_gib": before, "after_gib": after, "peak_gib": peak}


def _set_mode(mode: str, layers: list[int]) -> dict[str, str | None]:
    updates = {
        "LYNN_PACKED_PREFILL_SLOW": "1",
        "LYNN_PACKED_PREFILL_SLOW_MODE": mode,
        "LYNN_PACKED_PREFILL_P2E_LAYERS": ",".join(str(x) for x in layers),
        "LYNN_PACKED_PREFILL_P2E_BLOCK_T": "32",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_INTER": "8",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_HIDDEN": "128",
        "LYNN_PACKED_PREFILL_P2E_NUM_WARPS": "4",
        "LYNN_PACKED_PREFILL_P2E_DOWN_BLOCK_HIDDEN": "8",
        "LYNN_PACKED_PREFILL_P2E_DOWN_BLOCK_INTER": "512",
        "LYNN_PACKED_PREFILL_P2E_DOWN_NUM_WARPS": "8",
    }
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _new_state(text_cfg: dict[str, Any], seq_len: int, device: str) -> LynnInferenceState:
    return LynnInferenceState.from_config(
        text_cfg,
        batch=1,
        max_seq_len=seq_len,
        device=device,
        dtype=torch.bfloat16,
    )


def _run_prefill_layers(
    h: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    layers: list[tuple[int, str, dict[str, Any], dict[str, Any]]],
    text_cfg: dict[str, Any],
) -> torch.Tensor:
    state = _new_state(text_cfg, h.shape[1], h.device.type)
    out = h
    for layer_idx, layer_type, w, cfg in layers:
        out = _prefill_layer(out, position_ids, layer_type, w, cfg, state, layer_idx)
    state.seq_len = h.shape[1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--seq-lens", default="16,64")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    model_dir = Path(args.model)
    layer_ids = _parse_layers(args.layers)
    seq_lens = _parse_seqs(args.seq_lens)
    torch.manual_seed(args.seed)

    text_cfg = _model_cfg(model_dir)
    layer_types = infer_layer_types(text_cfg)
    num_experts = int(text_cfg.get("num_experts", 256))
    top_k = int(text_cfg.get("num_experts_per_tok", 8))
    selected_layers: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
    active_bf16_bytes = 0
    packed_bytes = 0
    for layer_idx in layer_ids:
        w, inferred_cfg = load_qwen36_layer(
            str(model_dir),
            layer_idx,
            num_experts=num_experts,
            device=device,
            dequant_dtype=torch.bfloat16,
        )
        layer_cfg = _with_inferred_layer_config(text_cfg, inferred_cfg, layer_idx)
        layer_cfg["num_experts"] = int(layer_cfg.get("num_experts", num_experts))
        layer_cfg["num_experts_per_tok"] = top_k
        layer_cfg["is_moe"] = True
        layer_cfg["layer_idx"] = int(layer_idx)
        packed = _attach_packed_moe(model_dir, layer_idx, w, device=device)
        active_bf16_bytes += _tensor_bytes([w.get("mlp.experts.gate_up_proj"), w.get("mlp.experts.down_proj")])
        packed_bytes += _tensor_bytes(list(packed.values()))
        selected_layers.append((layer_idx, layer_types[layer_idx], w, layer_cfg))

    hidden = int(selected_layers[0][2]["input_layernorm.weight"].shape[0])
    inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for seq_len in seq_lens:
        h = (torch.randn((1, seq_len, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        inputs[seq_len] = (h, pos)

    print("=============== STAGE 6 PHASE 2-H SELECTED-LAYER PREFILL SMOKE ===============", flush=True)
    print(f"model        : {model_dir}", flush=True)
    print(f"layers       : {layer_ids}", flush=True)
    print(f"layer_types  : {[layer_types[i] for i in layer_ids]}", flush=True)
    print(f"seq_lens     : {seq_lens}", flush=True)
    print(f"BF16 active  : {active_bf16_bytes / GIB:.3f} GiB", flush=True)
    print(f"packed active: {packed_bytes / GIB:.3f} GiB", flush=True)

    os.environ["LYNN_PACKED_PREFILL_SLOW"] = "0"
    refs: dict[int, torch.Tensor] = {}
    for seq_len, (h, pos) in inputs.items():
        print(f"[phase] bf16 reference T={seq_len}", flush=True)
        refs[seq_len] = _run_prefill_layers(
            h,
            position_ids=pos,
            layers=selected_layers,
            text_cfg=text_cfg,
        ).detach()
    bench_bf16: dict[str, Any] = {}
    for seq_len, (h, pos) in inputs.items():
        print(f"[phase] bf16 bench T={seq_len}", flush=True)
        bench_bf16[str(seq_len)] = _bench_cuda(
            lambda h=h, pos=pos: _run_prefill_layers(
                h,
                position_ids=pos,
                layers=selected_layers,
                text_cfg=text_cfg,
            ),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )

    for _layer_idx, _layer_type, w, _cfg in selected_layers:
        del w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_delete = _cuda_mem_gib()

    numeric: dict[str, Any] = {}
    bench_stream: dict[str, Any] = {}
    bench_p2e: dict[str, Any] = {}
    peak_stream: dict[str, Any] = {}
    peak_p2e: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for seq_len, (h, pos) in inputs.items():
        print(f"[phase] stream_bf16 T={seq_len}", flush=True)
        old = _set_mode("stream_bf16", layer_ids)
        stream_fn = lambda h=h, pos=pos: _run_prefill_layers(
            h,
            position_ids=pos,
            layers=selected_layers,
            text_cfg=text_cfg,
        )
        stream = stream_fn().detach()
        numeric[f"stream_T{seq_len}_vs_bf16"] = _diff_stats(stream, refs[seq_len])
        peak_stream[str(seq_len)] = _peak_once(stream_fn)
        bench_stream[str(seq_len)] = _bench_cuda(stream_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        _restore_env(old)

        print(f"[phase] p2e_hybrid T={seq_len}", flush=True)
        old = _set_mode("p2e_hybrid", layer_ids)
        p2e_fn = lambda h=h, pos=pos: _run_prefill_layers(
            h,
            position_ids=pos,
            layers=selected_layers,
            text_cfg=text_cfg,
        )
        p2e = p2e_fn().detach()
        numeric[f"p2e_T{seq_len}_vs_bf16"] = _diff_stats(p2e, refs[seq_len])
        numeric[f"p2e_T{seq_len}_vs_stream"] = _diff_stats(p2e, stream)
        peak_p2e[str(seq_len)] = _peak_once(p2e_fn)
        bench_p2e[str(seq_len)] = _bench_cuda(p2e_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        _restore_env(old)

        bf = bench_bf16[str(seq_len)]["median_us"]
        st = bench_stream[str(seq_len)]["median_us"]
        p2 = bench_p2e[str(seq_len)]["median_us"]
        rows.append({
            "seq_len": seq_len,
            "bf16_prefill_us": bf,
            "stream_bf16_us": st,
            "p2e_hybrid_us": p2,
            "p2e_vs_bf16": bf / p2 if p2 else None,
            "p2e_vs_stream": st / p2 if p2 else None,
        })
        print(
            f"[T={seq_len}] layers={len(layer_ids)} bf16={bf:.2f}us stream={st:.2f}us "
            f"p2e={p2:.2f}us p2e_vs_bf16={bf / p2 if p2 else float('nan'):.3f}x",
            flush=True,
        )

    numeric_pass = all(v["cosine"] > 0.999 and v["argmax_match"] for v in numeric.values())
    no_shadow_pass = all(
        "mlp.experts.gate_up_proj" not in w and "mlp.experts.down_proj" not in w
        for _layer_idx, _layer_type, w, _cfg in selected_layers
    )
    speed_vs_stream_pass = all((r["p2e_vs_stream"] or 0.0) >= 10.0 for r in rows)
    result = {
        "schema": "lynn-stage6-p2h-selected-layer-prefill-smoke-v1",
        "model": str(model_dir),
        "layers": layer_ids,
        "layer_types": [layer_types[i] for i in layer_ids],
        "seed": args.seed,
        "seq_lens": seq_lens,
        "env": {
            "LYNN_PACKED_PREFILL_SLOW": "1",
            "LYNN_PACKED_PREFILL_SLOW_MODE": "p2e_hybrid",
            "LYNN_PACKED_PREFILL_P2E_LAYERS": ",".join(str(x) for x in layer_ids),
            "LYNN_PACKED_PREFILL_P2E_BLOCK_INTER": "8",
        },
        "shape": {"hidden": hidden, "num_experts": num_experts, "top_k": top_k},
        "bytes": {
            "bf16_active_experts": active_bf16_bytes,
            "packed_active_experts": packed_bytes,
            "mem_after_deleting_bf16_active_gib": mem_after_delete,
        },
        "numeric": numeric,
        "bench": {
            "rows": rows,
            "bf16_prefill": bench_bf16,
            "stream_bf16": bench_stream,
            "p2e_hybrid": bench_p2e,
        },
        "memory": {
            "stream_peak": peak_stream,
            "p2e_peak": peak_p2e,
        },
        "passes": {
            "numeric": bool(numeric_pass),
            "no_bf16_active_shadow": bool(no_shadow_pass),
            "speed_vs_stream": bool(speed_vs_stream_pass),
            "all": bool(numeric_pass and no_shadow_pass and speed_vs_stream_pass),
        },
        "notes": [
            "This is selected-layer full transformer prefill on synthetic hidden states, not full tokenized end-to-end prefill.",
            "Each run creates a fresh LynnInferenceState and populates linear/full attention caches.",
            "The speed gate requires beating stream_bf16 by at least 10x; BF16 parity is reported but not required for P2-H.",
        ],
    }
    print("=============== RESULT JSON ===============", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
