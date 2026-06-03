#!/usr/bin/env python3
"""Stage 6 Phase 2-M: selected-layer smoke with block linear-attn + P2E MoE."""
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
from scripts.spark_stage6_p1_dense_projection_poc import _bench_cuda, _diff_stats, _nbytes  # noqa: E402
from scripts.spark_stage6_p2_grouped_moe_prefill_census import _attach_packed_moe  # noqa: E402
from scripts.spark_stage6_p2h_selected_layer_prefill_smoke import (  # noqa: E402
    DEFAULT_MODEL,
    GIB,
    _cuda_mem_gib,
    _model_cfg,
    _parse_layers,
    _parse_seqs,
    _peak_once,
    _restore_env,
    _set_mode,
    _tensor_bytes,
)


def _set_linear_block(enabled: bool) -> dict[str, str | None]:
    updates = {"LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA": "1" if enabled else "0"}
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    return old


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


def _with_modes(moe_mode: str | None, layer_ids: list[int], linear_block: bool, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    old_moe = _set_mode(moe_mode, layer_ids) if moe_mode is not None else {}
    old_linear = _set_linear_block(linear_block)
    try:
        if moe_mode is None:
            os.environ["LYNN_PACKED_PREFILL_SLOW"] = "0"
        return fn()
    finally:
        _restore_env(old_linear)
        if moe_mode is not None:
            _restore_env(old_moe)


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

    print("=============== STAGE 6 PHASE 2-M SELECTED-LAYER BLOCK-LINEAR SMOKE ===============", flush=True)
    print(f"model        : {model_dir}", flush=True)
    print(f"layers       : {layer_ids}", flush=True)
    print(f"layer_types  : {[layer_types[i] for i in layer_ids]}", flush=True)
    print(f"seq_lens     : {seq_lens}", flush=True)
    print(f"BF16 active  : {active_bf16_bytes / GIB:.3f} GiB", flush=True)
    print(f"packed active: {packed_bytes / GIB:.3f} GiB", flush=True)

    refs: dict[int, torch.Tensor] = {}
    bench_bf16: dict[str, Any] = {}
    for seq_len, (h, pos) in inputs.items():
        fn = lambda h=h, pos=pos: _run_prefill_layers(h, position_ids=pos, layers=selected_layers, text_cfg=text_cfg)
        print(f"[phase] bf16 reference T={seq_len}", flush=True)
        refs[seq_len] = _with_modes(None, layer_ids, False, fn).detach()
        bench_bf16[str(seq_len)] = _bench_cuda(
            lambda fn=fn: _with_modes(None, layer_ids, False, fn),
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
    bench_p2e: dict[str, Any] = {}
    bench_p2m: dict[str, Any] = {}
    peak_p2e: dict[str, Any] = {}
    peak_p2m: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for seq_len, (h, pos) in inputs.items():
        base_fn = lambda h=h, pos=pos: _run_prefill_layers(h, position_ids=pos, layers=selected_layers, text_cfg=text_cfg)

        print(f"[phase] p2e_hybrid linear-reference T={seq_len}", flush=True)
        p2e_fn = lambda base_fn=base_fn: _with_modes("p2e_hybrid", layer_ids, False, base_fn)
        p2e = p2e_fn().detach()
        numeric[f"p2e_T{seq_len}_vs_bf16"] = _diff_stats(p2e, refs[seq_len])
        peak_p2e[str(seq_len)] = _peak_once(p2e_fn)
        bench_p2e[str(seq_len)] = _bench_cuda(p2e_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)

        print(f"[phase] p2m_hybrid block-linear T={seq_len}", flush=True)
        p2m_fn = lambda base_fn=base_fn: _with_modes("p2e_hybrid", layer_ids, True, base_fn)
        p2m = p2m_fn().detach()
        numeric[f"p2m_T{seq_len}_vs_bf16"] = _diff_stats(p2m, refs[seq_len])
        numeric[f"p2m_T{seq_len}_vs_p2e"] = _diff_stats(p2m, p2e)
        peak_p2m[str(seq_len)] = _peak_once(p2m_fn)
        bench_p2m[str(seq_len)] = _bench_cuda(p2m_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)

        bf = bench_bf16[str(seq_len)]["median_us"]
        p2e_us = bench_p2e[str(seq_len)]["median_us"]
        p2m_us = bench_p2m[str(seq_len)]["median_us"]
        row = {
            "seq_len": seq_len,
            "bf16_prefill_us": bf,
            "p2e_hybrid_us": p2e_us,
            "p2m_block_linear_us": p2m_us,
            "p2m_vs_bf16": bf / p2m_us if p2m_us else None,
            "p2m_vs_p2e": p2e_us / p2m_us if p2m_us else None,
        }
        rows.append(row)
        print(
            f"[T={seq_len}] layers={len(layer_ids)} bf16={bf:.2f}us "
            f"p2e={p2e_us:.2f}us p2m={p2m_us:.2f}us "
            f"p2m_vs_bf16={row['p2m_vs_bf16']:.3f}x p2m_vs_p2e={row['p2m_vs_p2e']:.3f}x",
            flush=True,
        )

    numeric_pass = all(v["cosine"] > 0.999 and v["argmax_match"] for v in numeric.values())
    no_shadow_pass = all(
        "mlp.experts.gate_up_proj" not in w and "mlp.experts.down_proj" not in w
        for _layer_idx, _layer_type, w, _cfg in selected_layers
    )
    speed_vs_p2e_pass = all((r["p2m_vs_p2e"] or 0.0) >= 1.0 for r in rows)
    speed_vs_bf16_pass = all((r["p2m_vs_bf16"] or 0.0) >= 1.0 for r in rows)
    result = {
        "schema": "lynn-stage6-p2m-selected-layer-block-linear-smoke-v1",
        "model": str(model_dir),
        "layers": layer_ids,
        "layer_types": [layer_types[i] for i in layer_ids],
        "seed": args.seed,
        "seq_lens": seq_lens,
        "env": {
            "LYNN_PACKED_PREFILL_SLOW_MODE": "p2e_hybrid",
            "LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA": "1",
            "LYNN_PACKED_PREFILL_P2E_LAYERS": ",".join(str(x) for x in layer_ids),
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
            "p2e_hybrid_linear_reference": bench_p2e,
            "p2m_hybrid_block_linear": bench_p2m,
        },
        "memory": {
            "p2e_peak": peak_p2e,
            "p2m_peak": peak_p2m,
        },
        "passes": {
            "numeric": bool(numeric_pass),
            "no_bf16_active_shadow": bool(no_shadow_pass),
            "speed_vs_p2e": bool(speed_vs_p2e_pass),
            "speed_vs_bf16": bool(speed_vs_bf16_pass),
            "all": bool(numeric_pass and no_shadow_pass and speed_vs_p2e_pass and speed_vs_bf16_pass),
        },
        "notes": [
            "P2-M combines the P2-E packed MoE opt-in path with the P2-L block linear-attn flag.",
            "This is selected-layer full transformer prefill on synthetic hidden states, not full tokenized end-to-end prefill.",
            "Default path remains unchanged when the opt-in flags are unset.",
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
