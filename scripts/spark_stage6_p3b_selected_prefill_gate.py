#!/usr/bin/env python3
"""Stage 6 Phase 3-B: selected-layer prefill gate for the P3-A MoE contract.

P3-B moves the P3-A grouped active-MoE callable into a selected transformer
prefill stack. It is still not a fused P3 kernel and not a server/RC promotion:
router and shared expert remain on the existing BF16 paths, while active routed
experts must read packed NVFP4 tensors after their BF16 shadows are deleted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _with_inferred_layer_config  # noqa: E402
from engine.inference_state import infer_layer_types  # noqa: E402
from engine.loader import load_qwen36_layer  # noqa: E402
from scripts.spark_stage6_p1_dense_projection_poc import _bench_cuda, _diff_stats  # noqa: E402
from scripts.spark_stage6_p2_grouped_moe_prefill_census import _attach_packed_moe  # noqa: E402
from scripts.spark_stage6_p2h_selected_layer_prefill_smoke import (  # noqa: E402
    DEFAULT_MODEL,
    GIB,
    _cuda_mem_gib,
    _model_cfg,
    _parse_layers,
    _parse_seqs,
    _peak_once,
    _tensor_bytes,
)
from scripts.spark_stage6_p2m_selected_layer_block_linear_smoke import (  # noqa: E402
    _run_prefill_layers,
    _with_modes,
)


def _shadow_absent(layers: list[tuple[int, str, dict[str, Any], dict[str, Any]]]) -> bool:
    return all(
        "mlp.experts.gate_up_proj" not in w and "mlp.experts.down_proj" not in w
        for _layer_idx, _layer_type, w, _cfg in layers
    )


def _run_mode(
    mode: str | None,
    layer_ids: list[int],
    linear_block: bool,
    fn: Callable[[], torch.Tensor],
) -> torch.Tensor:
    return _with_modes(mode, layer_ids, linear_block, fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--seq-lens", "--tokens", dest="seq_lens", default="16,64")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--min-speed-vs-p2n", type=float, default=1.0)
    ap.add_argument(
        "--predecessors-pass",
        action="store_true",
        help="Assert P2-O basic, P2-O rc-mini, P3-A, and suite reports passed.",
    )
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

    mem_after_load = _cuda_mem_gib()
    hidden = int(selected_layers[0][2]["input_layernorm.weight"].shape[0])
    inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for seq_len in seq_lens:
        h = (torch.randn((1, seq_len, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        inputs[seq_len] = (h, pos)

    print("=============== STAGE 6 PHASE 3-B SELECTED-PREFILL GATE ===============", flush=True)
    print(f"model        : {model_dir}", flush=True)
    print(f"layers       : {layer_ids}", flush=True)
    print(f"layer_types  : {[layer_types[i] for i in layer_ids]}", flush=True)
    print(f"seq_lens     : {seq_lens}", flush=True)
    print(f"predecessors : {args.predecessors_pass}", flush=True)
    print(f"BF16 active  : {active_bf16_bytes / GIB:.3f} GiB", flush=True)
    print(f"packed active: {packed_bytes / GIB:.3f} GiB", flush=True)

    refs: dict[int, torch.Tensor] = {}
    bench_bf16: dict[str, Any] = {}
    for seq_len, (h, pos) in inputs.items():
        fn = lambda h=h, pos=pos: _run_prefill_layers(h, position_ids=pos, layers=selected_layers, text_cfg=text_cfg)
        print(f"[phase] bf16 reference T={seq_len}", flush=True)
        refs[seq_len] = _run_mode(None, layer_ids, False, fn).detach()
        bench_bf16[str(seq_len)] = _bench_cuda(
            lambda fn=fn: _run_mode(None, layer_ids, False, fn),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )

    for _layer_idx, _layer_type, w, _cfg in selected_layers:
        del w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_delete = _cuda_mem_gib()
    shadow_absence_checks = {"after_delete": _shadow_absent(selected_layers)}

    numeric: dict[str, Any] = {}
    bench_p2n: dict[str, Any] = {}
    bench_p3b: dict[str, Any] = {}
    peak_p2n: dict[str, Any] = {}
    peak_p3b: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    reload_calls: list[dict[str, Any]] = []
    reload_trap_installed = False
    reload_trap_status = "not installed"
    original_reload = None
    reload_owner = None
    try:
        from engine.resident_runner import LynnIncrementalRunner  # noqa: WPS433

        original_reload = LynnIncrementalRunner.reload_decode_bf16_shadows
        reload_owner = LynnIncrementalRunner

        def forbidden_reload(self: Any, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            reload_calls.append({
                "time": time.time(),
                "source": "LynnIncrementalRunner.reload_decode_bf16_shadows",
            })
            raise RuntimeError("P3-B forbids reload_decode_bf16_shadows() during selected-prefill gate")

        LynnIncrementalRunner.reload_decode_bf16_shadows = forbidden_reload  # type: ignore[method-assign]
        reload_trap_installed = True
        reload_trap_status = "installed"
    except Exception as exc:  # pragma: no cover - exercised on Spark if import is broken
        reload_trap_status = f"install_failed: {exc!r}"
        reload_calls.append({"time": time.time(), "source": "reload_trap_install", "error": repr(exc)})

    for seq_len, (h, pos) in inputs.items():
        base_fn = lambda h=h, pos=pos: _run_prefill_layers(h, position_ids=pos, layers=selected_layers, text_cfg=text_cfg)

        print(f"[phase] p2n_reference T={seq_len}", flush=True)
        p2n_fn = lambda base_fn=base_fn: _run_mode("p2e_hybrid", layer_ids, True, base_fn)
        p2n = p2n_fn().detach()
        numeric[f"p2n_T{seq_len}_vs_bf16"] = _diff_stats(p2n, refs[seq_len])
        peak_p2n[str(seq_len)] = _peak_once(p2n_fn)
        bench_p2n[str(seq_len)] = _bench_cuda(p2n_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        shadow_absence_checks[f"after_p2n_T{seq_len}"] = _shadow_absent(selected_layers)

        print(f"[phase] p3b_p3a_grouped T={seq_len}", flush=True)
        p3b_fn = lambda base_fn=base_fn: _run_mode("p3a_grouped", layer_ids, True, base_fn)
        p3b = p3b_fn().detach()
        numeric[f"p3b_T{seq_len}_vs_bf16"] = _diff_stats(p3b, refs[seq_len])
        numeric[f"p3b_T{seq_len}_vs_p2n"] = _diff_stats(p3b, p2n)
        peak_p3b[str(seq_len)] = _peak_once(p3b_fn)
        bench_p3b[str(seq_len)] = _bench_cuda(p3b_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        shadow_absence_checks[f"after_p3b_T{seq_len}"] = _shadow_absent(selected_layers)

        bf = bench_bf16[str(seq_len)]["median_us"]
        p2n_us = bench_p2n[str(seq_len)]["median_us"]
        p3b_us = bench_p3b[str(seq_len)]["median_us"]
        p3b_vs_bf16 = bf / p3b_us if p3b_us else None
        p3b_vs_p2n = p2n_us / p3b_us if p3b_us else None
        row = {
            "seq_len": seq_len,
            "bf16_prefill_us": bf,
            "p2n_reference_us": p2n_us,
            "p3b_selected_prefill_us": p3b_us,
            "p3b_vs_bf16": p3b_vs_bf16,
            "p3b_vs_p2n": p3b_vs_p2n,
            "p3b_cosine_vs_bf16": numeric[f"p3b_T{seq_len}_vs_bf16"]["cosine"],
            "p3b_argmax_vs_bf16": numeric[f"p3b_T{seq_len}_vs_bf16"]["argmax_match"],
        }
        rows.append(row)
        print(
            f"[T={seq_len}] layers={len(layer_ids)} bf16={bf:.2f}us "
            f"p2n={p2n_us:.2f}us p3b={p3b_us:.2f}us "
            f"p3b_vs_bf16={p3b_vs_bf16:.3f}x p3b_vs_p2n={p3b_vs_p2n:.3f}x "
            f"cos={row['p3b_cosine_vs_bf16']:.9f} argmax={row['p3b_argmax_vs_bf16']}",
            flush=True,
        )
    if reload_trap_installed and reload_owner is not None and original_reload is not None:
        reload_owner.reload_decode_bf16_shadows = original_reload  # type: ignore[method-assign]

    p3b_vs_bf16_stats = [v for k, v in numeric.items() if k.startswith("p3b_") and k.endswith("_vs_bf16")]
    final_stack_cosine = min((float(v["cosine"]) for v in p3b_vs_bf16_stats), default=0.0)
    final_stack_argmax = all(bool(v["argmax_match"]) for v in p3b_vs_bf16_stats)
    numeric_pass = final_stack_cosine >= args.min_cosine and final_stack_argmax
    speed_vs_p2n_pass = all((r["p3b_vs_p2n"] or 0.0) >= args.min_speed_vs_p2n for r in rows)
    active_shadow_absent = all(bool(v) for v in shadow_absence_checks.values())
    reload_not_called = reload_trap_installed and len(reload_calls) == 0

    result = {
        "schema": "lynn-stage6-p3b-selected-prefill-gate-v1",
        "verdict": "PASS"
        if args.predecessors_pass and numeric_pass and active_shadow_absent and reload_not_called and speed_vs_p2n_pass
        else "FAIL",
        "banked_fused_kernel": False,
        "banked_server_path": False,
        "model": str(model_dir),
        "layers": layer_ids,
        "layer_types": [layer_types[i] for i in layer_ids],
        "seed": args.seed,
        "seq_lens": seq_lens,
        "env": {
            "candidate_mode": "p3a_grouped",
            "reference_mode": "p2e_hybrid",
            "LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA": "1",
            "LYNN_PACKED_PREFILL_P2E_LAYERS": ",".join(str(x) for x in layer_ids),
        },
        "shape": {"hidden": hidden, "num_experts": num_experts, "top_k": top_k},
        "bytes": {
            "bf16_active_experts": active_bf16_bytes,
            "packed_active_experts": packed_bytes,
            "mem_after_load_gib": mem_after_load,
            "mem_after_deleting_bf16_active_gib": mem_after_delete,
            "mem_drop_after_deleting_bf16_active_gib": mem_after_load - mem_after_delete,
        },
        "numeric": numeric,
        "bench": {
            "rows": rows,
            "bf16_prefill": bench_bf16,
            "p2n_reference": bench_p2n,
            "p3b_selected_prefill": bench_p3b,
        },
        "memory": {
            "p2n_peak": peak_p2n,
            "p3b_peak": peak_p3b,
        },
        "shadow_absence_checks": shadow_absence_checks,
        "reload_trap": {
            "installed": bool(reload_trap_installed),
            "status": reload_trap_status,
        },
        "passes": {
            "predecessors_pass": bool(args.predecessors_pass),
            "numeric": bool(numeric_pass),
            "final_stack_cosine_min": final_stack_cosine,
            "final_stack_argmax_match": bool(final_stack_argmax),
            "no_active_bf16_shadow": bool(active_shadow_absent),
            "reload_trap_installed": bool(reload_trap_installed),
            "reload_not_called": bool(reload_not_called),
            "speed_vs_p2n_reference": bool(speed_vs_p2n_pass),
            "all": bool(
                args.predecessors_pass
                and numeric_pass
                and active_shadow_absent
                and reload_trap_installed
                and reload_not_called
                and speed_vs_p2n_pass
            ),
        },
        "reload_calls": reload_calls,
        "notes": [
            "P3-B is selected-layer composition only; it does not bank a fused P3 kernel.",
            "Router and shared expert remain on the existing BF16 paths.",
            "Active routed expert BF16 shadows are deleted before P2-N/P3-B candidates run.",
            "P3-C server readiness and RC quality promotion remain separate gates.",
        ],
    }
    print("=============== RESULT JSON ===============", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passes"]["all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
