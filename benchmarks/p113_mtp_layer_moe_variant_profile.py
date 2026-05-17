#!/usr/bin/env python3
"""P113: profile MTP decoder-layer MoE variants on collected decode states.

P112 showed that a one-token MTP draft spends most of its time inside the
sidecar decoder layer. This probe keeps the same states and weights, then
swaps only the MoE implementation used after full attention:

  - baseline_full_forward: engine.full_forward._layer_forward
  - decode_active_experts: engine.moe_optimized.moe_forward_decode_optimized
  - decode_bmm: engine.moe_optimized.moe_forward_decode_bmm

The goal is to identify whether the MTP sidecar can reuse Lynn's decode-only
MoE path instead of the full-forward training/eval path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _full_attn_forward, _layer_forward, _rms_norm  # noqa: E402
from engine.moe_optimized import moe_forward_decode_bmm, moe_forward_decode_optimized  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from scripts.a100_mtp_fc_calibration_train import _mtp_cfg  # noqa: E402
from scripts.a100_mtp_forward_smoke import _load_sidecar, _mtp_layer_weights  # noqa: E402
from scripts.a100_mtp_iterative_train import _collect_cases, _load_prompts  # noqa: E402


LayerFn = Callable[[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]], torch.Tensor]


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _timed(device: str, fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    _sync(device)
    t0 = time.time()
    out = fn()
    _sync(device)
    return out, time.time() - t0


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p90_ms": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))] * 1000.0,
    }


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).norm()
    denom = b.float().norm().clamp_min(1.0e-12)
    return float((diff / denom).item())


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _layer_forward_decode_moe(
    h: torch.Tensor,
    position_ids: torch.Tensor,
    w: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    moe_fn: Callable[[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]], torch.Tensor],
) -> torch.Tensor:
    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])
    attn_out = _full_attn_forward(h_norm, position_ids, w, cfg)
    h = residual + attn_out

    residual = h
    h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])
    moe_out = moe_fn(h_norm, w, cfg)
    return residual + moe_out


def _build_mtp_hidden(case: dict[str, Any], sidecar: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden_part = _rms_norm(case["base_hidden"], sidecar["mtp.pre_fc_norm_hidden.weight"])
    embed_part = _rms_norm(case["input_embed"], sidecar["mtp.pre_fc_norm_embedding.weight"])
    return F.linear(torch.cat([hidden_part, embed_part], dim=-1), sidecar["mtp.fc.weight"])


def _variant_fns() -> dict[str, LayerFn]:
    return {
        "baseline_full_forward": lambda h, pos, w, cfg: _layer_forward(h, pos, "full_attention", w, cfg),
        "decode_active_experts": lambda h, pos, w, cfg: _layer_forward_decode_moe(
            h, pos, w, cfg, moe_forward_decode_optimized
        ),
        "decode_bmm": lambda h, pos, w, cfg: _layer_forward_decode_moe(h, pos, w, cfg, moe_forward_decode_bmm),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar-file", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--max-cases", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--force-prefix-from-spec", action="store_true")
    ap.add_argument("--skip-forced-prefix-cases", action="store_true")
    ap.add_argument("--lm-head", choices=["current", "native_fp4", "bf16"], default="current")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompts(args.prompts_file, [])
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    if args.lm_head == "native_fp4":
        runner._prepare_native_fp4_lm_head()
    elif args.lm_head == "bf16":
        runner.native_fp4_lm_head_enabled = False

    sidecar, inventory = _load_sidecar(Path(args.sidecar_file), args.device, dtype)
    mtp_w = _mtp_layer_weights(sidecar)
    cfg = _mtp_cfg(runner, mtp_w)
    cases = _collect_cases(
        runner=runner,
        specs=specs,
        use_chat_template=False,
        force_prefix_from_spec=args.force_prefix_from_spec,
        skip_forced_prefix_cases=args.skip_forced_prefix_cases,
        max_new=args.max_new,
        first_token_weight=1.0,
        step1_weight=1.0,
        later_token_weight=1.0,
    )[: args.max_cases]

    variants = _variant_fns()
    timings = {name: [] for name in variants}
    max_abs_diff = {name: 0.0 for name in variants if name != "baseline_full_forward"}
    max_rel_l2 = {name: 0.0 for name in variants if name != "baseline_full_forward"}
    top1_match_baseline = {name: 0 for name in variants if name != "baseline_full_forward"}
    accepted = {name: 0 for name in variants}
    baseline_top1: list[int] = []
    variant_top1: dict[str, list[int]] = {name: [] for name in variants}

    with torch.no_grad():
        prepared: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for case in cases:
            prepared.append(
                (
                    _build_mtp_hidden(case, sidecar),
                    torch.tensor([[int(case["current_pos"])]], device=args.device, dtype=torch.long),
                    int(case["label_id"]),
                )
            )

        # Warm cuBLAS/cuDNN/Triton caches without polluting per-case timings.
        for mtp_hidden, pos, _label in prepared[: max(0, args.warmup)]:
            for fn in variants.values():
                _ = fn(mtp_hidden, pos, mtp_w, cfg)
        _sync(args.device)

        for mtp_hidden, pos, label_id in prepared:
            outputs: dict[str, torch.Tensor] = {}
            for name, fn in variants.items():
                out = None
                total_dt = 0.0
                for _ in range(max(1, args.repeats)):
                    out, dt = _timed(args.device, lambda fn=fn: fn(mtp_hidden, pos, mtp_w, cfg))
                    total_dt += dt
                assert out is not None
                outputs[name] = out
                timings[name].append(total_dt / max(1, args.repeats))

            baseline = outputs["baseline_full_forward"]
            for name, out in outputs.items():
                logits = runner._lm_head_logits(_rms_norm(out, sidecar["mtp.norm.weight"]))
                top1 = int(logits[0].argmax().item())
                variant_top1[name].append(top1)
                accepted[name] += int(top1 == label_id)
                if name == "baseline_full_forward":
                    baseline_top1.append(top1)
                    continue
                max_abs_diff[name] = max(max_abs_diff[name], _max_abs(out, baseline))
                max_rel_l2[name] = max(max_rel_l2[name], _rel_l2(out, baseline))
                top1_match_baseline[name] += int(top1 == baseline_top1[-1])

    summary = {name: _stats(values) for name, values in timings.items()}
    baseline_median = summary["baseline_full_forward"]["median_ms"]
    for name, item in summary.items():
        if baseline_median and item["median_ms"]:
            item["median_speedup_vs_baseline"] = float(baseline_median) / float(item["median_ms"])
        else:
            item["median_speedup_vs_baseline"] = None

    parity = {}
    for name in variants:
        if name == "baseline_full_forward":
            continue
        parity[name] = {
            "max_abs_vs_baseline": max_abs_diff[name],
            "max_rel_l2_vs_baseline": max_rel_l2[name],
            "top1_match_baseline": top1_match_baseline[name],
            "top1_match_rate_baseline": top1_match_baseline[name] / len(cases) if cases else None,
        }

    result = {
        "schema_version": "lynn-p113-mtp-layer-moe-variant-profile-v1",
        "decision": "AMBER: MTP layer MoE variant profile collected.",
        "model": args.model,
        "sidecar_file": args.sidecar_file,
        "prompts_file": args.prompts_file,
        "lm_head": args.lm_head,
        "case_count": len(cases),
        "repeats": args.repeats,
        "summary": summary,
        "accepted": accepted,
        "accept_rate": {name: accepted[name] / len(cases) if cases else None for name in variants},
        "parity_vs_baseline": parity,
        "sidecar_tensor_count": len(inventory.get("tensors", {})),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
