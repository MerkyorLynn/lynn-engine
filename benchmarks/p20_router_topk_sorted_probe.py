#!/usr/bin/env python3
"""P20: probe whether router top-k needs sorted=True.

PyTorch `topk` sorts results by default. For MoE decode, the final weighted sum
is order-independent as long as expert ids and routing weights stay paired. If
`sorted=False` returns the same top-k set faster, it is a safe micro-optimization
for every MoE layer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p15_moe_packed_segment_profile import _prepare_layer_moe_input, _prefill  # noqa: E402
from engine.inference_state import LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _bench(fn: Callable[[], tuple[torch.Tensor, torch.Tensor]], warmup: int, iters: int) -> float:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    token_id, decode_position = _prefill(runner, state, args.prompt)
    h_moe = _prepare_layer_moe_input(runner, state, token_id, decode_position, args.layer)
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    top_k = int(cfg["num_experts_per_tok"])
    logits = F.linear(h_flat, w["mlp.gate.weight"])

    def sorted_topk() -> tuple[torch.Tensor, torch.Tensor]:
        values, indices = torch.topk(logits, top_k, dim=-1, sorted=True)
        weights = F.softmax(values, dim=-1, dtype=torch.float32)[0]
        return indices[0], weights

    def unsorted_topk() -> tuple[torch.Tensor, torch.Tensor]:
        values, indices = torch.topk(logits, top_k, dim=-1, sorted=False)
        weights = F.softmax(values, dim=-1, dtype=torch.float32)[0]
        return indices[0], weights

    sorted_ids, sorted_weights = sorted_topk()
    unsorted_ids, unsorted_weights = unsorted_topk()
    sorted_pairs = sorted((int(i), float(w)) for i, w in zip(sorted_ids.tolist(), sorted_weights.tolist()))
    unsorted_pairs = sorted((int(i), float(w)) for i, w in zip(unsorted_ids.tolist(), unsorted_weights.tolist()))
    result = {
        "schema_version": "lynn-engine-p20-router-topk-sorted-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "decode_position": decode_position,
        "sorted_ids": [int(x) for x in sorted_ids.tolist()],
        "unsorted_ids": [int(x) for x in unsorted_ids.tolist()],
        "same_expert_set": sorted(int(x) for x in sorted_ids.tolist()) == sorted(int(x) for x in unsorted_ids.tolist()),
        "paired_weight_max_abs_after_sort": max(abs(a[1] - b[1]) for a, b in zip(sorted_pairs, unsorted_pairs)),
        "timing_ms": {
            "sorted_true_ms": _bench(sorted_topk, args.warmup, args.iters),
            "sorted_false_ms": _bench(unsorted_topk, args.warmup, args.iters),
        },
    }
    result["timing_ms"]["speedup"] = result["timing_ms"]["sorted_true_ms"] / result["timing_ms"]["sorted_false_ms"]
    result["pass"] = bool(result["same_expert_set"] and result["paired_weight_max_abs_after_sort"] < 1e-7)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
