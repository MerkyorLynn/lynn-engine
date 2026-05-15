#!/usr/bin/env python3
"""P14-C: graph-owned authoritative decode-state sequence probe.

P14-B measured a conservative state-refresh loop:
  real -> graph -> replay -> real

That is safe but caps near copy+replay cost. The higher-throughput design is to
copy prefill state into a graph-owned decode state once, then let CUDA graph
slots mutate that state authoritatively across tokens. Slots are still captured
per fixed position in this probe; later serving code can cache them by position.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p14b_graph_owned_state_slot_probe import (  # noqa: E402
    _capture_graph_owned_slot,
    _copy_state,
    _decode_one,
    _logit_diff,
    _new_state_like,
    _prefill,
)
from engine.resident_runner import FullTokenGraphSlot, LynnIncrementalRunner  # noqa: E402


def _eager_generate(
    runner: LynnIncrementalRunner,
    prompt: str,
    max_new: int,
) -> dict[str, Any]:
    token_id, state = _prefill(runner, prompt)
    ids = [int(token_id)]
    logits_trace = []
    for _ in range(1, max_new):
        token_id, logits = _decode_one(runner, state, token_id)
        ids.append(int(token_id))
        logits_trace.append(logits.clone())
    return {"ids": ids, "logits_trace": logits_trace}


def _capture_position_slots(
    runner: LynnIncrementalRunner,
    prompt: str,
    max_new: int,
) -> tuple[int, Any, list[FullTokenGraphSlot]]:
    token_id, prefill_state = _prefill(runner, prompt)
    graph_state = _new_state_like(runner)
    _copy_state(graph_state, prefill_state)
    slots: list[FullTokenGraphSlot] = []
    for _ in range(1, max_new):
        slot = _capture_graph_owned_slot(runner, graph_state, token_id)
        slots.append(slot)
        token_id, _ = _decode_one(runner, graph_state, token_id)
    return token_id, prefill_state, slots


def _replay_authoritative(
    graph_state,
    prefill_state,
    first_token_id: int,
    slots: list[FullTokenGraphSlot],
) -> dict[str, Any]:
    _copy_state(graph_state, prefill_state)
    token_id = int(first_token_id)
    ids = [token_id]
    logits_trace = []
    replay_ms: list[float] = []
    for slot in slots:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        logits = slot.replay(token_id).clone()
        graph_state.seq_len = slot.seq_len + 1
        end.record()
        torch.cuda.synchronize()
        replay_ms.append(float(start.elapsed_time(end)))
        token_id = int(logits[0].argmax().item())
        ids.append(token_id)
        logits_trace.append(logits)
    return {"ids": ids, "logits_trace": logits_trace, "replay_ms": replay_ms}


def _run_one(
    runner: LynnIncrementalRunner,
    *,
    prompt: str,
    max_new: int,
) -> dict[str, Any]:
    eager = _eager_generate(runner, prompt, max_new)
    first_id, prefill_state = _prefill(runner, prompt)
    graph_state = _new_state_like(runner)
    _copy_state(graph_state, prefill_state)

    capture_ms: list[float] = []
    slots: list[FullTokenGraphSlot] = []
    token_id = int(first_id)
    for _ in range(1, max_new):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        slot = _capture_graph_owned_slot(runner, graph_state, token_id)
        end.record()
        torch.cuda.synchronize()
        capture_ms.append(float(start.elapsed_time(end)))
        slots.append(slot)
        token_id, _ = _decode_one(runner, graph_state, token_id)

    replay = _replay_authoritative(graph_state, prefill_state, first_id, slots)
    diffs = [
        _logit_diff(eager_logits, graph_logits)
        for eager_logits, graph_logits in zip(eager["logits_trace"], replay["logits_trace"])
    ]
    replay_total_ms = sum(replay["replay_ms"])
    return {
        "prompt": prompt,
        "max_new": max_new,
        "slots": len(slots),
        "eager_ids": eager["ids"],
        "graph_ids": replay["ids"],
        "same_ids": eager["ids"] == replay["ids"],
        "capture_avg_ms": (sum(capture_ms) / len(capture_ms)) if capture_ms else None,
        "capture_total_ms": sum(capture_ms),
        "replay_avg_ms": (replay_total_ms / len(replay["replay_ms"])) if replay["replay_ms"] else None,
        "replay_tps": (len(replay["replay_ms"]) * 1000.0 / replay_total_ms) if replay_total_ms else None,
        "diffs": diffs,
        "min_cosine": min((d["cosine"] for d in diffs), default=None),
        "min_top10_overlap": min((d["top10_overlap"] for d in diffs), default=None),
        "all_top1_match": all(d["top1_match"] for d in diffs),
        "pass": eager["ids"] == replay["ids"] and all(d["top1_match"] for d in diffs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    case = _run_one(runner, prompt=args.prompt, max_new=args.max_new)
    result = {
        "schema_version": "lynn-engine-p14c-graph-owned-authoritative-sequence-v1",
        "model": args.model,
        "case": case,
        "pass": bool(case["pass"]),
        "note": "Replay-only graph-owned decode state sequence; capture cost excluded from replay TPS.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
