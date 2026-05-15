#!/usr/bin/env python3
"""P13 smoke: opt-in full-token graph-slot path inside runner.generate().

This validates the first production-shaped wiring of `FullTokenGraphSlot`:
`LynnIncrementalRunner.generate()` can capture/replay the current-token graph
slot under `LYNN_FULL_TOKEN_GRAPH_SLOT=1` and preserve greedy token IDs.

This is a correctness gate, not a speed claim. Capture-per-token is expected
to be slower than the stable eager/linear-block path; later P13 work can add a
cache/window lifecycle once this opt-in path is strict.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _run(model: str, prompt: str, max_new: int, graph_slot: bool) -> dict[str, Any]:
    old = os.environ.get("LYNN_FULL_TOKEN_GRAPH_SLOT")
    os.environ["LYNN_FULL_TOKEN_GRAPH_SLOT"] = "1" if graph_slot else "0"
    try:
        runner = LynnIncrementalRunner(model, device="cuda", dtype=torch.bfloat16, verbose=False)
        output = runner.generate(prompt, max_new=max_new)
        return {
            "new_ids": output["new_ids"],
            "completion_prefix": output["completion_text"][:160],
            "timings": output["timings"],
        }
    finally:
        if old is None:
            os.environ.pop("LYNN_FULL_TOKEN_GRAPH_SLOT", None)
        else:
            os.environ["LYNN_FULL_TOKEN_GRAPH_SLOT"] = old
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    eager = _run(args.model, args.prompt, args.max_new, graph_slot=False)
    graph = _run(args.model, args.prompt, args.max_new, graph_slot=True)
    capture = graph["timings"].get("full_token_graph_slot_capture_seconds") or []
    replay = graph["timings"].get("full_token_graph_slot_replay_seconds") or []
    result = {
        "schema_version": "lynn-engine-p13-full-token-graph-slot-generate-smoke-v1",
        "model": args.model,
        "prompt": args.prompt,
        "max_new": args.max_new,
        "eager": eager,
        "graph_slot": graph,
        "same_ids": eager["new_ids"] == graph["new_ids"],
        "capture_steps": len(capture),
        "replay_steps": len(replay),
        "avg_capture_ms": (sum(capture) / len(capture) * 1000.0) if capture else None,
        "avg_replay_ms": (sum(replay) / len(replay) * 1000.0) if replay else None,
        "avg_replay_tps": (len(replay) / sum(replay)) if replay else None,
        "pass": eager["new_ids"] == graph["new_ids"] and len(replay) == max(0, args.max_new - 1),
        "note": "Capture-per-token graph-slot generate path; correctness first, not a speed path yet.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
