#!/usr/bin/env python3
"""P7-C: quantify request-level overhead around the P6-S decode path.

P6-S proves the resident decode loop can sustain ~64-66 decode TPS once the
linear-attention CUDA graphs are captured. User-facing server requests still
pay prefill plus graph-capture overhead, so this probe measures how that cost
amortizes across different completion lengths before we change server state
reuse semantics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--max-new-list", default="8,32,128")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=False)
    rows = []
    for raw in args.max_new_list.split(","):
        max_new = int(raw.strip())
        if max_new <= 0:
            continue
        t0 = time.time()
        result = runner.generate(args.prompt, max_new=max_new, use_chat_template=True)
        wall = time.time() - t0
        completion_tokens = len(result["new_ids"])
        timings = result.get("timings", {})
        decode_seconds = timings.get("decode_step_seconds") or []
        decode_sum = float(sum(decode_seconds))
        capture_s = timings.get("linear_block_graph_capture_seconds") or 0.0
        prefill_s = timings.get("prefill_seconds") or 0.0
        rows.append(
            {
                "max_new": max_new,
                "completion_tokens": completion_tokens,
                "wall_s": wall,
                "request_tps": completion_tokens / max(wall, 1e-9),
                "prefill_s": prefill_s,
                "linear_block_graph_capture_s": capture_s,
                "decode_sum_s": decode_sum,
                "decode_tps": (len(decode_seconds) / decode_sum) if decode_sum > 0 else None,
                "overhead_s": max(wall - decode_sum, 0.0),
                "completion_preview": result.get("completion_text", "")[:120],
            }
        )

    report = {
        "schema_version": "lynn-engine-p7c-request-amortization-probe-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "load_seconds": runner.load_seconds,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
