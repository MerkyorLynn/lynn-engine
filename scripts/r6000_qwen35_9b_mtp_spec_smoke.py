#!/usr/bin/env python3
"""Small Qwen3.5-9B MTP serving smoke for R6000.

This deliberately keeps graph reuse off so the first gate answers a narrower
question: does the official inline MTP sidecar activate in resident serving,
and what accept rate do sequential and batched speculative paths get?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.resident_runner import LynnIncrementalRunner


PROMPTS = [
    "Explain why local inference matters in one paragraph.",
    "Write a concise Python function to compute Fibonacci numbers.",
    "A train travels 120 km in 90 minutes. What is its average speed in km/h?",
    "Summarize the difference between BF16 and 4-bit quantization.",
]


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _run_mode(
    *,
    model: str,
    sidecar: str,
    name: str,
    speculative: bool,
    batched: bool,
    max_new: int,
) -> dict[str, Any]:
    env = {
        "LYNN_MTP_SIDECAR": sidecar,
        "LYNN_MTP_VERIFY": "0",
        "LYNN_MTP_SHADOW_VERIFY": "0",
        "LYNN_MTP_SPECULATIVE": "1" if speculative else "0",
        "LYNN_MTP_SPECULATIVE_BATCHED": "1" if batched else "0",
        "LYNN_MOE_IMPL": "packed_nvfp4",
        "LYNN_PACKED_DECODE": "1",
        "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
        "LYNN_NATIVE_FP4_LM_HEAD": "1",
        "LYNN_LINEAR_BLOCK_GRAPH": "0",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    }
    old_env = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        runner = LynnIncrementalRunner(model)
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for prompt in PROMPTS:
            out = runner.generate(prompt, max_new=max_new)
            rows.append(
                {
                    "prompt": prompt,
                    "decode_tps": out.get("decode_tps"),
                    "tokens_generated": out.get("tokens_generated"),
                    "mtp_speculative": out.get("mtp_speculative"),
                    "mtp_shadow": out.get("mtp_shadow"),
                    "text_head": str(out.get("text", ""))[:160],
                }
            )
        elapsed = time.perf_counter() - started
    finally:
        _restore_env(old_env)

    spec_rows = [row.get("mtp_speculative") or {} for row in rows]
    event_count = sum(int(row.get("events") or 0) for row in spec_rows)
    accepted = sum(int(row.get("accepted_events") or 0) for row in spec_rows)
    committed = sum(int(row.get("tokens_committed") or 0) for row in spec_rows)
    decode_values = [
        float(
            row.get("decode_tps")
            or row.get("tokens_per_second")
            or row.get("decode_tokens_per_second")
            or 0.0
        )
        for row in rows
    ]
    spec_effective_values = [
        float((row.get("mtp_speculative") or {}).get("effective_token_tps") or 0.0)
        for row in rows
    ]
    return {
        "mode": name,
        "wall_seconds": elapsed,
        "mean_decode_tps": sum(decode_values) / len(decode_values),
        "mean_spec_effective_token_tps": (
            sum(spec_effective_values) / len(spec_effective_values)
            if any(spec_effective_values)
            else None
        ),
        "events": event_count,
        "accepted_events": accepted,
        "accept_rate": accepted / event_count if event_count else None,
        "tokens_committed": committed,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0",
    )
    parser.add_argument(
        "--sidecar",
        default="/root/autodl-tmp/models/mtp_sidecars/qwen35-9b-official-inline-lynn/mtp.safetensors",
    )
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument(
        "--modes",
        default="baseline,spec_k1,spec_k1_batched",
        help="Comma-separated subset: baseline,spec_k1,spec_k1_batched",
    )
    parser.add_argument(
        "--out-dir",
        default="/root/autodl-tmp/reports/qwen35_9b",
    )
    args = parser.parse_args()

    requested = {name.strip() for name in args.modes.split(",") if name.strip()}
    mode_specs = [
        ("baseline", False, False),
        ("spec_k1", True, False),
        ("spec_k1_batched", True, True),
    ]
    result = {
        "model": args.model,
        "sidecar": args.sidecar,
        "max_new": args.max_new,
        "modes": [
            _run_mode(
                model=args.model,
                sidecar=args.sidecar,
                name=name,
                speculative=speculative,
                batched=batched,
                max_new=args.max_new,
            )
            for name, speculative, batched in mode_specs
            if name in requested
        ],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qwen35_9b_mtp_spec_smoke_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out_path)
    print(
        json.dumps(
            {
                row["mode"]: {
                    "tps": row["mean_decode_tps"],
                    "spec_effective_tps": row["mean_spec_effective_token_tps"],
                    "events": row["events"],
                    "accept": row["accept_rate"],
                    "committed": row["tokens_committed"],
                }
                for row in result["modes"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
