#!/usr/bin/env python3
"""P186 · Qwen3.5-9B resident W4A8 generation gate.

Runs the safe 9B NVFP4 convstrict profile with W4A8 activation fake-quant
disabled, gate/up-only, and full.  This is a R6000 generation-path counterpart
to the Spark W4A8 quality result; it is not a final speed claim until native
FP8-active kernels replace the fake-quant round-trip.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from p148_qwen35_9b_nvfp4_fast_profile import BASELINE_ENV, _run_mode, _summarize_mode


def _merge(base: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    out = dict(base)
    out.update(updates)
    return out


CONVSTRICT_ENV = _merge(
    BASELINE_ENV,
    {
        "LYNN_LINEAR_STATE_UPDATE": "inplace",
        "LYNN_LINEAR_BLOCK_GRAPH": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "1",
        "LYNN_LINEAR_ATTN_CONV_BACKEND": "triton_torch_silu",
        "LYNN_W4A8_FAKE_QUANT_FORMAT": "e4m3",
        "LYNN_W4A8_FAKE_QUANT_GRANULARITY": "per16",
    },
)


def _load_prompts(path: Path, limit: int) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    prompts = []
    for row in rows:
        system = (row.get("system") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        if system:
            prompts.append(f"System: {system}\nUser: {prompt}")
        else:
            prompts.append(prompt)
    if limit:
        prompts = prompts[:limit]
    return prompts


def _prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    cand_by_key = {(r["prompt_id"], r["max_new"]): r for r in candidate["rows"]}
    rows = []
    exact = 0
    prefixes = []
    speedups = []
    for ref_row in reference["rows"]:
        key = (ref_row["prompt_id"], ref_row["max_new"])
        cand_row = cand_by_key.get(key)
        if cand_row is None:
            continue
        ref_ids = [int(x) for x in ref_row.get("new_ids", [])]
        cand_ids = [int(x) for x in cand_row.get("new_ids", [])]
        ids_equal = ref_ids == cand_ids
        exact += int(ids_equal)
        prefix = _prefix_len(ref_ids, cand_ids)
        prefixes.append(prefix)
        ref_tps = ref_row.get("decode_tps")
        cand_tps = cand_row.get("decode_tps")
        speedup = (float(cand_tps) / float(ref_tps)) if ref_tps and cand_tps else None
        if speedup is not None:
            speedups.append(speedup)
        rows.append(
            {
                "prompt_id": key[0],
                "max_new": key[1],
                "exact": ids_equal,
                "prefix_len": prefix,
                "reference_decode_tps": ref_tps,
                "candidate_decode_tps": cand_tps,
                "speedup": speedup,
                "reference_prefix": ref_ids[:16],
                "candidate_prefix": cand_ids[:16],
            }
        )
    total = len(rows)
    return {
        "exact_count": exact,
        "total": total,
        "all_exact": exact == total,
        "min_prefix": min(prefixes) if prefixes else None,
        "mean_prefix": sum(prefixes) / len(prefixes) if prefixes else None,
        "speedup_mean": sum(speedups) / len(speedups) if speedups else None,
        "speedup_min": min(speedups) if speedups else None,
        "rows": rows,
    }


def _decision(comparison_full: dict[str, Any], comparison_gateup: dict[str, Any]) -> str:
    if comparison_full.get("all_exact"):
        return "DENSE_W4A8_FULL_GENERATION_EXACT"
    if comparison_gateup.get("all_exact"):
        return "DENSE_W4A8_GATEUP_EXACT_FULL_DRIFT"
    full_min = comparison_full.get("min_prefix") or 0
    gate_min = comparison_gateup.get("min_prefix") or 0
    if full_min >= 8 or gate_min >= 8:
        return "DENSE_W4A8_GENERATION_AMBER_PREFIX"
    return "DENSE_W4A8_GENERATION_DRIFT"


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5-9B resident W4A8 generation gate.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--limit", type=int, default=70)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = _load_prompts(Path(args.prompts_json), args.limit)
    modes: list[dict[str, Any]] = []
    env_off = _merge(CONVSTRICT_ENV, {"LYNN_W4A8_FAKE_QUANT_ACTIVE": "off"})
    env_gateup = _merge(CONVSTRICT_ENV, {"LYNN_W4A8_FAKE_QUANT_ACTIVE": "gateup"})
    env_full = _merge(CONVSTRICT_ENV, {"LYNN_W4A8_FAKE_QUANT_ACTIVE": "full"})

    reference = _run_mode(
        model=args.model,
        label="convstrict_w4a16_reference",
        env=env_off,
        max_new_values=[args.max_new],
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    modes.append(reference)
    gateup = _run_mode(
        model=args.model,
        label="convstrict_w4a8_gateup",
        env=env_gateup,
        max_new_values=[args.max_new],
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    modes.append(gateup)
    full = _run_mode(
        model=args.model,
        label="convstrict_w4a8_full",
        env=env_full,
        max_new_values=[args.max_new],
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    modes.append(full)

    cmp_gateup = _compare(reference, gateup)
    cmp_full = _compare(reference, full)
    report = {
        "schema": "lynn-qwen35-9b-dense-w4a8-resident-gate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "prompts_json": args.prompts_json,
        "limit": args.limit,
        "max_new": args.max_new,
        "max_seq_len": args.max_seq_len,
        "note": "W4A8 fake-quant validates generation drift; speed is emulation-only until native FP8-active kernels land.",
        "summaries": [_summarize_mode(m) for m in modes],
        "comparison_gateup": cmp_gateup,
        "comparison_full": cmp_full,
        "modes": modes,
        "decision": _decision(cmp_full, cmp_gateup),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"],
        "summaries": report["summaries"],
        "gateup": {
            "exact": f"{cmp_gateup.get('exact_count')}/{cmp_gateup.get('total')}",
            "min_prefix": cmp_gateup.get("min_prefix"),
            "mean_prefix": cmp_gateup.get("mean_prefix"),
        },
        "full": {
            "exact": f"{cmp_full.get('exact_count')}/{cmp_full.get('total')}",
            "min_prefix": cmp_full.get("min_prefix"),
            "mean_prefix": cmp_full.get("mean_prefix"),
        },
        "out": str(out_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
