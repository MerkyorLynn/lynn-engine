#!/usr/bin/env python3
"""Structured generation gate with optional first-token format guard.

The regular W4A8 generation gate intentionally compares raw greedy decode, but
the current failures are mostly first-token / first-few-token format-domain
drift on structured prompts. This gate uses explicit prompt specs with an
optional forced prefix so we can separate:

* teacher prompt cleanliness;
* W4A8 parity after format anchoring;
* whether the raw top-1 token already wanted the anchored prefix.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _same_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if int(x) != int(y):
            break
        n += 1
    return n


def _format_eval(text: str, spec: dict[str, Any]) -> dict[str, Any]:
    stripped = text.lstrip()
    expected = spec.get("expected_start")
    forbid = list(spec.get("forbid") or [])
    starts_ok = True if not expected else stripped.startswith(str(expected))
    forbidden_hits = [item for item in forbid if item in text]
    return {
        "starts_ok": starts_ok,
        "forbidden_hits": forbidden_hits,
        "ok": starts_ok and not forbidden_hits,
        "expected_start": expected,
    }


def _load_specs(path: str) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    specs = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            specs.append({"id": str(idx), "prompt": item})
        elif isinstance(item, dict):
            specs.append({"id": item.get("id", str(idx)), **item})
        else:
            raise TypeError(f"prompt spec must be string or object, got {type(item)}")
    return specs


def _run_model(
    model_dir: str,
    *,
    specs: list[dict[str, Any]],
    max_new: int,
    top_k: int,
    label: str,
    device: str,
    dtype: torch.dtype,
    force_prefix: bool,
    use_chat_template: bool,
) -> dict[str, Any]:
    runner = LynnIncrementalRunner(model_dir, device=device, dtype=dtype, max_seq_len=4096, verbose=True)
    rows: list[dict[str, Any]] = []
    try:
        for prompt_id, spec in enumerate(specs):
            prompt = str(spec["prompt"])
            forced = str(spec.get("forced_prefix", "")) if force_prefix and spec.get("forced_prefix") is not None else None
            per_prompt: dict[str, Any] = {
                "prompt_id": prompt_id,
                "id": spec.get("id", str(prompt_id)),
                "prompt": prompt,
                "forced_prefix": forced,
                "expected_start": spec.get("expected_start"),
            }
            for mode in ("off", "full"):
                os.environ["LYNN_W4A8_FAKE_QUANT_ACTIVE"] = mode
                out = runner.generate(
                    prompt,
                    max_new=max_new,
                    top_k=top_k,
                    use_chat_template=use_chat_template,
                    forced_prefix_text=forced,
                )
                per_prompt[mode] = {
                    "new_ids": out["new_ids"],
                    "completion_text": out["completion_text"],
                    "completion_text_raw": out["completion_text_raw"],
                    "stopped_reason": out["stopped_reason"],
                    "decode_tps": out["timings"].get("decode_tps"),
                    "format": _format_eval(out["completion_text"], spec),
                    "forced_prefix": out.get("forced_prefix"),
                    "topk_trace": out.get("topk_trace", []),
                }
            prefix = _same_prefix(per_prompt["off"]["new_ids"], per_prompt["full"]["new_ids"])
            per_prompt["self_compare"] = {
                "exact": per_prompt["off"]["new_ids"] == per_prompt["full"]["new_ids"],
                "same_prefix_tokens": prefix,
                "first_diff_index": None if prefix == max_new else prefix,
            }
            rows.append(per_prompt)
    finally:
        os.environ["LYNN_W4A8_FAKE_QUANT_ACTIVE"] = "off"
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    exact = sum(1 for row in rows if row["self_compare"]["exact"])
    prefixes = [row["self_compare"]["same_prefix_tokens"] for row in rows]
    return {
        "label": label,
        "model_dir": model_dir,
        "prompt_count": len(rows),
        "exact": exact,
        "min_same_prefix_tokens": min(prefixes) if prefixes else None,
        "mean_same_prefix_tokens": sum(prefixes) / len(prefixes) if prefixes else None,
        "format_ok_off": sum(1 for row in rows if row["off"]["format"]["ok"]),
        "format_ok_full": sum(1 for row in rows if row["full"]["format"]["ok"]),
        "forced_raw_match_off": sum(
            1
            for row in rows
            if not row["off"].get("forced_prefix")
            or row["off"]["forced_prefix"].get("raw_all_matched")
        ),
        "forced_raw_match_full": sum(
            1
            for row in rows
            if not row["full"].get("forced_prefix")
            or row["full"]["forced_prefix"].get("raw_all_matched")
        ),
        "rows": rows,
    }


def _cross_compare(reference: dict[str, Any], candidate: dict[str, Any], max_new: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ref, cand in zip(reference["rows"], candidate["rows"]):
        prefix = _same_prefix(ref["off"]["new_ids"], cand["full"]["new_ids"])
        rows.append(
            {
                "prompt_id": ref["prompt_id"],
                "id": ref.get("id"),
                "prompt": ref["prompt"],
                "forced_prefix": ref.get("forced_prefix"),
                "exact": ref["off"]["new_ids"] == cand["full"]["new_ids"],
                "same_prefix_tokens": prefix,
                "first_diff_index": None if prefix == max_new else prefix,
                "reference_format": ref["off"]["format"],
                "candidate_format": cand["full"]["format"],
                "candidate_forced_prefix": cand["full"].get("forced_prefix"),
                "reference_off_text": ref["off"]["completion_text"],
                "candidate_full_text": cand["full"]["completion_text"],
            }
        )
    prefixes = [row["same_prefix_tokens"] for row in rows]
    return {
        "reference": reference["label"] + ":off",
        "candidate": candidate["label"] + ":full",
        "exact": sum(1 for row in rows if row["exact"]),
        "min_same_prefix_tokens": min(prefixes) if prefixes else None,
        "mean_same_prefix_tokens": sum(prefixes) / len(prefixes) if prefixes else None,
        "reference_format_ok": sum(1 for row in rows if row["reference_format"]["ok"]),
        "candidate_format_ok": sum(1 for row in rows if row["candidate_format"]["ok"]),
        "candidate_raw_prefix_match": sum(
            1
            for row in rows
            if not row.get("candidate_forced_prefix")
            or row["candidate_forced_prefix"].get("raw_all_matched")
        ),
        "rows": rows,
    }


def _decision(cross: dict[str, Any], total: int, max_new: int) -> str:
    if cross["candidate_format_ok"] < total:
        return "RED: candidate still violates structured format under this gate."
    if cross["exact"] == total:
        return "GREEN: W4A8 is token-exact and format-clean under this gate."
    if cross["min_same_prefix_tokens"] is not None and cross["min_same_prefix_tokens"] >= max_new * 0.75:
        return "AMBER: format is clean and W4A8 diverges late under this gate."
    return "RED: format is clean but W4A8 still diverges early under this gate."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--folded-model", required=True)
    parser.add_argument("--prompt-specs-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force-prefix", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("LYNN_MOE_IMPL", "bmm")
    os.environ.setdefault("LYNN_W4A8_FAKE_QUANT_FORMAT", "e4m3")
    os.environ.setdefault("LYNN_W4A8_FAKE_QUANT_GRANULARITY", "per16")

    specs = _load_specs(args.prompt_specs_file)
    dtype = torch.bfloat16
    original = _run_model(
        args.model,
        specs=specs,
        max_new=args.max_new,
        top_k=args.top_k,
        label="original",
        device=args.device,
        dtype=dtype,
        force_prefix=args.force_prefix,
        use_chat_template=args.use_chat_template,
    )
    folded = _run_model(
        args.folded_model,
        specs=specs,
        max_new=args.max_new,
        top_k=args.top_k,
        label="folded",
        device=args.device,
        dtype=dtype,
        force_prefix=args.force_prefix,
        use_chat_template=args.use_chat_template,
    )
    cross = _cross_compare(original, folded, args.max_new)
    result = {
        "schema_version": "lynn-a100-w4a8-format-guard-gate-v1",
        "decision": _decision(cross, len(specs), args.max_new),
        "force_prefix": args.force_prefix,
        "use_chat_template": args.use_chat_template,
        "max_new": args.max_new,
        "top_k": args.top_k,
        "env": {
            "LYNN_MOE_IMPL": os.environ.get("LYNN_MOE_IMPL"),
            "LYNN_W4A8_FAKE_QUANT_FORMAT": os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT"),
            "LYNN_W4A8_FAKE_QUANT_GRANULARITY": os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY"),
        },
        "models": [original, folded],
        "cross_model_compare": cross,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
