#!/usr/bin/env python3
"""P196 · Qwen3.5-9B W4A8 structured content gate.

P185/P186 answer "does W4A8 drift from W4A16?".  This gate answers the
promotion question more directly: under the hard structured prompt set, does
the W4A8 activation route still produce usable JSON/code/YAML/math outputs?

The W4A8 modes here use fake quantization.  Their TPS is therefore diagnostic
only; the pass/fail signal is quality/format stability.  Native FP8-active
kernels must still pass this gate before any default promotion.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from p148_qwen35_9b_nvfp4_fast_profile import BASELINE_ENV, _summarize_mode


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


def _load_specs(path: Path, limit: int) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("prompt specs file must be a JSON list")
    specs = [dict(item) for item in raw]
    return specs[:limit] if limit else specs


def _prompt_from_spec(spec: dict[str, Any]) -> str:
    system = str(spec.get("system") or "").strip()
    prompt = str(spec.get("prompt") or "").strip()
    if system:
        return f"{system}\n\n{prompt}"
    return prompt


def _format_checks(text: str, spec: dict[str, Any]) -> dict[str, Any]:
    stripped = text.lstrip()
    starts_with = spec.get("starts_with")
    starts_any = [str(item) for item in spec.get("starts_with_any") or []]
    if starts_with is not None:
        starts_ok = stripped.startswith(str(starts_with))
    elif starts_any:
        starts_ok = any(stripped.startswith(item) for item in starts_any)
    else:
        starts_ok = True

    must_contain = [str(item) for item in spec.get("must_contain") or []]
    missing = [item for item in must_contain if item not in text]
    forbid = [str(item) for item in spec.get("forbid") or []]
    forbidden_hits = [item for item in forbid if item in text]

    json_ok = True
    json_error = None
    parsed_kind = None
    if bool(spec.get("parse_json")):
        try:
            parsed = json.loads(text)
            json_ok = isinstance(parsed, dict)
            parsed_kind = type(parsed).__name__
        except Exception as exc:  # noqa: BLE001 - diagnostic gate
            json_ok = False
            json_error = str(exc)

    empty = not stripped
    repeated_first_token = False
    return {
        "starts_ok": starts_ok,
        "missing": missing,
        "forbidden_hits": forbidden_hits,
        "json_ok": json_ok,
        "json_error": json_error,
        "parsed_kind": parsed_kind,
        "empty": empty,
        "repeated_first_token": repeated_first_token,
        "ok": starts_ok and not missing and not forbidden_hits and json_ok and not empty,
    }


def _mode_env(active: str) -> dict[str, str]:
    return _merge(CONVSTRICT_ENV, {"LYNN_W4A8_FAKE_QUANT_ACTIVE": active})


def _set_env(env: dict[str, str]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _run_chat_mode(
    *,
    model: str,
    label: str,
    env: dict[str, str],
    max_new: int,
    prompts: list[str],
    max_seq_len: int,
) -> dict[str, Any]:
    from engine.resident_runner import LynnIncrementalRunner

    print(f"[p196] loading mode={label}", flush=True)
    old = _set_env(env)
    try:
        t_load0 = time.time()
        runner = LynnIncrementalRunner(
            model,
            device="cuda",
            dtype=torch.bfloat16,
            max_seq_len=max_seq_len,
            verbose=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_seconds = time.time() - t_load0
        rows: list[dict[str, Any]] = []
        for prompt_id, prompt in enumerate(prompts):
            print(f"[p196] {label} prompt={prompt_id} max_new={max_new}", flush=True)
            out = runner.generate(prompt, max_new=max_new, use_chat_template=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings = out.get("timings", {})
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "max_new": max_new,
                    "new_ids": out.get("new_ids", []),
                    "completion_text": out.get("completion_text", ""),
                    "decode_tps": timings.get("decode_tps"),
                    "wall_tps": timings.get("wall_tps"),
                    "timings": timings,
                }
            )
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {
            "label": label,
            "env": env,
            "load_seconds": load_seconds,
            "rows": rows,
        }
    finally:
        _restore_env(old)


def _score_mode(mode: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in mode["rows"]:
        prompt_id = int(row["prompt_id"])
        spec = specs[prompt_id]
        text = str(row.get("completion_text") or "")
        checks = _format_checks(text, spec)
        new_ids = [int(x) for x in row.get("new_ids") or []]
        if len(new_ids) >= 8 and len(set(new_ids[:8])) == 1:
            checks["repeated_first_token"] = True
            checks["ok"] = False
        rows.append(
            {
                "prompt_id": prompt_id,
                "id": spec.get("id", f"spec_{prompt_id}"),
                "decode_tps": row.get("decode_tps"),
                "wall_tps": row.get("wall_tps"),
                "new_ids_prefix": new_ids[:16],
                "text_prefix": text[:240],
                "format": checks,
            }
        )

    decode = [float(row["decode_tps"]) for row in rows if row.get("decode_tps") is not None]
    pass_count = sum(1 for row in rows if row["format"]["ok"])
    return {
        "label": mode["label"],
        "request_count": len(rows),
        "pass_count": pass_count,
        "pass_rate": (pass_count / len(rows)) if rows else None,
        "all_format_ok": pass_count == len(rows),
        "decode_tps_mean": statistics.fmean(decode) if decode else None,
        "decode_tps_min": min(decode) if decode else None,
        "decode_tps_max": max(decode) if decode else None,
        "failures": [row for row in rows if not row["format"]["ok"]],
        "rows": rows,
    }


def _decision(scores: dict[str, dict[str, Any]], full_min_rate: float, gateup_min_rate: float) -> str:
    off = scores["convstrict_w4a16_reference"]
    gateup = scores["convstrict_w4a8_gateup"]
    full = scores["convstrict_w4a8_full"]
    if not off["all_format_ok"]:
        return "W4A16_REFERENCE_RED"
    if full["all_format_ok"]:
        return "W4A8_FULL_CONTENT_GREEN"
    if gateup["all_format_ok"]:
        return "W4A8_GATEUP_CONTENT_GREEN_FULL_DRIFT"
    if (full["pass_rate"] or 0.0) >= full_min_rate and (gateup["pass_rate"] or 0.0) >= gateup_min_rate:
        return "W4A8_CONTENT_AMBER"
    return "W4A8_CONTENT_RED_FALLBACK_A16"


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5-9B W4A8 structured content gate.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--limit", type=int, default=70)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--full-min-rate", type=float, default=0.985)
    ap.add_argument("--gateup-min-rate", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    specs = _load_specs(Path(args.prompts_json), args.limit)
    prompts = [_prompt_from_spec(spec) for spec in specs]
    modes = []
    for label, active in (
        ("convstrict_w4a16_reference", "off"),
        ("convstrict_w4a8_gateup", "gateup"),
        ("convstrict_w4a8_full", "full"),
    ):
        modes.append(
            _run_chat_mode(
                model=args.model,
                label=label,
                env=_mode_env(active),
                max_new=args.max_new,
                prompts=prompts,
                max_seq_len=args.max_seq_len,
            )
        )

    scores = {mode["label"]: _score_mode(mode, specs) for mode in modes}
    report = {
        "schema": "lynn-qwen35-9b-w4a8-structured-content-gate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "prompts_json": args.prompts_json,
        "limit": args.limit,
        "max_new": args.max_new,
        "max_seq_len": args.max_seq_len,
        "thresholds": {
            "full_min_rate": args.full_min_rate,
            "gateup_min_rate": args.gateup_min_rate,
        },
        "note": "W4A8 fake-quant checks structured stability only; speed is not a native FP8-active claim.",
        "summaries": [_summarize_mode(mode) for mode in modes],
        "scores": scores,
        "decision": _decision(scores, args.full_min_rate, args.gateup_min_rate),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"],
        "scores": {
            key: {
                "pass_count": value["pass_count"],
                "request_count": value["request_count"],
                "pass_rate": value["pass_rate"],
                "decode_tps_mean": value["decode_tps_mean"],
                "failures": [row["id"] for row in value["failures"][:10]],
            }
            for key, value in scores.items()
        },
        "out": str(out_path),
    }, ensure_ascii=False, indent=2))
    return 0 if report["decision"] in {"W4A8_FULL_CONTENT_GREEN", "W4A8_GATEUP_CONTENT_GREEN_FULL_DRIFT"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
