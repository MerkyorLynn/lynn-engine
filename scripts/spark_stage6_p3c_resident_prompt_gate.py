#!/usr/bin/env python3
"""Stage 6 Phase 3-C: resident-runner real-prompt gate for P3-B/P3-A path."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spark_stage6_p2o_packed_prefill_rc_smoke import (  # noqa: E402
    CANONICAL_ENV,
    DEFAULT_MODEL,
    _degenerate,
    _norm,
    _prompt_preset,
    _restore_env,
)


def _prefill_updates(enabled: bool) -> dict[str, str]:
    return {
        "LYNN_PACKED_PREFILL_SLOW": "1" if enabled else "0",
        "LYNN_PACKED_PREFILL_SLOW_MODE": "p3a_grouped",
        "LYNN_PACKED_PREFILL_P2E_LAYERS": "all",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_T": "32",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_INTER": "8",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_HIDDEN": "128",
        "LYNN_PACKED_PREFILL_P2E_NUM_WARPS": "4",
        "LYNN_PACKED_PREFILL_P2E_DOWN_BLOCK_HIDDEN": "8",
        "LYNN_PACKED_PREFILL_P2E_DOWN_BLOCK_INTER": "512",
        "LYNN_PACKED_PREFILL_P2E_DOWN_NUM_WARPS": "8",
        "LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA": "1" if enabled else "0",
    }


def _set_prefill_optin(enabled: bool) -> dict[str, str | None]:
    updates = _prefill_updates(enabled)
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    return old


def _run_batch(
    runner: Any,
    prompts: list[str],
    *,
    max_new: int,
    optin: bool,
) -> list[dict[str, Any]]:
    old = _set_prefill_optin(optin)
    rows: list[dict[str, Any]] = []
    try:
        for idx, prompt in enumerate(prompts):
            t0 = time.time()
            out = runner.generate(prompt, max_new=max_new, top_k=0, use_chat_template=True)
            elapsed = time.time() - t0
            timings = out.get("timings", {}) or {}
            rows.append({
                "index": idx,
                "prompt": prompt,
                "new_ids": out.get("new_ids", []),
                "completion_text": out.get("completion_text", ""),
                "completion_text_raw": out.get("completion_text_raw", ""),
                "stopped_reason": out.get("stopped_reason"),
                "prefill_seconds": timings.get("prefill_seconds"),
                "decode_tps": timings.get("decode_tps"),
                "elapsed_seconds": elapsed,
                "degenerate": _degenerate(out.get("completion_text", "")),
            })
    finally:
        _restore_env(old)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--prompts-json", default="")
    ap.add_argument("--preset", choices=("basic", "rc-mini"), default="basic")
    ap.add_argument("--min-release-gib", type=float, default=30.0)
    ap.add_argument("--p3b-pass", action="store_true", help="Assert P3-B selected-prefill predecessor passed.")
    args = ap.parse_args()

    if args.max_new <= 0:
        raise ValueError("--max-new must be positive")
    prompts = _prompt_preset(args.preset)
    if args.prompts_json:
        prompts = json.loads(Path(args.prompts_json).read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt list must be non-empty")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("all prompts must be non-empty strings")

    import torch  # noqa: WPS433
    from engine.resident_runner import LynnIncrementalRunner  # noqa: WPS433

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("=============== STAGE 6 PHASE 3-C RESIDENT PROMPT GATE ===============", flush=True)
    print(f"model       : {args.model}", flush=True)
    print(f"preset      : {args.preset}", flush=True)
    print(f"prompts     : {len(prompts)}", flush=True)
    print(f"max_new     : {args.max_new}", flush=True)
    print(f"p3b_pass    : {args.p3b_pass}", flush=True)

    runner = LynnIncrementalRunner(
        args.model,
        device="cuda",
        dtype=torch.bfloat16,
        max_seq_len=args.max_seq_len,
        verbose=False,
    )
    torch.cuda.synchronize()
    mem_loaded_gib = float(torch.cuda.memory_allocated() / (1024**3))

    print("[phase] baseline BF16-prefill generate", flush=True)
    baseline = _run_batch(runner, prompts, max_new=args.max_new, optin=False)

    print("[phase] release active MoE BF16 shadows only", flush=True)
    release = runner.release_decode_bf16_shadows(
        include_moe_experts=True,
        include_projection_aliases=False,
    )
    mem_after_release_gib = float(torch.cuda.memory_allocated() / (1024**3))

    print("[phase] p3a_grouped prefill opt-in no-reload generate", flush=True)
    reload_calls: list[dict[str, Any]] = []
    original_reload = runner.reload_decode_bf16_shadows

    def forbidden_reload(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        reload_calls.append({"time": time.time(), "source": "runner.reload_decode_bf16_shadows"})
        raise RuntimeError("P3-C forbids reload_decode_bf16_shadows() in candidate no-reload phase")

    runner.reload_decode_bf16_shadows = forbidden_reload  # type: ignore[method-assign]
    try:
        candidate = _run_batch(runner, prompts, max_new=args.max_new, optin=True)
    finally:
        runner.reload_decode_bf16_shadows = original_reload  # type: ignore[method-assign]

    comparisons: list[dict[str, Any]] = []
    for base, cand in zip(baseline, candidate):
        base_ids = [int(x) for x in base.get("new_ids", [])]
        cand_ids = [int(x) for x in cand.get("new_ids", [])]
        prefix_n = min(len(base_ids), len(cand_ids))
        comparisons.append({
            "index": base["index"],
            "token_exact": base_ids == cand_ids,
            "token_prefix_match": sum(1 for a, b in zip(base_ids, cand_ids) if a == b),
            "token_prefix_n": prefix_n,
            "text_prefix_200_match": _norm(base.get("completion_text", ""))[:200] == _norm(cand.get("completion_text", ""))[:200],
            "baseline_ids": base_ids,
            "candidate_ids": cand_ids,
            "baseline_text": base.get("completion_text", ""),
            "candidate_text": cand.get("completion_text", ""),
        })

    prompt_count_pass = (
        len(prompts) > 0
        and len(baseline) == len(prompts)
        and len(candidate) == len(prompts)
        and len(comparisons) == len(prompts)
    )
    functional_pass = prompt_count_pass and all(not row["degenerate"] for row in baseline + candidate)
    token_exact = prompt_count_pass and all(row["token_exact"] for row in comparisons)
    text_prefix_match = prompt_count_pass and all(row["text_prefix_200_match"] for row in comparisons)
    release_gib = float(release.get("released_gib", 0.0) or 0.0)
    release_tensors = int(release.get("released_tensors", 0) or 0)
    memory_drop_gib = mem_loaded_gib - mem_after_release_gib
    release_meaningful = release_tensors > 0 and release_gib >= args.min_release_gib
    memory_drop_meaningful = memory_drop_gib >= (args.min_release_gib * 0.5)
    reload_not_called = len(reload_calls) == 0
    all_pass = bool(
        args.p3b_pass
        and prompt_count_pass
        and functional_pass
        and token_exact
        and release_meaningful
        and memory_drop_meaningful
        and reload_not_called
    )
    result = {
        "schema": "lynn-stage6-p3c-resident-prompt-gate-v1",
        "verdict": "PASS" if all_pass else "FAIL",
        "banked_server_path": False,
        "banked_rc_quality": False,
        "model": args.model,
        "preset": args.preset,
        "max_new": args.max_new,
        "min_release_gib": args.min_release_gib,
        "env": {
            "canonical_effective": {k: os.environ.get(k) for k in sorted(CANONICAL_ENV)},
            "prefill_baseline": _prefill_updates(False),
            "prefill_candidate": _prefill_updates(True),
        },
        "memory": {
            "loaded_gib": mem_loaded_gib,
            "after_release_gib": mem_after_release_gib,
            "drop_gib": memory_drop_gib,
            "release": release,
        },
        "baseline": baseline,
        "candidate_no_reload": candidate,
        "comparisons": comparisons,
        "passes": {
            "p3b_pass": bool(args.p3b_pass),
            "prompt_count": bool(prompt_count_pass),
            "functional_non_degenerate": bool(functional_pass),
            "generated_token_exact": bool(token_exact),
            "token_exact": bool(token_exact),
            "text_prefix_200_match": bool(text_prefix_match),
            "release_meaningful": bool(release_meaningful),
            "memory_drop_meaningful": bool(memory_drop_meaningful),
            "reload_not_called": bool(reload_not_called),
            "all": all_pass,
        },
        "reload_calls": reload_calls,
        "notes": [
            "P3-C uses real prompts in the resident runner with candidate mode p3a_grouped.",
            "Projection shadows intentionally remain resident in this gate.",
            "P3-C is not a server/default/RC promotion; P3-D owns promotion.",
            "Default path remains unchanged when opt-in flags are unset.",
        ],
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passes"]["all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
