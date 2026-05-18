#!/usr/bin/env python3
"""P119: MTP verify parity with in-place KV and linear-state scratch.

P118 froze K=2 accept/reject semantics with full cloned states.  P119 moves one
step closer to the production verifier contract: verify tokens are decoded
in-place, full-attention KV is not restored after reject, and only linear
recurrent/conv intermediates plus seq_len are committed.

This is still Python, but it validates the key native assumption:

  stale full-attention KV beyond seq_len is harmless because future reads are
  clipped and future writes overwrite the rejected draft position.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.p118_mtp_verify_state_parity import (  # noqa: E402
    _choose_reject_token,
    _decode_one,
    _diff_values,
    _diffs_pass,
    _load_prompt_specs,
    _prefill_prompt,
    _restore_to_new_state,
    _state_diffs,
)
from engine.inference_state import LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _linear_scratch_bytes(row: dict[str, Any]) -> int:
    total = 0
    for tensor in row["recurrent"].values():
        total += tensor.element_size() * tensor.numel()
    for tensor in row["conv"].values():
        total += tensor.element_size() * tensor.numel()
    return total


@torch.no_grad()
def _verify_tokens_inplace_linear_scratch(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    verify_tokens: list[int],
) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []
    argmax_after: list[int] = []
    argmax_text_after: list[str] = []
    positions: list[int] = []
    for token_id in verify_tokens:
        row = _decode_one(runner, state, token_id)
        snapshots.append(
            {
                "seq_len": int(state.seq_len),
                "recurrent": {i: t.clone() for i, t in state.recurrent_state.items()},
                "conv": {i: t.clone() for i, t in state.conv_state.items()},
            }
        )
        argmax_after.append(int(row["argmax_id"]))
        argmax_text_after.append(str(row["argmax_text"]))
        positions.append(int(row["position"]))
    return {
        "verify_tokens": [int(x) for x in verify_tokens],
        "positions": positions,
        "argmax_after": argmax_after,
        "argmax_text_after": argmax_text_after,
        "snapshots": snapshots,
        "scratch_bytes_per_token": [
            _linear_scratch_bytes(row)
            for row in snapshots
        ],
    }


def _commit_linear_scratch(
    state: LynnInferenceState,
    scratch: dict[str, Any],
    *,
    commit_count: int,
) -> None:
    if commit_count < 1 or commit_count > len(scratch["snapshots"]):
        raise ValueError(f"invalid commit_count={commit_count}")
    commit_snap = scratch["snapshots"][commit_count - 1]

    # Deliberately do not restore full-attention KV.  The verifier contract
    # relies on seq_len clipping and later overwrites for rejected positions.
    state.seq_len = int(commit_snap["seq_len"])
    for layer_idx, tensor in commit_snap["recurrent"].items():
        state.recurrent_state[layer_idx].copy_(tensor)
    for layer_idx, tensor in commit_snap["conv"].items():
        state.conv_state[layer_idx].copy_(tensor)


@torch.no_grad()
def _probe_prompt(
    *,
    runner: LynnIncrementalRunner,
    prompt_id: str,
    prompt: str,
    use_chat_template: bool,
    max_events: int,
    tolerance: float,
) -> dict[str, Any]:
    canonical_state, pending_id, prompt_tokens = _prefill_prompt(
        runner,
        prompt,
        use_chat_template=use_chat_template,
    )

    rows: list[dict[str, Any]] = []
    for event in range(max_events):
        if pending_id in runner.stop_token_ids:
            break
        before = runner._snapshot_state(canonical_state)

        after_x_state = _restore_to_new_state(runner, before)
        x_row = _decode_one(runner, after_x_state, pending_id)
        after_x_snap = runner._snapshot_state(after_x_state)
        accept_draft = int(x_row["argmax_id"])
        reject_draft = _choose_reject_token(x_row["logits"], accept_draft)

        expected_accept_state = _restore_to_new_state(runner, before)
        _decode_one(runner, expected_accept_state, pending_id)
        accept_second = _decode_one(runner, expected_accept_state, accept_draft)

        verify_accept_state = _restore_to_new_state(runner, before)
        accept_scratch = _verify_tokens_inplace_linear_scratch(
            runner,
            verify_accept_state,
            [pending_id, accept_draft],
        )
        accept_commit_count = 2 if accept_draft == accept_scratch["argmax_after"][0] else 1
        _commit_linear_scratch(
            verify_accept_state,
            accept_scratch,
            commit_count=accept_commit_count,
        )
        accept_diffs = _state_diffs(expected_accept_state, verify_accept_state)

        expected_reject_state = _restore_to_new_state(runner, after_x_snap)
        verify_reject_state = _restore_to_new_state(runner, before)
        reject_scratch = _verify_tokens_inplace_linear_scratch(
            runner,
            verify_reject_state,
            [pending_id, reject_draft],
        )
        reject_commit_count = 2 if reject_draft == reject_scratch["argmax_after"][0] else 1
        _commit_linear_scratch(
            verify_reject_state,
            reject_scratch,
            commit_count=reject_commit_count,
        )
        reject_diffs = _state_diffs(expected_reject_state, verify_reject_state)

        row = {
            "event": event,
            "position_start": int(before["seq_len"]),
            "pending_id": int(pending_id),
            "pending_text": runner.tokenizer.decode([int(pending_id)]),
            "base_after_pending_id": accept_draft,
            "base_after_pending_text": runner.tokenizer.decode([accept_draft]),
            "accept_case": {
                "draft_id": accept_draft,
                "draft_text": runner.tokenizer.decode([accept_draft]),
                "argmax_after": accept_scratch["argmax_after"],
                "commit_count": accept_commit_count,
                "expected_next_id": int(accept_second["argmax_id"]),
                "passed": accept_commit_count == 2 and _diffs_pass(accept_diffs, tolerance),
                "diffs": accept_diffs,
                "scratch_bytes_per_token": accept_scratch["scratch_bytes_per_token"],
            },
            "reject_case": {
                "draft_id": reject_draft,
                "draft_text": runner.tokenizer.decode([reject_draft]),
                "argmax_after": reject_scratch["argmax_after"],
                "commit_count": reject_commit_count,
                "passed": reject_commit_count == 1 and _diffs_pass(reject_diffs, tolerance),
                "diffs": reject_diffs,
                "scratch_bytes_per_token": reject_scratch["scratch_bytes_per_token"],
            },
        }
        rows.append(row)

        canonical_state = _restore_to_new_state(runner, after_x_snap)
        pending_id = accept_draft

    all_passed = all(
        bool(row["accept_case"]["passed"]) and bool(row["reject_case"]["passed"])
        for row in rows
    )
    kv_diffs = _diff_values(rows, "max_kv_abs")
    recurrent_diffs = _diff_values(rows, "max_recurrent_abs")
    conv_diffs = _diff_values(rows, "max_conv_abs")
    max_diffs = {
        "kv": max(kv_diffs) if kv_diffs else None,
        "recurrent": max(recurrent_diffs) if recurrent_diffs else None,
        "conv": max(conv_diffs) if conv_diffs else None,
    }
    scratch_bytes = [
        int(value)
        for row in rows
        for case in ("accept_case", "reject_case")
        for value in row[case]["scratch_bytes_per_token"]
    ]
    return {
        "id": prompt_id,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "events": len(rows),
        "passed": all_passed,
        "max_diffs": max_diffs,
        "scratch_bytes_per_token": {
            "min": min(scratch_bytes) if scratch_bytes else None,
            "max": max(scratch_bytes) if scratch_bytes else None,
            "mean": statistics.fmean(scratch_bytes) if scratch_bytes else None,
        },
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-file")
    ap.add_argument(
        "--prompts",
        nargs="*",
        default=[
            "Return one JSON object with keys city and unit for Berlin in metric units. No markdown.",
            "Write a Python function is_palindrome(s) that returns a boolean. Output code only.",
            "用一句中文短句说明 MoE router 的作用。必须以 router 开头。",
        ],
    )
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--max-events", type=int, default=8)
    ap.add_argument("--tolerance", type=float, default=0.0)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompt_specs(args.prompts_file, args.prompts)
    runner = LynnIncrementalRunner(
        args.base_model,
        device=args.device,
        dtype=dtype,
        max_seq_len=args.max_seq_len,
        verbose=True,
    )

    prompt_reports = [
        _probe_prompt(
            runner=runner,
            prompt_id=spec["id"],
            prompt=spec["prompt"],
            use_chat_template=args.use_chat_template,
            max_events=args.max_events,
            tolerance=args.tolerance,
        )
        for spec in specs
    ]
    total_events = sum(int(row["events"]) for row in prompt_reports)
    passed_events = sum(
        1
        for prompt_report in prompt_reports
        for row in prompt_report["rows"]
        if row["accept_case"]["passed"] and row["reject_case"]["passed"]
    )
    all_diffs = [
        float(prompt_report["max_diffs"][key])
        for prompt_report in prompt_reports
        for key in ("kv", "recurrent", "conv")
        if prompt_report["max_diffs"][key] is not None
    ]
    scratch_bytes = [
        float(prompt_report["scratch_bytes_per_token"][key])
        for prompt_report in prompt_reports
        for key in ("min", "max", "mean")
        if prompt_report["scratch_bytes_per_token"][key] is not None
    ]
    result = {
        "schema_version": "lynn-p119-mtp-inplace-scratch-parity-v1",
        "decision": (
            "GREEN: K=2 in-place KV plus linear-state scratch matches direct base decode."
            if total_events and passed_events == total_events
            else "RED: K=2 in-place scratch parity failed or produced no events."
        ),
        "base_model": args.base_model,
        "use_chat_template": args.use_chat_template,
        "dtype": args.dtype,
        "max_seq_len": args.max_seq_len,
        "max_events_per_prompt": args.max_events,
        "tolerance": args.tolerance,
        "summary": {
            "prompt_count": len(prompt_reports),
            "events": total_events,
            "passed_events": passed_events,
            "pass_rate": passed_events / total_events if total_events else 0.0,
            "mean_prompt_events": statistics.fmean([p["events"] for p in prompt_reports])
            if prompt_reports
            else 0.0,
            "max_abs_diff": max(all_diffs) if all_diffs else None,
            "scratch_bytes_per_token_min": min(scratch_bytes) if scratch_bytes else None,
            "scratch_bytes_per_token_max": max(scratch_bytes) if scratch_bytes else None,
        },
        "prompts": prompt_reports,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
