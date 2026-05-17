#!/usr/bin/env python3
"""P118: Python-only MTP verify/commit state-parity probe.

This freezes the K=2 verifier ABI before native code exists.  It uses cloned
LynnInferenceState objects as the scratch buffer, then checks that accept and
reject commits leave the canonical state identical to a direct base decode path.

The probe is intentionally independent from any MTP sidecar.  Draft tokens are
chosen synthetically:

- accept case: draft == base argmax after the pending token
- reject case: draft != base argmax after the pending token

That isolates the verifier state contract from proposer quality.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _load_prompt_specs(path: str | None, inline: list[str]) -> list[dict[str, str]]:
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        raw = inline
    specs: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            specs.append({"id": str(idx), "prompt": item})
        elif isinstance(item, dict):
            specs.append({"id": str(item.get("id", idx)), "prompt": str(item["prompt"])})
        else:
            raise TypeError(f"prompt spec must be string or object, got {type(item)}")
    return specs


def _new_state(runner: LynnIncrementalRunner) -> LynnInferenceState:
    return LynnInferenceState(
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )


def _restore_to_new_state(
    runner: LynnIncrementalRunner,
    snap: dict[str, Any],
) -> LynnInferenceState:
    state = _new_state(runner)
    runner._restore_state(state, snap)
    return state


@torch.no_grad()
def _prefill_prompt(
    runner: LynnIncrementalRunner,
    prompt: str,
    *,
    use_chat_template: bool,
) -> tuple[LynnInferenceState, int, int]:
    ids = _encode_prompt(
        runner.tokenizer,
        prompt,
        runner.device,
        use_chat_template=use_chat_template,
    )
    state = _new_state(runner)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for layer_idx in range(runner.n_layers):
        h = _prefill_layer(
            h,
            pos,
            LAYER_TYPES[layer_idx],
            runner.layer_weights[layer_idx],
            runner.layer_cfgs[layer_idx],
            state,
            layer_idx,
        )
    state.seq_len = int(ids.shape[1])
    logits = runner._lm_head_logits(
        _rms_norm(h[:, -1:, :], runner.outside["model.language_model.norm.weight"])
    )
    pending_id = int(logits[0].argmax().item())
    return state, pending_id, int(ids.numel())


@torch.no_grad()
def _decode_one(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    token_id: int,
) -> dict[str, Any]:
    token = torch.tensor([[int(token_id)]], device=runner.device, dtype=torch.long)
    pos_id = int(state.seq_len)
    pos = torch.tensor([[pos_id]], device=runner.device, dtype=torch.long)
    h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    for layer_idx in range(runner.n_layers):
        h = runner._decode_layer_fast(h, pos, state, layer_idx)
    state.seq_len += 1
    logits = runner._lm_head_logits(_rms_norm(h, runner.outside["model.language_model.norm.weight"]))
    argmax_id = int(logits[0].argmax().item())
    return {
        "input_id": int(token_id),
        "position": pos_id,
        "argmax_id": argmax_id,
        "argmax_text": runner.tokenizer.decode([argmax_id]),
        "logits": logits,
    }


def _choose_reject_token(logits: torch.Tensor, accept_id: int) -> int:
    top = torch.topk(logits[0], k=min(8, logits.shape[-1])).indices.tolist()
    for token_id in top:
        if int(token_id) != int(accept_id):
            return int(token_id)
    return int((accept_id + 1) % logits.shape[-1])


def _verify_tokens_python(
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
        snapshots.append(runner._snapshot_state(state))
        argmax_after.append(int(row["argmax_id"]))
        argmax_text_after.append(str(row["argmax_text"]))
        positions.append(int(row["position"]))
    return {
        "verify_tokens": [int(x) for x in verify_tokens],
        "positions": positions,
        "argmax_after": argmax_after,
        "argmax_text_after": argmax_text_after,
        "snapshots": snapshots,
    }


def _commit_verify_python(
    state: LynnInferenceState,
    scratch: dict[str, Any],
    *,
    commit_count: int,
) -> None:
    if commit_count < 1 or commit_count > len(scratch["snapshots"]):
        raise ValueError(f"invalid commit_count={commit_count}")
    commit_snap = scratch["snapshots"][commit_count - 1]

    # Full-attention KV positions beyond seq_len may stay stale after reject.
    # Recurrent and conv states must roll back to the selected intermediate.
    state.seq_len = int(commit_snap["seq_len"])
    for layer_idx, tensor in commit_snap["recurrent"].items():
        state.recurrent_state[layer_idx].copy_(tensor)
    for layer_idx, tensor in commit_snap["conv"].items():
        state.conv_state[layer_idx].copy_(tensor)


def _state_diffs(
    expected: LynnInferenceState,
    observed: LynnInferenceState,
) -> dict[str, Any]:
    if int(expected.seq_len) != int(observed.seq_len):
        return {
            "seq_len_match": False,
            "expected_seq_len": int(expected.seq_len),
            "observed_seq_len": int(observed.seq_len),
            "max_kv_abs": None,
            "max_recurrent_abs": None,
            "max_conv_abs": None,
        }

    seq_len = int(expected.seq_len)
    max_kv = 0.0
    for layer_idx, (exp_k, exp_v) in expected.kv_cache.items():
        obs_k, obs_v = observed.kv_cache[layer_idx]
        max_kv = max(
            max_kv,
            float((exp_k[:, :, :seq_len, :].float() - obs_k[:, :, :seq_len, :].float()).abs().max().item()),
            float((exp_v[:, :, :seq_len, :].float() - obs_v[:, :, :seq_len, :].float()).abs().max().item()),
        )

    max_recurrent = 0.0
    for layer_idx, exp_tensor in expected.recurrent_state.items():
        obs_tensor = observed.recurrent_state[layer_idx]
        max_recurrent = max(
            max_recurrent,
            float((exp_tensor.float() - obs_tensor.float()).abs().max().item()),
        )

    max_conv = 0.0
    for layer_idx, exp_tensor in expected.conv_state.items():
        obs_tensor = observed.conv_state[layer_idx]
        max_conv = max(
            max_conv,
            float((exp_tensor.float() - obs_tensor.float()).abs().max().item()),
        )

    return {
        "seq_len_match": True,
        "expected_seq_len": seq_len,
        "observed_seq_len": int(observed.seq_len),
        "max_kv_abs": max_kv,
        "max_recurrent_abs": max_recurrent,
        "max_conv_abs": max_conv,
    }


def _diffs_pass(diffs: dict[str, Any], tolerance: float) -> bool:
    if not diffs["seq_len_match"]:
        return False
    return all(
        float(diffs[key]) <= tolerance
        for key in ("max_kv_abs", "max_recurrent_abs", "max_conv_abs")
    )


def _diff_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        for case in ("accept_case", "reject_case"):
            value = row[case]["diffs"][key]
            if value is not None:
                values.append(float(value))
    return values


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
        accept_scratch = _verify_tokens_python(
            runner,
            verify_accept_state,
            [pending_id, accept_draft],
        )
        accept_commit_count = 2 if accept_draft == accept_scratch["argmax_after"][0] else 1
        _commit_verify_python(verify_accept_state, accept_scratch, commit_count=accept_commit_count)
        accept_diffs = _state_diffs(expected_accept_state, verify_accept_state)

        expected_reject_state = _restore_to_new_state(runner, after_x_snap)
        verify_reject_state = _restore_to_new_state(runner, before)
        reject_scratch = _verify_tokens_python(
            runner,
            verify_reject_state,
            [pending_id, reject_draft],
        )
        reject_commit_count = 2 if reject_draft == reject_scratch["argmax_after"][0] else 1
        _commit_verify_python(verify_reject_state, reject_scratch, commit_count=reject_commit_count)
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
            },
            "reject_case": {
                "draft_id": reject_draft,
                "draft_text": runner.tokenizer.decode([reject_draft]),
                "argmax_after": reject_scratch["argmax_after"],
                "commit_count": reject_commit_count,
                "passed": reject_commit_count == 1 and _diffs_pass(reject_diffs, tolerance),
                "diffs": reject_diffs,
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
    return {
        "id": prompt_id,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "events": len(rows),
        "passed": all_passed,
        "max_diffs": max_diffs,
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
    result = {
        "schema_version": "lynn-p118-mtp-verify-state-parity-v1",
        "decision": (
            "GREEN: Python K=2 verify accept/reject commit semantics match direct base decode."
            if total_events and passed_events == total_events
            else "RED: Python K=2 verify state parity failed or produced no events."
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
