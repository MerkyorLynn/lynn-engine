#!/usr/bin/env python3
"""Trace MTP batched speculative commits against greedy baseline.

This is a narrow post-M13 diagnostic.  The layer-level K2-vs-two-T1 diff probe
can pass while end-to-end speculative generation still diverges, because the
remaining bug may live in commit/reject state updates or next-pending handoff.
This script runs one prompt and compares the generated token prefix after each
batched speculative event.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LynnInferenceState  # noqa: E402
from engine.mtp_serving import speculative_step_k1_batched  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from engine.nvfp4_layout import detect_nvfp4_layout  # noqa: E402


BASE_ENV = {
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_MTP_SHADOW_VERIFY": "0",
    "LYNN_MTP_SPECULATIVE": "0",
    "LYNN_LINEAR_BLOCK_GRAPH": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _set_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _prefix_match_len(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _tail(ids: list[int], idx: int, radius: int = 8) -> list[int]:
    lo = max(0, idx - radius)
    hi = min(len(ids), idx + radius)
    return ids[lo:hi]


def _prefill_runner(
    runner: LynnIncrementalRunner,
    prompt: str,
) -> tuple[torch.Tensor, LynnInferenceState, int, int, torch.Tensor]:
    tok = runner.tokenizer
    ids = _encode_prompt(tok, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState.from_config(
        runner.cfg,
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(
            h,
            pos,
            runner.layer_types[i],
            runner.layer_weights[i],
            runner.layer_cfgs[i],
            state,
            i,
        )
    state.seq_len = int(ids.shape[1])
    logits = runner._lm_head_logits(_rms_norm(h, runner.outside["model.language_model.norm.weight"]))
    next_id = int(logits[0].argmax().item())
    return ids, state, next_id, int(ids.shape[1] - 1), h[:, -1:, :].contiguous()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    env = dict(BASE_ENV)
    env["LYNN_MTP_SIDECAR"] = args.sidecar
    previous = _set_env(env)
    try:
        layout = detect_nvfp4_layout(args.model)
        if layout.layout_kind != "lynn_native_per16_variable":
            raise SystemExit(f"expected Lynn-native NVFP4 model, got {layout.layout_kind}")

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
        runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=False)
        baseline = runner.generate(args.prompt, max_new=args.max_new)
        baseline_ids = [int(x) for x in baseline["new_ids"]]

        _ids, state, pending_id, pending_pos, pending_hidden = _prefill_runner(runner, args.prompt)
        generated = [int(pending_id)]
        events: list[dict[str, Any]] = []
        first_divergence: dict[str, Any] | None = None
        first_step = True
        while len(generated) < args.max_new and pending_id not in runner.stop_token_ids:
            t0 = time.time()
            result = speculative_step_k1_batched(
                runner,
                state,
                int(pending_id),
                pending_hidden,
                int(pending_pos),
            )
            if runner.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            emit_from = 1 if first_step else 0
            emitted: list[int] = []
            for token_id in result.committed_tokens[emit_from:]:
                if len(generated) >= args.max_new:
                    break
                generated.append(int(token_id))
                emitted.append(int(token_id))
                if int(token_id) in runner.stop_token_ids:
                    break
            first_step = False
            pref = _prefix_match_len(generated, baseline_ids)
            event = {
                "event": len(events),
                "elapsed_seconds": elapsed,
                "state_seq_len": int(state.seq_len),
                "accepted": bool(result.accepted),
                "draft_id": int(result.draft_id),
                "draft_text": runner.tokenizer.decode([int(result.draft_id)]),
                "committed_tokens": [int(x) for x in result.committed_tokens],
                "emitted_tokens": emitted,
                "next_pending_id": int(result.next_pending_id),
                "next_pending_text": runner.tokenizer.decode([int(result.next_pending_id)]),
                "next_pos": int(result.next_pos),
                "generated_len": len(generated),
                "prefix_match_len": pref,
            }
            if pref < len(generated) and first_divergence is None:
                event["first_divergence_here"] = True
                first_divergence = {
                    "event": len(events),
                    "prefix_match_len": pref,
                    "baseline_tail": _tail(baseline_ids, pref),
                    "generated_tail": _tail(generated, pref),
                    "baseline_tail_text": runner.tokenizer.decode(_tail(baseline_ids, pref)),
                    "generated_tail_text": runner.tokenizer.decode(_tail(generated, pref)),
                    "event_snapshot": dict(event),
                }
                events.append(event)
                break
            events.append(event)
            if len(generated) >= args.max_new or (generated and generated[-1] in runner.stop_token_ids):
                break
            pending_id = int(result.next_pending_id)
            pending_hidden = result.next_base_hidden
            pending_pos = int(result.next_pos)

        report = {
            "schema_version": "lynn-mtp-batched-commit-trace-v1",
            "model": args.model,
            "sidecar": args.sidecar,
            "prompt": args.prompt,
            "max_new": args.max_new,
            "baseline_ids": baseline_ids,
            "generated_ids": generated,
            "exact_match": generated == baseline_ids[: len(generated)] and len(generated) == len(baseline_ids),
            "prefix_match_len": _prefix_match_len(generated, baseline_ids),
            "baseline_head": baseline.get("completion_text", "")[:240],
            "generated_head": runner.tokenizer.decode(generated, skip_special_tokens=True)[:240],
            "first_divergence": first_divergence,
            "events": events,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "out": str(out),
            "exact_match": report["exact_match"],
            "prefix_match_len": report["prefix_match_len"],
            "first_divergence": first_divergence,
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        _restore_env(previous)


if __name__ == "__main__":
    raise SystemExit(main())
