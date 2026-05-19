#!/usr/bin/env python3
"""Trace batched MTP state boundaries against canonical T=1 decode.

This probe is narrower than a full smoke. It advances two states from the same
prefill snapshot:

* baseline_state: commits tokens with canonical T=1 decode;
* spec_state: commits through ``speculative_step_k1_batched``.

After every speculative event it compares state at the committed-token boundary
and verifies the next pending token handoff. It is meant to localize multi-event
drift that a single K2-vs-two-T1 layer probe cannot see.
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
from engine.mtp_serving import decode_one_to_logits_and_hidden, speculative_step_k1_batched  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


BASE_ENV = {
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_MTP_SHADOW_VERIFY": "0",
    "LYNN_MTP_SPECULATIVE": "0",
    "LYNN_LINEAR_BLOCK_GRAPH": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    "LYNN_FULL_ATTN_K2_BACKEND": "t1_loop",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _set_env(updates: dict[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    diff = af - bf
    denom = torch.linalg.vector_norm(af).clamp_min(1e-12)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
    }


def _new_state_like(runner: LynnIncrementalRunner) -> LynnInferenceState:
    return LynnInferenceState.from_config(
        runner.cfg,
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )


def _prefill(
    runner: LynnIncrementalRunner,
    prompt: str,
) -> tuple[LynnInferenceState, int, torch.Tensor, int]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = _new_state_like(runner)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for layer_idx in range(runner.n_layers):
        h = _prefill_layer(
            h,
            pos,
            runner.layer_types[layer_idx],
            runner.layer_weights[layer_idx],
            runner.layer_cfgs[layer_idx],
            state,
            layer_idx,
        )
    state.seq_len = int(ids.shape[1])
    logits = runner._lm_head_logits(_rms_norm(h, runner.outside["model.language_model.norm.weight"]))
    next_id = int(logits[0].argmax().item())
    return state, next_id, h[:, -1:, :].contiguous(), int(ids.shape[1] - 1)


def _state_worst(
    a: LynnInferenceState,
    b: LynnInferenceState,
    *,
    include_kv: bool,
) -> dict[str, Any]:
    worst: dict[str, Any] | None = None

    def add(kind: str, layer: int, name: str, x: torch.Tensor, y: torch.Tensor) -> None:
        nonlocal worst
        c = _cmp(x, y)
        row = {"kind": kind, "layer": int(layer), "name": name, **c}
        if worst is None or row["max_abs"] > worst["max_abs"]:
            worst = row

    for layer, tensor in a.recurrent_state.items():
        add("recurrent", int(layer), "recurrent", tensor, b.recurrent_state[layer])
    for layer, tensor in a.conv_state.items():
        add("conv", int(layer), "conv", tensor, b.conv_state[layer])
    if include_kv:
        upto = min(int(a.seq_len), int(b.seq_len))
        for layer, (ka, va) in a.kv_cache.items():
            kb, vb = b.kv_cache[layer]
            add("kv", int(layer), "K", ka[:, :, :upto, :], kb[:, :, :upto, :])
            add("kv", int(layer), "V", va[:, :, :upto, :], vb[:, :, :upto, :])
    return worst or {"kind": "none", "layer": -1, "name": "none", "max_abs": 0.0, "cosine": 1.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.")
    ap.add_argument("--max-events", type=int, default=48)
    ap.add_argument("--include-kv", action="store_true")
    ap.add_argument("--drift-max-abs", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    env = dict(BASE_ENV)
    env["LYNN_MTP_SIDECAR"] = args.sidecar
    prev = _set_env(env)
    try:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
        t0 = time.time()
        runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=False)
        if not runner.mtp_sidecar_loaded:
            raise SystemExit(f"MTP sidecar did not load: {args.sidecar}")

        prefill_state, pending_id, pending_hidden, pending_pos = _prefill(runner, args.prompt)
        snap = runner._snapshot_state(prefill_state)
        baseline_state = _new_state_like(runner)
        spec_state = _new_state_like(runner)
        runner._restore_state(baseline_state, snap)
        runner._restore_state(spec_state, snap)

        baseline_pending = int(pending_id)
        spec_pending = int(pending_id)
        spec_hidden = pending_hidden
        spec_pos = int(pending_pos)
        generated: list[int] = [int(pending_id)]
        events: list[dict[str, Any]] = []
        first_bad: dict[str, Any] | None = None

        for event_idx in range(args.max_events):
            result = speculative_step_k1_batched(
                runner,
                spec_state,
                spec_pending,
                spec_hidden,
                spec_pos,
            )
            committed = [int(x) for x in result.committed_tokens]

            baseline_hidden = None
            baseline_next = baseline_pending
            for token_id in committed:
                baseline_hidden, _logits, baseline_next = decode_one_to_logits_and_hidden(
                    runner,
                    baseline_state,
                    int(token_id),
                )
            generated.extend(committed[1:] if event_idx == 0 else committed)

            worst = _state_worst(baseline_state, spec_state, include_kv=args.include_kv)
            next_pending_match = int(result.next_pending_id) == int(baseline_next)
            hidden_cmp = (
                _cmp(baseline_hidden, result.next_base_hidden)
                if baseline_hidden is not None
                else {"max_abs": 0.0, "cosine": 1.0}
            )
            row = {
                "event": event_idx,
                "accepted": bool(result.accepted),
                "committed_tokens": committed,
                "draft_id": int(result.draft_id),
                "baseline_next_pending": int(baseline_next),
                "spec_next_pending": int(result.next_pending_id),
                "next_pending_match": next_pending_match,
                "baseline_seq_len": int(baseline_state.seq_len),
                "spec_seq_len": int(spec_state.seq_len),
                "state_worst": worst,
                "next_hidden": hidden_cmp,
            }
            events.append(row)
            bad = (
                not next_pending_match
                or int(baseline_state.seq_len) != int(spec_state.seq_len)
                or worst["max_abs"] > args.drift_max_abs
                or hidden_cmp["max_abs"] > args.drift_max_abs
            )
            if bad and first_bad is None:
                first_bad = dict(row)
                break
            if int(result.next_pending_id) in runner.stop_token_ids:
                break
            baseline_pending = int(baseline_next)
            spec_pending = int(result.next_pending_id)
            spec_hidden = result.next_base_hidden
            spec_pos = int(result.next_pos)

        report = {
            "schema_version": "lynn-mtp-state-boundary-trace-v1",
            "model": args.model,
            "sidecar": args.sidecar,
            "prompt": args.prompt,
            "max_events": args.max_events,
            "include_kv": args.include_kv,
            "first_bad": first_bad,
            "events": events,
            "elapsed_seconds": time.time() - t0,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({
            "out": str(out),
            "first_bad_event": None if first_bad is None else first_bad["event"],
            "events": len(events),
            "elapsed_seconds": report["elapsed_seconds"],
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        _restore_env(prev)


if __name__ == "__main__":
    raise SystemExit(main())
