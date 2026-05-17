#!/usr/bin/env python3
"""P11 smoke: release BF16 shadows after prefill, then decode from packed aliases.

This is a session-scoped experiment, not a default serving mode. It proves that
once prefill has populated KV/recurrent state, decode can continue after the
large MoE/projection BF16 shadows are removed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _decode_layer, _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


REQUIRED_ENV = {
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_LINEAR_ATTN_GQA_RECURRENT": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
}


def _memory() -> dict[str, float]:
    torch.cuda.synchronize()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "max_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
    }


def _prefill(runner: LynnIncrementalRunner, prompt: str) -> tuple[int, LynnInferenceState]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = ids.shape[1]
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_final)
    return int(logits[0].argmax().item()), state


def _decode_ids(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    first_id: int,
    max_new: int,
) -> tuple[list[int], dict[str, float]]:
    ids = [int(first_id)]
    token_id = int(first_id)
    token = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    pos_tensor = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    step_seconds: list[float] = []
    for _ in range(1, max_new):
        t0 = time.time()
        token.fill_(token_id)
        pos_tensor.fill_(int(state.seq_len))
        h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        state.seq_len += 1
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        logits = runner._lm_head_logits(h_final)
        token_id = int(logits[0].argmax().item())
        ids.append(token_id)
        torch.cuda.synchronize()
        step_seconds.append(time.time() - t0)
    total = sum(step_seconds)
    return ids, {
        "decode_steps": len(step_seconds),
        "decode_seconds": total,
        "decode_tps": (len(step_seconds) / total) if total > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    env = {key: os.environ.get(key) for key in sorted(REQUIRED_ENV)}
    env_ok = all(os.environ.get(key) == value for key, value in REQUIRED_ENV.items())
    if not env_ok:
        raise RuntimeError(f"P11 smoke requires env {REQUIRED_ENV}, got {env}")

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    first_id, state = _prefill(runner, args.prompt)
    prefill_snap = runner._snapshot_state(state)
    memory_before_release = _memory()

    baseline_ids, baseline_timing = _decode_ids(runner, state, first_id, args.max_new)
    runner._restore_state(state, prefill_snap)
    release = runner.release_decode_bf16_shadows()
    memory_after_release = _memory()
    released_ids, released_timing = _decode_ids(runner, state, first_id, args.max_new)
    memory_after_decode = _memory()

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p11-decode-shadow-release-smoke-v1",
        "model": args.model,
        "prompt": args.prompt,
        "max_new": args.max_new,
        "required_env": REQUIRED_ENV,
        "env": env,
        "baseline_ids": baseline_ids,
        "released_ids": released_ids,
        "baseline_timing": baseline_timing,
        "released_timing": released_timing,
        "same_ids": baseline_ids == released_ids,
        "release": release,
        "memory_before_release": memory_before_release,
        "memory_after_release": memory_after_release,
        "memory_after_decode": memory_after_decode,
        "allocated_drop_gib": memory_before_release["allocated_gib"] - memory_after_release["allocated_gib"],
        "reserved_drop_gib": memory_before_release["reserved_gib"] - memory_after_release["reserved_gib"],
        "pass": baseline_ids == released_ids and release["released_gib"] > 50.0,
        "note": "Session-scoped proof only: prefill still needs BF16 shadows in the current default runner.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
