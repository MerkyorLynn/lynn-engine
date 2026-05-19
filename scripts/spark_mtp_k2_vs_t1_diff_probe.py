#!/usr/bin/env python3
"""Locate Qwen3.6 MTP K=2 verifier drift against two sequential T=1 steps.

This is a diagnostic companion to ``spark_mtp_speculative_smoke.py``.  It runs
one prompt through prefill, picks the baseline pending token and the official
MTP draft token, then compares:

* sequential verifier: pending T=1 full decode, then draft T=1 full decode;
* batched verifier: one K=2 decode over [pending, draft].

The output records per-layer hidden-state drift for both positions and final
logit drift.  It is intentionally not a promotion gate; it is a localization
probe for the remaining MTP batched correctness failure.
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
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


BASE_ENV = {
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_LINEAR_BLOCK_GRAPH": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _set_env(updates: dict[str, str]) -> dict[str, str | None]:
    prev: dict[str, str | None] = {}
    for key, value in updates.items():
        prev[key] = os.environ.get(key)
        os.environ[key] = value
    return prev


def _restore_env(prev: dict[str, str | None]) -> None:
    for key, value in prev.items():
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


def _argmax_id(logits: torch.Tensor) -> int:
    return int(logits[0].argmax().item())


def _state_cmp(seq_snap: dict[str, Any], k2_snap: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    worst: dict[str, Any] | None = None

    def add(kind: str, layer: int, name: str, a: torch.Tensor, b: torch.Tensor) -> None:
        nonlocal worst
        c = _cmp(a, b)
        row = {"kind": kind, "layer": int(layer), "name": name, **c}
        rows.append(row)
        if worst is None or row["max_abs"] > worst["max_abs"]:
            worst = row

    for layer, tensor in seq_snap.get("recurrent", {}).items():
        add("recurrent", int(layer), "recurrent", tensor, k2_snap["recurrent"][layer])
    for layer, tensor in seq_snap.get("conv", {}).items():
        add("conv", int(layer), "conv", tensor, k2_snap["conv"][layer])
    for layer, (k_seq, v_seq) in seq_snap.get("kv", {}).items():
        k_k2, v_k2 = k2_snap["kv"][layer]
        add("kv", int(layer), "K", k_seq, k_k2)
        add("kv", int(layer), "V", v_seq, v_k2)
    return {
        "worst": worst,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--drift-max-abs", type=float, default=1e-3)
    ap.add_argument("--drift-cos-min", type=float, default=0.99999)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    env = dict(BASE_ENV)
    env["LYNN_MTP_SIDECAR"] = args.sidecar
    prev = _set_env(env)
    try:
        t0 = time.time()
        runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=False)
        if not runner.mtp_sidecar_loaded:
            raise SystemExit(f"MTP sidecar did not load: {args.sidecar}")

        tok = runner.tokenizer
        ids = _encode_prompt(tok, args.prompt, runner.device, use_chat_template=False)
        T = int(ids.shape[1])
        state = LynnInferenceState.from_config(
            runner.cfg,
            batch=1,
            max_seq_len=runner.max_seq_len,
            device=runner.device,
            dtype=runner.dtype,
        )

        h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
        pos = torch.arange(T, device=runner.device, dtype=torch.long).unsqueeze(0)
        for i in range(runner.n_layers):
            h = _prefill_layer(h, pos, runner.layer_types[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        state.seq_len = T

        h_norm = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        pending_id = _argmax_id(runner._lm_head_logits(h_norm))
        draft_logits = runner._mtp_draft_logits(
            base_hidden=h[:, -1:, :],
            current_token_id=int(ids[0, -1].item()),
            current_pos=T - 1,
        )
        draft_id = _argmax_id(draft_logits)
        pre_snap = runner._snapshot_state(state)

        # Sequential T=1 + T=1 verifier.
        runner._restore_state(state, pre_snap)
        token0 = torch.tensor([[pending_id]], device=runner.device, dtype=torch.long)
        pos0 = torch.tensor([[T]], device=runner.device, dtype=torch.long)
        h0 = F.embedding(token0, runner.outside["model.language_model.embed_tokens.weight"])
        seq0_by_layer: list[torch.Tensor] = []
        for i in range(runner.n_layers):
            h0 = runner._decode_layer_fast(h0, pos0, state, i)
            seq0_by_layer.append(h0.detach().clone())
        state.seq_len += 1

        token1 = torch.tensor([[draft_id]], device=runner.device, dtype=torch.long)
        pos1 = torch.tensor([[T + 1]], device=runner.device, dtype=torch.long)
        h1 = F.embedding(token1, runner.outside["model.language_model.embed_tokens.weight"])
        seq1_by_layer: list[torch.Tensor] = []
        for i in range(runner.n_layers):
            h1 = runner._decode_layer_fast(h1, pos1, state, i)
            seq1_by_layer.append(h1.detach().clone())
        state.seq_len += 1
        seq_h = torch.cat([h0, h1], dim=1)
        seq_logits = runner._lm_head_logits(
            _rms_norm(seq_h, runner.outside["model.language_model.norm.weight"]),
            all_positions=True,
        )
        seq_snap = runner._snapshot_state(state)

        # Batched K=2 verifier.
        runner._restore_state(state, pre_snap)
        tokens_k2 = torch.tensor([[pending_id, draft_id]], device=runner.device, dtype=torch.long)
        pos_k2 = torch.tensor([[T, T + 1]], device=runner.device, dtype=torch.long)
        hk2 = F.embedding(tokens_k2, runner.outside["model.language_model.embed_tokens.weight"])
        layer_rows: list[dict[str, Any]] = []
        first_bad: dict[str, Any] | None = None
        for i in range(runner.n_layers):
            hk2 = runner._decode_layer_k2_fast(hk2, pos_k2, state, i)
            c0 = _cmp(seq0_by_layer[i], hk2[:, 0:1, :])
            c1 = _cmp(seq1_by_layer[i], hk2[:, 1:2, :])
            row = {
                "layer": i,
                "layer_type": runner.layer_types[i],
                "pos0": c0,
                "pos1": c1,
            }
            layer_rows.append(row)
            if first_bad is None:
                bad0 = c0["max_abs"] > args.drift_max_abs or c0["cosine"] < args.drift_cos_min
                bad1 = c1["max_abs"] > args.drift_max_abs or c1["cosine"] < args.drift_cos_min
                if bad0 or bad1:
                    first_bad = {
                        "layer": i,
                        "layer_type": runner.layer_types[i],
                        "pos0_bad": bad0,
                        "pos1_bad": bad1,
                        "pos0": c0,
                        "pos1": c1,
                    }
        state.seq_len += 2
        k2_logits = runner._lm_head_logits(
            _rms_norm(hk2, runner.outside["model.language_model.norm.weight"]),
            all_positions=True,
        )
        k2_snap = runner._snapshot_state(state)
        state_diff = _state_cmp(seq_snap, k2_snap)

        report = {
            "schema_version": "lynn-mtp-k2-vs-two-t1-diff-probe-v1",
            "model": args.model,
            "sidecar": args.sidecar,
            "prompt": args.prompt,
            "prompt_len": T,
            "pending_id": pending_id,
            "pending_text": tok.decode([pending_id]),
            "draft_id": draft_id,
            "draft_text": tok.decode([draft_id]),
            "seq_argmax_pos0": int(seq_logits[0, 0].argmax().item()),
            "seq_argmax_pos1": int(seq_logits[0, 1].argmax().item()),
            "k2_argmax_pos0": int(k2_logits[0, 0].argmax().item()),
            "k2_argmax_pos1": int(k2_logits[0, 1].argmax().item()),
            "logits_pos0": _cmp(seq_logits[:, 0, :], k2_logits[:, 0, :]),
            "logits_pos1": _cmp(seq_logits[:, 1, :], k2_logits[:, 1, :]),
            "state_diff": state_diff,
            "first_bad_layer": first_bad,
            "layers": layer_rows,
            "elapsed_seconds": time.time() - t0,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[mtp-k2-diff] wrote {out}")
        if first_bad:
            print("[mtp-k2-diff] first_bad_layer", json.dumps(first_bad, ensure_ascii=False))
        else:
            print("[mtp-k2-diff] all layers within threshold")
        return 0
    finally:
        _restore_env(prev)


if __name__ == "__main__":
    raise SystemExit(main())
