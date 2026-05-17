#!/usr/bin/env python3
"""P33: first-divergence probe for the native active-MoE backend.

P32 split the P31 signal into two facts:
  - cuda_scalar + reusable linear-block CUDA graphs can silently replay token 0.
  - cuda_scalar without graphs is coherent but not greedy-identical to Triton.

This probe feeds both backends the same reference tokens and compares layer
outputs plus logits after each decode step. It is intentionally diagnostic, not
a speed benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
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


COMMON_ENV = {
    "LYNN_PREFILL_WARMUP": "1",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_LINEAR_ATTN_GQA_RECURRENT": "1",
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
    "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_LINEAR_STATE_UPDATE": "inplace",
    "LYNN_LINEAR_BLOCK_GRAPH": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "0",
    "LYNN_PACKED_DECODE": "0",
    "LYNN_PACKED_DECODE_PREPARE_NATIVE": "0",
    "LYNN_PACKED_SHARED_EXPERT": "0",
    "LYNN_NATIVE_ACTIVE_MOE_LAYERS": "",
}


def _set_common_env() -> dict[str, str | None]:
    old = {k: os.environ.get(k) for k in COMMON_ENV}
    os.environ.update(COMMON_ENV)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _diff(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    rf = ref.float().reshape(-1)
    of = out.float().reshape(-1)
    delta = of - rf
    denom = torch.linalg.vector_norm(rf).clamp_min(1e-20)
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(delta) / denom).item()),
        "cosine": float(F.cosine_similarity(rf, of, dim=0).item()),
    }


def _topk(logits: torch.Tensor, k: int) -> dict[str, Any]:
    values, indices = torch.topk(logits[0].float(), k=k)
    return {
        "ids": [int(x) for x in indices.tolist()],
        "values": [float(x) for x in values.tolist()],
        "margin": float((values[0] - values[1]).item()) if k >= 2 else None,
    }


def _restore_to_new_state(runner: LynnIncrementalRunner, snap: dict[str, Any]) -> LynnInferenceState:
    state = LynnInferenceState(
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )
    runner._restore_state(state, snap)
    return state


def _decode_step(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    token_id: int,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    old = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_BACKEND")
    os.environ["LYNN_NATIVE_ACTIVE_MOE_BACKEND"] = backend
    try:
        token = torch.tensor([[int(token_id)]], device=runner.device, dtype=torch.long)
        pos = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
        h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        layer_hiddens: list[dict[str, Any]] = []
        for layer_idx in range(runner.n_layers):
            h = _decode_layer(
                h,
                pos,
                LAYER_TYPES[layer_idx],
                runner.layer_weights[layer_idx],
                runner.layer_cfgs[layer_idx],
                state,
                layer_idx,
            )
            layer_hiddens.append({"layer": layer_idx, "hidden": h.detach().clone()})
        state.seq_len += 1
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        logits = runner._lm_head_logits(h_final)
        return h.detach().clone(), logits.detach().clone(), layer_hiddens
    finally:
        if old is None:
            os.environ.pop("LYNN_NATIVE_ACTIVE_MOE_BACKEND", None)
        else:
            os.environ["LYNN_NATIVE_ACTIVE_MOE_BACKEND"] = old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Python 写一个递归阶乘函数")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--layer-cos-threshold", type=float, default=0.999999)
    args = ap.parse_args()

    old_env = _set_common_env()
    try:
        runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
        ids = _encode_prompt(runner.tokenizer, args.prompt, runner.device, use_chat_template=False)
        state0 = LynnInferenceState(
            batch=1,
            max_seq_len=runner.max_seq_len,
            device=runner.device,
            dtype=runner.dtype,
        )
        h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
        pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
        for layer_idx in range(runner.n_layers):
            h = _prefill_layer(
                h,
                pos,
                LAYER_TYPES[layer_idx],
                runner.layer_weights[layer_idx],
                runner.layer_cfgs[layer_idx],
                state0,
                layer_idx,
            )
        state0.seq_len = int(ids.shape[1])
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        logits0 = runner._lm_head_logits(h_final)
        next_id = int(logits0[0].argmax().item())
        snap = runner._snapshot_state(state0)

        state_triton = _restore_to_new_state(runner, snap)
        state_cuda = _restore_to_new_state(runner, snap)
        rows: list[dict[str, Any]] = []
        first_top1_divergence: dict[str, Any] | None = None
        first_layer_divergence: dict[str, Any] | None = None

        for step in range(args.steps):
            h_t, logits_t, layers_t = _decode_step(runner, state_triton, next_id, "triton")
            h_c, logits_c, layers_c = _decode_step(runner, state_cuda, next_id, "cuda_scalar")
            logits_diff = _diff(logits_t, logits_c)
            top_t = _topk(logits_t, args.topk)
            top_c = _topk(logits_c, args.topk)
            layer_rows = []
            step_first_layer = None
            for lt, lc in zip(layers_t, layers_c, strict=True):
                d = _diff(lt["hidden"], lc["hidden"])
                compact = {
                    "layer": int(lt["layer"]),
                    "cosine": d["cosine"],
                    "max_abs": d["max_abs"],
                    "rel_l2": d["rel_l2"],
                }
                layer_rows.append(compact)
                if step_first_layer is None and d["cosine"] < args.layer_cos_threshold:
                    step_first_layer = compact
                    if first_layer_divergence is None:
                        first_layer_divergence = {"step": step, **compact}

            row = {
                "step": step,
                "input_token_id": int(next_id),
                "triton_topk": top_t,
                "cuda_scalar_topk": top_c,
                "top1_match": top_t["ids"][0] == top_c["ids"][0],
                "logits_diff": logits_diff,
                "first_layer_below_threshold": step_first_layer,
                "layers": layer_rows,
            }
            rows.append(row)
            if first_top1_divergence is None and not row["top1_match"]:
                first_top1_divergence = {
                    "step": step,
                    "triton_top1": top_t["ids"][0],
                    "cuda_scalar_top1": top_c["ids"][0],
                    "triton_margin": top_t["margin"],
                    "cuda_scalar_margin": top_c["margin"],
                }

            # Follow the Triton/reference greedy trajectory so both backends see
            # the same input token stream until a later analyzer chooses otherwise.
            next_id = int(top_t["ids"][0])

        result = {
            "schema_version": "lynn-engine-p33-native-active-moe-first-divergence-v1",
            "model": args.model,
            "prompt": args.prompt,
            "steps": args.steps,
            "initial_topk": _topk(logits0, args.topk),
            "first_top1_divergence": first_top1_divergence,
            "first_layer_divergence": first_layer_divergence,
            "rows": rows,
            "pass": first_top1_divergence is None,
            "notes": [
                "Both backends are fed the Triton/reference greedy token stream.",
                "This probe runs with LYNN_LINEAR_BLOCK_GRAPH=0 to isolate numeric/backend drift from graph replay.",
            ],
        }
    finally:
        _restore_env(old_env)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": result["pass"],
        "initial_top1": result["initial_topk"]["ids"][0],
        "first_top1_divergence": result["first_top1_divergence"],
        "first_layer_divergence": result["first_layer_divergence"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
