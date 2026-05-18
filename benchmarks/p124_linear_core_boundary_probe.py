#!/usr/bin/env python3
"""P124: linear-attention boundary regression + strict parity probe.

Stream B Task 4 from ``docs/QWEN36_W4A16_KERNEL_REFACTOR_PLAN_20260518.md``.

Where P10C measures latency for each linear-attention core segment, P124 adds:

1. Strict numerical parity between (a) the production fused/recurrent path and
   (b) a plain ``lynn_linear_attn_forward``-style reference rebuilt from the
   torch HF math. Used to gate any boundary reduction or fused-op rewrite.
2. Backend sweep that toggles known opt-in env vars
   (``LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4``, ``LYNN_LINEAR_ATTN_RECURRENT_INPLACE``,
   ``LYNN_LINEAR_ATTN_GQA_RECURRENT``) so a candidate "boundary collapse" branch
   reports cosine + max_abs against the strict baseline before it merges.

Per the refactor plan, the per-segment targets are:

| segment | target ms/layer |
|---|---:|
| native FP4 in-proj | ~0.077 |
| recurrent fused prepare | ~0.036 |
| conv update | ~0.026–0.033 |
| gated RMSNorm | ~0.020 |

P124 reports both the measured ms and the parity envelope. A boundary reduction
that drops cosine below 0.9999 or pushes max_abs above 5e-3 must close — even
if it wins on latency.

Output JSON schema:
``lynn-engine-p124-linear-core-boundary-probe-v1``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.incremental_decode import (  # noqa: E402
    _linear,
    _linear_conv_update_decode,
    _rms_norm_gated_decode,
)
from engine.qwen36_linear_attn_block import (  # noqa: E402
    HEAD_K_DIM,
    HEAD_V_DIM,
    KEY_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    RMS_EPS,
    VALUE_DIM,
    V_PER_K,
)
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.gated_delta import (  # noqa: E402
    recurrent_gated_delta_fused_prepare,
    recurrent_gated_delta_fused_prepare_gqa,
)


def _bench(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    denom = af.norm() * bf.norm()
    if denom.item() == 0.0:
        return float("nan")
    return float((af @ bf) / denom)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def _first_linear_attn_layer() -> int:
    for i, t in enumerate(LAYER_TYPES):
        if t == "linear_attention":
            return i
    raise RuntimeError("no linear_attention layer in LAYER_TYPES")


def _prefill(runner: LynnIncrementalRunner, prompt: str):
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(
        batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype
    )
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(
            h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i
        )
    state.seq_len = int(ids.shape[1])
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    return int(logits[0].argmax().item()), state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument(
        "--prompt", default="用一句话解释 MoE active parameters,然后写 100 字技术 note",
    )
    args = ap.parse_args()

    if args.layer < 0:
        args.layer = _first_linear_attn_layer()
    if LAYER_TYPES[args.layer] != "linear_attention":
        raise ValueError(
            f"layer {args.layer} is {LAYER_TYPES[args.layer]!r}, expected linear_attention"
        )

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    next_id, state = _prefill(runner, args.prompt)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h0 = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    w = runner.layer_weights[args.layer]
    h_new = _rms_norm(h0, w["input_layernorm.weight"])
    B = h_new.shape[0]

    fused_key = "linear_attn._in_proj_qkv_z_b_a.weight"
    fused_native_available = fused_key in w
    use_gqa_recurrent = (
        V_PER_K > 1
        and os.environ.get("LYNN_LINEAR_ATTN_GQA_RECURRENT", "0") == "1"
    )
    recurrent_inplace = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_INPLACE", "0") == "1"

    # ─── baseline reference: separate-projection torch math ───
    # Closely mirrors lynn_linear_attn_forward but for the decode 1-token path.
    # Used as the strict parity reference for any boundary collapse candidate.
    def reference_inproj():
        qkv = F.linear(h_new, w["linear_attn.in_proj_qkv.weight"])
        z = F.linear(h_new, w["linear_attn.in_proj_z.weight"])
        b = F.linear(h_new, w["linear_attn.in_proj_b.weight"])
        a = F.linear(h_new, w["linear_attn.in_proj_a.weight"])
        return qkv, z, b, a

    def fused_inproj_native_fp4():
        if not fused_native_available:
            return reference_inproj()
        proj_all = _linear(h_new, w[fused_key])
        qkv, z, b, a = torch.split(
            proj_all,
            [KEY_DIM + KEY_DIM + VALUE_DIM, VALUE_DIM, NUM_V_HEADS, NUM_V_HEADS],
            dim=-1,
        )
        return qkv, z, b, a

    ref_qkv, ref_z, ref_b, ref_a = reference_inproj()
    if fused_native_available:
        fp4_qkv, fp4_z, fp4_b, fp4_a = fused_inproj_native_fp4()
    else:
        fp4_qkv, fp4_z, fp4_b, fp4_a = ref_qkv, ref_z, ref_b, ref_a

    inproj_parity = {
        "qkv_cosine": _cosine(ref_qkv, fp4_qkv),
        "qkv_max_abs": _max_abs(ref_qkv, fp4_qkv),
        "z_cosine": _cosine(ref_z, fp4_z),
        "z_max_abs": _max_abs(ref_z, fp4_z),
        "b_cosine": _cosine(ref_b, fp4_b),
        "b_max_abs": _max_abs(ref_b, fp4_b),
        "a_cosine": _cosine(ref_a, fp4_a),
        "a_max_abs": _max_abs(ref_a, fp4_a),
    }

    # Choose the production path (fp4 if available) as the operating point
    qkv, z, b, a = (fp4_qkv, fp4_z, fp4_b, fp4_a)

    mixed_new_t = qkv.transpose(1, 2)
    conv_out, _conv_state_after = _linear_conv_update_decode(
        mixed_new_t,
        state.conv_state[args.layer],
        w["linear_attn.conv1d.weight"],
    )
    q_full, k_full, v_full = torch.split(conv_out, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q_full = q_full.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
    k_full = k_full.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
    v_full = v_full.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
    if V_PER_K > 1 and not use_gqa_recurrent:
        q_full = q_full.repeat_interleave(V_PER_K, dim=2)
        k_full = k_full.repeat_interleave(V_PER_K, dim=2)
    z_view = z.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
    beta = b.sigmoid()
    g_decay = -w["linear_attn.A_log"].float().exp() * F.softplus(
        a.float() + w["linear_attn.dt_bias"].float()
    )

    # Reference recurrent (snapshot state to allow strict re-run)
    rec_state_ref = state.recurrent_state[args.layer].clone()

    def recurrent_no_gqa(state_buf):
        return recurrent_gated_delta_fused_prepare(q_full, k_full, v_full, g_decay, beta, state_buf)

    def recurrent_gqa(state_buf):
        return recurrent_gated_delta_fused_prepare_gqa(
            q_full, k_full, v_full, g_decay, beta, state_buf
        )

    # Strict-baseline core: replay separate projection path with a fresh state
    # buffer so the parity check is bitwise-comparable to a re-run from prefill.
    state_no_gqa = rec_state_ref.clone()
    core_no_gqa, _ = recurrent_no_gqa(state_no_gqa)
    if V_PER_K > 1:
        state_gqa = rec_state_ref.clone()
        core_gqa, _ = recurrent_gqa(state_gqa)
        recurrent_parity = {
            "no_gqa_vs_gqa_cosine": _cosine(core_no_gqa, core_gqa),
            "no_gqa_vs_gqa_max_abs": _max_abs(core_no_gqa, core_gqa),
        }
    else:
        recurrent_parity = {"no_gqa_vs_gqa_cosine": float("nan"), "no_gqa_vs_gqa_max_abs": float("nan")}

    core_attn_out = core_gqa if use_gqa_recurrent else core_no_gqa

    flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
    flat_z = z_view.reshape(-1, HEAD_V_DIM)
    normed = _rms_norm_gated_decode(flat_x, w["linear_attn.norm.weight"], flat_z)
    normed_view = normed.reshape(B, 1, NUM_V_HEADS * HEAD_V_DIM)

    # ─── timing pass ───
    timing = {
        "ref_inproj_separate": _bench(reference_inproj, args.warmup, args.iters),
    }
    if fused_native_available:
        timing["fused_inproj_native_fp4"] = _bench(
            fused_inproj_native_fp4, args.warmup, args.iters
        )

    def conv_update_call():
        return _linear_conv_update_decode(
            mixed_new_t,
            state.conv_state[args.layer],
            w["linear_attn.conv1d.weight"],
        )

    timing["conv_update_decode"] = _bench(conv_update_call, args.warmup, args.iters)

    def split_qkv_call():
        out_q, out_k, out_v = torch.split(conv_out, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        out_q = out_q.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        out_k = out_k.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        out_v = out_v.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        if V_PER_K > 1 and not use_gqa_recurrent:
            out_q = out_q.repeat_interleave(V_PER_K, dim=2)
            out_k = out_k.repeat_interleave(V_PER_K, dim=2)
        return out_q, out_k, out_v

    timing["split_qkv_repeat"] = _bench(split_qkv_call, args.warmup, args.iters)

    # Recurrent path benches use a temporary state buffer so the underlying
    # production state is not mutated by the probe.
    def recurrent_no_gqa_bench():
        buf = rec_state_ref.clone()
        return recurrent_no_gqa(buf)

    def recurrent_gqa_bench():
        buf = rec_state_ref.clone()
        return recurrent_gqa(buf)

    timing["recurrent_fused_prepare"] = _bench(
        recurrent_no_gqa_bench, args.warmup, args.iters
    )
    if V_PER_K > 1:
        timing["recurrent_fused_prepare_gqa"] = _bench(
            recurrent_gqa_bench, args.warmup, args.iters
        )

    def gated_rmsnorm_call():
        return _rms_norm_gated_decode(flat_x, w["linear_attn.norm.weight"], flat_z)

    timing["gated_rmsnorm_decode"] = _bench(gated_rmsnorm_call, args.warmup, args.iters)

    def out_proj_call():
        return _linear(normed_view, w["linear_attn.out_proj.weight"])

    timing["out_proj_bf16"] = _bench(out_proj_call, args.warmup, args.iters)

    # ─── boundary-collapsed candidate: fused inproj→conv→split→recurrent ───
    # This is the operating point Stream B should beat. Any further candidate
    # MUST keep parity vs the reference-separate path documented above.
    def full_core_boundary():
        proj_qkv, proj_z, proj_b, proj_a = (
            fused_inproj_native_fp4() if fused_native_available else reference_inproj()
        )
        mixed_t = proj_qkv.transpose(1, 2)
        conv, _ = _linear_conv_update_decode(
            mixed_t, state.conv_state[args.layer], w["linear_attn.conv1d.weight"]
        )
        q0, k0, v0 = torch.split(conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q0 = q0.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        k0 = k0.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        v0 = v0.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        if V_PER_K > 1 and not use_gqa_recurrent:
            q0 = q0.repeat_interleave(V_PER_K, dim=2)
            k0 = k0.repeat_interleave(V_PER_K, dim=2)
        z0 = proj_z.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        beta0 = proj_b.sigmoid()
        g0 = -w["linear_attn.A_log"].float().exp() * F.softplus(
            proj_a.float() + w["linear_attn.dt_bias"].float()
        )
        buf = rec_state_ref.clone()
        if use_gqa_recurrent:
            attn, _ = recurrent_gated_delta_fused_prepare_gqa(q0, k0, v0, g0, beta0, buf)
        else:
            attn, _ = recurrent_gated_delta_fused_prepare(q0, k0, v0, g0, beta0, buf)
        flat = attn.reshape(-1, HEAD_V_DIM)
        flat_zz = z0.reshape(-1, HEAD_V_DIM)
        out_n = _rms_norm_gated_decode(flat, w["linear_attn.norm.weight"], flat_zz)
        out_n = out_n.reshape(B, 1, NUM_V_HEADS * HEAD_V_DIM)
        return _linear(out_n, w["linear_attn.out_proj.weight"])

    timing["full_core_boundary"] = _bench(
        full_core_boundary, max(2, args.warmup // 4), max(20, args.iters // 4)
    )

    full_core_out = full_core_boundary()

    # ─── ceiling reference: re-run separate-projection path + check parity ───
    def full_core_reference():
        proj_qkv, proj_z, proj_b, proj_a = reference_inproj()
        mixed_t = proj_qkv.transpose(1, 2)
        conv, _ = _linear_conv_update_decode(
            mixed_t, state.conv_state[args.layer], w["linear_attn.conv1d.weight"]
        )
        q0, k0, v0 = torch.split(conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q0 = q0.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        k0 = k0.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        v0 = v0.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        if V_PER_K > 1 and not use_gqa_recurrent:
            q0 = q0.repeat_interleave(V_PER_K, dim=2)
            k0 = k0.repeat_interleave(V_PER_K, dim=2)
        z0 = proj_z.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        beta0 = proj_b.sigmoid()
        g0 = -w["linear_attn.A_log"].float().exp() * F.softplus(
            proj_a.float() + w["linear_attn.dt_bias"].float()
        )
        buf = rec_state_ref.clone()
        if use_gqa_recurrent:
            attn, _ = recurrent_gated_delta_fused_prepare_gqa(q0, k0, v0, g0, beta0, buf)
        else:
            attn, _ = recurrent_gated_delta_fused_prepare(q0, k0, v0, g0, beta0, buf)
        flat = attn.reshape(-1, HEAD_V_DIM)
        flat_zz = z0.reshape(-1, HEAD_V_DIM)
        out_n = _rms_norm_gated_decode(flat, w["linear_attn.norm.weight"], flat_zz)
        out_n = out_n.reshape(B, 1, NUM_V_HEADS * HEAD_V_DIM)
        return _linear(out_n, w["linear_attn.out_proj.weight"])

    ref_core_out = full_core_reference()
    boundary_parity = {
        "fused_vs_reference_cosine": _cosine(full_core_out, ref_core_out),
        "fused_vs_reference_max_abs": _max_abs(full_core_out, ref_core_out),
    }

    top_segments = sorted(
        [
            {"segment": k, "latency_ms": v}
            for k, v in timing.items()
            if not (k.startswith("full_core") or k.startswith("ref_"))
        ],
        key=lambda row: row["latency_ms"],
        reverse=True,
    )

    env_snapshot = {
        name: os.environ.get(name)
        for name in (
            "LYNN_LINEAR_ATTN_RECURRENT_BACKEND",
            "LYNN_LINEAR_ATTN_RECURRENT_INPLACE",
            "LYNN_LINEAR_ATTN_GQA_RECURRENT",
            "LYNN_LINEAR_ATTN_CONV_BACKEND",
            "LYNN_LINEAR_ATTN_INPROJ_FUSED",
            "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4",
            "LYNN_PACKED_DECODE",
            "LYNN_PACKED_DECODE_BACKEND",
        )
    }

    targets_ms = {
        "fused_inproj_native_fp4": 0.077,
        "recurrent_fused_prepare": 0.036,
        "conv_update_decode": 0.030,
        "gated_rmsnorm_decode": 0.020,
    }

    target_deltas: dict[str, dict[str, float | None]] = {}
    for key, target in targets_ms.items():
        measured = timing.get(key)
        if measured is None:
            target_deltas[key] = {"target_ms": target, "measured_ms": None, "delta_ms": None}
            continue
        target_deltas[key] = {
            "target_ms": target,
            "measured_ms": measured,
            "delta_ms": measured - target,
        }

    result = {
        "schema_version": "lynn-engine-p124-linear-core-boundary-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "device": torch.cuda.get_device_name("cuda"),
        "config": {
            "V_PER_K": V_PER_K,
            "fused_native_inproj_available": fused_native_available,
            "use_gqa_recurrent": use_gqa_recurrent,
            "recurrent_inplace": recurrent_inplace,
        },
        "env": env_snapshot,
        "timing_ms": timing,
        "top_segments": top_segments,
        "target_deltas_ms": target_deltas,
        "inproj_parity_native_fp4_vs_bf16": inproj_parity,
        "recurrent_no_gqa_vs_gqa_parity": recurrent_parity,
        "boundary_collapse_parity_vs_reference": boundary_parity,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
