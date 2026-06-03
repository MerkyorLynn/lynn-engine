#!/usr/bin/env python3
"""Stage 6: does DECODE (not prefill) read the BF16 shadow? (2026-06-03, sharpened)

The earlier audit's post-release `generate()` errored in PREFILL (per-expert BF16), which does
NOT prove DECODE needs BF16. This isolates DECODE: prefill once (shadows present) to build a valid
state, decode a stretch, then `release_decode_bf16_shadows()` and KEEP DECODING from the same state
(no re-prefill). Outcome:
  - decode continues shadow-free, ~same TPS  => DECODE already reads packed FP4; the 60 GiB BF16 is
    prefill-only / resident-unused-by-decode. Stage-6 "delete shadow for SPEED" is a MEMORY win, not a
    decode-bandwidth win -> real decode lever = kernel efficiency / dispatch (graph), re-scope.
  - decode continues shadow-free but FASTER => decode WAS reading the shadow -> zero-shadow is the lever.
  - decode ERRORS shadow-free => decode depends on a BF16 weight (which one) -> that's the Stage-6 target.

Run in docker lynn-eval-base:cu13, PYTHONNOUSERSITE=1, APEX stopped.
"""
import os, sys, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_ENV = {
    "LYNN_MOE_IMPL": "packed_nvfp4", "LYNN_MOE_FAST_FIXED": "1",
    "LYNN_NATIVE_ACTIVE_MOE_BACKEND": "triton", "LYNN_NATIVE_GATEUP_BACKEND": "triton_fast_decode",
    "LYNN_NATIVE_DOWN_BACKEND": "triton", "LYNN_ROUTER_TOPK_SORTED": "0",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare", "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1", "LYNN_LINEAR_STATE_UPDATE": "inplace",
    "LYNN_PACKED_DECODE": "0", "LYNN_PACKED_SHARED_EXPERT": "0", "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair", "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_FULL_ATTN_ROPE_CACHE": "1", "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
for k, v in BASE_ENV.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("LYNN_MOE_DOWN_BLOCK_HIDDEN", "4")
os.environ.setdefault("LYNN_LINEAR_ATTN_GQA_RECURRENT", "1")
os.environ.setdefault("LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV", "1")
os.environ.setdefault("LYNN_RMSNORM_FUSED", "1")
os.environ.setdefault("LYNN_FULL_ATTN_FUSED", "1")
os.environ.setdefault("LYNN_SHARED_EXPERT_FUSED", "1")
os.environ.setdefault("LYNN_LINEAR_ATTN_FUSE_GBETA", "1")
os.environ.setdefault("LYNN_NVFP4_BF16_OUT", "1")
os.environ.setdefault("LYNN_DECODE_OPROJ_NOCOPY", "1")

import torch
import torch.nn.functional as F
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt
from engine.full_forward import _prefill_layer, _rms_norm
from engine.inference_state import LynnInferenceState
from engine.mtp_serving import decode_one_to_logits_and_hidden

EMBED = "model.language_model.embed_tokens.weight"
NORM = "model.language_model.norm.weight"
MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
P = "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step."
N = int(os.environ.get("DECODE_STEPS", "40"))


def decode_stretch(runner, state, cur, n, tok):
    toks = []
    t0 = time.time()
    for _ in range(n):
        _h, _lg, am = decode_one_to_logits_and_hidden(runner, state, cur)
        cur = am
        toks.append(int(am))
    if runner.device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    return cur, toks, n / max(dt, 1e-9), tok.decode(toks)


def main():
    runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
    tok = runner.tokenizer
    dev = runner.device
    ids = _encode_prompt(tok, P, dev, use_chat_template=False)
    T = ids.shape[1]
    state = LynnInferenceState.from_config(runner.cfg, batch=1, max_seq_len=runner.max_seq_len, device=dev, dtype=runner.dtype)

    # ---- manual prefill (shadows present) ----
    h = F.embedding(ids, runner.outside[EMBED])
    pos = torch.arange(T, device=dev, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, runner.layer_types[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = T
    cur = int(runner._lm_head_logits(_rms_norm(h, runner.outside[NORM]))[0].argmax().item())
    print(f"prefill ok, first token={cur!r}", flush=True)

    # ---- decode stretch A: shadows PRESENT ----
    cur, toksA, tpsA, textA = decode_stretch(runner, state, cur, N, tok)
    print(f"\n[A] decode WITH shadows: {tpsA:.2f} tok/s", flush=True)
    print(f"    textA: {textA[:90]!r}", flush=True)

    # ---- release BF16 shadows ----
    try:
        rep = runner.release_decode_bf16_shadows()
        print(f"\nreleased_gib = {rep.get('released_gib'):.2f} GiB", flush=True)
    except Exception as e:
        print(f"\nrelease ERROR: {e!r}", flush=True)

    # ---- decode stretch B: shadow-FREE, SAME state, no re-prefill ----
    print("\n[B] continue DECODE shadow-free (same state, no re-prefill)...", flush=True)
    try:
        cur, toksB, tpsB, textB = decode_stretch(runner, state, cur, N, tok)
        print(f"[B] decode SHADOW-FREE: {tpsB:.2f} tok/s  ({tpsB/max(tpsA,1e-9):.3f}x vs A)", flush=True)
        print(f"    textB: {textB[:90]!r}", flush=True)
        ok = True
    except Exception as e:
        print(f"[B] decode SHADOW-FREE ERROR: {e!r}", flush=True)
        ok = False
        tpsB = 0.0

    print("\n=============== DECODE SHADOW-FREE VERDICT ===============", flush=True)
    if not ok:
        print("=> DECODE depends on a BF16 weight (errored shadow-free). That weight = the Stage-6 target.", flush=True)
    elif tpsB > tpsA * 1.05:
        print(f"=> DECODE was reading the shadow ({tpsB:.1f} > {tpsA:.1f}). Zero-shadow IS a decode-speed lever.", flush=True)
    else:
        print(f"=> DECODE runs shadow-free at ~same TPS ({tpsB:.1f} ~ {tpsA:.1f}).", flush=True)
        print("   => decode ALREADY reads packed FP4; the 60 GiB BF16 is prefill-only / resident-unused-by-decode.", flush=True)
        print("   => Stage-6 'delete shadow' = MEMORY win (run decode-only at ~27 GiB), NOT a decode-speed lever.", flush=True)
        print("   => the real 45->70 decode lever is kernel-efficiency / dispatch (reusable decode graph), re-scope.", flush=True)


if __name__ == "__main__":
    main()
