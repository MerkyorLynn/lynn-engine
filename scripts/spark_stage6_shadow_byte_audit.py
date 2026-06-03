#!/usr/bin/env python3
"""Stage 6 evidence-lock: BF16-shadow footprint + does decode actually read it? (2026-06-03)

Stage 6's premise: decode reads a BF16 dequant-shadow (~6GB/tok) instead of true 4-bit
(~1.7GB/tok), and THAT is the ~40 TPS bandwidth wall. Before writing any read-4bit/
dequant-in-register kernel we evidence-lock the real upside:

  1. AUDIT resident weight bytes: total, BF16, packed-NVFP4 -- on the full RC stack.
  2. BASELINE decode TPS (shadows present).
  3. release_decode_bf16_shadows() -> reports freed GiB = the "delete-shadow" CEILING.
  4. POST-release decode TPS + coherence:
       - faster  => decode WAS reading the shadow (Stage 6 bandwidth upside is real & large)
       - same    => shadow not on the per-token read path (win must come from register-dequant
                    of a transient BF16 temp, or it's dispatch-bound -> need graph not zero-shadow)
       - errors  => packed decode path still depends on the BF16 weight (not shadow-free yet)
  5. Implied bytes/token from TPS (ms/tok * 240 GB/s) to sanity-check bandwidth-bound vs dispatch.

Run in docker lynn-eval-base:cu13, PYTHONNOUSERSITE=1, APEX stopped.
"""
import os, sys, pathlib

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
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get(
    "MODEL",
    "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526",
)
P = "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step."
GiB = 1024 ** 3


def audit(runner):
    """Sum resident weight bytes by dtype across outside + all layer weights."""
    by_dtype = {}
    total = 0
    n = 0
    dicts = [("outside", runner.outside)]
    for i, w in enumerate(runner.layer_weights):
        dicts.append((f"L{i}", w))
    for _name, d in dicts:
        for k, t in d.items():
            if not isinstance(t, torch.Tensor):
                continue
            b = int(t.numel() * t.element_size())
            by_dtype[str(t.dtype)] = by_dtype.get(str(t.dtype), 0) + b
            total += b
            n += 1
    return total, by_dtype, n


def tps(runner, n=3):
    best = 0.0
    last = None
    for _ in range(n):
        out = runner.generate(P, max_new=64)
        best = max(best, float((out.get("timings", {}) or {}).get("decode_tps") or 0.0))
        last = out
    return best, (last.get("text") or "")


def cuda_mem():
    try:
        return torch.cuda.memory_allocated() / GiB
    except Exception:
        return float("nan")


def main():
    runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=True)

    total, by_dtype, n = audit(runner)
    print("\n=============== RESIDENT WEIGHT AUDIT (pre-release) ===============", flush=True)
    print(f"tensors={n}  total={total/GiB:.2f} GiB", flush=True)
    for dt, b in sorted(by_dtype.items(), key=lambda x: -x[1]):
        print(f"  {dt:>22} : {b/GiB:7.2f} GiB", flush=True)
    bf16 = sum(b for dt, b in by_dtype.items() if "bfloat16" in dt or "float16" in dt)
    packed = sum(b for dt, b in by_dtype.items() if any(s in dt for s in ("uint8", "int8", "float4", "uint16")))
    print(f"  -> BF16/FP16 (incl shadows) = {bf16/GiB:.2f} GiB ; packed/int = {packed/GiB:.2f} GiB", flush=True)

    runner.generate("warm up.", max_new=8)
    print(f"\ncuda_mem_allocated pre  = {cuda_mem():.2f} GiB", flush=True)
    tps_pre, text_pre = tps(runner)
    print(f"BASELINE decode TPS (shadows present) = {tps_pre:.2f}", flush=True)
    print(f"  implied bytes/token @240GB/s = {(1.0/max(tps_pre,1e-9))*240:.2f} GB  (= ms/tok * 240)", flush=True)

    print("\n=============== release_decode_bf16_shadows() ===============", flush=True)
    err = None
    rep = None
    try:
        rep = runner.release_decode_bf16_shadows()
        print(f"released_gib = {rep.get('released_gib'):.2f} GiB  ({len(rep.get('released', []))} tensors)", flush=True)
    except Exception as e:
        err = repr(e)
        print(f"release ERROR: {err}", flush=True)

    print(f"cuda_mem_allocated post = {cuda_mem():.2f} GiB", flush=True)
    total2, _, n2 = audit(runner)
    print(f"resident weight total post-release = {total2/GiB:.2f} GiB  (delta {(total-total2)/GiB:+.2f})  tensors {n}->{n2}", flush=True)

    print("\n=============== POST-release decode (shadow-free?) ===============", flush=True)
    post_err = None
    try:
        tps_post, text_post = tps(runner)
        coherent = text_post[:60] == text_pre[:60]
        print(f"POST-release decode TPS = {tps_post:.2f}  ({tps_post/max(tps_pre,1e-9):.3f}x baseline)", flush=True)
        print(f"  coherent vs baseline (first 60 chars) = {coherent}", flush=True)
        print(f"  text_post head: {text_post[:80]!r}", flush=True)
    except Exception as e:
        post_err = repr(e)
        tps_post = 0.0
        coherent = None
        print(f"POST-release decode ERROR: {post_err}", flush=True)

    print("\n=============== STAGE 6 EVIDENCE VERDICT ===============", flush=True)
    shadow_gib = (rep.get("released_gib") if rep else 0.0) or 0.0
    print(f"BF16-shadow footprint (delete-shadow ceiling) = {shadow_gib:.2f} GiB", flush=True)
    if post_err:
        print("=> packed decode STILL DEPENDS on the BF16 weight (errors when shadow dropped):", flush=True)
        print("   the native_fast_2d/packed path is NOT shadow-free yet -> Stage 6 must first wire a", flush=True)
        print("   true packed-only decode, then the register-dequant kernel. This IS the work.", flush=True)
    elif tps_post > tps_pre * 1.05:
        print(f"=> decode WAS reading the shadow: post-release {tps_post:.1f} > baseline {tps_pre:.1f}.", flush=True)
        print("   Stage 6 bandwidth upside is REAL and large -> read-4bit/zero-shadow is the right lever.", flush=True)
    elif coherent:
        print(f"=> shadow droppable + decode runs shadow-free at ~same TPS ({tps_post:.1f} vs {tps_pre:.1f}).", flush=True)
        print("   => the per-token read is NOT the resident BF16 shadow (likely a transient dequant temp,", flush=True)
        print("      or decode is dispatch-bound). Stage 6 win = register-dequant of the temp +/- graph,", flush=True)
        print("      not merely deleting the resident shadow. Re-scope before kernel work.", flush=True)
    else:
        print("=> ambiguous (incoherent post-release) -- inspect text + per-config before scoping.", flush=True)


if __name__ == "__main__":
    main()
