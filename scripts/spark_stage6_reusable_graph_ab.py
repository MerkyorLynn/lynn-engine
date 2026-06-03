#!/usr/bin/env python3
"""Stage 6 — the deciding cheap test: NVFP4 reusable decode CUDA graph delta vs baseline (2026-06-03).

Evidence-lock proved decode is launch/dispatch-bound (read-4bit done, shadow is prefill-only, FP4-ing
attn gives 0/-). The ONLY remaining lever is dispatch reduction via the reusable decode graph
(LYNN_REUSABLE_DECODE_GRAPH=1). The FP8-revival run measured +10% there; the NVFP4 delta is UNMEASURED.
This A/B settles it on ONE load:
  >+15% over 44.68 -> graph is worth pursuing toward ~50+.
  ~+10% or less   -> accept ~45-50 + the 60 GiB decode-only memory win; no kernel.

The reusable-graph capture (resident_runner.py:1145) auto-sets LYNN_FULL_ATTN_FIXED_SHAPE=1 +
LYNN_FP8_MOE_GRAPH_SAFE=1 (graph-safe MoE dispatch), so it applies to the 35B MoE. decode_tps measured
by the runner is pure per-token REPLAY speed (one-time capture is timed separately).

Run in docker lynn-eval-base:cu13, PYTHONNOUSERSITE=1, APEX stopped.
"""
import os, sys, pathlib, json

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

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
MAXNEW = int(os.environ.get("MAXNEW", "96"))
P = "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step."


def _tps(out):
    return float((out.get("timings", {}) or {}).get("decode_tps") or 0.0)


def run(n=3):
    best, last = 0.0, None
    for _ in range(n):
        out = runner.generate(P, max_new=MAXNEW)
        best = max(best, _tps(out)); last = out
    return best, (last.get("text") or "")


runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
runner.generate("warm up.", max_new=8)

os.environ["LYNN_REUSABLE_DECODE_GRAPH"] = "0"
tps_base, text_base = run()
print(f"\nbaseline (graph OFF) = {tps_base:.2f} TPS", flush=True)
print(f"base text: {text_base[:80]!r}", flush=True)

os.environ["LYNN_REUSABLE_DECODE_GRAPH"] = "1"
err = None
try:
    runner.generate("warm graph.", max_new=8)  # first call captures
    tps_g, text_g = run()
    coherent = text_g[:60] == text_base[:60]
    print(f"\ngraph ON  = {tps_g:.2f} TPS  ({tps_g/max(tps_base,1e-9):.3f}x vs baseline)", flush=True)
    print(f"graph text: {text_g[:80]!r}", flush=True)
    print(f"coherent vs baseline (first 60) = {coherent}", flush=True)
except Exception as e:
    err = repr(e)
    tps_g, coherent = 0.0, None
    print(f"\ngraph ON ERROR: {err}", flush=True)

print("\n================ REUSABLE-GRAPH A/B VERDICT ================", flush=True)
if err:
    print(f"=> graph capture/replay FAILED on the 35B NVFP4 MoE stack: {err}", flush=True)
    print("   => the reusable graph is not drop-in for this config; dispatch lever needs graph-safe work first.", flush=True)
else:
    x = tps_g / max(tps_base, 1e-9)
    print(f"baseline {tps_base:.2f} -> graph {tps_g:.2f}  = {x:.3f}x ; coherent={coherent}", flush=True)
    if x >= 1.15 and coherent:
        print("   => graph delta >= +15% AND coherent: dispatch lever is REAL -> pursue toward ~50+.", flush=True)
    elif x >= 1.04 and coherent:
        print(f"   => modest +{(x-1)*100:.0f}% (matches the FP8 ~+10%): graph helps a little but won't reach 70.", flush=True)
        print("      Recommend: bank it as an opt-in, accept ~45-50 + the 60 GiB decode-only memory win.", flush=True)
    elif not coherent:
        print("   => graph changes outputs (not coherent): fixed-shape/MoE-graph-safe numerics need a fix before use.", flush=True)
    else:
        print("   => no gain: decode dispatch isn't the lever either -> Spark NVFP4 is at its structural ceiling.", flush=True)
print(json.dumps({"baseline": tps_base, "graph": tps_g, "x": (tps_g/tps_base if tps_base else 0), "coherent": coherent, "error": err}, ensure_ascii=False), flush=True)
