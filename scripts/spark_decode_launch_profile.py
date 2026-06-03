#!/usr/bin/env python3
"""Stage-4 re-profile: count CUDA kernel LAUNCHES per decode token on the full RC-validated
fusion stack (bh4 + RMSNorm + full-attn + shared-expert + g/beta), to find the biggest
REMAINING launch cluster. Launch count is the trustworthy metric (decode is launch-bound;
section timing is cuda-sync-inflated per the campaign).

Method: profile N and 2N decode tokens; the DELTA in per-kernel call counts / N = launches
per decode token (cancels prefill + warmup constants). Ranks kernels by per-token launches.

Run in docker: lynn-eval-base:cu13 with PYTHONNOUSERSITE=1.
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
# full RC-validated stack ON:
os.environ.setdefault("LYNN_RMSNORM_FUSED", "1")
os.environ.setdefault("LYNN_FULL_ATTN_FUSED", "1")
os.environ.setdefault("LYNN_SHARED_EXPERT_FUSED", "1")
os.environ.setdefault("LYNN_LINEAR_ATTN_FUSE_GBETA", "1")

import torch
from torch.profiler import profile, ProfilerActivity
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
P = "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step."
runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)

def call_counts(max_new):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        runner.generate(P, max_new=max_new)
    torch.cuda.synchronize()
    counts = {}
    cuda_total = 0
    for e in prof.key_averages():
        dev_t = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        if dev_t and dev_t > 0:  # only entries that actually ran on the GPU
            counts[e.key] = counts.get(e.key, 0) + e.count
            cuda_total += e.count
    return counts, cuda_total

runner.generate("warm up.", max_new=8)
N = 16
c1, t1 = call_counts(N)
c2, t2 = call_counts(2 * N)

# per-token launches = (counts at 2N - counts at N) / N  (cancels prefill+constant)
per_tok = {}
for k in set(list(c1.keys()) + list(c2.keys())):
    d = (c2.get(k, 0) - c1.get(k, 0)) / N
    if d > 0.05:
        per_tok[k] = d
total_per_tok = (t2 - t1) / N

print("=== DECODE LAUNCH CENSUS (full 5-fusion stack) ===")
print("total CUDA kernel launches / decode token  ≈ %.1f" % total_per_tok)
print("\n=== top launch clusters (launches per decode token) ===")
for k, v in sorted(per_tok.items(), key=lambda x: x[1], reverse=True)[:30]:
    print("  %6.1f/tok   %s" % (v, k[:72]))
