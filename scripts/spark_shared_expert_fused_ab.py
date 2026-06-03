#!/usr/bin/env python3
"""Same-process A/B: shared-expert fusion OFF vs ON (LYNN_SHARED_EXPERT_FUSED), stacked
on the full current best (bh4 + linear-attn flags + fused RMSNorm + fused full-attn).
Verifies claude-internal's shared_expert_decode_fused_triton: token-coherence + e2e TPS.
Biggest remaining launch cluster (~160-200/token across 40 MoE layers' shared expert).
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
os.environ["LYNN_SHARED_EXPERT_FUSED"] = "0"

import torch
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
P = "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step."
runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)


def tps(n=3):
    vals = []
    for _ in range(n):
        out = runner.generate(P, max_new=64)
        vals.append((out.get("timings", {}) or {}).get("decode_tps") or 0)
    return max(vals), out

runner.generate("warm up.", max_new=8)
os.environ["LYNN_SHARED_EXPERT_FUSED"] = "0"
tpsA, outA = tps()
os.environ["LYNN_SHARED_EXPERT_FUSED"] = "1"
runner.generate("warm up fused.", max_new=8)
tpsB, outB = tps()

print("=== shared-expert fusion: OFF vs ON (full stack: bh4+flags+RMSNorm+full-attn) ===")
print("A  shared-expert-fused OFF : %.3f TPS" % tpsA)
print("B  shared-expert-fused ON  : %.3f TPS  (%.3fx vs A)" % (tpsB, tpsB / max(tpsA, 1e-9)))
ta = outA.get("text") or ""; tb = outB.get("text") or ""
print("COHERENT_A:", repr(ta[:80]))
print("COHERENT_B:", repr(tb[:80]))
print("TOKEN_EXACT_A_vs_B:", ta == tb)
