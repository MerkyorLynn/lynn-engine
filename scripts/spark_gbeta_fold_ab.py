#!/usr/bin/env python3
"""Stage-3 A/B: linear-attn g/beta-fold OFF vs ON (LYNN_LINEAR_ATTN_FUSE_GBETA), stacked on
the full RC-validated best (bh4 + linear-attn flags + fused RMSNorm + full-attn + shared-expert).
Verifies the g/beta-into-recurrent-kernel fold: token-EXACTness (pure launch fold, math
identical -> expect True) + e2e TPS. ~4 elementwise launches/layer x ~38 layers folded away.

Run in docker: lynn-eval-base:cu13 with PYTHONNOUSERSITE=1 (see RC harness).
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
# keep all prior RC-validated wins ON:
os.environ.setdefault("LYNN_RMSNORM_FUSED", "1")
os.environ.setdefault("LYNN_FULL_ATTN_FUSED", "1")
os.environ.setdefault("LYNN_SHARED_EXPERT_FUSED", "1")
os.environ["LYNN_LINEAR_ATTN_FUSE_GBETA"] = "0"

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
os.environ["LYNN_LINEAR_ATTN_FUSE_GBETA"] = "0"
tpsA, outA = tps()
os.environ["LYNN_LINEAR_ATTN_FUSE_GBETA"] = "1"
runner.generate("warm up fused.", max_new=8)
tpsB, outB = tps()

print("=== linear-attn g/beta-fold: OFF vs ON (full stack: bh4+flags+RMSNorm+full-attn+shared-expert) ===")
print("A  gbeta-fold OFF : %.3f TPS" % tpsA)
print("B  gbeta-fold ON  : %.3f TPS  (%.3fx vs A)" % (tpsB, tpsB / max(tpsA, 1e-9)))
ta = outA.get("text") or ""; tb = outB.get("text") or ""
print("COHERENT_A:", repr(ta[:80]))
print("COHERENT_B:", repr(tb[:80]))
print("TOKEN_EXACT_A_vs_B:", ta == tb, "(expect True -- pure launch fold, math identical)")
