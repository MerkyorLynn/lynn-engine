#!/usr/bin/env python3
"""Stage-4A A/B: NVFP4 decode matmul fp16-out vs bf16-out (LYNN_NVFP4_BF16_OUT), stacked on
the full RC-validated stack (bh4 + RMSNorm + full-attn + shared-expert + g/beta). bf16-out
removes the per-projection fp16->bf16 aten::copy_ (the #1 launch cluster, ~230/tok census).
Checks coherence + TPS (TOKEN_EXACT not expected — different rounding, quality-neutral-or-better).

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
os.environ["LYNN_NVFP4_BF16_OUT"] = "0"

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
os.environ["LYNN_NVFP4_BF16_OUT"] = "0"
tpsA, outA = tps()
os.environ["LYNN_NVFP4_BF16_OUT"] = "1"
runner.generate("warm up bf16out.", max_new=8)
tpsB, outB = tps()

print("=== NVFP4 decode matmul: fp16-out (OFF) vs bf16-out (ON) — removes per-proj fp16->bf16 copy ===")
print("A  bf16-out OFF (fp16+cast) : %.3f TPS" % tpsA)
print("B  bf16-out ON  (no cast)   : %.3f TPS  (%.3fx vs A)" % (tpsB, tpsB / max(tpsA, 1e-9)))
ta = outA.get("text") or ""; tb = outB.get("text") or ""
print("COHERENT_A:", repr(ta[:80]))
print("COHERENT_B:", repr(tb[:80]))
print("TOKEN_EXACT_A_vs_B:", ta == tb)
