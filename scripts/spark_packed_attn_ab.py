#!/usr/bin/env python3
"""Same-process A/B: full-attn q/k/v/o read as BF16 shadow vs packed 4-bit NVFP4.

The attn projections ship PACKED 4-bit but the runner dequantizes to a BF16 shadow.
_prepare_full_attn_qkv_native_fp4() attaches the packed objects; LYNN_PACKED_DECODE_FULL_ATTN
(read at runtime in _decode_weight) toggles which decode reads. Same load/prompt, on top
of bh4 + linear-attn flags. Should be ~token-exact (same quantized weights) + faster
(4x less attn-proj traffic).
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
os.environ["LYNN_PACKED_DECODE_FULL_ATTN"] = "0"   # start OFF (BF16 attn shadow)

import torch
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
P = "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step."
runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
runner._prepare_full_attn_qkv_native_fp4()
attached = getattr(runner, "full_attn_qkv_native_fp4_attached", -1)


def tps(n=3):
    vals = []
    for _ in range(n):
        out = runner.generate(P, max_new=64)
        vals.append((out.get("timings", {}) or {}).get("decode_tps") or 0)
    return max(vals), out

runner.generate("warm up.", max_new=8)
os.environ["LYNN_PACKED_DECODE_FULL_ATTN"] = "0"
tpsA, outA = tps()
os.environ["LYNN_PACKED_DECODE_FULL_ATTN"] = "1"
runner.generate("warm up packed.", max_new=8)
tpsB, outB = tps()

print("=== full-attn q/k/v/o: BF16 shadow vs packed-4bit (bh4 + linear-attn flags) ===")
print("attached packed projections: %d (expect 40 = 10 full-attn layers x 4)" % attached)
print("A  BF16 attn shadow : %.3f TPS" % tpsA)
print("B  packed-4bit attn : %.3f TPS  (%.3fx vs A)" % (tpsB, tpsB / max(tpsA, 1e-9)))
ta = outA.get("text") or ""; tb = outB.get("text") or ""
print("COHERENT_A:", repr(ta[:70]))
print("COHERENT_B:", repr(tb[:70]))
print("TOKEN_EXACT_A_vs_B:", ta == tb)
