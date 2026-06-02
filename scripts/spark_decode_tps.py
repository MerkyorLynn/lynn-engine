#!/usr/bin/env python3
"""Clean (no-profile) NVFP4 35B decode TPS under a given env config.

Run with extra env to A/B levers, e.g.:
  LYNN_LINEAR_ATTN_GQA_RECURRENT=1 LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV=1 python scripts/spark_decode_tps.py
  LYNN_REUSABLE_DECODE_GRAPH=1 LYNN_FULL_ATTN_FIXED_SHAPE=1 ... python scripts/spark_decode_tps.py
Prints decode_tps + the active levers + an output snippet (coherence check).
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

import torch
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
runner.generate("Hello there, friend.", max_new=8)   # warmup (+ graph capture if enabled)
out = runner.generate("If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step.", max_new=64)
dtps = (out.get("timings", {}) or {}).get("decode_tps")
txt = out.get("text")
if not isinstance(txt, str):
    txt = str(out.get("new_ids", [])[:24])
print("CONFIG_TPS decode_tps=%.3f" % (dtps or 0))
print("LEVERS reusable_graph=%s gqa=%s outconv=%s down_bh=%s packed_linear=%s" % (
    os.environ.get("LYNN_REUSABLE_DECODE_GRAPH"), os.environ.get("LYNN_LINEAR_ATTN_GQA_RECURRENT"),
    os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV"), os.environ.get("LYNN_MOE_DOWN_BLOCK_HIDDEN"),
    os.environ.get("LYNN_PACKED_DECODE_LINEAR_ATTN")))
print("OUTPUT:", repr(txt[:200]))
