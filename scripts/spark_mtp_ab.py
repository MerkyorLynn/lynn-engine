#!/usr/bin/env python3
"""Stage-5 MTP eval A/B: speculative decode OFF vs ON (K>=2), on the full RC-validated stack
(bh4 + RMSNorm + full-attn + shared-expert + g/beta + bf16-out + o_proj-nocopy).

Uses the trained sidecar qwen36-35b-a3b-mtp. MTP is VERIFIED speculative decode -> output must
be TOKEN_EXACT vs non-MTP (it's a pure speedup). Measures TPS gain + accept stats. Expect
~+13% byte-capped (llama.cpp APEX proves 79 vs 69.77 on this model).

Run in docker: lynn-eval-base:cu13 with PYTHONNOUSERSITE=1.
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
# full RC-validated stack ON:
os.environ.setdefault("LYNN_RMSNORM_FUSED", "1")
os.environ.setdefault("LYNN_FULL_ATTN_FUSED", "1")
os.environ.setdefault("LYNN_SHARED_EXPERT_FUSED", "1")
os.environ.setdefault("LYNN_LINEAR_ATTN_FUSE_GBETA", "1")
os.environ.setdefault("LYNN_NVFP4_BF16_OUT", "1")
os.environ.setdefault("LYNN_DECODE_OPROJ_NOCOPY", "1")
# MTP: set sidecar + speculative=1 at INIT so the sidecar loads; toggle speculative per-call.
SIDECAR = os.environ.get("LYNN_MTP_SIDECAR",
                         "/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp/mtp.safetensors")
os.environ["LYNN_MTP_SIDECAR"] = SIDECAR
os.environ["LYNN_MTP_SPECULATIVE"] = "1"   # init: mtp_wanted -> load sidecar
SPEC_K = os.environ.get("SPEC_K", "2")

import torch
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
P = "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step."
runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=True)
print("MTP sidecar loaded?:", getattr(runner, "mtp_sidecar_loaded", None), "from", SIDECAR)


def run(n=3):
    vals = []
    last = None
    for _ in range(n):
        out = runner.generate(P, max_new=64)
        vals.append((out.get("timings", {}) or {}).get("decode_tps") or 0)
        last = out
    return max(vals), last

runner.generate("warm up.", max_new=8)
# A: MTP off (sidecar loaded but speculative disabled -> normal decode)
os.environ["LYNN_MTP_SPECULATIVE"] = "0"
tpsA, outA = run()
# B: MTP on, K>=2 batched
os.environ["LYNN_MTP_SPECULATIVE"] = "1"
os.environ["LYNN_MTP_SPECULATIVE_BATCHED"] = "1"
os.environ["LYNN_MTP_SPECULATIVE_K"] = SPEC_K
runner.generate("warm up mtp.", max_new=8)
tpsB, outB = run()

print("=== MTP speculative decode: OFF vs ON (K=%s) on full RC stack ===" % SPEC_K)
print("A  MTP OFF : %.3f TPS" % tpsA)
print("B  MTP ON  : %.3f TPS  (%.3fx vs A)" % (tpsB, tpsB / max(tpsA, 1e-9)))
ta = outA.get("text") or ""; tb = outB.get("text") or ""
print("COHERENT_A:", repr(ta[:80]))
print("COHERENT_B:", repr(tb[:80]))
print("TOKEN_EXACT_A_vs_B:", ta == tb, "(MUST be True -- verified speculative decode)")
print("=== B timings (accept stats) ===")
try:
    print(json.dumps({k: v for k, v in (outB.get("timings", {}) or {}).items()
                      if any(s in k.lower() for s in ("mtp", "spec", "accept", "tps", "multiplier", "draft"))},
                     indent=2, default=str))
except Exception as e:
    print("timings dump err:", e)
