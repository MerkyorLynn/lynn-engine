#!/usr/bin/env python3
"""Profile the 35B-A3B NVFP4 W4A16 *baseline* decode on Spark to find the real
bottleneck (attention vs MoE vs lm_head vs linear-attn vs glue) before optimizing.

Uses the production NVFP4 fast path (the smoke's BASE_ENV) + the banked
LYNN_MOE_DOWN_BLOCK_HIDDEN=4 win, with LYNN_MTP_PROFILE to break decode into
sections. The full_attn_t1.* sections are instrumented; whatever is left
(MoE + lm_head + glue) shows up as the remainder vs measured ms/tok.
"""
import os, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_ENV = {
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_MOE_FAST_FIXED": "1",
    "LYNN_NATIVE_ACTIVE_MOE_BACKEND": "triton",
    "LYNN_NATIVE_GATEUP_BACKEND": "triton_fast_decode",
    "LYNN_NATIVE_DOWN_BACKEND": "triton",
    "LYNN_ROUTER_TOPK_SORTED": "0",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_LINEAR_STATE_UPDATE": "inplace",
    "LYNN_PACKED_DECODE": "0",
    "LYNN_PACKED_SHARED_EXPERT": "0",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
    "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_FULL_ATTN_ROPE_CACHE": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
for k, v in BASE_ENV.items():
    os.environ.setdefault(k, v)
os.environ["LYNN_MOE_DOWN_BLOCK_HIDDEN"] = "4"   # banked +11% config win
os.environ["LYNN_MTP_PROFILE"] = "1"

import torch
from engine import mtp_profile
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
runner.generate("Hello.", max_new=4)          # warmup
mtp_profile.reset()
out = runner.generate("If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step.", max_new=64)
snap = mtp_profile.snapshot()

n = len(out["new_ids"])
dtps = (out.get("timings", {}) or {}).get("decode_tps")
print(f"=== NVFP4 35B baseline decode profile: tokens={n} decode_tps={dtps} ===")
rows = sorted(snap.items(), key=lambda kv: -kv[1]["total_seconds"])
prof_total = sum(r["total_seconds"] for _, r in snap.items())
for name, r in rows[:30]:
    print(f"  {r['total_seconds']*1000/max(n,1):8.2f} ms/tok  {name}  (count={r['count']}, mean={r['mean_ms']:.3f}ms)")
measured = 1000.0 / dtps if dtps else 0
print(f"  --- profiled sections ~= {prof_total*1000/max(n,1):.1f} ms/tok | measured ~= {measured:.1f} ms/tok | remainder ~= {measured - prof_total*1000/max(n,1):.1f} ms/tok ---")
_txt = out.get("text")
if not isinstance(_txt, str):
    _txt = str(out.get("new_ids", [])[:24])
print("OUTPUT_SNIPPET:", repr(_txt[:240]))
print("FLAGS: GQA_RECURRENT=%s FROM_OUTCONV=%s PACKED_LINEAR=%s" % (
    os.environ.get("LYNN_LINEAR_ATTN_GQA_RECURRENT"), os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV"),
    os.environ.get("LYNN_PACKED_DECODE_LINEAR_ATTN")))
pathlib.Path("/home/merkyor/reports/qwen36_35b/nvfp4_decode_profile.json").write_text(
    json.dumps({"tokens": n, "decode_tps": dtps, "sections": snap}, indent=2))
print("wrote /home/merkyor/reports/qwen36_35b/nvfp4_decode_profile.json")
