#!/usr/bin/env python3
"""Profile the 35B-A3B FP8 decode to pinpoint the ~290ms/token bottleneck.

35B FP8 decode is ~3.5 TPS vs the same model's NVFP4 ~38.96 TPS (~11x slower).
Graph (P1) only gave +10% -> NOT dispatch-bound. This run uses LYNN_MTP_PROFILE
to break decode time into sections (attention vs the rest) and decide the fix:
grouped FP8 GEMM (P2) vs pivoting the reusable graph onto the faster NVFP4 path.
"""
import os, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LYNN_W4A8_FP8_PATH", "1")
os.environ["LYNN_MTP_PROFILE"] = "1"
os.environ["LYNN_FP8_MOE_GRAPH_SAFE"] = os.environ.get("LYNN_FP8_MOE_GRAPH_SAFE", "1")
os.environ["LYNN_LINEAR_STATE_UPDATE"] = "inplace"
os.environ["LYNN_NATIVE_FP4_LM_HEAD"] = "0"

import torch
from engine import mtp_profile
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8")
runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
runner.generate("Hello.", max_new=4)          # warmup
mtp_profile.reset()
out = runner.generate("If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step.", max_new=64)
snap = mtp_profile.snapshot()

n = len(out["new_ids"])
dtps = (out.get("timings", {}) or {}).get("decode_tps")
print(f"=== 35B FP8 decode profile: tokens={n} decode_tps={dtps} ===")
rows = sorted(snap.items(), key=lambda kv: -kv[1]["total_seconds"])
prof_total = sum(r["total_seconds"] for _, r in snap.items())
for name, r in rows[:30]:
    print(f"  {r['total_seconds']*1000/max(n,1):8.2f} ms/tok  {name}  (count={r['count']}, mean={r['mean_ms']:.3f}ms)")
print(f"  --- profiled-sections total ~= {prof_total*1000/max(n,1):.1f} ms/tok (vs ~{1000.0/dtps if dtps else 0:.1f} ms/tok measured) ---")
pathlib.Path("/home/merkyor/reports/qwen36_35b/fp8_decode_profile.json").write_text(
    json.dumps({"tokens": n, "decode_tps": dtps, "sections": snap}, indent=2))
print("wrote /home/merkyor/reports/qwen36_35b/fp8_decode_profile.json")
