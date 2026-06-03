#!/usr/bin/env python3
"""Stage 6 Phase-0 lever #1: packed-FP4 decode for attn/out_proj — config sweep (2026-06-03).

codex trace: the RC BASE_ENV sets LYNN_PACKED_DECODE=0, so full-attn q/k/v/o (~545 MB/tok) and
linear-attn out_proj (~503 MB/tok) decode in BF16 even though `.packed` FP4 aliases are resident
(`self_attn.*_proj.weight.packed`). `_decode_weight` (incremental_decode.py:145) switches to
`.packed` per-call when LYNN_PACKED_DECODE / _FULL_ATTN / _LINEAR_ATTN is set. The MoE routed
experts + linear-attn in-proj + lm_head already read FP4. So this is a CONFIG test, not a kernel:
does routing attn/out_proj through packed FP4 cut ~1 GB/token of BF16 reads and raise TPS while
staying token-coherent?

One model load; toggles the flags per generate; reports decode_tps + xbase + token-coherence vs
the current RC baseline (PACKED_DECODE=0). Run in docker lynn-eval-base:cu13, PYTHONNOUSERSITE=1, APEX stopped.
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
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
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

TOGGLE = ["LYNN_PACKED_DECODE", "LYNN_PACKED_DECODE_FULL_ATTN", "LYNN_PACKED_DECODE_LINEAR_ATTN", "LYNN_PACKED_SHARED_EXPERT"]
CONFIGS = [
    ("baseline_PD0",     {}),
    ("full_attn_fp4",    {"LYNN_PACKED_DECODE_FULL_ATTN": "1"}),
    ("linear_outproj_fp4", {"LYNN_PACKED_DECODE_LINEAR_ATTN": "1"}),
    ("attn_both_fp4",    {"LYNN_PACKED_DECODE_FULL_ATTN": "1", "LYNN_PACKED_DECODE_LINEAR_ATTN": "1"}),
    ("all_packed_PD1",   {"LYNN_PACKED_DECODE": "1"}),
    ("attn_both+shared", {"LYNN_PACKED_DECODE_FULL_ATTN": "1", "LYNN_PACKED_DECODE_LINEAR_ATTN": "1", "LYNN_PACKED_SHARED_EXPERT": "1"}),
]


def _apply(cfg):
    for k in TOGGLE:
        os.environ.pop(k, None)
    os.environ["LYNN_PACKED_SHARED_EXPERT"] = "0"  # base default
    for k, v in cfg.items():
        os.environ[k] = v


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
_apply({})
tps_base, text_base = run()
print(f"\nbaseline (PACKED_DECODE=0) = {tps_base:.2f} TPS", flush=True)
print(f"base text: {text_base[:80]!r}", flush=True)

rows = []
for name, cfg in CONFIGS[1:]:
    _apply(cfg)
    runner.generate("warm.", max_new=4)
    try:
        tps, text = run()
        coherent = text[:60] == text_base[:60]
        rows.append({"config": name, "tps": round(tps, 2), "xbase": round(tps / max(tps_base, 1e-9), 3), "coherent": coherent})
        print(f"  [{name}] {tps:.2f} TPS ({tps/max(tps_base,1e-9):.3f}x) coherent={coherent}", flush=True)
    except Exception as e:
        rows.append({"config": name, "error": repr(e)[:160]})
        print(f"  [{name}] ERROR {e!r}", flush=True)

print("\n================ PACKED-DECODE SWEEP SUMMARY ================", flush=True)
print(f"baseline PACKED_DECODE=0 = {tps_base:.2f} TPS", flush=True)
print(f"{'config':>20} | {'TPS':>7} | {'xbase':>6} | coherent", flush=True)
for r in rows:
    if "error" in r:
        print(f"{r['config']:>20} | ERROR {r['error']}", flush=True); continue
    print(f"{r['config']:>20} | {r['tps']:>7.2f} | {r['xbase']:>6.3f} | {r['coherent']}", flush=True)
print("\nINTERPRETATION: any config with xbase>1.0 AND coherent=True => free TPS win from routing", flush=True)
print("attn/out_proj through resident packed-FP4 aliases (no kernel work). If all ~1.0x or slower,", flush=True)
print("the Spark _scaled_mm FP4 attn path isn't faster than BF16 F.linear -> lever is dispatch/graph.", flush=True)
print(json.dumps({"baseline": tps_base, "rows": rows}, ensure_ascii=False), flush=True)
