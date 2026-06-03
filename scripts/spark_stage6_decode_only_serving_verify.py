#!/usr/bin/env python3
"""Stage 6 productize: verify the decode-only shadow-free serving path (2026-06-03).

The 60 GiB BF16 dequant-shadow is PREFILL-only; decode reads packed NVFP4. The primitives already
exist: release_decode_bf16_shadows() (drop, after prefill) + reload_decode_bf16_shadows() (rebuild
from resident packed, NO disk I/O, before next prefill). The product cycle is:
    reload -> prefill -> release -> decode   (per request; decode sits at ~27 GiB).

This verifies + measures the cycle end-to-end:
  1. baseline generate() (no release) -> reference text + TPS + resident (~87 GiB).
  2. multi-request loop x3 using generate(release_decode_shadows_after_prefill=True) with
     reload_decode_bf16_shadows() between requests:
       - token-exact vs baseline (decode-only shadow-free must NOT change output)
       - resident after decode (~27 GiB, the freed 60 GiB headroom)
       - reload cost (seconds; the per-request price of the cycle)
       - decode TPS unchanged (~45)
Run in docker lynn-eval-base:cu13, PYTHONNOUSERSITE=1, APEX stopped.
"""
import os, sys, pathlib, json, time

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
os.environ.setdefault("LYNN_SHARED_EXPERT_FUSED", "1")
os.environ.setdefault("LYNN_LINEAR_ATTN_FUSE_GBETA", "1")
os.environ.setdefault("LYNN_NVFP4_BF16_OUT", "1")
os.environ.setdefault("LYNN_DECODE_OPROJ_NOCOPY", "1")

import torch
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
MAXNEW = int(os.environ.get("MAXNEW", "64"))
GiB = 1024 ** 3
PROMPTS = [
    "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step.",
    "Write a haiku about the sea, then explain its imagery.",
    "What is the capital of France, and name two famous landmarks there?",
]


def mem():
    try:
        return torch.cuda.memory_allocated() / GiB
    except Exception:
        return float("nan")


def tps(out):
    return float((out.get("timings", {}) or {}).get("decode_tps") or 0.0)


runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
runner.generate("warm up.", max_new=8)
print(f"resident after load+warm = {mem():.2f} GiB", flush=True)

# --- baseline references (no release): per-prompt text + tps, shadows present ---
refs = []
for p in PROMPTS:
    out = runner.generate(p, max_new=MAXNEW)
    refs.append((out.get("text") or "", tps(out)))
mem_full = mem()
print(f"resident with shadows (baseline) = {mem_full:.2f} GiB", flush=True)
for i, (t, x) in enumerate(refs):
    print(f"  baseline[{i}] tps={x:.2f} text={t[:60]!r}", flush=True)

# --- multi-request decode-only cycle: reload -> generate(release after prefill) -> (decode @ ~27) ---
print("\n=== multi-request cycle: reload -> prefill -> release -> decode ===", flush=True)
rows = []
for i, p in enumerate(PROMPTS):
    rl = runner.reload_decode_bf16_shadows()   # req0: noop; later: rebuild from packed (no disk)
    mem_after_reload = mem()
    out = runner.generate(p, max_new=MAXNEW, release_decode_shadows_after_prefill=True)
    mem_after_decode = mem()
    text = out.get("text") or ""
    exact = text == refs[i][0]
    rows.append({
        "req": i,
        "reload_s": round(float(rl.get("seconds", 0.0)), 2),
        "reload_gib": round(float(rl.get("reloaded_gib", 0.0)), 1),
        "mem_after_reload": round(mem_after_reload, 1),
        "mem_after_decode": round(mem_after_decode, 1),
        "tps": round(tps(out), 2),
        "token_exact_vs_baseline": exact,
    })
    print(f"  req{i}: reload {rl.get('seconds',0.0):.2f}s (+{rl.get('reloaded_gib',0.0):.1f}GiB) "
          f"-> prefill@{mem_after_reload:.1f} -> decode@{mem_after_decode:.1f}GiB "
          f"tps={tps(out):.2f} exact={exact}", flush=True)

print("\n================ DECODE-ONLY SERVING VERDICT ================", flush=True)
all_exact = all(r["token_exact_vs_baseline"] for r in rows)
decode_mem = sum(r["mem_after_decode"] for r in rows) / max(len(rows), 1)
reload_costs = [r["reload_s"] for r in rows if r["reload_gib"] > 0]
avg_reload = sum(reload_costs) / max(len(reload_costs), 1) if reload_costs else 0.0
avg_tps = sum(r["tps"] for r in rows) / max(len(rows), 1)
base_tps = sum(x for _, x in refs) / max(len(refs), 1)
print(f"token-exact across all requests : {all_exact}", flush=True)
print(f"resident during decode          : ~{decode_mem:.1f} GiB  (baseline {mem_full:.1f}; freed ~{mem_full-decode_mem:.1f} GiB)", flush=True)
print(f"reload cost per request         : ~{avg_reload:.2f} s  (GPU re-dequant from resident packed, no disk)", flush=True)
print(f"decode TPS cycle vs baseline    : {avg_tps:.2f} vs {base_tps:.2f}  ({avg_tps/max(base_tps,1e-9):.3f}x)", flush=True)
if all_exact and (mem_full - decode_mem) > 40:
    print("=> PRODUCTIZABLE: decode-only runs token-exact at ~27 GiB; reload makes it multi-request-safe.", flush=True)
    print(f"   60 GiB freed during decode (the long phase) -> multi-service / batch headroom on the 128 GB box.", flush=True)
    print(f"   Tradeoff: ~{avg_reload:.1f}s reload per prefill (only if interleaving prefills); pure long-session pays it ZERO.", flush=True)
else:
    print("=> NOT clean: check token-exactness / freed bytes above before wiring into serving.", flush=True)
print(json.dumps({"baseline_mem": mem_full, "rows": rows}, ensure_ascii=False), flush=True)
