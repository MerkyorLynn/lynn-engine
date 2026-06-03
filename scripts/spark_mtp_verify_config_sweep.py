#!/usr/bin/env python3
"""MTP verify-config sweep -- accept + TPS + token-exact, ONE model load (2026-06-03).

Stage 5 / task A step 3. The offset probe PROVED the MTP draft head is excellent
(91.5% top-1 for x_{p+2}); the block-verify probe localized the 2.4% A/B accept to
the T>=2 batched verify path (linear-attn k2/block divergence per M16/M17), not the
draft. This sweep measures, in a single process (toggling env per generate), each
candidate verify config end-to-end:

  base            : LYNN_MTP_SPECULATIVE=0           (truth text + baseline TPS)
  seq_k1          : BATCHED=0  K=1                    (verify = decode_one, T=1, correct) -> accept ceiling
  k1b_default     : BATCHED=1  K=1                    (T=2 k2 verify, linear-attn k2 default)
  k1b_lin_t1loop  : BATCHED=1  K=1  LIN=t1_loop       (T=2, linear-attn strict T=1 -> correctness fix?)
  k2_default      : BATCHED=1  K=2                    (reproduces the 2.4% A/B; T=3 block)
  k2_lin_t1loop   : BATCHED=1  K=2  LIN=t1_loop       (knob only affects T=2 k2, not T=3 block -> control)
  k1b_fast        : BATCHED=1  K=1  LIN=t1_loop  FULL_ATTN_K2=k2  SMALLM=1   (true-batched verify attempt -> TPS win?)

For each: accept = accepted_draft_tokens/draft_tokens_proposed, decode_tps,
tps vs base, token-exact vs base text. Run in docker lynn-eval-base:cu13,
PYTHONNOUSERSITE=1, APEX stopped.
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
os.environ.setdefault("LYNN_RMSNORM_FUSED", "1")
os.environ.setdefault("LYNN_FULL_ATTN_FUSED", "1")
os.environ.setdefault("LYNN_SHARED_EXPERT_FUSED", "1")
os.environ.setdefault("LYNN_LINEAR_ATTN_FUSE_GBETA", "1")
os.environ.setdefault("LYNN_NVFP4_BF16_OUT", "1")
os.environ.setdefault("LYNN_DECODE_OPROJ_NOCOPY", "1")

SIDECAR = os.environ.get(
    "LYNN_MTP_SIDECAR",
    "/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp/mtp.safetensors",
)
os.environ["LYNN_MTP_SIDECAR"] = SIDECAR
os.environ["LYNN_MTP_SPECULATIVE"] = "1"  # init-time: load sidecar

import torch
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get(
    "MODEL",
    "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526",
)
MAXNEW = int(os.environ.get("MAXNEW", "64"))
P = "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step."

# env knobs we toggle between configs (reset each iteration to avoid leakage)
TOGGLE_KEYS = [
    "LYNN_MTP_SPECULATIVE", "LYNN_MTP_SPECULATIVE_BATCHED", "LYNN_MTP_SPECULATIVE_K",
    "LYNN_MTP_K2_LINEAR_ATTN_MODE", "LYNN_FULL_ATTN_K2_BACKEND", "LYNN_MTP_VERIFY_SMALLM",
]

CONFIGS = [
    ("seq_k1",         {"LYNN_MTP_SPECULATIVE": "1", "LYNN_MTP_SPECULATIVE_BATCHED": "0", "LYNN_MTP_SPECULATIVE_K": "1"}),
    ("k1b_default",    {"LYNN_MTP_SPECULATIVE": "1", "LYNN_MTP_SPECULATIVE_BATCHED": "1", "LYNN_MTP_SPECULATIVE_K": "1"}),
    ("k1b_lin_t1loop", {"LYNN_MTP_SPECULATIVE": "1", "LYNN_MTP_SPECULATIVE_BATCHED": "1", "LYNN_MTP_SPECULATIVE_K": "1",
                        "LYNN_MTP_K2_LINEAR_ATTN_MODE": "t1_loop"}),
    ("k2_default",     {"LYNN_MTP_SPECULATIVE": "1", "LYNN_MTP_SPECULATIVE_BATCHED": "1", "LYNN_MTP_SPECULATIVE_K": "2"}),
    ("k2_lin_t1loop",  {"LYNN_MTP_SPECULATIVE": "1", "LYNN_MTP_SPECULATIVE_BATCHED": "1", "LYNN_MTP_SPECULATIVE_K": "2",
                        "LYNN_MTP_K2_LINEAR_ATTN_MODE": "t1_loop"}),
    ("k1b_fast",       {"LYNN_MTP_SPECULATIVE": "1", "LYNN_MTP_SPECULATIVE_BATCHED": "1", "LYNN_MTP_SPECULATIVE_K": "1",
                        "LYNN_MTP_K2_LINEAR_ATTN_MODE": "t1_loop", "LYNN_FULL_ATTN_K2_BACKEND": "k2",
                        "LYNN_MTP_VERIFY_SMALLM": "1"}),
]


def _apply(cfg):
    for k in TOGGLE_KEYS:
        os.environ.pop(k, None)
    for k, v in cfg.items():
        os.environ[k] = v


def _tps(out):
    return float((out.get("timings", {}) or {}).get("decode_tps") or 0.0)


def _accept(out):
    m = out.get("mtp_speculative", {}) or {}
    p = float(m.get("draft_tokens_proposed") or 0)
    a = float(m.get("accepted_draft_tokens") or 0)
    return (a / p if p else 0.0), int(a), int(p), int(m.get("tokens_committed") or 0), int(m.get("events") or 0)


def run(n=3):
    tps = 0.0
    last = None
    for _ in range(n):
        out = runner.generate(P, max_new=MAXNEW)
        tps = max(tps, _tps(out))
        last = out
    return tps, last


runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=True)
print("MTP sidecar loaded?:", getattr(runner, "mtp_sidecar_loaded", None), flush=True)
runner.generate("warm up.", max_new=8)

# baseline (no speculation) = truth text + reference TPS
_apply({"LYNN_MTP_SPECULATIVE": "0"})
tps_base, out_base = run()
text_base = out_base.get("text") or ""
print(f"\nbase (no-spec): {tps_base:.2f} TPS", flush=True)
print("base text head:", repr(text_base[:90]), flush=True)

rows = []
for name, cfg in CONFIGS:
    _apply(cfg)
    runner.generate("warm up mtp.", max_new=8)
    try:
        tps, out = run()
        acc, a, p, committed, events = _accept(out)
        text = out.get("text") or ""
        rows.append({
            "config": name, "tps": round(tps, 2), "x_base": round(tps / max(tps_base, 1e-9), 3),
            "accept": round(acc, 3), "acc_raw": f"{a}/{p}", "committed": committed, "events": events,
            "token_exact": text == text_base,
        })
        print(f"  [{name}] tps={tps:.2f} ({tps/max(tps_base,1e-9):.3f}x) accept={acc:.1%} ({a}/{p}) "
              f"committed={committed} exact={text == text_base}", flush=True)
    except Exception as e:
        rows.append({"config": name, "error": repr(e)})
        print(f"  [{name}] ERROR {e!r}", flush=True)

print("\n================ VERIFY-CONFIG SWEEP SUMMARY ================", flush=True)
print(f"base no-spec TPS = {tps_base:.2f}", flush=True)
print(f"{'config':>16} | {'TPS':>7} | {'xbase':>6} | {'accept':>8} | {'raw':>8} | {'exact':>5}", flush=True)
for r in rows:
    if "error" in r:
        print(f"{r['config']:>16} | ERROR {r['error']}", flush=True)
        continue
    print(f"{r['config']:>16} | {r['tps']:>7.2f} | {r['x_base']:>6.3f} | {r['accept']:>7.1%} | {r['acc_raw']:>8} | {str(r['token_exact']):>5}", flush=True)
print("\nINTERPRETATION:", flush=True)
print(" - seq_k1 accept = the head+verify ceiling (verify is correct T=1). If ~0.85-0.9 -> draft pipeline is sound.", flush=True)
print(" - k1b_default low but k1b_lin_t1loop high -> linear-attn k2 divergence is the correctness bug (knob fixes it).", flush=True)
print(" - any config with xbase>1.0 AND high accept AND exact -> a real TPS win. If none -> need true-batched verify kernels.", flush=True)
print(json.dumps(rows, ensure_ascii=False), flush=True)
