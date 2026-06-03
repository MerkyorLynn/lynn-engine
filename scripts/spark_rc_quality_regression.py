#!/usr/bin/env python3
"""RELEASE-CANDIDATE quality regression for the fused decode stack.

Confirms the 43.5-TPS stack (bh4 + RMSNorm-fused + full-attn-fused + shared-expert-fused)
is QUALITY-PRESERVING, not merely sample-coherent. The 3 fusions are non-bit-identical
(reduction-order), so we must prove capability didn't silently drop.

Same-process A/B, greedy (top_k=0):
  BASELINE = LYNN_RMSNORM_FUSED/FULL_ATTN_FUSED/SHARED_EXPERT_FUSED all 0 (original paths)
  FUSED    = all 3 on
Suites: structured-gate (scored) / V9 holdout (gold) / GPQA (answer) / tool-call / long-output.
Verdict = does FUSED match BASELINE on accuracy + agreement, no degeneration.
"""
import os, sys, json, re, pathlib
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

FUSION_FLAGS = ["LYNN_RMSNORM_FUSED", "LYNN_FULL_ATTN_FUSED", "LYNN_SHARED_EXPERT_FUSED",
                "LYNN_LINEAR_ATTN_FUSE_GBETA",            # Stage 3
                "LYNN_NVFP4_BF16_OUT", "LYNN_DECODE_OPROJ_NOCOPY"]  # Stage 4A copy-hunt — re-validate full stack
def set_fusion(on):
    for f in FUSION_FLAGS:
        os.environ[f] = "1" if on else "0"

import torch
from engine.resident_runner import LynnIncrementalRunner

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
EVP = "/home/merkyor/eval_prompts"
V9B = "/home/merkyor/lynn-v9-bench/data"

def norm(s): return re.sub(r"\s+", " ", (s or "").strip().lower())

# ---- load suites ----
def load_structured(n=12):
    p = ROOT / "scripts" / "qwen36_structured_hard_prompts_70.json"
    d = json.load(open(p))[:n]
    return d

def load_v9(n=8):
    out = []
    for line in open(f"{EVP}/v9_holdout.jsonl"):
        line = line.strip()
        if not line: continue
        try: out.append(json.loads(line))
        except Exception: pass
    return out[:n]

def load_gpqa(n=10):
    out = []
    for f in ("gpqa_physics3.json", "gpqa_chemistry3.json", "gpqa_biology3.json"):
        try: out += json.load(open(f"{V9B}/{f}"))
        except Exception: pass
    return out[:n]

def load_toolcall(n=8):
    out = []
    for line in open(f"{EVP}/stage1_tool_calling.jsonl"):
        line = line.strip()
        if not line: continue
        try: out.append(json.loads(line))
        except Exception: pass
    return out[:n]

STRUCT = load_structured(); V9 = load_v9(); GPQA = load_gpqa(); TOOLS = load_toolcall()
LONG = ["Explain how a CPU cache hierarchy (L1/L2/L3) works and why it speeds up programs.",
        "Write a short essay on why the sky appears blue, covering Rayleigh scattering."]

runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
def gen(prompt, max_new=256):
    out = runner.generate(prompt, max_new=max_new, use_chat_template=True)
    return (out.get("text") or "", (out.get("timings", {}) or {}).get("decode_tps") or 0)

# ---- scorers ----
def score_struct(item, text):
    ok = True; reasons = []
    sw = item.get("starts_with")
    if sw and not text.lstrip().startswith(sw): ok = False; reasons.append("startswith")
    for m in (item.get("must_contain") or []):
        if str(m).lower() not in text.lower(): ok = False; reasons.append(f"miss:{m}")
    for fb in (item.get("forbid") or []):
        if str(fb) in text: ok = False; reasons.append(f"forbid:{fb}")
    if item.get("parse_json"):
        try: json.loads(text.strip())
        except Exception: ok = False; reasons.append("badjson")
    return ok, reasons

def degenerate(text):
    t = text.strip()
    if len(t) < 20: return True
    for w in (12, 24):
        chunk = t[:w]
        if chunk and t.count(chunk) >= 6: return True
    return False

def run_phase(label):
    R = {}
    # structured
    sc = 0
    for it in STRUCT:
        p = (it.get("system","") + "\n\n" + it.get("prompt","")).strip()
        txt, _ = gen(p, 220)
        ok, _ = score_struct(it, txt)
        sc += int(ok)
        R.setdefault("structured_out", []).append(txt)
    R["structured_pass"] = sc; R["structured_n"] = len(STRUCT)
    # v9 (gold)
    v9c = 0
    for it in V9:
        txt, _ = gen(it.get("problem","") + " /no_think", 256)
        g = norm(str(it.get("gold_answer","")))
        if g and g in norm(txt): v9c += 1
        R.setdefault("v9_out", []).append(txt)
    R["v9_acc"] = v9c; R["v9_n"] = len(V9)
    # gpqa (answer contains)
    gc = 0
    for it in GPQA:
        txt, _ = gen(it.get("problem","") + " /no_think", 256)
        a = norm(str(it.get("answer","")))
        if a and a in norm(txt): gc += 1
        R.setdefault("gpqa_out", []).append(txt)
    R["gpqa_acc"] = gc; R["gpqa_n"] = len(GPQA)
    # tool-call (emits the function name)
    tc = 0
    for it in TOOLS:
        tools = it.get("tools", []); fn = ""
        try: fn = tools[0]["function"]["name"]
        except Exception: pass
        p = ("You can call tools: " + json.dumps(tools, ensure_ascii=False) +
             "\nRespond with a JSON tool call.\nUser: " + it.get("user",""))
        txt, _ = gen(p, 200)
        if fn and fn in txt: tc += 1
        R.setdefault("tool_out", []).append(txt)
    R["tool_pass"] = tc; R["tool_n"] = len(TOOLS)
    # long-output coherence
    deg = 0; tpsv = []
    for p in LONG:
        txt, tp = gen(p + " /no_think", 384)
        if degenerate(txt): deg += 1
        tpsv.append(tp)
        R.setdefault("long_out", []).append(txt)
    R["long_degenerate"] = deg; R["long_n"] = len(LONG); R["tps"] = max(tpsv) if tpsv else 0
    print(f"[{label}] struct {R['structured_pass']}/{R['structured_n']}  v9 {R['v9_acc']}/{R['v9_n']}  "
          f"gpqa {R['gpqa_acc']}/{R['gpqa_n']}  tool {R['tool_pass']}/{R['tool_n']}  "
          f"long-degen {R['long_degenerate']}/{R['long_n']}  tps {R['tps']:.1f}", flush=True)
    return R

def agree(a, b):
    n = min(len(a), len(b)); m = 0
    for x, y in zip(a, b):
        if norm(x)[:200] == norm(y)[:200]: m += 1
    return m, n

print("=== RC QUALITY REGRESSION: baseline (fusions OFF) vs fused stack (ON) ===", flush=True)
print(f"suites: structured={len(STRUCT)} v9={len(V9)} gpqa={len(GPQA)} tool={len(TOOLS)} long={len(LONG)}", flush=True)
set_fusion(False); runner.generate("warm up.", max_new=8, use_chat_template=True)
A = run_phase("BASELINE")
set_fusion(True); runner.generate("warm up fused.", max_new=8, use_chat_template=True)
B = run_phase("FUSED   ")

print("\n=== AGREEMENT (fused vs baseline, normalized first 200 chars) ===", flush=True)
for k, name in [("structured_out","structured"),("v9_out","v9"),("gpqa_out","gpqa"),("tool_out","tool"),("long_out","long")]:
    m, n = agree(A.get(k,[]), B.get(k,[]))
    print(f"  {name:11s}: {m}/{n} identical-greedy", flush=True)

print("\n=== VERDICT ===", flush=True)
ok = (B["structured_pass"] >= A["structured_pass"] - 1 and
      B["v9_acc"] >= A["v9_acc"] and B["gpqa_acc"] >= A["gpqa_acc"] - 1 and
      B["tool_pass"] >= A["tool_pass"] - 1 and B["long_degenerate"] <= A["long_degenerate"])
print("RC_QUALITY_PRESERVED:", ok)
print("DELTA tps: %.1f -> %.1f (%.3fx)" % (A["tps"], B["tps"], B["tps"]/max(A["tps"],1e-9)))
