#!/usr/bin/env python3
"""Stage 6 BANK: verify the 60 GiB decode-only memory win as a SERVING capability.

Stage-6 step2/step3 proved DECODE runs shadow-free (release_decode_bf16_shadows
drops ~60 GiB, resident 87->27, decode keeps going at ~same TPS). That was a
probe. This banks it: adds reload_decode_bf16_shadows() (rebuild the BF16 shadow
from the resident packed NVFP4, no disk I/O) and wires the server to the
per-request cycle  reload -> prefill -> release -> decode  (no longer one-shot).

What this verifies (option (b), the cheaper bank the user chose):
  1. resident drops 87 -> ~27 GiB during decode (release), back to ~87 on reload.
  2. TOKEN-EXACT: a request whose PREFILL uses the RELOADED shadow produces the
     same token ids as the pristine BF16 baseline  (== reload is correct).
  3. per-request reload cost is measured (the price of option (b)).
  4. NO ~45 TPS decode regression (shadow-free decode is >= shadows-present).
  5. KV headroom: the freed ~60 GiB is genuinely reclaimable -- a big-max_seq_len
     KV cache that does NOT fit in free@87 allocates fine at 27 (proven by a real
     safe allocation + mem_get_info arithmetic; we do NOT trigger a kernel OOM on
     the shared unified-memory box).
  6. SERVER wiring: LynnEngineHandle.generate() can be called repeatedly (the old
     one-shot 409 is gone) and stays token-exact across reload cycles.

Run in docker lynn-eval-base:cu13, PYTHONNOUSERSITE=1, APEX stopped.
"""
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same RC-validated ~45 TPS serving stack as the Stage-6 probes.
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
from engine.inference_state import LynnInferenceState

MODEL = os.environ.get("MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526")
P = os.environ.get(
    "PROMPT",
    "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step, "
    "then explain why distance equals speed times time.",
)
N = int(os.environ.get("DECODE_STEPS", "48"))
GIB = 1024 ** 3


def mem_alloc_gib() -> float:
    return torch.cuda.memory_allocated() / GIB if torch.cuda.is_available() else 0.0


def mem_free_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / GIB


def gen(runner, release: bool):
    """One full generate; return (new_ids, decode_tps, prefill_seconds, release_gib)."""
    r = runner.generate(P, max_new=N, release_decode_shadows_after_prefill=release)
    t = r["timings"]
    rel = t.get("decode_bf16_shadow_release")
    return (
        [int(x) for x in r["new_ids"]],
        t.get("decode_tps"),
        t.get("prefill_seconds"),
        (rel or {}).get("released_gib") if isinstance(rel, dict) else None,
    )


def kv_bytes_per_token(runner) -> int:
    """KV bytes/token = full_attn_layers * num_kv_heads * head_dim * 2(K,V) * elt."""
    s = LynnInferenceState.from_config(runner.cfg, batch=1, max_seq_len=8, device="cpu", dtype=runner.dtype)
    n_full = sum(1 for t in s.layer_types if t == "full_attention")
    elt = torch.tensor([], dtype=runner.dtype).element_size()
    return n_full * s.num_kv_heads * s.head_dim * 2 * elt


def main():
    out = {"model": MODEL, "decode_steps": N}
    runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
    torch.cuda.synchronize()
    out["mem_after_load_gib"] = round(mem_alloc_gib(), 2)
    print(f"[load] resident = {out['mem_after_load_gib']:.2f} GiB", flush=True)

    # warmup (no release)
    runner.generate("warm up.", max_new=8)

    # ---- Reference: pristine BF16 baseline, shadows present throughout ----
    refA_ids, refA_tps, refA_prefill, _ = gen(runner, release=False)
    print(f"[refA] baseline (shadows present): decode {refA_tps:.2f} tok/s, "
          f"prefill {refA_prefill:.3f}s", flush=True)
    out["baseline"] = {"decode_tps": refA_tps, "prefill_seconds": refA_prefill,
                       "first8_ids": refA_ids[:8]}

    # ---- Request 1: release shadows after prefill, decode at ~27 GiB ----
    r1_ids, r1_tps, r1_prefill, r1_rel = gen(runner, release=True)
    torch.cuda.synchronize()
    out["mem_after_release_gib"] = round(mem_alloc_gib(), 2)
    out["released_gib_reported"] = round(r1_rel, 2) if r1_rel else None
    print(f"[req1] released {r1_rel:.2f} GiB -> resident {out['mem_after_release_gib']:.2f} GiB; "
          f"shadow-free decode {r1_tps:.2f} tok/s", flush=True)

    # ---- KV headroom demo (SAFE: real alloc at 27, arithmetic at 87) ----
    kv_per_tok = kv_bytes_per_token(runner)
    free27 = mem_free_gib()
    target_gib = free27 * 0.55  # comfortably fits at 27, and (by construction) > free@87
    big_T = int(target_gib * GIB / kv_per_tok)
    headroom = {"kv_bytes_per_token": kv_per_tok, "free_at_27_gib": round(free27, 2),
                "target_kv_gib": round(target_gib, 2), "big_max_seq_len": big_T}
    print(f"[headroom] KV={kv_per_tok}B/tok  free@27={free27:.1f}GiB  "
          f"alloc target={target_gib:.1f}GiB (max_seq_len={big_T:,})", flush=True)
    try:
        big = LynnInferenceState.from_config(runner.cfg, batch=1, max_seq_len=big_T,
                                             device="cuda", dtype=runner.dtype)
        torch.cuda.synchronize()
        headroom["alloc_at_27_gib"] = round(big.memory_bytes() / GIB, 2)
        headroom["alloc_at_27_ok"] = True
        print(f"[headroom] OK: allocated {headroom['alloc_at_27_gib']:.1f} GiB KV at 27-resident",
              flush=True)
        del big
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        headroom["alloc_at_27_ok"] = False
        headroom["alloc_at_27_error"] = repr(e)
        torch.cuda.empty_cache()
        print(f"[headroom] alloc@27 FAILED: {e!r}", flush=True)

    # ---- Reload shadows (rebuild BF16 from resident packed) ----
    rep = runner.reload_decode_bf16_shadows()
    torch.cuda.synchronize()
    out["mem_after_reload_gib"] = round(mem_alloc_gib(), 2)
    out["reload_seconds"] = round(rep["seconds"], 3)
    out["reload_gib"] = round(rep["reloaded_gib"], 2)
    print(f"[reload] rebuilt {rep['reloaded_gib']:.2f} GiB in {rep['seconds']:.3f}s "
          f"-> resident {out['mem_after_reload_gib']:.2f} GiB", flush=True)

    free87 = mem_free_gib()
    headroom["free_at_87_gib"] = round(free87, 2)
    headroom["would_oom_at_87"] = bool(target_gib > free87)
    print(f"[headroom] free@87={free87:.1f}GiB ; same {target_gib:.1f}GiB KV "
          f"{'WOULD OOM' if headroom['would_oom_at_87'] else 'would fit'} at 87-resident", flush=True)
    out["headroom"] = headroom

    # ---- Request 2: PREFILL now uses the RELOADED shadow (reload-correctness) ----
    r2_ids, r2_tps, r2_prefill, _ = gen(runner, release=True)
    torch.cuda.synchronize()
    out["mem_after_req2_release_gib"] = round(mem_alloc_gib(), 2)
    print(f"[req2] reloaded-shadow prefill {r2_prefill:.3f}s; shadow-free decode {r2_tps:.2f} tok/s "
          f"-> resident {out['mem_after_req2_release_gib']:.2f} GiB", flush=True)

    # ---- Server wiring: LynnEngineHandle, repeated calls, no one-shot 409 ----
    from server.openai_http import LynnEngineHandle, EngineConfig
    handle = LynnEngineHandle(EngineConfig(model_dir=MODEL))
    handle.runner = runner
    handle.tokenizer = runner.tokenizer
    handle.ready = True
    handle.release_decode_shadows_after_prefill = True
    h1 = handle.generate(P, N)   # detects released-from-req2 -> reload -> prefill -> release -> decode
    h2 = handle.generate(P, N)   # again (old code would 409 here)
    hA_ids = [int(x) for x in h1["new_token_ids"]]
    hB_ids = [int(x) for x in h2["new_token_ids"]]
    out["handle"] = {
        "call1_reload_seconds": h1.get("reload_decode_shadows_seconds"),
        "call2_reload_seconds": h2.get("reload_decode_shadows_seconds"),
        "release_reload_count": h2.get("release_reload_count"),
    }
    print(f"[handle] 2 sequential requests OK (no one-shot 409); "
          f"reload_count={out['handle']['release_reload_count']}", flush=True)

    # ---- Verdicts ----
    tok_exact_decode = (r1_ids == refA_ids)              # shadow-free decode == baseline
    tok_exact_reload = (r2_ids == refA_ids)              # reloaded-shadow prefill == baseline
    tok_exact_handle = (hA_ids == refA_ids == hB_ids)    # server path == baseline, both calls
    resident_dropped = out["mem_after_release_gib"] <= out["mem_after_load_gib"] - 45
    resident_restored = out["mem_after_reload_gib"] >= out["mem_after_load_gib"] - 5
    worst_decode = min(r1_tps, r2_tps)
    no_tps_regress = worst_decode >= refA_tps * 0.97

    out["verdict"] = {
        "token_exact_shadow_free_decode": tok_exact_decode,
        "token_exact_reloaded_prefill": tok_exact_reload,
        "token_exact_server_path": tok_exact_handle,
        "resident_dropped_>=45GiB": resident_dropped,
        "resident_restored_on_reload": resident_restored,
        "no_decode_tps_regression": no_tps_regress,
        "kv_headroom_reclaimable": bool(headroom.get("alloc_at_27_ok") and headroom.get("would_oom_at_87")),
        "decode_tps": {"baseline": refA_tps, "req1_shadowfree": r1_tps, "req2_shadowfree": r2_tps},
    }
    out["ALL_PASS"] = all([
        tok_exact_decode, tok_exact_reload, tok_exact_handle,
        resident_dropped, resident_restored, no_tps_regress,
        out["verdict"]["kv_headroom_reclaimable"],
    ])

    print("\n=============== BANK VERDICT ===============", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    print(f"\nALL_PASS = {out['ALL_PASS']}", flush=True)


if __name__ == "__main__":
    main()
