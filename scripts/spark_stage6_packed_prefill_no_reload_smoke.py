#!/usr/bin/env python3
"""Stage 6 P0.1: packed-prefill no-reload smoke.

This is the first gate after banking the 60 GiB decode-only serving win.

The already-productized serving cycle is:

    reload_decode_bf16_shadows() -> prefill -> release -> decode

It is correct, but reload costs ~23-24 s because it rebuilds the BF16 MoE
shadow. P0.1 proves the next contract: after releasing that MoE shadow, a second
request can prefill without calling reload by using the intentionally slow
`LYNN_PACKED_PREFILL_SLOW=1` proof path. This is not a speed benchmark.

Run on Spark in docker lynn-eval-base:cu13, PYTHONNOUSERSITE=1, APEX stopped.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Same stable Stage-6 serving stack used by the 60 GiB bank verifier. Keep
# projection/shared alias release out of P0.1; this gate is specifically the
# 60 GiB MoE shadow no-reload proof. Full projection/shared zero-shadow is the
# next kernel phase.
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
    "LYNN_PACKED_DECODE_LINEAR_ATTN": "0",
    "LYNN_PACKED_DECODE_FULL_ATTN": "0",
    "LYNN_PACKED_SHARED_EXPERT": "0",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
    "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_FULL_ATTN_ROPE_CACHE": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_MTP_VERIFY": "0",
    "LYNN_MTP_SHADOW_VERIFY": "0",
    "LYNN_MTP_SPECULATIVE": "0",
    "LYNN_PACKED_PREFILL_SLOW_MODE": "stream_bf16",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
for key, value in BASE_ENV.items():
    os.environ.setdefault(key, value)
os.environ.setdefault("LYNN_MOE_DOWN_BLOCK_HIDDEN", "4")
os.environ.setdefault("LYNN_LINEAR_ATTN_GQA_RECURRENT", "1")
os.environ.setdefault("LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV", "1")
os.environ.setdefault("LYNN_RMSNORM_FUSED", "1")
os.environ.setdefault("LYNN_FULL_ATTN_FUSED", "1")
os.environ.setdefault("LYNN_SHARED_EXPERT_FUSED", "1")
os.environ.setdefault("LYNN_LINEAR_ATTN_FUSE_GBETA", "1")
os.environ.setdefault("LYNN_NVFP4_BF16_OUT", "1")
os.environ.setdefault("LYNN_DECODE_OPROJ_NOCOPY", "1")

# Baseline must use normal BF16 prefill. The slow packed proof path is enabled
# only after the runner has released the MoE BF16 shadow.
os.environ["LYNN_PACKED_PREFILL_SLOW"] = "0"

import torch  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


MODEL = os.environ.get(
    "MODEL",
    "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526",
)
PROMPT = os.environ.get("PROMPT", "Solve: 2 + 2 =")
MAXNEW = int(os.environ.get("MAXNEW", "8"))
GIB = 1024**3


def mem_alloc_gib() -> float:
    return torch.cuda.memory_allocated() / GIB if torch.cuda.is_available() else 0.0


def peak_alloc_gib() -> float:
    return torch.cuda.max_memory_allocated() / GIB if torch.cuda.is_available() else 0.0


def gen_ids(out: dict[str, Any]) -> list[int]:
    return [int(x) for x in out.get("new_ids", [])]


def main() -> None:
    out: dict[str, Any] = {
        "schema": "lynn-stage6-p0.1-packed-prefill-no-reload-smoke-v1",
        "model": MODEL,
        "prompt": PROMPT,
        "max_new": MAXNEW,
        "gate_scope": "MoE BF16 shadow only; projection/shared alias full zero-shadow is later P0.2/P1",
        "packed_prefill_slow_mode": os.environ.get("LYNN_PACKED_PREFILL_SLOW_MODE", "stream_bf16"),
    }

    runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
    torch.cuda.synchronize()
    out["mem_after_load_gib"] = round(mem_alloc_gib(), 2)
    print(f"[load] resident={out['mem_after_load_gib']:.2f} GiB", flush=True)

    runner.generate("warm up.", max_new=2)
    torch.cuda.synchronize()

    ref = runner.generate(PROMPT, max_new=MAXNEW, release_decode_shadows_after_prefill=False)
    torch.cuda.synchronize()
    ref_ids = gen_ids(ref)
    ref_tps = float(ref["timings"].get("decode_tps") or 0.0)
    ref_prefill = float(ref["timings"].get("prefill_seconds") or 0.0)
    mem_full = mem_alloc_gib()
    out["baseline"] = {
        "new_ids": ref_ids,
        "text_prefix": (ref.get("text") or "")[:120],
        "prefill_seconds": ref_prefill,
        "decode_tps": ref_tps,
        "resident_gib": round(mem_full, 2),
    }
    print(
        f"[baseline] prefill={ref_prefill:.3f}s decode={ref_tps:.2f} tok/s "
        f"resident={mem_full:.2f} GiB ids={ref_ids[:8]}",
        flush=True,
    )

    release = runner.release_decode_bf16_shadows(
        include_moe_experts=True,
        include_projection_aliases=False,
    )
    torch.cuda.synchronize()
    mem_after_release = mem_alloc_gib()
    out["release"] = {
        "released_tensors": int(release["released_tensors"]),
        "released_gib": round(float(release["released_gib"]), 2),
        "resident_after_release_gib": round(mem_after_release, 2),
        "sample_items": release.get("items", [])[:8],
    }
    print(
        f"[release] dropped={release['released_gib']:.2f} GiB "
        f"resident={mem_after_release:.2f} GiB tensors={release['released_tensors']}",
        flush=True,
    )

    reload_calls: list[float] = []
    original_reload = runner.reload_decode_bf16_shadows

    def forbidden_reload(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        reload_calls.append(time.time())
        raise RuntimeError("P0.1 forbids reload_decode_bf16_shadows()")

    runner.reload_decode_bf16_shadows = forbidden_reload  # type: ignore[method-assign]
    os.environ["LYNN_PACKED_PREFILL_SLOW"] = "1"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    probe_t0 = time.time()
    probe_error: str | None = None
    probe_traceback: str | None = None
    probe: dict[str, Any] | None = None
    try:
        probe = runner.generate(PROMPT, max_new=MAXNEW, release_decode_shadows_after_prefill=True)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        probe_error = repr(exc)
        probe_traceback = traceback.format_exc()
        print(f"[probe] FAILED: {probe_error}", flush=True)
    finally:
        runner.reload_decode_bf16_shadows = original_reload  # type: ignore[method-assign]
    probe_seconds = time.time() - probe_t0

    peak_probe = peak_alloc_gib()
    mem_after_probe = mem_alloc_gib()
    probe_ids = gen_ids(probe) if probe is not None else []
    probe_timing = (probe or {}).get("timings", {}) if probe is not None else {}
    release2 = probe_timing.get("decode_bf16_shadow_release") if isinstance(probe_timing, dict) else None
    out["probe"] = {
        "error": probe_error,
        "traceback_tail": "\n".join((probe_traceback or "").splitlines()[-12:]) if probe_traceback else None,
        "new_ids": probe_ids,
        "text_prefix": ((probe or {}).get("text") or "")[:120] if probe is not None else "",
        "prefill_seconds": probe_timing.get("prefill_seconds") if isinstance(probe_timing, dict) else None,
        "decode_tps": probe_timing.get("decode_tps") if isinstance(probe_timing, dict) else None,
        "wall_seconds": round(probe_seconds, 3),
        "resident_after_probe_gib": round(mem_after_probe, 2),
        "peak_alloc_during_probe_gib": round(peak_probe, 2),
        "second_release_report": release2,
        "reload_calls": len(reload_calls),
    }
    if probe is not None:
        print(
            f"[probe] prefill={float(probe_timing.get('prefill_seconds') or 0.0):.3f}s "
            f"decode={float(probe_timing.get('decode_tps') or 0.0):.2f} tok/s "
            f"resident={mem_after_probe:.2f} GiB peak={peak_probe:.2f} GiB ids={probe_ids[:8]}",
            flush=True,
        )

    token_exact = probe_ids == ref_ids
    release_big_enough = float(release["released_gib"]) >= 45.0
    resident_dropped = mem_after_release <= mem_full - 40.0
    no_reload = len(reload_calls) == 0
    no_exception = probe_error is None
    # This is a smoke threshold, not a hard memory optimization claim. It catches
    # accidental 60 GiB reloads while allowing normal allocator/KV noise.
    no_hidden_reload_peak = peak_probe <= mem_full - 30.0
    resident_stays_low = mem_after_probe <= mem_full - 30.0

    out["verdict"] = {
        "token_exact_vs_bf16_prefill": token_exact,
        "released_moe_shadow_gib_>=45": release_big_enough,
        "resident_dropped_>=40GiB": resident_dropped,
        "no_reload_call": no_reload,
        "no_exception": no_exception,
        "no_hidden_reload_peak": no_hidden_reload_peak,
        "resident_stays_low_after_probe": resident_stays_low,
    }
    out["ALL_PASS"] = all(out["verdict"].values())

    print("\n=============== STAGE 6 P0.1 NO-RELOAD VERDICT ===============", flush=True)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print(f"\nALL_PASS = {out['ALL_PASS']}", flush=True)
    if not out["ALL_PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
