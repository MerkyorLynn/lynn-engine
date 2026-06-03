#!/usr/bin/env python3
"""Stage 6 P0.2: resident BF16 inventory after the P0.1 MoE-shadow release.

P0.1 proved no-reload prefill can be token-exact while the 60 GiB grouped-MoE
shadow is non-resident. P0.2 answers the next question before writing kernels:
what BF16 tensors still live after that release, and which of them have packed
aliases/could plausibly move to packed-prefill?

This script is an inventory gate, not a benchmark. It does not run generation.
Run on Spark in docker lynn-eval-base:cu13, PYTHONNOUSERSITE=1, APEX stopped.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import defaultdict
from typing import Any

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
os.environ["LYNN_PACKED_PREFILL_SLOW"] = "0"

if os.environ.get("LYNN_P02_ATTACH_PROJECTION_ALIASES", "0") == "1":
    # Inventory-only mode. Do not run decode with these flags unless explicitly
    # testing semantics; P0.2 first needs the byte/candidate table.
    os.environ["LYNN_PACKED_DECODE_LINEAR_ATTN"] = "1"
    os.environ["LYNN_PACKED_DECODE_FULL_ATTN"] = "1"
    os.environ["LYNN_PACKED_SHARED_EXPERT"] = "1"

import torch  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


MODEL = os.environ.get(
    "MODEL",
    "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526",
)
GIB = 1024**3


def mem_alloc_gib() -> float:
    return torch.cuda.memory_allocated() / GIB if torch.cuda.is_available() else 0.0


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def category(key: str, scope: str) -> str:
    if scope == "outside":
        if "embed_tokens" in key:
            return "outside.embed"
        if "lm_head" in key:
            return "outside.lm_head"
        if "norm" in key:
            return "outside.norm"
        return "outside.other"
    if "mlp.experts." in key:
        return "moe.grouped_expert_shadow"
    if "mlp.shared_expert" in key:
        return "moe.shared_expert"
    if "mlp.shared_expert_gate" in key:
        return "moe.shared_gate"
    if "mlp.gate.weight" in key:
        return "moe.router"
    if key.endswith("layernorm.weight"):
        return "layernorm"
    if key.startswith("self_attn."):
        return "full_attn.projection"
    if key.startswith("linear_attn."):
        return "linear_attn.projection"
    if key.startswith("mlp."):
        return "dense_ffn"
    return "other"


def packed_alias_exists(w: dict[str, Any], key: str) -> bool:
    return key + ".packed" in w


def collect_layer_bf16(runner: LynnIncrementalRunner) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_idx, w in enumerate(runner.layer_weights):
        for key, value in sorted(w.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if value.dtype != torch.bfloat16:
                continue
            nbytes = tensor_nbytes(value)
            rows.append(
                {
                    "scope": "layer",
                    "layer": layer_idx,
                    "key": key,
                    "category": category(key, "layer"),
                    "bytes": nbytes,
                    "gib": nbytes / GIB,
                    "shape": list(value.shape),
                    "packed_alias": packed_alias_exists(w, key),
                }
            )
    return rows


def collect_outside_bf16(runner: LynnIncrementalRunner) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in sorted(runner.outside.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if value.dtype != torch.bfloat16:
            continue
        nbytes = tensor_nbytes(value)
        rows.append(
            {
                "scope": "outside",
                "layer": None,
                "key": key,
                "category": category(key, "outside"),
                "bytes": nbytes,
                "gib": nbytes / GIB,
                "shape": list(value.shape),
                "packed_alias": False,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, int] = defaultdict(int)
    alias_bytes = 0
    for row in rows:
        by_cat[str(row["category"])] += int(row["bytes"])
        if row.get("packed_alias"):
            alias_bytes += int(row["bytes"])
    return {
        "total_tensors": len(rows),
        "total_gib": sum(int(r["bytes"]) for r in rows) / GIB,
        "packed_alias_candidate_gib": alias_bytes / GIB,
        "by_category_gib": {
            key: value / GIB
            for key, value in sorted(by_cat.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "top_tensors": sorted(rows, key=lambda r: int(r["bytes"]), reverse=True)[:40],
    }


def main() -> None:
    out: dict[str, Any] = {
        "schema": "lynn-stage6-p0.2-resident-inventory-v1",
        "model": MODEL,
        "attach_projection_aliases": os.environ.get("LYNN_P02_ATTACH_PROJECTION_ALIASES", "0") == "1",
    }
    runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=False)
    torch.cuda.synchronize()
    out["mem_after_load_gib"] = round(mem_alloc_gib(), 2)
    before = collect_outside_bf16(runner) + collect_layer_bf16(runner)
    out["before_release"] = summarize(before)
    print(f"[load] resident={out['mem_after_load_gib']:.2f} GiB", flush=True)
    print(f"[before] bf16_total={out['before_release']['total_gib']:.2f} GiB", flush=True)

    release = runner.release_decode_bf16_shadows(
        include_moe_experts=True,
        include_projection_aliases=False,
    )
    torch.cuda.synchronize()
    out["release"] = {
        "released_tensors": int(release["released_tensors"]),
        "released_gib": round(float(release["released_gib"]), 2),
        "resident_after_release_gib": round(mem_alloc_gib(), 2),
    }
    after = collect_outside_bf16(runner) + collect_layer_bf16(runner)
    out["after_moe_shadow_release"] = summarize(after)

    print(
        f"[release] dropped={out['release']['released_gib']:.2f} GiB "
        f"resident={out['release']['resident_after_release_gib']:.2f} GiB",
        flush=True,
    )
    print(f"[after] bf16_total={out['after_moe_shadow_release']['total_gib']:.2f} GiB", flush=True)
    print("[after] by category:", flush=True)
    for key, value in out["after_moe_shadow_release"]["by_category_gib"].items():
        print(f"  {key:28s} {value:8.3f} GiB", flush=True)

    out["next_gate"] = {
        "p0_2": "decide which remaining BF16 residents can be converted without changing decode semantics",
        "p1": "batched packed projection prefill kernels",
        "p2": "grouped M>1 packed MoE prefill kernel, token-exact vs stream_bf16",
    }
    print("\n=============== STAGE 6 P0.2 RESIDENT INVENTORY ===============", flush=True)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
