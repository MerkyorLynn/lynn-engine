#!/usr/bin/env python3
"""P32: full generate gate for native active-MoE runtime backend.

P31 showed that `LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar` is exact and faster
at the MoE function boundary. P32 checks whether that signal survives a full
runner.generate path with linear block graphs and native FP4 lm_head enabled.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from statistics import mean, median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


PROMPTS = [
    "用一句话解释 MoE active parameters",
    "Python 写一个递归阶乘函数",
    "比较 RoPE 与 ALiBi 的优缺点",
]


def _set_runtime_env(
    backend: str,
    *,
    linear_block_graph: bool,
    native_layers: str | None,
) -> dict[str, str | None]:
    graph_flag = "1" if linear_block_graph else "0"
    updates = {
        "LYNN_PREFILL_WARMUP": "1",
        "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
        "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
        "LYNN_MOE_IMPL": "packed_nvfp4",
        "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
        "LYNN_RMSNORM_GATED_BACKEND": "triton",
        "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
        "LYNN_NATIVE_FP4_LM_HEAD": "1",
        "LYNN_LINEAR_STATE_UPDATE": "inplace",
        "LYNN_LINEAR_BLOCK_GRAPH": graph_flag,
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": graph_flag,
        "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": graph_flag,
        "LYNN_PACKED_DECODE": "0",
        "LYNN_PACKED_DECODE_PREPARE_NATIVE": "0",
        "LYNN_PACKED_SHARED_EXPERT": "0",
        "LYNN_NATIVE_ACTIVE_MOE_BACKEND": backend,
    }
    if native_layers is not None:
        updates["LYNN_NATIVE_ACTIVE_MOE_LAYERS"] = native_layers
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["timings"]["decode_tps"]) for r in rows if r["timings"].get("decode_tps")]
    return {
        "count": len(rows),
        "decode_tps_mean": mean(vals) if vals else None,
        "decode_tps_median": median(vals) if vals else None,
        "new_ids_all_match_reference": all(r.get("new_ids_match_reference", True) for r in rows),
    }


def _run_backend(
    model: str,
    backend: str,
    *,
    max_new: int,
    prompts: list[str],
    linear_block_graph: bool,
    native_layers: str | None,
) -> list[dict[str, Any]]:
    old = _set_runtime_env(backend, linear_block_graph=linear_block_graph, native_layers=native_layers)
    try:
        runner = LynnIncrementalRunner(model, device="cuda", dtype=torch.bfloat16, verbose=False)
        rows = []
        for i, prompt in enumerate(prompts):
            out = runner.generate(prompt, max_new=max_new, use_chat_template=False)
            rows.append({
                "prompt_id": f"prompt_{i:03d}",
                "prompt": prompt,
                "backend": backend,
                "linear_block_graph": linear_block_graph,
                "native_active_moe_layers": native_layers,
                "new_ids": out["new_ids"],
                "completion_text": out["completion_text"],
                "timings": out["timings"],
                "stopped_reason": out["stopped_reason"],
            })
        return rows
    finally:
        _restore_env(old)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--prompts-jsonl")
    ap.add_argument(
        "--linear-block-graph",
        choices=("0", "1"),
        default="1",
        help="Enable the production reusable linear-block CUDA graph path.",
    )
    ap.add_argument(
        "--native-active-moe-layers",
        default=None,
        help="Optional cuda_scalar allowlist, e.g. 'full_attention', 'linear_attention', or comma-separated layer ids.",
    )
    args = ap.parse_args()
    linear_block_graph = args.linear_block_graph == "1"

    prompts = PROMPTS
    if args.prompts_jsonl:
        prompts = []
        with open(args.prompts_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("input") or item.get("question") or item.get("problem")
                if prompt:
                    prompts.append(str(prompt))
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompts_jsonl}")

    triton_rows = _run_backend(
        args.model,
        "triton",
        max_new=args.max_new,
        prompts=prompts,
        linear_block_graph=linear_block_graph,
        native_layers=args.native_active_moe_layers,
    )
    cuda_rows = _run_backend(
        args.model,
        "cuda_scalar",
        max_new=args.max_new,
        prompts=prompts,
        linear_block_graph=linear_block_graph,
        native_layers=args.native_active_moe_layers,
    )
    for ref, cand in zip(triton_rows, cuda_rows, strict=True):
        cand["new_ids_match_reference"] = cand["new_ids"] == ref["new_ids"]
        cand["reference_new_ids"] = ref["new_ids"]

    result = {
        "schema_version": "lynn-engine-p32-native-active-moe-generate-gate-v1",
        "model": args.model,
        "max_new": args.max_new,
        "prompt_count": len(prompts),
        "linear_block_graph": linear_block_graph,
        "native_active_moe_layers": args.native_active_moe_layers,
        "triton": {
            "rows": triton_rows,
            "summary": _summary(triton_rows),
        },
        "cuda_scalar": {
            "rows": cuda_rows,
            "summary": _summary(cuda_rows),
        },
        "pass": all(row["new_ids_match_reference"] for row in cuda_rows),
        "promote_default": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
