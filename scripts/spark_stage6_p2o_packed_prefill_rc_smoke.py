#!/usr/bin/env python3
"""Stage 6 Phase 2-O: real-prompt no-reload packed-prefill RC smoke.

P2-N proves selected-layer synthetic-hidden coverage. P2-O moves to a small
resident-runner smoke:

1. Generate a few real prompts with the normal BF16 prefill path.
2. Release the active MoE BF16 expert shadow only.
3. Enable P2E packed MoE prefill + P2-L block linear-attn.
4. Generate the same prompts without reloading the released MoE shadow.

This does not promote the path by itself; it verifies that the combined opt-in
path can survive real tokenized prefill/decode in the resident runner.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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
    "LYNN_LINEAR_ATTN_GQA_RECURRENT": "1",
    "LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_LINEAR_STATE_UPDATE": "inplace",
    "LYNN_PACKED_DECODE": "0",
    "LYNN_PACKED_SHARED_EXPERT": "0",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
    "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_FULL_ATTN_ROPE_CACHE": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_MOE_DOWN_BLOCK_HIDDEN": "4",
    "LYNN_MTP_VERIFY": "0",
    "LYNN_MTP_SHADOW_VERIFY": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
FUSION_ENV = {
    "LYNN_RMSNORM_FUSED": "1",
    "LYNN_FULL_ATTN_FUSED": "1",
    "LYNN_SHARED_EXPERT_FUSED": "1",
    "LYNN_LINEAR_ATTN_FUSE_GBETA": "1",
    "LYNN_NVFP4_BF16_OUT": "1",
    "LYNN_DECODE_OPROJ_NOCOPY": "1",
}
for k, v in {**BASE_ENV, **FUSION_ENV}.items():
    os.environ.setdefault(k, v)

import torch  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


DEFAULT_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"
DEFAULT_PROMPTS = [
    "Answer in one sentence: what is 17 + 25?",
    "Return compact JSON with keys answer and reason for this question: is water wet?",
    "Write a tiny Python function named add_one that returns x + 1.",
]


def _long_context_prompt() -> str:
    filler = "\n".join(
        f"Context line {i:04d}: keep reading; the answer key is not on this line."
        for i in range(220)
    )
    return (
        "You are testing long-context retrieval. The secret code is LYNN-ZERO-SHADOW-42.\n"
        f"{filler}\n"
        "Question: answer with only the secret code."
    )


def _prompt_preset(name: str) -> list[str]:
    if name == "basic":
        return list(DEFAULT_PROMPTS)
    if name == "rc-mini":
        return [
            "Answer exactly and briefly: 17 + 25 = ?",
            "Return compact JSON only with keys tool and arguments for a weather lookup in Shanghai.",
            "Write a tiny Python function named add_one that returns x + 1.",
            "用一句中文解释为什么水会结冰。",
            "V9 prompt-format smoke: reply with exactly two bullet points about Lynn engine.",
            _long_context_prompt(),
        ]
    raise ValueError(f"unknown prompt preset: {name}")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _degenerate(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return True
    for w in (8, 16):
        chunk = t[:w]
        if chunk and t.count(chunk) >= 4:
            return True
    return False


def _set_prefill_optin(enabled: bool) -> dict[str, str | None]:
    updates = {
        "LYNN_PACKED_PREFILL_SLOW": "1" if enabled else "0",
        "LYNN_PACKED_PREFILL_SLOW_MODE": "p2e_hybrid",
        "LYNN_PACKED_PREFILL_P2E_LAYERS": "all",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_T": "32",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_INTER": "8",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_HIDDEN": "128",
        "LYNN_PACKED_PREFILL_P2E_NUM_WARPS": "4",
        "LYNN_PACKED_PREFILL_P2E_DOWN_BLOCK_HIDDEN": "8",
        "LYNN_PACKED_PREFILL_P2E_DOWN_BLOCK_INTER": "512",
        "LYNN_PACKED_PREFILL_P2E_DOWN_NUM_WARPS": "8",
        "LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA": "1" if enabled else "0",
    }
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _run_batch(
    runner: LynnIncrementalRunner,
    prompts: list[str],
    *,
    max_new: int,
    optin: bool,
) -> list[dict[str, Any]]:
    old = _set_prefill_optin(optin)
    rows: list[dict[str, Any]] = []
    try:
        for idx, prompt in enumerate(prompts):
            t0 = time.time()
            out = runner.generate(prompt, max_new=max_new, top_k=0, use_chat_template=True)
            elapsed = time.time() - t0
            timings = out.get("timings", {}) or {}
            rows.append({
                "index": idx,
                "prompt": prompt,
                "new_ids": out.get("new_ids", []),
                "completion_text": out.get("completion_text", ""),
                "completion_text_raw": out.get("completion_text_raw", ""),
                "stopped_reason": out.get("stopped_reason"),
                "prefill_seconds": timings.get("prefill_seconds"),
                "decode_tps": timings.get("decode_tps"),
                "elapsed_seconds": elapsed,
                "degenerate": _degenerate(out.get("completion_text", "")),
            })
    finally:
        _restore_env(old)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--prompts-json", default="")
    ap.add_argument("--preset", choices=("basic", "rc-mini"), default="basic")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    prompts = _prompt_preset(args.preset)
    if args.prompts_json:
        prompts = json.loads(Path(args.prompts_json).read_text())

    print("=============== STAGE 6 PHASE 2-O PACKED-PREFILL RC SMOKE ===============", flush=True)
    print(f"model       : {args.model}", flush=True)
    print(f"prompts     : {len(prompts)}", flush=True)
    print(f"max_new     : {args.max_new}", flush=True)

    runner = LynnIncrementalRunner(
        args.model,
        device="cuda",
        dtype=torch.bfloat16,
        max_seq_len=args.max_seq_len,
        verbose=False,
    )
    torch.cuda.synchronize()
    mem_loaded_gib = float(torch.cuda.memory_allocated() / (1024**3))

    print("[phase] baseline BF16-prefill generate", flush=True)
    baseline = _run_batch(runner, prompts, max_new=args.max_new, optin=False)

    print("[phase] release active MoE BF16 shadows only", flush=True)
    release = runner.release_decode_bf16_shadows(
        include_moe_experts=True,
        include_projection_aliases=False,
    )
    mem_after_release_gib = float(torch.cuda.memory_allocated() / (1024**3))

    print("[phase] packed-prefill opt-in no-reload generate", flush=True)
    optin = _run_batch(runner, prompts, max_new=args.max_new, optin=True)

    comparisons: list[dict[str, Any]] = []
    for base, cand in zip(baseline, optin):
        base_ids = [int(x) for x in base.get("new_ids", [])]
        cand_ids = [int(x) for x in cand.get("new_ids", [])]
        prefix_n = min(len(base_ids), len(cand_ids))
        comparisons.append({
            "index": base["index"],
            "token_exact": base_ids == cand_ids,
            "token_prefix_match": sum(1 for a, b in zip(base_ids, cand_ids) if a == b),
            "token_prefix_n": prefix_n,
            "text_prefix_200_match": _norm(base.get("completion_text", ""))[:200] == _norm(cand.get("completion_text", ""))[:200],
            "baseline_ids": base_ids,
            "optin_ids": cand_ids,
            "baseline_text": base.get("completion_text", ""),
            "optin_text": cand.get("completion_text", ""),
        })

    functional_pass = all(not row["degenerate"] for row in baseline + optin)
    token_exact = all(row["token_exact"] for row in comparisons)
    text_prefix_match = all(row["text_prefix_200_match"] for row in comparisons)
    result = {
        "schema": "lynn-stage6-p2o-packed-prefill-rc-smoke-v1",
        "model": args.model,
        "preset": args.preset,
        "max_new": args.max_new,
        "env": {
            **BASE_ENV,
            **FUSION_ENV,
            "LYNN_PACKED_PREFILL_SLOW_MODE": "p2e_hybrid",
            "LYNN_PACKED_PREFILL_P2E_LAYERS": "all",
            "LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA": "1",
        },
        "memory": {
            "loaded_gib": mem_loaded_gib,
            "after_release_gib": mem_after_release_gib,
            "release": release,
        },
        "baseline": baseline,
        "optin_no_reload": optin,
        "comparisons": comparisons,
        "passes": {
            "functional_non_degenerate": bool(functional_pass),
            "token_exact": bool(token_exact),
            "text_prefix_200_match": bool(text_prefix_match),
            "all": bool(functional_pass and token_exact),
        },
        "notes": [
            "This releases only active MoE BF16 expert shadows; projection shadows remain resident.",
            "The gate is a real-prompt resident-runner smoke, not a full RC quality battery.",
            "Default path remains unchanged when opt-in flags are unset.",
        ],
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
