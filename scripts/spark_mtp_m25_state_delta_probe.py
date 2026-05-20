#!/usr/bin/env python3
"""M25 state-delta probe — direct byte-diff K=2 vs canonical T=1 commit state.

M24 ruled out cheap commit-output repair (next_base_hidden and
next_pending_id are byte-equal to canonical T=1 chain outputs). Only
``state``/``full_canonical`` repair brings exact ≥ 5/6. By elimination
the bug lives in the **committed state after K=2 vs after a sequential
T=1 chain**, even though M18 confirmed layer-output bit-strict equality
under ``LYNN_FULL_ATTN_K2_BACKEND=t1_loop``.

This probe pins which tensor (KV per layer, recurrent_state per layer,
conv_state per layer) actually diverges between the two commit paths.
If any captured tensor diverges, the bug is in that compute path. If
all captured tensors are byte-equal, the bug is process-side
(autotune cache, allocator, native FP4 lm_head workspace, runner lazy
buffers) and M26 must probe those directly.

Methodology per prompt (1-2 prompts to keep cost low):

  1. Prefill (build canonical post-prompt state).
  2. snap_pre   = runner._snapshot_state(state)  (clones)
  3. Pick pending = prefill argmax, draft = MTP head draft.
  4. K=2 forward: F.embedding → 40 layers _decode_layer_k2_fast →
     state.seq_len += 2.
  5. snap_post_k2 = runner._snapshot_state(state)
  6. runner._restore_state(state, snap_pre)
  7. T=1 chain: decode_one_to_logits_and_hidden(state, pending) then
     decode_one_to_logits_and_hidden(state, draft).
  8. snap_post_t1 = runner._snapshot_state(state)
  9. Per-layer per-tensor diff post_k2 vs post_t1:
       kv_cache K, V          — slice [pre:pre+2] of seq_len axis
       recurrent_state        — full FP32 tensor
       conv_state             — full tensor
  10. Report first non-zero diff layer + tensor + magnitude.

Usage on Spark::

    /home/merkyor/comfyui/ComfyUI/.venv/bin/python -u \
        scripts/spark_mtp_m25_state_delta_probe.py \
        --model /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
        --sidecar /home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
        --out /tmp/lynn_m25/mtp_m25_state_delta_$(date +%Y%m%d_%H%M%S).json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_ENV: dict[str, str] = {
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_PACKED_DECODE": "1",
    "LYNN_PACKED_SHARED_EXPERT": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_FULL_ATTN_QKV_FUSED": "1",
    "LYNN_FULL_ATTN_K2_BACKEND": "t1_loop",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    # Apples-to-apples eager + shadow off.
    "LYNN_LINEAR_BLOCK_GRAPH": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    "LYNN_MTP_SHADOW_VERIFY": "0",
}


DEFAULT_PROMPTS = [
    "Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.",
    "用一句话解释 speculative decoding 的核心思想。",
]


def _set_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    prev: dict[str, str | None] = {}
    for key, value in updates.items():
        prev[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    return prev


def _restore_env(prev: dict[str, str | None]) -> None:
    for key, value in prev.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Per-tensor numerical diff (flattened): max_abs, rel_l2, cosine."""
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    if af.numel() != bf.numel():
        return {"shape_mismatch": True, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = af - bf
    a_norm = float(af.norm().item())
    diff_norm = float(diff.norm().item())
    return {
        "max_abs": float(diff.abs().max().item()),
        "rel_l2": (diff_norm / a_norm) if a_norm > 1e-12 else float("nan"),
        "cosine": float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()),
        "all_equal": bool(diff.abs().max().item() == 0.0),
        "a_norm": a_norm,
        "shape": list(a.shape),
    }


def _diff_state_snapshots(
    snap_k2: dict[str, Any],
    snap_t1: dict[str, Any],
    cached_seq_len_pre: int,
    layer_types: list[str],
    n_layers: int,
) -> dict[str, Any]:
    """Diff two state snapshots (returned by runner._snapshot_state)."""
    per_layer: list[dict[str, Any]] = []
    first_bad: dict[str, Any] | None = None
    any_diff = False

    for i in range(n_layers):
        ltype = layer_types[i]
        entry: dict[str, Any] = {"layer": i, "layer_type": ltype}
        # KV cache (full_attention only); for linear_attention layers it's not used.
        if i in snap_k2.get("kv", {}) and i in snap_t1.get("kv", {}):
            K_k2, V_k2 = snap_k2["kv"][i]
            K_t1, V_t1 = snap_t1["kv"][i]
            # Diff full KV tensor (positions outside [pre:pre+2] should not have changed in either path).
            kv_diff = {
                "K_full": _diff(K_k2, K_t1),
                "V_full": _diff(V_k2, V_t1),
            }
            # Also slice the newly-written positions for focused analysis.
            K_slice_k2 = K_k2[:, :, cached_seq_len_pre:cached_seq_len_pre + 2, :]
            V_slice_k2 = V_k2[:, :, cached_seq_len_pre:cached_seq_len_pre + 2, :]
            K_slice_t1 = K_t1[:, :, cached_seq_len_pre:cached_seq_len_pre + 2, :]
            V_slice_t1 = V_t1[:, :, cached_seq_len_pre:cached_seq_len_pre + 2, :]
            kv_diff["K_at_new_positions"] = _diff(K_slice_k2, K_slice_t1)
            kv_diff["V_at_new_positions"] = _diff(V_slice_k2, V_slice_t1)
            entry["kv_diff"] = kv_diff
        # Recurrent state (linear_attention only).
        if i in snap_k2.get("recurrent", {}) and i in snap_t1.get("recurrent", {}):
            entry["recurrent_diff"] = _diff(snap_k2["recurrent"][i], snap_t1["recurrent"][i])
        # Conv state (linear_attention only).
        if i in snap_k2.get("conv", {}) and i in snap_t1.get("conv", {}):
            entry["conv_diff"] = _diff(snap_k2["conv"][i], snap_t1["conv"][i])

        # Aggregate any_diff flag for this layer.
        layer_has_diff = False
        for d in ("kv_diff", "recurrent_diff", "conv_diff"):
            if d not in entry:
                continue
            if d == "kv_diff":
                for sub_key, sub in entry[d].items():
                    if isinstance(sub, dict) and not sub.get("all_equal", True):
                        layer_has_diff = True
                        if first_bad is None:
                            first_bad = {"layer": i, "layer_type": ltype, "field": f"kv.{sub_key}", "diff": sub}
            else:
                if isinstance(entry[d], dict) and not entry[d].get("all_equal", True):
                    layer_has_diff = True
                    if first_bad is None:
                        first_bad = {"layer": i, "layer_type": ltype, "field": d, "diff": entry[d]}
        entry["layer_has_diff"] = layer_has_diff
        any_diff = any_diff or layer_has_diff
        per_layer.append(entry)

    return {
        "any_diff": any_diff,
        "first_bad_layer": first_bad,
        "seq_len_k2": snap_k2.get("seq_len"),
        "seq_len_t1": snap_t1.get("seq_len"),
        "seq_len_equal": snap_k2.get("seq_len") == snap_t1.get("seq_len"),
        "per_layer": per_layer,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--prompts-json", default=None)
    args = ap.parse_args()

    prompts: list[str] = list(DEFAULT_PROMPTS)
    if args.prompts_json:
        raw = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        prompts = [str(item["prompt"]) if isinstance(item, dict) else str(item) for item in raw]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    base_env_with_sidecar = dict(BASE_ENV)
    base_env_with_sidecar["LYNN_MTP_SIDECAR"] = args.sidecar
    base_prev = _set_env(base_env_with_sidecar)

    try:
        from engine.nvfp4_layout import detect_nvfp4_layout
        layout = detect_nvfp4_layout(args.model)
        if layout.layout_kind != "lynn_native_per16_variable":
            raise SystemExit(f"[m25] {args.model} layout {layout.layout_kind!r} not Lynn-native")

        from engine.inference_state import LAYER_TYPES, LynnInferenceState
        from engine.full_forward import _prefill_layer, _rms_norm
        from engine.mtp_serving import decode_one_to_logits_and_hidden
        from engine.resident_runner import LynnIncrementalRunner

        runner = LynnIncrementalRunner(
            args.model, device=args.device, dtype=dtype, verbose=False,
        )
        if not runner.mtp_sidecar_loaded:
            raise SystemExit(f"[m25] MTP sidecar not loaded: {args.sidecar}")

        layer_types = list(LAYER_TYPES)
        results: list[dict[str, Any]] = []

        for prompt_idx, prompt in enumerate(prompts):
            print(f"[m25] prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]!r}", flush=True)

            # 1. Prefill.
            ids = runner.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(runner.device)
            T = int(ids.shape[1])
            state = LynnInferenceState(
                batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype,
            )
            h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
            pos = torch.arange(T, device=runner.device, dtype=torch.long).unsqueeze(0)
            for i in range(runner.n_layers):
                h = _prefill_layer(
                    h, pos, layer_types[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i,
                )
            state.seq_len = T
            h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
            logits = runner._lm_head_logits(h_final)
            pending_id = int(logits[0].argmax().item())
            pending_base_hidden = h[:, -1:, :].contiguous()
            pending_pos = T - 1

            # 2. MTP draft.
            draft_logits = runner._mtp_draft_logits(
                base_hidden=pending_base_hidden,
                current_token_id=pending_id,
                current_pos=pending_pos,
            )
            draft_id = int(draft_logits[0].argmax().item())

            # 3. Snap pre-K=2.
            snap_pre = runner._snapshot_state(state)
            cached_seq_len_pre = int(state.seq_len)

            # 4. K=2 forward (no lm_head/commit needed for state diff).
            tokens_k2 = torch.tensor(
                [[pending_id, draft_id]], device=runner.device, dtype=torch.long,
            )
            pos_k2 = torch.tensor(
                [[cached_seq_len_pre, cached_seq_len_pre + 1]],
                device=runner.device, dtype=torch.long,
            )
            h_k2 = F.embedding(tokens_k2, runner.outside["model.language_model.embed_tokens.weight"])
            for i in range(runner.n_layers):
                h_k2 = runner._decode_layer_k2_fast(h_k2, pos_k2, state, i)
            state.seq_len += 2

            # 5. Snap post-K=2.
            snap_post_k2 = runner._snapshot_state(state)

            # 6. Restore pre-K=2.
            runner._restore_state(state, snap_pre)

            # 7. T=1 chain.
            _h_p, _, _ = decode_one_to_logits_and_hidden(runner, state, pending_id)
            _h_d, _, _ = decode_one_to_logits_and_hidden(runner, state, draft_id)

            # 8. Snap post-T=1.
            snap_post_t1 = runner._snapshot_state(state)

            # 9. Diff.
            diff_report = _diff_state_snapshots(
                snap_post_k2, snap_post_t1, cached_seq_len_pre, layer_types, runner.n_layers,
            )

            results.append({
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "prompt_len": T,
                "pending_id": pending_id,
                "draft_id": draft_id,
                "pending_text": runner.tokenizer.decode([pending_id]),
                "draft_text": runner.tokenizer.decode([draft_id]),
                "diff": diff_report,
            })

            print(
                f"[m25]   any_diff={diff_report['any_diff']} "
                f"first_bad_layer={diff_report['first_bad_layer']['layer'] if diff_report['first_bad_layer'] else None} "
                f"({diff_report['first_bad_layer']['field'] if diff_report['first_bad_layer'] else '-'})",
                flush=True,
            )

        # Aggregate verdict.
        any_diff_count = sum(1 for r in results if r["diff"]["any_diff"])
        verdict: dict[str, Any] = {
            "n_prompts": len(results),
            "n_prompts_with_diff": any_diff_count,
        }
        if any_diff_count == 0:
            verdict["verdict_class"] = "CAPTURED_STATE_BYTE_EQUAL__BUG_IS_PROCESS_SIDE"
        elif any_diff_count == len(results):
            verdict["verdict_class"] = "CAPTURED_STATE_DIFFERS_ALL_PROMPTS"
        else:
            verdict["verdict_class"] = "CAPTURED_STATE_DIFFERS_SOME_PROMPTS"

        # First-bad-field across prompts.
        first_bads = [r["diff"]["first_bad_layer"] for r in results if r["diff"]["first_bad_layer"]]
        if first_bads:
            verdict["first_bad_layer_idx_min"] = min(fb["layer"] for fb in first_bads)
            verdict["first_bad_fields"] = sorted({fb["field"] for fb in first_bads})

        report = {
            "schema_version": "lynn-mtp-m25-state-delta-v1",
            "generated_at": datetime.now().isoformat(timespec="seconds") + "Z",
            "model": args.model,
            "sidecar": args.sidecar,
            "base_env": BASE_ENV,
            "prompts": results,
            "verdict": verdict,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[m25] wrote {out_path}", flush=True)
        print(f"[m25] verdict_class = {verdict['verdict_class']}", flush=True)
        if "first_bad_layer_idx_min" in verdict:
            print(f"[m25] first_bad_layer_idx_min = {verdict['first_bad_layer_idx_min']}", flush=True)
            print(f"[m25] first_bad_fields = {verdict['first_bad_fields']}", flush=True)
        return 0
    finally:
        _restore_env(base_prev)


if __name__ == "__main__":
    raise SystemExit(main())
