#!/usr/bin/env python3
"""Inventory the official Qwen3.6 MTP FP8 safetensors (V2).

Reads `mtp.safetensors` header and produces a structured inventory with:
- key_count, expert_count (inferred from keys, asserted contiguous 0..E-1)
- dtype_counts
- self_attn_summary: per q/k/v/o weight AND weight_scale_inv shapes
- expert_shapes: gate/up/down weight + scale_inv shapes
- shared_expert_shapes
- fc / pre_fc_norm / norm shapes

Dims are inferred from tensor shapes, NOT from config defaults.
config.json is read for model_type context only.

CPU-only. No GPU.

Usage:
  python scripts/qwen36_official_mtp_inventory.py \\
    --mtp /home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp.safetensors \\
    --config /home/merkyor/models/Qwen3.6-35B-A3B-FP8/config.json \\
    --out reports/mtp/qwen36_official_mtp_inventory.json
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def _parse_config(path: Path | None) -> dict[str, Any]:
    """Read config.json, supporting nested text_config."""
    if not path or not path.exists():
        return {}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    # Qwen3.6 models often nest the real config under text_config
    text_cfg = cfg.get("text_config", cfg)
    return {
        "model_type": cfg.get("model_type", text_cfg.get("model_type", "unknown")),
        "hidden_size": text_cfg.get("hidden_size"),
        "intermediate_size": text_cfg.get("intermediate_size", text_cfg.get("moe_intermediate_size")),
        "num_experts": text_cfg.get("num_experts", text_cfg.get("num_local_experts")),
        "num_experts_per_tok": text_cfg.get("num_experts_per_tok"),
        "mtp_num_hidden_layers": text_cfg.get("mtp_num_hidden_layers", text_cfg.get("num_nextn_predict_layers")),
        "vocab_size": text_cfg.get("vocab_size"),
        "num_attention_heads": text_cfg.get("num_attention_heads"),
        "num_key_value_heads": text_cfg.get("num_key_value_heads"),
        "head_dim": text_cfg.get("head_dim"),
    }


def _parse_expert_id(key: str) -> int | None:
    m = re.search(r"experts?[._](\d+)", key)
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.6 Official MTP Inventory V2")
    ap.add_argument("--mtp", required=True, help="Path to official mtp.safetensors")
    ap.add_argument("--config", default=None, help="Path to config.json (optional, for context)")
    ap.add_argument("--out", default=None, help="Output JSON path (default: stdout)")
    args = ap.parse_args()

    mtp_path = Path(args.mtp)
    if not mtp_path.exists():
        print(f"ERROR: mtp file not found: {mtp_path}", file=sys.stderr)
        return 1

    config_info = _parse_config(Path(args.config) if args.config else None)
    header = _read_safetensors_header(mtp_path)
    key_count = len(header)
    print(f"[inventory] path: {mtp_path}", file=sys.stderr)
    print(f"[inventory] keys: {key_count}", file=sys.stderr)

    # ── Dtype counts ──
    dtype_counts: Counter[str] = Counter()
    for info in header.values():
        dtype_counts[info.get("dtype", "unknown")] += 1

    # ── Expert detection (from per-expert keys) ──
    expert_ids: set[int] = set()
    expert_gate_keys: dict[int, dict[str, Any]] = {}
    expert_up_keys: dict[int, dict[str, Any]] = {}
    expert_down_keys: dict[int, dict[str, Any]] = {}
    expert_gate_scale_keys: dict[int, dict[str, Any]] = {}
    expert_up_scale_keys: dict[int, dict[str, Any]] = {}
    expert_down_scale_keys: dict[int, dict[str, Any]] = {}

    # ── Self-attn ──
    self_attn_weights: dict[str, dict[str, Any]] = {}  # "q_proj" → {shape, dtype}
    self_attn_scales: dict[str, dict[str, Any]] = {}

    # ── Shared expert ──
    shared_expert_weights: dict[str, dict[str, Any]] = {}
    shared_expert_scales: dict[str, dict[str, Any]] = {}

    # ── Other ──
    fc_info: dict[str, Any] = {}
    pre_fc_norm_embed_info: dict[str, Any] = {}
    pre_fc_norm_hidden_info: dict[str, Any] = {}
    output_norm_info: dict[str, Any] = {}
    router_gate_info: dict[str, Any] = {}

    for key, info in header.items():
        kl = key.lower()
        shape = info.get("shape", [])
        dtype = info.get("dtype", "unknown")
        entry = {"key": key, "shape": shape, "dtype": dtype}

        eid = _parse_expert_id(key)

        # Scale keys
        is_scale = "scale_inv" in kl or (kl.endswith("_scale") and "expert" not in kl.split("scale")[0][-5:])

        # Per-expert (not shared)
        if eid is not None and "shared" not in kl:
            expert_ids.add(eid)
            if "gate_proj" in kl:
                if "scale" in kl:
                    expert_gate_scale_keys[eid] = entry
                else:
                    expert_gate_keys[eid] = entry
            elif "up_proj" in kl:
                if "scale" in kl:
                    expert_up_scale_keys[eid] = entry
                else:
                    expert_up_keys[eid] = entry
            elif "down_proj" in kl:
                if "scale" in kl:
                    expert_down_scale_keys[eid] = entry
                else:
                    expert_down_keys[eid] = entry
            continue

        # Self-attn
        if "self_attn" in kl:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                if proj in kl:
                    if "scale" in kl:
                        self_attn_scales[proj] = entry
                    else:
                        self_attn_weights[proj] = entry
                    break
            continue

        # Shared expert
        if "shared_expert" in kl and "shared_expert_gate" not in kl:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                if proj in kl:
                    if "scale" in kl:
                        shared_expert_scales[proj] = entry
                    else:
                        shared_expert_weights[proj] = entry
                    break
            continue

        # Shared expert gate (router for shared)
        if "shared_expert_gate" in kl:
            router_gate_info = entry
            continue

        # Router gate
        if re.search(r"mlp[._]gate[._]weight", kl) or (kl.endswith("gate.weight") and "proj" not in kl and "expert" not in kl):
            router_gate_info = entry
            continue

        # Pre-FC norms
        if "pre_fc_norm_embedding" in kl:
            pre_fc_norm_embed_info = entry
        elif "pre_fc_norm_hidden" in kl:
            pre_fc_norm_hidden_info = entry
        # FC projection
        elif re.search(r"\bfc\b", kl) and "norm" not in kl and "scale" not in kl:
            fc_info = entry
        # Output norm (not input_layernorm, not post_attention_layernorm)
        elif re.search(r"(^|[._])norm[._]weight$", kl) and "layer" not in kl and "fc" not in kl and "attn" not in kl:
            output_norm_info = entry

    # ── Validate expert contiguity ──
    expert_count = len(expert_ids)
    expert_contiguous = (expert_ids == set(range(expert_count))) if expert_count > 0 else True
    blockers: list[str] = []
    if not expert_contiguous:
        blockers.append(f"Expert IDs not contiguous 0..{expert_count-1}: found {sorted(expert_ids)[:10]}...")

    # ── Infer dims from tensor shapes ──
    inferred_hidden: int | None = None
    inferred_intermediate: int | None = None

    # From a gate_proj: [intermediate, hidden]
    if expert_gate_keys:
        sample = next(iter(expert_gate_keys.values()))
        if len(sample["shape"]) == 2:
            inferred_intermediate = sample["shape"][0]
            inferred_hidden = sample["shape"][1]

    # ── Infer scale block size ──
    inferred_scale_block: tuple[int, int] | None = None
    if expert_gate_scale_keys and expert_gate_keys:
        w_shape = next(iter(expert_gate_keys.values()))["shape"]
        s_shape = next(iter(expert_gate_scale_keys.values()))["shape"]
        if len(w_shape) == 2 and len(s_shape) == 2:
            block_r = w_shape[0] // s_shape[0] if s_shape[0] > 0 else None
            block_c = w_shape[1] // s_shape[1] if s_shape[1] > 0 else None
            if block_r and block_c:
                inferred_scale_block = (block_r, block_c)

    # ── Build report ──
    report: dict[str, Any] = {
        "schema": "lynn-qwen36-official-mtp-inventory-v2",
        "mtp_path": str(mtp_path),
        "config_context": config_info,
        "key_count": key_count,
        "dtype_counts": dict(dtype_counts),
        "expert_count": expert_count,
        "expert_contiguous": expert_contiguous,
        "inferred_hidden": inferred_hidden,
        "inferred_intermediate": inferred_intermediate,
        "inferred_scale_block": list(inferred_scale_block) if inferred_scale_block else None,
        "has_pre_fc_norm_embedding": bool(pre_fc_norm_embed_info),
        "has_pre_fc_norm_hidden": bool(pre_fc_norm_hidden_info),
        "self_attn_summary": {
            "weights": {k: {"shape": v["shape"], "dtype": v["dtype"]} for k, v in self_attn_weights.items()},
            "scales": {k: {"shape": v["shape"], "dtype": v["dtype"]} for k, v in self_attn_scales.items()},
        },
        "expert_shapes": {
            "gate_proj_sample": next(iter(expert_gate_keys.values()), {}).get("shape"),
            "gate_proj_scale_sample": next(iter(expert_gate_scale_keys.values()), {}).get("shape"),
            "up_proj_sample": next(iter(expert_up_keys.values()), {}).get("shape"),
            "up_proj_scale_sample": next(iter(expert_up_scale_keys.values()), {}).get("shape"),
            "down_proj_sample": next(iter(expert_down_keys.values()), {}).get("shape"),
            "down_proj_scale_sample": next(iter(expert_down_scale_keys.values()), {}).get("shape"),
            "gate_proj_dtype": next(iter(expert_gate_keys.values()), {}).get("dtype"),
            "scale_dtype": next(iter(expert_gate_scale_keys.values()), {}).get("dtype"),
        },
        "shared_expert_shapes": {
            "weights": {k: {"shape": v["shape"], "dtype": v["dtype"]} for k, v in shared_expert_weights.items()},
            "scales": {k: {"shape": v["shape"], "dtype": v["dtype"]} for k, v in shared_expert_scales.items()},
        },
        "fc_info": {"shape": fc_info.get("shape"), "dtype": fc_info.get("dtype"), "key": fc_info.get("key")},
        "pre_fc_norm_embedding": {"shape": pre_fc_norm_embed_info.get("shape"), "dtype": pre_fc_norm_embed_info.get("dtype"), "key": pre_fc_norm_embed_info.get("key")},
        "pre_fc_norm_hidden": {"shape": pre_fc_norm_hidden_info.get("shape"), "dtype": pre_fc_norm_hidden_info.get("dtype"), "key": pre_fc_norm_hidden_info.get("key")},
        "output_norm": {"shape": output_norm_info.get("shape"), "dtype": output_norm_info.get("dtype"), "key": output_norm_info.get("key")},
        "router_gate": {"shape": router_gate_info.get("shape") if isinstance(router_gate_info, dict) else None, "dtype": router_gate_info.get("dtype") if isinstance(router_gate_info, dict) else None, "key": router_gate_info.get("key") if isinstance(router_gate_info, dict) else None},
        "blockers": blockers,
    }

    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"[inventory] wrote: {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(output)

    # Summary
    print(f"[inventory] expert_count: {expert_count}", file=sys.stderr)
    print(f"[inventory] expert_contiguous: {expert_contiguous}", file=sys.stderr)
    print(f"[inventory] inferred hidden: {inferred_hidden}", file=sys.stderr)
    print(f"[inventory] inferred intermediate: {inferred_intermediate}", file=sys.stderr)
    print(f"[inventory] scale_block: {inferred_scale_block}", file=sys.stderr)
    print(f"[inventory] dtypes: {dict(dtype_counts)}", file=sys.stderr)
    print(f"[inventory] self_attn weights:", file=sys.stderr)
    for k, v in self_attn_weights.items():
        s = self_attn_scales.get(k, {})
        print(f"  {k:10s} weight={v['shape']} ({v['dtype']})  scale={s.get('shape', 'N/A')} ({s.get('dtype', '')})", file=sys.stderr)
    if blockers:
        print(f"[inventory] BLOCKERS: {blockers}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
