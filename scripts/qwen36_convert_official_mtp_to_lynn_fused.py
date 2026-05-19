#!/usr/bin/env python3
"""Convert official Qwen3.6 MTP FP8 safetensors to Lynn fused layout (V2).

V2 fixes:
- Block-scale FP8 dequant: scale_inv [R/128, C/128] expanded via repeat_interleave
- Expert count inferred from keys (assert contiguous 0..E-1)
- Dims inferred from tensor shapes, not config defaults
- Nested text_config support
- --self-test for CPU mock validation

Reads official `mtp.safetensors` with per-expert FP8 E4M3 layout + block scale_inv
and produces Lynn-compatible fused BF16 safetensors.

Usage:
  python scripts/qwen36_convert_official_mtp_to_lynn_fused.py \\
    --mtp /home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp.safetensors \\
    --config /home/merkyor/models/Qwen3.6-35B-A3B-FP8/config.json \\
    --out /path/to/output/mtp_lynn_fused.safetensors

  python scripts/qwen36_convert_official_mtp_to_lynn_fused.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


SCALE_BLOCK_SIZE = 128  # Qwen FP8 uses 128×128 block quantization


def _read_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def _sha256_prefix(path: Path, *, chars: int = 32) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:chars]


def _load_tensor_cpu(path: Path, key: str) -> torch.Tensor:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _parse_config(path: Path | None) -> dict[str, Any]:
    """Read config.json supporting nested text_config."""
    if not path or not path.exists():
        return {}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config", cfg)
    return {
        "model_type": cfg.get("model_type", text_cfg.get("model_type", "unknown")),
        "hidden_size": text_cfg.get("hidden_size"),
        "intermediate_size": text_cfg.get("intermediate_size", text_cfg.get("moe_intermediate_size")),
        "num_experts": text_cfg.get("num_experts", text_cfg.get("num_local_experts")),
    }


def dequant_fp8_block_scale(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = SCALE_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize FP8 E4M3 tensor with block-wise scale_inv.

    weight_fp8: [R, C] (float8_e4m3fn or uint8 viewed as fp8)
    scale_inv: [R // block_size, C // block_size]
    Returns: [R, C] bfloat16

    Each 128×128 block of weight shares one scale value.
    Dequant: bf16_value = fp8_value.float() * scale_inv_expanded
    """
    if weight_fp8.dtype == torch.uint8:
        # View as FP8 then to float
        fp8_float = weight_fp8.view(torch.float8_e4m3fn).float()
    elif weight_fp8.dtype == torch.float8_e4m3fn:
        fp8_float = weight_fp8.float()
    else:
        # Already float-like, just convert
        return weight_fp8.to(torch.bfloat16)

    R, C = fp8_float.shape
    sr, sc = scale_inv.shape

    # Validate block alignment
    expected_sr = (R + block_size - 1) // block_size
    expected_sc = (C + block_size - 1) // block_size
    assert sr == expected_sr and sc == expected_sc, (
        f"Scale shape mismatch: weight [{R},{C}] → expected scale [{expected_sr},{expected_sc}], "
        f"got [{sr},{sc}] with block_size={block_size}"
    )

    # Expand scale to weight shape via repeat_interleave
    scale_expanded = scale_inv.float()
    scale_expanded = scale_expanded.repeat_interleave(block_size, dim=0)[:R]
    scale_expanded = scale_expanded.repeat_interleave(block_size, dim=1)[:, :C]

    result = fp8_float * scale_expanded
    return result.to(torch.bfloat16)


def _find_scale_key(key: str, all_keys: set[str]) -> str | None:
    """Find the matching scale_inv key for a weight tensor."""
    candidates = [
        key.replace(".weight", ".weight_scale_inv"),
        key + "_scale_inv",
        key.replace(".weight", "_scale_inv"),
    ]
    for c in candidates:
        if c in all_keys:
            return c
    # Fuzzy: same prefix + scale_inv
    base = key.rsplit(".", 1)[0]
    for k in all_keys:
        if k.startswith(base) and "scale_inv" in k:
            return k
    return None


def _parse_expert_id(key: str) -> int | None:
    m = re.search(r"experts?[._](\d+)", key)
    return int(m.group(1)) if m else None


def _lynn_layer_key(official_key: str) -> str | None:
    """Map official key → Lynn mtp.layers.0.* key for passthrough tensors."""
    patterns = [
        (r"input_layernorm\.weight", "mtp.layers.0.input_layernorm.weight"),
        (r"post_attention_layernorm\.weight", "mtp.layers.0.post_attention_layernorm.weight"),
        (r"self_attn\.q_proj\.weight$", "mtp.layers.0.self_attn.q_proj.weight"),
        (r"self_attn\.k_proj\.weight$", "mtp.layers.0.self_attn.k_proj.weight"),
        (r"self_attn\.v_proj\.weight$", "mtp.layers.0.self_attn.v_proj.weight"),
        (r"self_attn\.o_proj\.weight$", "mtp.layers.0.self_attn.o_proj.weight"),
        (r"self_attn\.q_norm\.weight$", "mtp.layers.0.self_attn.q_norm.weight"),
        (r"self_attn\.k_norm\.weight$", "mtp.layers.0.self_attn.k_norm.weight"),
        (r"mlp\.gate\.weight$", "mtp.layers.0.mlp.gate.weight"),
        (r"shared_expert\.gate_proj\.weight$", "mtp.layers.0.mlp.shared_expert.gate_proj.weight"),
        (r"shared_expert\.up_proj\.weight$", "mtp.layers.0.mlp.shared_expert.up_proj.weight"),
        (r"shared_expert\.down_proj\.weight$", "mtp.layers.0.mlp.shared_expert.down_proj.weight"),
        (r"shared_expert_gate\.weight$", "mtp.layers.0.mlp.shared_expert_gate.weight"),
    ]
    for pat, target in patterns:
        if re.search(pat, official_key):
            return target
    return None


def _self_test() -> int:
    """CPU mock test: verify block-scale dequant works correctly."""
    print("[self-test] Testing block-scale FP8 dequant...", file=sys.stderr)

    # Simulate: weight [512, 2048], scale [4, 16] with block=128
    R, C = 512, 2048
    block = 128
    sr, sc = R // block, C // block  # [4, 16]

    # Create a known pattern: fp8 values = 1.0 everywhere, scale = varying
    weight_fp32 = torch.ones(R, C, dtype=torch.float32)
    weight_fp8 = weight_fp32.to(torch.float8_e4m3fn)
    scale_inv = torch.arange(1, sr * sc + 1, dtype=torch.float32).view(sr, sc)

    result = dequant_fp8_block_scale(weight_fp8, scale_inv, block_size=block)

    # Verify shape
    assert result.shape == (R, C), f"Shape mismatch: {result.shape} vs ({R}, {C})"
    assert result.dtype == torch.bfloat16, f"Dtype mismatch: {result.dtype}"

    # Verify values: each 128×128 block should equal its scale value (since fp8=1.0)
    for bi in range(sr):
        for bj in range(sc):
            block_val = result[bi * block:(bi + 1) * block, bj * block:(bj + 1) * block]
            expected = scale_inv[bi, bj].item()
            actual = block_val[0, 0].float().item()
            assert abs(actual - expected) < 0.01, (
                f"Block [{bi},{bj}]: expected {expected}, got {actual}"
            )

    print(f"[self-test] PASS: dequant [{R},{C}] with scale [{sr},{sc}] block={block}", file=sys.stderr)

    # Test non-aligned case: weight [384, 2048], scale [3, 16]
    R2, C2 = 384, 2048
    sr2, sc2 = R2 // block, C2 // block
    w2 = torch.ones(R2, C2, dtype=torch.float32).to(torch.float8_e4m3fn)
    s2 = torch.ones(sr2, sc2, dtype=torch.float32) * 2.5
    r2 = dequant_fp8_block_scale(w2, s2, block_size=block)
    assert r2.shape == (R2, C2), f"Shape mismatch: {r2.shape}"
    assert abs(r2[0, 0].float().item() - 2.5) < 0.01
    print(f"[self-test] PASS: dequant [{R2},{C2}] with scale [{sr2},{sc2}]", file=sys.stderr)

    print("[self-test] ALL TESTS PASSED", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert official Qwen3.6 MTP FP8 → Lynn fused V2")
    ap.add_argument("--mtp", default=None, help="Path to official mtp.safetensors")
    ap.add_argument("--config", default=None, help="config.json for context (dims inferred from tensors)")
    ap.add_argument("--out", default=None, help="Output Lynn fused safetensors path")
    ap.add_argument("--dry-run", action="store_true", help="Print key mapping without writing")
    ap.add_argument("--self-test", action="store_true", help="Run CPU mock dequant test and exit")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if not args.mtp:
        print("ERROR: --mtp is required (or use --self-test)", file=sys.stderr)
        return 1
    if not args.out and not args.dry_run:
        print("ERROR: --out is required (or use --dry-run)", file=sys.stderr)
        return 1

    mtp_path = Path(args.mtp)
    if not mtp_path.exists():
        print(f"ERROR: {mtp_path} not found", file=sys.stderr)
        return 1

    config_info = _parse_config(Path(args.config) if args.config else None)
    header = _read_header(mtp_path)
    all_keys = set(header.keys())
    print(f"[converter] source: {mtp_path} ({len(header)} keys)", file=sys.stderr)

    # ── Classify keys ──
    expert_gate: dict[int, str] = {}
    expert_up: dict[int, str] = {}
    expert_down: dict[int, str] = {}
    passthrough: list[tuple[str, str]] = []  # (official, lynn)
    fc_key: str | None = None
    pre_fc_norm_embed_key: str | None = None
    pre_fc_norm_hidden_key: str | None = None
    output_norm_key: str | None = None
    scale_keys: set[str] = set()

    for key in header:
        kl = key.lower()
        if "scale_inv" in kl or (kl.endswith("_scale") and "gate.weight" not in kl):
            scale_keys.add(key)
            continue

        eid = _parse_expert_id(key)
        if eid is not None and "shared" not in kl:
            if "gate_proj" in kl:
                expert_gate[eid] = key
            elif "up_proj" in kl:
                expert_up[eid] = key
            elif "down_proj" in kl:
                expert_down[eid] = key
            continue

        if "pre_fc_norm_embedding" in kl:
            pre_fc_norm_embed_key = key
        elif "pre_fc_norm_hidden" in kl:
            pre_fc_norm_hidden_key = key
        elif re.search(r"\bfc\b", kl) and "norm" not in kl and "scale" not in kl:
            fc_key = key
        elif re.search(r"(^|[._])norm[._]weight$", kl) and "layer" not in kl and "fc" not in kl and "attn" not in kl:
            output_norm_key = key
        else:
            lynn_key = _lynn_layer_key(key)
            if lynn_key:
                passthrough.append((key, lynn_key))

    # ── Validate experts ──
    expert_count = max(len(expert_gate), len(expert_up), len(expert_down))
    expert_ids = set(expert_gate.keys()) | set(expert_up.keys()) | set(expert_down.keys())
    expected_ids = set(range(expert_count))
    if expert_ids != expected_ids:
        missing = expected_ids - expert_ids
        print(f"[converter] BLOCKER: expert IDs not contiguous 0..{expert_count-1}", file=sys.stderr)
        print(f"  missing: {sorted(missing)[:20]}", file=sys.stderr)
        if not args.dry_run:
            return 1

    # ── Infer dims from tensor shapes ──
    hidden: int | None = None
    intermediate: int | None = None
    if expert_gate:
        sample_key = next(iter(expert_gate.values()))
        shape = header[sample_key]["shape"]
        if len(shape) == 2:
            intermediate = shape[0]
            hidden = shape[1]
    print(f"[converter] experts={expert_count} hidden={hidden} intermediate={intermediate}", file=sys.stderr)

    # ── Infer block size from first expert scale ──
    block_size = SCALE_BLOCK_SIZE
    sample_gate_key = next(iter(expert_gate.values()), None)
    if sample_gate_key:
        gate_scale_key = _find_scale_key(sample_gate_key, all_keys)
        if gate_scale_key:
            ws = header[sample_gate_key]["shape"]
            ss = header[gate_scale_key]["shape"]
            if len(ws) == 2 and len(ss) == 2 and ss[0] > 0 and ss[1] > 0:
                block_size = ws[0] // ss[0]
                print(f"[converter] inferred block_size={block_size} from {ws} / {ss}", file=sys.stderr)

    # ── Print mapping ──
    print(f"\n[converter] Key mapping ({expert_count} experts, block={block_size}):", file=sys.stderr)
    if fc_key:
        print(f"  {fc_key} → mtp.fc.weight", file=sys.stderr)
    if pre_fc_norm_embed_key:
        print(f"  {pre_fc_norm_embed_key} → mtp.pre_fc_norm_embedding.weight", file=sys.stderr)
    if pre_fc_norm_hidden_key:
        print(f"  {pre_fc_norm_hidden_key} → mtp.pre_fc_norm_hidden.weight", file=sys.stderr)
    if output_norm_key:
        print(f"  {output_norm_key} → mtp.norm.weight", file=sys.stderr)
    for off_key, lynn_key in passthrough:
        print(f"  {off_key} → {lynn_key}", file=sys.stderr)
    print(f"  {expert_count}× gate+up → mtp.layers.0.mlp.experts.gate_up_proj [{expert_count}, {2*(intermediate or 0)}, {hidden}]", file=sys.stderr)
    print(f"  {expert_count}× down → mtp.layers.0.mlp.experts.down_proj [{expert_count}, {hidden}, {intermediate}]", file=sys.stderr)
    print(f"  scale keys: {len(scale_keys)} (block dequant applied)", file=sys.stderr)

    if args.dry_run:
        print(f"\n[converter] DRY_RUN — not writing.", file=sys.stderr)
        return 0

    # ── Load and convert ──
    print(f"\n[converter] Loading and dequantizing...", file=sys.stderr)
    output_tensors: dict[str, torch.Tensor] = {}
    fp8_count = 0

    def load_dequant(key: str) -> torch.Tensor:
        nonlocal fp8_count
        t = _load_tensor_cpu(mtp_path, key)
        dtype_str = header[key].get("dtype", "")
        if "F8_E4M3" in dtype_str.upper() or t.dtype == torch.float8_e4m3fn or (t.dtype == torch.uint8 and t.element_size() == 1 and "F8" in dtype_str.upper()):
            scale_key = _find_scale_key(key, all_keys)
            if scale_key:
                scale = _load_tensor_cpu(mtp_path, scale_key)
                result = dequant_fp8_block_scale(t, scale, block_size=block_size)
                fp8_count += 1
                return result
            else:
                print(f"  [WARN] no scale for FP8 tensor: {key}", file=sys.stderr)
                return t.view(torch.float8_e4m3fn).float().to(torch.bfloat16)
        return t.to(torch.bfloat16) if t.is_floating_point() else t.float().to(torch.bfloat16)

    # Top-level
    if fc_key:
        output_tensors["mtp.fc.weight"] = load_dequant(fc_key)
    if pre_fc_norm_embed_key:
        output_tensors["mtp.pre_fc_norm_embedding.weight"] = load_dequant(pre_fc_norm_embed_key)
    if pre_fc_norm_hidden_key:
        output_tensors["mtp.pre_fc_norm_hidden.weight"] = load_dequant(pre_fc_norm_hidden_key)
    if output_norm_key:
        output_tensors["mtp.norm.weight"] = load_dequant(output_norm_key)

    # Passthrough layer keys
    for off_key, lynn_key in passthrough:
        output_tensors[lynn_key] = load_dequant(off_key)

    # Fuse experts
    if expert_count > 0 and hidden and intermediate:
        print(f"[converter] Fusing {expert_count} experts...", file=sys.stderr)
        gate_up_list: list[torch.Tensor] = []
        down_list: list[torch.Tensor] = []

        for eid in range(expert_count):
            g_key = expert_gate.get(eid)
            u_key = expert_up.get(eid)
            d_key = expert_down.get(eid)

            if g_key and u_key:
                g = load_dequant(g_key)
                u = load_dequant(u_key)
                gate_up_list.append(torch.cat([g, u], dim=0))
            else:
                gate_up_list.append(torch.zeros(2 * intermediate, hidden, dtype=torch.bfloat16))

            if d_key:
                down_list.append(load_dequant(d_key))
            else:
                down_list.append(torch.zeros(hidden, intermediate, dtype=torch.bfloat16))

            if (eid + 1) % 64 == 0:
                print(f"  ... {eid + 1}/{expert_count} experts done", file=sys.stderr)

        output_tensors["mtp.layers.0.mlp.experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0)
        output_tensors["mtp.layers.0.mlp.experts.down_proj"] = torch.stack(down_list, dim=0)
        print(f"  gate_up_proj: {list(output_tensors['mtp.layers.0.mlp.experts.gate_up_proj'].shape)}", file=sys.stderr)
        print(f"  down_proj: {list(output_tensors['mtp.layers.0.mlp.experts.down_proj'].shape)}", file=sys.stderr)

    # ── Write ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_path": str(mtp_path),
        "source_sha256_prefix": _sha256_prefix(mtp_path),
        "source_key_count": str(len(header)),
        "detected_expert_count": str(expert_count),
        "hidden_inferred": str(hidden or ""),
        "intermediate_inferred": str(intermediate or ""),
        "scale_block": f"{block_size},{block_size}",
        "conversion": "fp8_block_scale_to_bf16_fused",
        "fp8_tensors_dequanted": str(fp8_count),
        "output_key_count": str(len(output_tensors)),
        "converter": "qwen36_convert_official_mtp_to_lynn_fused.py_v2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    save_file(output_tensors, str(out_path), metadata=metadata)

    # SHA256
    h = hashlib.sha256()
    with out_path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    sha_prefix = h.hexdigest()[:32]

    print(f"\n[converter] Output: {out_path}", file=sys.stderr)
    print(f"[converter] Keys: {len(output_tensors)}", file=sys.stderr)
    print(f"[converter] Size: {out_path.stat().st_size / (1024*1024):.1f} MiB", file=sys.stderr)
    print(f"[converter] SHA256: {sha_prefix}", file=sys.stderr)
    print(f"[converter] FP8 dequanted: {fp8_count}", file=sys.stderr)
    print(f"[converter] Metadata: {json.dumps(metadata, indent=2)}", file=sys.stderr)
    print(f"\n[converter] Output keys:", file=sys.stderr)
    for key in sorted(output_tensors.keys()):
        t = output_tensors[key]
        print(f"  {key:60s} {str(t.dtype):12s} {list(t.shape)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
