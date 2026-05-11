"""
Lynn Engine · Phase 2 · safetensors loader for Qwen 3.6 35B-A3B-FP8

Reads safetensors files for a SINGLE layer + dequantizes FP8 weights to BF16.
Used for the A1 single-layer integration test.

vLLM FP8 layout convention:
  - weights stored as float8_e4m3fn
  - per-block scales in `weight_scale_inv` (block_size from quantization_config)
  - dequant: y = (fp8_weight * scale).to(bf16)
  - For matmul: dynamic activation quant + GEMM in FP8 → BF16 accumulator

For our spike test we just dequant to BF16 once and use standard PyTorch matmul.
This is slow but correct, and verifies the weights load + dequant pipeline.
"""
import json
import re
import time
from pathlib import Path

import torch


# ============================================================================
# L4 (Phase 4 fail-loud guard, 2026-05-11):
# Prior behavior fell through `v.to(dequant_dtype)` on unrecognized weight keys,
# producing garbage tensors silently (NVFP4 .weight_packed uint8 → garbage BF16
# — 776 packed tensors per layer x 40 layers, all "loaded successfully" but raw).
# Refuse to load formats this loader cannot dequant correctly.
# See zhihu postmortem 2026-05-11: https://zhuanlan.zhihu.com/p/2036443846322680848
# ============================================================================

# Quant formats Lynn engine loader v1 can handle.
# None / "fp8" / "float8": Qwen 3.6 FP8 base (block-scaled, .weight + .weight_scale_inv).
_SUPPORTED_QUANT_METHODS = {None, "fp8", "float8"}

# Weight key suffixes that indicate NVFP4 compressed-tensors packed format.
# v8-RTN ckpt has 4-suffix-per-Linear schema; presence of any indicates NVFP4.
_NVFP4_PACKED_KEY_SUFFIXES = (
    ".weight_packed",
    ".weight_global_scale",
    ".input_global_scale",
)


def _detect_unsupported_quant_format(quant_cfg: dict, weight_keys: list) -> None:
    """
    Phase 4 fail-loud guard. Raises NotImplementedError on any quant format
    this loader cannot dequant correctly.

    Loud > silent corruption. NVFP4 .weight_packed (uint8) silently went through
    `v.to(bf16)` before this guard, producing garbage hidden states.

    Phase 4 backlog L1-L5 will add the actual NVFP4 dequant path.
    """
    quant_method = quant_cfg.get("quant_method")
    quant_format = quant_cfg.get("format")

    # Explicit compressed-tensors NVFP4
    if quant_method == "compressed-tensors" and quant_format and "nvfp4" in quant_format.lower():
        raise NotImplementedError(
            f"NVFP4 compressed-tensors checkpoint detected "
            f"(quant_method={quant_method!r}, format={quant_format!r}).\n"
            f"Lynn engine loader v1 cannot dequant NVFP4 .weight_packed (uint8) tensors.\n"
            f"This is Phase 4 backlog (L1-L5, ~8-12h sprint).\n"
            f"For NVFP4 inference today, use SGLang dev-cu13 + "
            f"`--quantization compressed-tensors`.\n"
            f"Phase 4 plan: https://zhuanlan.zhihu.com/p/2036443846322680848"
        )

    # Explicit modelopt_fp4 (5/15 V4-Distill output format)
    if quant_method in ("modelopt", "modelopt_fp4"):
        raise NotImplementedError(
            f"modelopt_fp4 checkpoint detected (quant_method={quant_method!r}).\n"
            f"Lynn engine loader v1 does not yet support modelopt_fp4 — Phase 4 L3 backlog.\n"
            f"For modelopt_fp4 inference today, use SGLang stable v0.5.9 + "
            f"`--quantization modelopt_fp4`."
        )

    # Key-level NVFP4 sniff (defensive: catches checkpoints where quant_method is unset
    # but packed-FP4 weight keys are present).
    for k in weight_keys[:200]:
        for suffix in _NVFP4_PACKED_KEY_SUFFIXES:
            if k.endswith(suffix):
                raise NotImplementedError(
                    f"Detected NVFP4-style packed weight key {k!r} (suffix {suffix!r}).\n"
                    f"Lynn engine loader v1 does not support NVFP4 native loading.\n"
                    f"Phase 4 backlog: https://zhuanlan.zhihu.com/p/2036443846322680848"
                )

    # Catch-all: any unknown quant_method
    if quant_method is not None and quant_method not in _SUPPORTED_QUANT_METHODS:
        raise NotImplementedError(
            f"Unsupported quantization: quant_method={quant_method!r}, "
            f"format={quant_format!r}.\n"
            f"Lynn engine loader v1 supports: {sorted(_SUPPORTED_QUANT_METHODS, key=str)}.\n"
            f"To add support: (1) extend this guard to allow the new method, "
            f"(2) add the dequant path in `load_qwen36_layer`.\n"
            f"Universal rule: fail-loud on unknown formats, never fall through "
            f"to v.to(dtype) (produces garbage). See zhihu postmortem § 5."
        )


def load_qwen36_layer(
    model_dir: str,
    layer_idx: int,
    num_experts: int = 256,
    device: str = "cuda",
    dequant_dtype: torch.dtype = torch.bfloat16,
):
    """
    Load all weights for a single Qwen 3.6 35B-A3B-FP8 layer + dequantize to BF16.

    Args:
        model_dir: path to /home/merkyor/models/Qwen3.6-35B-A3B-FP8
        layer_idx: which transformer layer to load (full_attention layers: 3, 7, 11, ...)
        num_experts: 256 for Qwen 3.6
        device: cuda / cpu
        dequant_dtype: bfloat16 (matches model native dtype) or float16

    Returns:
        weights: dict of {short_key: tensor}, all BF16/FP16, ready for forward
        config: dict of dimensions extracted from weight shapes
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)

    # Read weight map. Some Lynn/NVFP4 artifacts are single-file
    # `model.safetensors` without `model.safetensors.index.json`; support both
    # layouts so 27B single-file exports do not fail at loader startup.
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        quantization_config = index.get("metadata", {})
    elif single_path.exists():
        with safe_open(single_path, framework="pt", device="cpu") as st:
            weight_map = {k: single_path.name for k in st.keys()}
        quantization_config = {}
    else:
        raise FileNotFoundError(
            f"Expected {index_path.name} or {single_path.name} under {model_dir}"
        )

    # Read full config to know block_size
    with open(model_dir / "config.json") as f:
        full_config = json.load(f)
    quant_cfg = full_config.get("quantization_config", {})
    weight_block_size = quant_cfg.get("weight_block_size", [128, 128])

    # L4 (Phase 4 fail-loud guard, 2026-05-11): refuse silent NVFP4 / unknown mis-load.
    _detect_unsupported_quant_format(quant_cfg, list(weight_map.keys()))

    # Filter keys for our target layer
    prefix = f"model.language_model.layers.{layer_idx}."
    layer_keys = [k for k in weight_map if k.startswith(prefix)]
    print(f"  Found {len(layer_keys)} keys for layer {layer_idx}", flush=True)

    # Group keys by their file
    file_keys = {}
    for k in layer_keys:
        f = weight_map[k]
        file_keys.setdefault(f, []).append(k)

    print(f"  Spans {len(file_keys)} safetensors files", flush=True)

    # Load and dequantize
    out = {}

    def shorten(k):
        """Strip prefix → short key."""
        return k[len(prefix):]

    def dequant(weight_fp8, scale_inv, dtype):
        """vLLM FP8 e4m3 block-scaled dequantization.

        scale_inv tensor shape is [out//block, in//block].
        Each block is [block_size[0], block_size[1]] of FP8 weights.
        Dequant: weight_bf16 = (fp8_weight * scale_inv_broadcast).to(dtype)
        """
        if scale_inv is None:
            return weight_fp8.to(dtype)
        # scale_inv shape: [out_blocks, in_blocks]
        bs0, bs1 = weight_block_size
        # Repeat scale to match weight shape
        out_dim, in_dim = weight_fp8.shape
        # Pad weights divisible by block; safetensors usually stores it padded already
        scale_full = scale_inv.repeat_interleave(bs0, dim=0).repeat_interleave(bs1, dim=1)
        scale_full = scale_full[:out_dim, :in_dim]
        return (weight_fp8.to(torch.float32) * scale_full.to(torch.float32)).to(dtype)

    t_load_start = time.time()
    bytes_loaded = 0
    for f, ks in file_keys.items():
        with safe_open(model_dir / f, framework="pt", device=device) as st:
            for k in ks:
                short = shorten(k)
                tensor = st.get_tensor(k)
                bytes_loaded += tensor.element_size() * tensor.numel()
                out[short] = tensor

    print(f"  Raw load: {bytes_loaded/1e9:.2f} GB in {time.time()-t_load_start:.1f}s", flush=True)

    # Pair up weight + weight_scale_inv and dequant
    t_dq_start = time.time()
    final = {}
    for k, v in out.items():
        if k.endswith(".weight_scale_inv"):
            continue
        if k.endswith(".weight"):
            scale_key = k.replace(".weight", ".weight_scale_inv")
            scale = out.get(scale_key, None)
            if v.dtype == torch.float8_e4m3fn:
                final[k] = dequant(v, scale, dequant_dtype)
            else:
                final[k] = v.to(dequant_dtype)
        else:
            final[k] = v

    print(f"  Dequant: {time.time()-t_dq_start:.1f}s", flush=True)

    # Extract config from weight shapes
    config = {}
    if "self_attn.q_proj.weight" in final:
        q_w = final["self_attn.q_proj.weight"]
        # Linear weight is [out_features, in_features]
        config["q_proj_out"] = q_w.shape[0]
        config["hidden_size"] = q_w.shape[1]
    if "self_attn.k_proj.weight" in final:
        config["k_proj_out"] = final["self_attn.k_proj.weight"].shape[0]
    if "self_attn.v_proj.weight" in final:
        config["v_proj_out"] = final["self_attn.v_proj.weight"].shape[0]
    if "mlp.gate.weight" in final:
        config["num_experts"] = final["mlp.gate.weight"].shape[0]
    if "mlp.experts.0.gate_proj.weight" in final:
        config["expert_intermediate"] = final["mlp.experts.0.gate_proj.weight"].shape[0]
    if "mlp.shared_expert.gate_proj.weight" in final:
        config["shared_intermediate"] = final["mlp.shared_expert.gate_proj.weight"].shape[0]

    return final, config


def report(weights, config):
    """Pretty-print loaded weights summary."""
    total_bytes = sum(v.element_size() * v.numel() for v in weights.values())
    print(f"\n📦 Loaded {len(weights)} tensors, {total_bytes/1e9:.2f} GB total ({list(weights.values())[0].dtype})")
    print(f"   Config inferred from weights:")
    for k, v in config.items():
        print(f"     {k}: {v}")
    # Categorize
    cats = {"norm": 0, "attn": 0, "moe_router": 0, "experts": 0, "shared": 0, "other": 0}
    for k in weights:
        if "norm" in k or "_norm" in k:
            cats["norm"] += 1
        elif k.startswith("self_attn"):
            cats["attn"] += 1
        elif k == "mlp.gate.weight":
            cats["moe_router"] += 1
        elif "experts" in k and "shared" not in k:
            cats["experts"] += 1
        elif "shared" in k:
            cats["shared"] += 1
        else:
            cats["other"] += 1
    print(f"   By category: {cats}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/models/Qwen3.6-35B-A3B-FP8",
                    help="container-mounted path to Qwen 3.6 model dir")
    ap.add_argument("--layer", type=int, default=3, help="layer index to load")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"⚙️  Loading Qwen 3.6 35B-A3B-FP8 layer {args.layer} from {args.model_dir}")
    weights, config = load_qwen36_layer(args.model_dir, args.layer, device=args.device)
    report(weights, config)

    # Sanity: print a sample tensor norm
    print()
    sample = "self_attn.q_proj.weight"
    if sample in weights:
        t = weights[sample]
        print(f"Sample {sample}: shape={tuple(t.shape)} dtype={t.dtype} norm={t.float().norm().item():.4f}")
