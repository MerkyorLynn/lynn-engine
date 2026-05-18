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

# Quant formats Lynn engine loader can handle.
# None / "fp8" / "float8": Qwen 3.6 FP8/BF16 path.
# compressed-tensors + nvfp4-pack-quantized: Phase 4 P2 slow dequant path.
_SUPPORTED_QUANT_METHODS = {None, "fp8", "float8", "compressed-tensors"}

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

    # Explicit compressed-tensors NVFP4 is handled by the P2 slow path. Other
    # compressed-tensors formats still fall through to the catch-all below.
    if quant_method == "compressed-tensors" and quant_format and "nvfp4" in quant_format.lower():
        return

    # Explicit modelopt_fp4 (5/15 V4-Distill output format)
    if quant_method in ("modelopt", "modelopt_fp4"):
        raise NotImplementedError(
            f"modelopt_fp4 checkpoint detected (quant_method={quant_method!r}).\n"
            f"Lynn engine loader v1 does not yet support modelopt_fp4 — Phase 4 L3 backlog.\n"
            f"For modelopt_fp4 inference today, use SGLang stable v0.5.9 + "
            f"`--quantization modelopt_fp4`."
        )

    # Key-level NVFP4 sniff (defensive: catches checkpoints where quant_method
    # is unset but packed-FP4 weight keys are present).
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


def _is_nvfp4_v8_rtn(quant_cfg: dict) -> bool:
    quant_method = quant_cfg.get("quant_method")
    quant_format = quant_cfg.get("format")
    return (
        quant_method == "compressed-tensors"
        and quant_format is not None
        and "nvfp4" in str(quant_format).lower()
    )


def _is_lynn_variable_nvfp4(model_dir: Path) -> bool:
    manifest_path = model_dir / "lynn_quant_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    return manifest.get("schema_version") == "lynn-variable-nvfp4-pack-v1"


def _dequantize_lynn_variable_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    original_shape: list[int],
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize Lynn-native row-wise-per-16 NVFP4 tensors.

    The 27B variable-expert artifact stores every matrix-like tensor as
    flattened rows: `[prod(original_shape[:-1]), K/2]` uint8 packed FP4 plus
    `[rows, K/16]` FP16 scales. Restore the original tensor shape after the
    explicit slow dequant path.
    """
    try:
        from engine.dequant import unpack_fp4_e2m1_from_uint8
    except ModuleNotFoundError:
        from dequant import unpack_fp4_e2m1_from_uint8

    unpacked = unpack_fp4_e2m1_from_uint8(packed, dtype=torch.float32)
    scale = scale.to(torch.float32)
    global_scale = global_scale.to(torch.float32)
    if unpacked.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "Lynn variable NVFP4 expects packed/scale tensors to be 2D; "
            f"got packed={tuple(packed.shape)} unpacked={tuple(unpacked.shape)} scale={tuple(scale.shape)}"
        )
    if unpacked.shape[1] % scale.shape[1] != 0:
        raise ValueError(
            "unpacked last dim must be divisible by scale columns; "
            f"unpacked={tuple(unpacked.shape)} scale={tuple(scale.shape)}"
        )
    group_size = unpacked.shape[1] // scale.shape[1]
    if group_size != 16:
        raise ValueError(f"expected Lynn variable NVFP4 group_size=16, got {group_size}")
    scale_full = scale.repeat_interleave(group_size, dim=-1) / global_scale
    out = (unpacked * scale_full).reshape(original_shape)
    return out.to(output_dtype)


def _load_qwen36_layer_lynn_variable_nvfp4(
    model_dir: Path,
    layer_idx: int,
    device: str,
    dequant_dtype: torch.dtype,
):
    """Load one Lynn-native variable-expert NVFP4 layer via slow dequant.

    This supports artifacts produced by `lynn-variable-nvfp4-pack-v1`, where
    physical expert tensors already have per-layer variable expert counts.
    Unlike compressed-tensors v8-RTN, no per-expert unfusing is needed.
    """
    from safetensors import safe_open

    manifest = json.loads((model_dir / "lynn_quant_manifest.json").read_text())
    with open(model_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"model.language_model.layers.{layer_idx}."
    final = {}
    bytes_loaded = 0
    t_load_start = time.time()

    def shorten(k: str) -> str:
        return k[len(prefix):]

    def load_tensor(key: str) -> torch.Tensor:
        file_name = weight_map[key]
        with safe_open(model_dir / file_name, framework="pt", device=device) as st:
            return st.get_tensor(key)

    for key, rec in manifest.get("kept_tensors", {}).items():
        if not key.startswith(prefix):
            continue
        tensor = load_tensor(key)
        bytes_loaded += tensor.element_size() * tensor.numel()
        if key.endswith(".weight") and tensor.dtype != dequant_dtype:
            tensor = tensor.to(dequant_dtype)
        final[shorten(key)] = tensor

    for key, rec in manifest.get("quantized_tensors", {}).items():
        if not key.startswith(prefix):
            continue
        packed = load_tensor(rec["packed_key"])
        scale = load_tensor(rec["scale_key"])
        global_scale = load_tensor(rec["global_scale_key"])
        tensor = _dequantize_lynn_variable_nvfp4(
            packed,
            scale,
            global_scale,
            rec["original_shape"],
            output_dtype=dequant_dtype,
        ).to(device)
        bytes_loaded += tensor.element_size() * tensor.numel()
        final[shorten(key)] = tensor

    print(
        f"  Lynn variable NVFP4 slow dequant load: {bytes_loaded/1e9:.2f} GB "
        f"in {time.time()-t_load_start:.1f}s",
        flush=True,
    )

    config = {}
    if "self_attn.q_proj.weight" in final:
        q_w = final["self_attn.q_proj.weight"]
        config["q_proj_out"] = q_w.shape[0]
        config["hidden_size"] = q_w.shape[1]
    if "self_attn.k_proj.weight" in final:
        config["k_proj_out"] = final["self_attn.k_proj.weight"].shape[0]
    if "self_attn.v_proj.weight" in final:
        config["v_proj_out"] = final["self_attn.v_proj.weight"].shape[0]
    if "mlp.gate.weight" in final:
        config["num_experts"] = final["mlp.gate.weight"].shape[0]
    if "mlp.shared_expert.gate_proj.weight" in final:
        config["shared_intermediate"] = final["mlp.shared_expert.gate_proj.weight"].shape[0]
    if "mlp.experts.gate_up_proj" in final:
        config["expert_intermediate"] = final["mlp.experts.gate_up_proj"].shape[1] // 2
    return final, config


def _load_qwen36_layer_nvfp4_v8_rtn(
    model_dir: Path,
    layer_idx: int,
    num_experts: int,
    device: str,
    dequant_dtype: torch.dtype,
):
    """Load one compressed-tensors NVFP4 v8-RTN layer via slow dequant.

    v8-RTN stores MoE experts as per-expert HF-style keys, while Lynn engine's
    BF16 path uses fused tensors:

    - `mlp.experts.down_proj`: [E, hidden, intermediate]
    - `mlp.experts.gate_up_proj`: [E, 2 * intermediate, hidden]

    This function normalizes NVFP4 into the same fused layout returned by the
    BF16 loader, so downstream forward code does not know which checkpoint
    format was used.
    """
    from safetensors import safe_open
    from engine.dequant import dequantize_nvfp4_v8_rtn_weight

    single_path = model_dir / "model.safetensors"
    if not single_path.exists():
        raise FileNotFoundError(f"NVFP4 v8-RTN loader expected {single_path}")

    prefix = f"model.language_model.layers.{layer_idx}."
    final = {}
    expert_parts: dict[int, dict[str, torch.Tensor]] = {}
    bytes_loaded = 0

    def shorten(k: str) -> str:
        return k[len(prefix):]

    def load_dequant(st, base_key: str) -> torch.Tensor:
        packed = st.get_tensor(base_key + ".weight_packed")
        scale = st.get_tensor(base_key + ".weight_scale")
        global_scale = st.get_tensor(base_key + ".weight_global_scale")
        return dequantize_nvfp4_v8_rtn_weight(
            packed,
            scale,
            global_scale,
            output_dtype=dequant_dtype,
        ).to(device)

    t_load_start = time.time()
    with safe_open(single_path, framework="pt", device=device) as st:
        layer_keys = [k for k in st.keys() if k.startswith(prefix)]
        print(f"  Found {len(layer_keys)} NVFP4 keys for layer {layer_idx}", flush=True)
        print("  Spans 1 safetensors file", flush=True)

        for k in layer_keys:
            short = shorten(k)

            if short.endswith((".weight_scale", ".weight_global_scale", ".input_global_scale")):
                continue

            if short.endswith(".weight_packed"):
                base_key = k[: -len(".weight_packed")]
                base_short = short[: -len(".weight_packed")]
                tensor = load_dequant(st, base_key)
                bytes_loaded += tensor.element_size() * tensor.numel()

                # Convert per-expert HF layout to Lynn fused expert layout.
                m = re.match(r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$", base_short)
                if m:
                    expert_idx = int(m.group(1))
                    proj = m.group(2)
                    expert_parts.setdefault(expert_idx, {})[proj] = tensor
                else:
                    final[base_short + ".weight"] = tensor
                continue

            # Non-packed tensors are BF16/F32 metadata or residual weights.
            tensor = st.get_tensor(k)
            bytes_loaded += tensor.element_size() * tensor.numel()
            if short.endswith(".weight") and tensor.dtype != dequant_dtype:
                tensor = tensor.to(dequant_dtype)
            final[short] = tensor

    if expert_parts:
        missing = [
            e
            for e in range(num_experts)
            if set(expert_parts.get(e, {})) != {"gate_proj", "up_proj", "down_proj"}
        ]
        if missing:
            raise ValueError(f"NVFP4 expert tensor group incomplete; missing experts: {missing[:8]}")
        downs = []
        gate_ups = []
        for e in range(num_experts):
            parts = expert_parts[e]
            downs.append(parts["down_proj"])
            gate_ups.append(torch.cat([parts["gate_proj"], parts["up_proj"]], dim=0))
        final["mlp.experts.down_proj"] = torch.stack(downs, dim=0).to(dequant_dtype)
        final["mlp.experts.gate_up_proj"] = torch.stack(gate_ups, dim=0).to(dequant_dtype)

    print(
        f"  NVFP4 slow dequant load: {bytes_loaded/1e9:.2f} GB "
        f"in {time.time()-t_load_start:.1f}s",
        flush=True,
    )

    config = {}
    if "mlp.gate.weight" in final:
        config["num_experts"] = final["mlp.gate.weight"].shape[0]
    if "mlp.shared_expert.gate_proj.weight" in final:
        config["shared_intermediate"] = final["mlp.shared_expert.gate_proj.weight"].shape[0]
    if "mlp.experts.gate_up_proj" in final:
        config["expert_intermediate"] = final["mlp.experts.gate_up_proj"].shape[1] // 2
    return final, config


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

    if _is_lynn_variable_nvfp4(model_dir):
        return _load_qwen36_layer_lynn_variable_nvfp4(
            model_dir=model_dir,
            layer_idx=layer_idx,
            device=device,
            dequant_dtype=dequant_dtype,
        )

    if _is_nvfp4_v8_rtn(quant_cfg):
        return _load_qwen36_layer_nvfp4_v8_rtn(
            model_dir=model_dir,
            layer_idx=layer_idx,
            num_experts=num_experts,
            device=device,
            dequant_dtype=dequant_dtype,
        )

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
        elif k in ("mlp.experts.down_proj", "mlp.experts.gate_up_proj"):
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
    # Dense FFN: detect gate_proj (no MoE gate.weight present)
    if "mlp.gate_proj.weight" in final and "mlp.gate.weight" not in final:
        config["num_experts"] = 0
        config["ffn_intermediate"] = final["mlp.gate_proj.weight"].shape[0]

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
