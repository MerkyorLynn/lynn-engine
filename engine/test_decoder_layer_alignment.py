"""
Compare Lynn `_layer_forward` (whole DecoderLayer) vs HF Qwen3_5MoeDecoderLayer
on real layer 0 weights. Localizes the orchestration bug from full_forward.py.
"""
import sys
import time

import torch
import torch.nn.functional as F


def main():
    sys.path.insert(0, "/work")
    from engine.loader import load_qwen36_layer
    from engine.full_forward import _layer_forward, _rms_norm

    device = "cuda"
    dtype = torch.bfloat16
    model_dir = "/models/Qwen3.6-35B-A3B-FP8"
    layer_idx = 0
    seq_len = 16

    # config
    import json
    cfg_full = json.load(open(f"{model_dir}/config.json"))
    tc = cfg_full["text_config"]
    cfg = {
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "num_experts": tc["num_experts"],
        "num_experts_per_tok": tc["num_experts_per_tok"],
        "rope_theta": tc.get("rope_theta", 1e6),
    }
    layer_type = tc["layer_types"][layer_idx]
    print(f"Layer {layer_idx} type: {layer_type}")
    print(f"rope_theta from config: {cfg['rope_theta']}")
    print(f"All text_config rope-related: { {k:v for k,v in tc.items() if 'rope' in k.lower()} }")

    # Load layer 0 weights
    weights, _ = load_qwen36_layer(model_dir, layer_idx, device=device, dequant_dtype=dtype)
    print(f"\nLoaded {len(weights)} tensors")

    # Instantiate HF model — only need 1 decoder layer
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    full_cfg_obj = AutoConfig.from_pretrained(model_dir)
    text_cfg = full_cfg_obj.text_config
    print(f"\nHF text_cfg.rope_theta: {getattr(text_cfg, 'rope_theta', '?')}")

    hf_layer = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx).to(device=device, dtype=dtype)
    hf_layer.eval()

    # Map our weights into HF layer state_dict
    sd = hf_layer.state_dict()
    # keys in our weights are like "input_layernorm.weight", "linear_attn.in_proj_qkv.weight", etc.
    # HF layer state_dict has same names — verify
    hf_keys = set(sd.keys())
    our_keys = set(weights.keys())

    missing_in_ours = hf_keys - our_keys
    missing_in_hf = our_keys - hf_keys
    print(f"\nKeys in HF layer not in our weights: {len(missing_in_ours)}")
    if missing_in_ours:
        for k in sorted(missing_in_ours)[:10]:
            print(f"  - {k}")
    print(f"Keys in our weights not in HF layer: {len(missing_in_hf)}")
    if missing_in_hf:
        for k in sorted(missing_in_hf)[:10]:
            print(f"  - {k}")

    # Copy what we can
    new_sd = {}
    for k in hf_keys:
        if k in our_keys:
            new_sd[k] = weights[k].to(sd[k].dtype).to(sd[k].device)
        else:
            new_sd[k] = sd[k]  # keep HF init
    hf_layer.load_state_dict(new_sd, strict=False)
    print("Loaded weights into HF layer")

    # Synthetic input
    torch.manual_seed(42)
    h_in = torch.randn(1, seq_len, cfg["hidden_size"], device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    print(f"\nh_in shape={tuple(h_in.shape)} mag={h_in.float().abs().mean().item():.3f}")

    # HF forward — DecoderLayer signature requires position_embeddings (cos, sin), not just position_ids
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    rope = Qwen3_5MoeTextRotaryEmbedding(text_cfg).to(device)
    cos, sin = rope(h_in, position_ids)

    with torch.no_grad():
        out_hf = hf_layer(
            h_in,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )
    if isinstance(out_hf, tuple):
        out_hf = out_hf[0]
    print(f"HF out shape={tuple(out_hf.shape)} mag={out_hf.float().abs().mean().item():.3f}")

    # Lynn forward
    with torch.no_grad():
        out_lynn = _layer_forward(h_in, position_ids, layer_type, weights, cfg)
    print(f"Lynn out shape={tuple(out_lynn.shape)} mag={out_lynn.float().abs().mean().item():.3f}")

    diff = (out_lynn - out_hf).float().abs()
    rel = diff.max().item() / max(out_hf.float().abs().mean().item(), 1e-8) * 100
    print(f"\ndiff max={diff.max().item():.3e}  mean={diff.mean().item():.3e}  rel={rel:.3f}%")
    print("✅ PASS" if rel < 5.0 else "❌ FAIL")


if __name__ == "__main__":
    main()
