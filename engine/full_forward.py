"""
Lynn Engine · 40-layer full forward (memory-bounded, layer-by-layer load).

The Lynn-engine end-to-end forward without any HF dependency. Architecture:

    input_ids → embed_tokens → h0
    for layer i in 0..39:
        residual = h
        h = input_layernorm(h)
        if layer_types[i] == 'linear_attention':
            h = lynn_linear_attn_forward(h, layer_i_weights)
        else:  # 'full_attention'
            h = lynn_full_attn_forward(h, layer_i_weights)  # (P1.1 path)
        h = residual + h
        residual = h
        h = post_attention_layernorm(h)
        h = MoE_forward(h, layer_i_weights)  # 256 experts top-8 + shared
        h = residual + h
        # free layer_i_weights
    h = final_norm(h)
    logits = h @ lm_head.T
    return logits

Memory profile:
  embeddings + lm_head:          1.0 GB BF16   (kept resident)
  per-layer weights, peak:       1.7 GB BF16   (loaded then freed)
  hidden state activation:       few MB        (B=1, T<=256)
  Total GPU peak:               ~3 GB

Doesn't disturb running vLLM (which uses 60 GB at mem-fraction 0.5).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3_5MoeRMSNorm — note the `(1.0 + weight)` factor, not plain `weight`.

    From HF transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:806 ::
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())   # the +1 offset
        return output.type_as(x)

    See https://github.com/huggingface/transformers/pull/29402 — Qwen-family
    diverges from Llama-style RMSNorm (which is `weight * x` only).
    """
    in_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    x_n = x_f * torch.rsqrt(var + eps)
    return (x_n * (1.0 + weight.float())).to(in_dtype)


def _full_attn_forward(h: torch.Tensor, position_ids: torch.Tensor,
                       w: dict, cfg: dict) -> torch.Tensor:
    """Full-attention forward (Qwen 3.6 specifics: GQA, attn_output_gate,
    q_norm/k_norm, partial-rotary GPT-NeoX-style RoPE with theta=1e7).

    Note on RoPE: Qwen 3.6 uses MROPE (multi-modal: T/H/W position grids)
    with `partial_rotary_factor=0.25` (only first 64 of 256 head dims rotate)
    and `rope_theta=1e7`. For text-only input, T=H=W positions, so MROPE
    collapses to standard GPT-NeoX RoPE on the first 64 dims. The remaining
    192 dims pass through unrotated.
    """
    B, M, D = h.shape
    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    rope_theta = cfg["rope_theta"]
    partial = cfg["partial_rotary_factor"]
    rotary_dim = int(head_dim * partial)

    q_full = F.linear(h, w["self_attn.q_proj.weight"])
    k = F.linear(h, w["self_attn.k_proj.weight"])
    v = F.linear(h, w["self_attn.v_proj.weight"])

    # Critical: q_proj output is [B, M, H_Q*2*head_dim]. HF first reshapes to
    # [B, M, H_Q, 2*head_dim] (per-head 2x slot) then chunks along last dim
    # into [q, gate]. Doing chunk(2, dim=-1) on the flat representation
    # incorrectly mixes head0_gate into "q" and head_last_q into "gate".
    q_full_view = q_full.view(B, M, H_Q, head_dim * 2)
    q, attn_output_gate = q_full_view.chunk(2, dim=-1)
    q = q.transpose(1, 2)                              # [B, H_Q, M, head_dim]
    attn_output_gate = attn_output_gate.transpose(1, 2)
    k = k.view(B, M, H_KV, head_dim).transpose(1, 2)
    v = v.view(B, M, H_KV, head_dim).transpose(1, 2)

    # q_norm and k_norm (Qwen3 trick)
    q = _rms_norm(q, w["self_attn.q_norm.weight"])
    k = _rms_norm(k, w["self_attn.k_norm.weight"])

    # RoPE — GPT-NeoX split-halves style on first `rotary_dim` channels
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, device=h.device, dtype=torch.float32) / rotary_dim)
    )  # [rotary_dim // 2]
    freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]  # [B, M, rotary_dim // 2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [B, M, rotary_dim]
    cos = emb.cos()[:, None, :, :]  # [B, 1, M, rotary_dim] (broadcast over H)
    sin = emb.sin()[:, None, :, :]

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def apply_partial_rope(x):
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        c, s = cos.to(x.dtype), sin.to(x.dtype)
        x_rotated = (x_rot * c) + (rotate_half(x_rot) * s)
        return torch.cat([x_rotated, x_pass], dim=-1)

    q = apply_partial_rope(q)
    k = apply_partial_rope(k)

    # GQA: repeat k, v
    if H_KV != H_Q:
        k = k.repeat_interleave(H_Q // H_KV, dim=1)
        v = v.repeat_interleave(H_Q // H_KV, dim=1)

    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_out = attn_out * torch.sigmoid(attn_output_gate.float()).to(attn_out.dtype)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, H_Q * head_dim)
    return F.linear(attn_out, w["self_attn.o_proj.weight"])


def _moe_forward(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """MoE forward: 256 experts, top-K=8 routing, shared expert with sigmoid gate."""
    B, M, D = h.shape
    E = cfg["num_experts"]
    K = cfg["num_experts_per_tok"]

    h_flat = h.view(B * M, D)
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)

    moe_out = torch.zeros_like(h_flat)
    for e in range(E):
        mask = (expert_indices == e)
        if not mask.any():
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]
        gate_e = F.linear(x_e, w[f"mlp.experts.{e}.gate_proj.weight"])
        up_e = F.linear(x_e, w[f"mlp.experts.{e}.up_proj.weight"])
        ffn_e = F.linear(F.silu(gate_e) * up_e, w[f"mlp.experts.{e}.down_proj.weight"])
        weight_e = routing_weights[token_idx, slot_idx].unsqueeze(-1)
        moe_out.index_add_(0, token_idx, ffn_e * weight_e)

    # Shared expert
    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return moe_out.view(B, M, D)


def _layer_forward(h: torch.Tensor, position_ids: torch.Tensor, layer_type: str,
                   w: dict, cfg: dict) -> torch.Tensor:
    """One transformer block."""
    # Pre-norm
    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])

    # Attention path
    if layer_type == "linear_attention":
        from engine.qwen36_linear_attn_block import lynn_linear_attn_forward
        attn_out = lynn_linear_attn_forward(h_norm, w)
    elif layer_type == "full_attention":
        attn_out = _full_attn_forward(h_norm, position_ids, w, cfg)
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")
    h = residual + attn_out

    # Post-norm + MoE
    residual = h
    h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])
    moe_out = _moe_forward(h_norm, w, cfg)
    return residual + moe_out


# ----------------- outside-weights loader -----------------

def load_outside_weights(model_dir: str, device: str, dtype=torch.bfloat16):
    """Load embeddings + lm_head + final norm from outside.safetensors."""
    from safetensors import safe_open
    path = Path(model_dir) / "outside.safetensors"
    keys = [
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.language_model.norm.weight",
    ]
    out = {}
    with safe_open(path, framework="pt", device=device) as f:
        for k in keys:
            out[k] = f.get_tensor(k).to(dtype)
    return out


# ----------------- end-to-end forward -----------------

def run_forward(model_dir: str, prompt: str, max_new: int = 1, device: str = "cuda",
                dtype=torch.bfloat16, verbose: bool = True):
    """End-to-end Lynn-engine forward on `prompt`. Returns logits + top-1 token."""
    from engine.loader import load_qwen36_layer

    # Config
    with open(Path(model_dir) / "config.json") as f:
        full_config = json.load(f)
    tc = full_config["text_config"]
    rope_p = tc.get("rope_parameters", {})
    cfg = {
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "num_experts": tc["num_experts"],
        "num_experts_per_tok": tc["num_experts_per_tok"],
        # Qwen 3.6 stores rope params under text_config.rope_parameters
        "rope_theta": rope_p.get("rope_theta", tc.get("rope_theta", 1e6)),
        "partial_rotary_factor": rope_p.get("partial_rotary_factor", 1.0),
    }
    layer_types = tc["layer_types"]
    n_layers = tc["num_hidden_layers"]

    # Tokenize
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    B, T = input_ids.shape
    position_ids = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)

    if verbose:
        print(f"prompt: {prompt!r}")
        print(f"input_ids shape: {input_ids.shape}, tokens: {input_ids[0].tolist()[:20]}...")

    # Load embeddings + final norm + lm_head (kept resident)
    if verbose:
        print(f"\nLoading outside weights ...")
    t0 = time.time()
    outside = load_outside_weights(model_dir, device, dtype)
    if verbose:
        print(f"  done in {time.time()-t0:.1f}s")

    # Embed
    h = F.embedding(input_ids, outside["model.language_model.embed_tokens.weight"])
    if verbose:
        print(f"\nh0 shape: {tuple(h.shape)}, mag: {h.float().abs().mean().item():.3f}")

    # Forward through layers
    t_total = time.time()
    for i in range(n_layers):
        layer_type = layer_types[i]
        t0 = time.time()
        weights, _ = load_qwen36_layer(model_dir, i, num_experts=cfg["num_experts"],
                                       device=device, dequant_dtype=dtype)
        t_load = time.time() - t0

        t0 = time.time()
        h = _layer_forward(h, position_ids, layer_type, weights, cfg)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_fwd = time.time() - t0

        if verbose:
            print(f"  L{i:2} ({layer_type[:6]}) load {t_load:5.1f}s  fwd {t_fwd*1000:5.0f}ms  "
                  f"h_mag {h.float().abs().mean().item():.3f}")

        del weights
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if verbose:
        print(f"\nTotal forward: {time.time()-t_total:.1f}s")

    # Final norm
    h = _rms_norm(h, outside["model.language_model.norm.weight"])

    # lm_head
    last_h = h[:, -1, :]
    logits = F.linear(last_h, outside["lm_head.weight"])

    # Top-K
    top_k = 10
    topv, topi = torch.topk(logits[0], top_k)
    if verbose:
        print(f"\nTop-{top_k} next tokens (logit | id | text):")
        for v, i in zip(topv.tolist(), topi.tolist()):
            text = tok.decode([i]).replace("\n", "\\n")
            print(f"  {v:8.3f}  {i:8d}  {text!r}")

    return {
        "logits": logits.detach().cpu(),
        "top_token_id": topi[0].item(),
        "top_token_text": tok.decode([topi[0].item()]),
        "input_ids": input_ids,
        "tokenizer": tok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, "/work")
    out = run_forward(args.model, args.prompt, device=args.device)
    print(f"\n=== Lynn Engine top-1 next token: {out['top_token_id']} ({out['top_token_text']!r}) ===")


if __name__ == "__main__":
    main()
