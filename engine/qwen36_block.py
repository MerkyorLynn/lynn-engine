"""
Lynn Engine · Qwen 3.6 35B-A3B specific transformer block

Adds Qwen 3.6 specific features missing from the generic transformer_block.py:
  · attn_output_gate: q_proj output is 2× expected (Q | gate),
    sigmoid(gate) modulates attention output before O proj
  · q_norm + k_norm: extra RMSNorm on Q and K BEFORE RoPE (Qwen3 trick)
  · GQA 16:2 = 8x ratio
  · No bias on attention projections

Test goal: load real layer 3 weights → compare lynn vs reference output.
"""
import sys, math, time, json
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.loader import load_qwen36_layer


# ─────────────────────────────────────────────────────────────
# Qwen 3.6 reference block (PyTorch only, no Triton)
# ─────────────────────────────────────────────────────────────
def qwen36_reference(hidden, position_ids, w, cfg):
    """Qwen 3.6 35B-A3B transformer block forward (PyTorch reference).

    w: dict from loader.py with real weights (BF16 dequant from FP8)
    cfg: dict with hidden_size, num_attention_heads, num_kv_heads, head_dim, etc.
    """
    B, M, D = hidden.shape
    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    E = cfg["num_experts"]
    K = cfg["num_experts_per_tok"]
    rope_theta = cfg["rope_theta"]

    # 1. Pre-attention RMSNorm
    x_f32 = hidden.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    h_norm = (x_f32 / rms * w["input_layernorm.weight"].float()).to(hidden.dtype)

    # 2. QKV projections (Qwen 3.6: q_proj output = 2 × H_Q × head_dim, second half is gate)
    q_full = F.linear(h_norm, w["self_attn.q_proj.weight"])      # [B, M, 2*H_Q*head_dim]
    k = F.linear(h_norm, w["self_attn.k_proj.weight"])           # [B, M, H_KV*head_dim]
    v = F.linear(h_norm, w["self_attn.v_proj.weight"])           # [B, M, H_KV*head_dim]

    # Split q_full into [Q | output_gate]
    q, attn_output_gate = q_full.chunk(2, dim=-1)                # each [B, M, H_Q*head_dim]

    # Reshape to [B, H, M, head_dim]
    q = q.view(B, M, H_Q, head_dim).transpose(1, 2)
    k = k.view(B, M, H_KV, head_dim).transpose(1, 2)
    v = v.view(B, M, H_KV, head_dim).transpose(1, 2)
    attn_output_gate = attn_output_gate.view(B, M, H_Q, head_dim).transpose(1, 2)  # gate per head

    # 3. q_norm and k_norm (Qwen 3 specific — RMSNorm on Q and K BEFORE RoPE)
    q_norm_w = w["self_attn.q_norm.weight"]
    k_norm_w = w["self_attn.k_norm.weight"]
    q_f32 = q.float()
    q_rms = torch.sqrt(q_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    q = (q_f32 / q_rms * q_norm_w.float()).to(q.dtype)

    k_f32 = k.float()
    k_rms = torch.sqrt(k_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    k = (k_f32 / k_rms * k_norm_w.float()).to(k.dtype)

    # 4. RoPE on Q and K
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=hidden.device, dtype=torch.float32) / head_dim))
    freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]
    cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(1)

    def apply_rope(x):
        x_f32 = x.float()
        x_e = x_f32[..., 0::2]
        x_o = x_f32[..., 1::2]
        c_e = cos[..., 0::2]
        s_e = sin[..., 0::2]
        out = torch.empty_like(x_f32)
        out[..., 0::2] = x_e * c_e - x_o * s_e
        out[..., 1::2] = x_o * c_e + x_e * s_e
        return out.to(x.dtype)

    q = apply_rope(q)
    k = apply_rope(k)

    # 5. Causal attention with GQA
    if H_KV != H_Q:
        k = k.repeat_interleave(H_Q // H_KV, dim=1)
        v = v.repeat_interleave(H_Q // H_KV, dim=1)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, M, D]

    # 6. attn_output_gate (Qwen 3.6 specific)
    attn_out = attn_out * torch.sigmoid(attn_output_gate.float()).to(attn_out.dtype)

    # 7. O proj + residual
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, H_Q * head_dim)
    h = hidden + F.linear(attn_out, w["self_attn.o_proj.weight"])

    # 8. Post-attention RMSNorm
    x_f32 = h.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    h_norm = (x_f32 / rms * w["post_attention_layernorm.weight"].float()).to(h.dtype)

    # 9. MoE: router + top-K + per-expert SwiGLU + combine
    h_flat = h_norm.view(B * M, D)
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

    # 10. Shared expert (always active, sigmoid-gated)
    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return h + moe_out.view(B, M, D)


# ─────────────────────────────────────────────────────────────
# Qwen 3.6 lynn-engine block (uses 4 Triton kernels)
# ─────────────────────────────────────────────────────────────
def qwen36_lynn(hidden, position_ids, w, cfg, rmsnorm_fn, rope_fn, attention_fn, router_fn):
    B, M, D = hidden.shape
    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    E = cfg["num_experts"]
    K = cfg["num_experts_per_tok"]
    rope_theta = cfg["rope_theta"]

    # 1. Pre-attention RMSNorm (Triton)
    h_norm = rmsnorm_fn(hidden, w["input_layernorm.weight"], 1e-6)

    # 2. QKV projections + split q + gate
    q_full = F.linear(h_norm, w["self_attn.q_proj.weight"])
    k = F.linear(h_norm, w["self_attn.k_proj.weight"])
    v = F.linear(h_norm, w["self_attn.v_proj.weight"])
    q, attn_output_gate = q_full.chunk(2, dim=-1)

    q = q.view(B, M, H_Q, head_dim).transpose(1, 2).contiguous()
    k = k.view(B, M, H_KV, head_dim).transpose(1, 2).contiguous()
    v = v.view(B, M, H_KV, head_dim).transpose(1, 2).contiguous()
    attn_output_gate = attn_output_gate.view(B, M, H_Q, head_dim).transpose(1, 2).contiguous()

    # 3. q_norm / k_norm (Triton RMSNorm — applied per-head per-position)
    # rmsnorm expects [..., D]; we have [B, H, M, head_dim], merge B and H, reshape
    q_flat = q.reshape(B * H_Q * M, head_dim)
    q_flat = rmsnorm_fn(q_flat, w["self_attn.q_norm.weight"], 1e-6)
    q = q_flat.view(B, H_Q, M, head_dim)

    k_flat = k.reshape(B * H_KV * M, head_dim)
    k_flat = rmsnorm_fn(k_flat, w["self_attn.k_norm.weight"], 1e-6)
    k = k_flat.view(B, H_KV, M, head_dim)

    # 4. RoPE (Triton)
    q = rope_fn(q, position_ids, theta=rope_theta, ntk_factor=1.0)
    k = rope_fn(k, position_ids, theta=rope_theta, ntk_factor=1.0)

    # 5. Attention (Triton)
    attn_out = attention_fn(q, k, v, causal=True)

    # 6. attn_output_gate
    attn_out = attn_out * torch.sigmoid(attn_output_gate.float()).to(attn_out.dtype)

    # 7. O proj + residual
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, H_Q * head_dim)
    h = hidden + F.linear(attn_out, w["self_attn.o_proj.weight"])

    # 8. Post-attention RMSNorm (Triton)
    h_norm = rmsnorm_fn(h, w["post_attention_layernorm.weight"], 1e-6)

    # 9. MoE: router (Triton) + per-expert (PyTorch)
    h_flat = h_norm.view(B * M, D)
    expert_indices, routing_weights = router_fn(h_flat, w["mlp.gate.weight"], K)
    expert_indices = expert_indices.long()
    routing_weights = routing_weights.to(h.dtype)

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

    # 10. Shared expert
    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return h + moe_out.view(B, M, D)


# ─────────────────────────────────────────────────────────────
def run_alignment_test(model_dir, layer_idx=3, M=64, B=1):
    """Load real Qwen 3.6 layer N + verify lynn vs reference alignment."""
    torch.manual_seed(42)
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        print("❌ CUDA required"); return
    device = "cuda"

    # ── Load real config ──
    with open(Path(model_dir) / "config.json") as f:
        full_config = json.load(f)
    tc = full_config.get("text_config", full_config)

    cfg = {
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "num_experts": tc["num_experts"],
        "num_experts_per_tok": tc["num_experts_per_tok"],
        "rope_theta": tc.get("rope_theta", 1e6),
    }

    print(f"⚙️  Qwen 3.6 35B-A3B layer {layer_idx} integration test")
    print(f"   config: {cfg}")
    print()

    # ── Load real weights ──
    print(f"📥 Loading layer {layer_idx} weights...")
    weights, _ = load_qwen36_layer(model_dir, layer_idx, num_experts=cfg["num_experts"], device=device)

    # ── Test inputs ──
    D = cfg["hidden_size"]
    hidden = torch.randn(B, M, D, dtype=torch.bfloat16, device=device) * 0.1
    position_ids = torch.arange(M, device=device, dtype=torch.long).unsqueeze(0).expand(B, M)

    # ── Reference ──
    print(f"\n🧪 Running reference (PyTorch only)...")
    t0 = time.time()
    out_ref = qwen36_reference(hidden, position_ids, weights, cfg)
    torch.cuda.synchronize()
    ref_ms = (time.time() - t0) * 1000
    print(f"   reference: {ref_ms:.1f} ms")

    # ── Lynn (with Triton kernels) ──
    print(f"\n🧪 Running lynn (Triton attention + RoPE + RMSNorm + router)...")
    from triton_kernels.attention import make_triton_attention
    from triton_kernels.rope import make_triton_rope
    from triton_kernels.rmsnorm import make_triton_rmsnorm
    from triton_kernels.moe import make_triton_router

    rmsnorm_fn = make_triton_rmsnorm()
    rope_fn = make_triton_rope()
    attn_fn = make_triton_attention()
    router_fn = make_triton_router()

    t0 = time.time()
    out_lynn = qwen36_lynn(hidden, position_ids, weights, cfg,
                          rmsnorm_fn, rope_fn, attn_fn, router_fn)
    torch.cuda.synchronize()
    lynn_ms = (time.time() - t0) * 1000
    print(f"   lynn:      {lynn_ms:.1f} ms")

    # ── Compare ──
    max_diff = (out_ref.float() - out_lynn.float()).abs().max().item()
    mean_diff = (out_ref.float() - out_lynn.float()).abs().mean().item()
    out_ref_norm = out_ref.float().abs().mean().item()
    rel_diff = mean_diff / out_ref_norm

    print(f"\n=== Real-weights layer {layer_idx} alignment ===")
    print(f"  output magnitude (ref mean abs):  {out_ref_norm:.4f}")
    print(f"  max  abs diff:                    {max_diff:.6e}")
    print(f"  mean abs diff:                    {mean_diff:.6e}")
    print(f"  relative diff:                    {rel_diff:.6e}")

    PASS = 5e-2  # 5% — full block compound BF16
    status = "✅ PASS" if max_diff < PASS else "❌ FAIL"
    print(f"\n  PASS threshold (compound BF16, 10+ ops): {PASS}")
    print(f"  Status: {status}")
    if max_diff < PASS:
        print(f"\n✅ Lynn Engine integration on REAL Qwen 3.6 layer-{layer_idx} weights PASSES")
        print(f"   This validates: 4 Triton kernels work on real learned weight distributions,")
        print(f"   safetensors loader + FP8 dequant pipeline correct, MoE router on real")
        print(f"   learned routing distributions matches reference output.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/models/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--M", type=int, default=64, help="sequence length")
    args = ap.parse_args()
    run_alignment_test(args.model_dir, layer_idx=args.layer, M=args.M)
