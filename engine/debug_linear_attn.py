"""
Compare Lynn linear_attn forward vs HF reference at every intermediate
to localize the divergence.
"""
import sys
import time

import torch
import torch.nn.functional as F


sys.path.insert(0, "/work")
from engine.loader import load_qwen36_layer
from engine.qwen36_linear_attn_block import (
    HIDDEN_SIZE, NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, CONV_KERNEL,
    CONV_DIM, KEY_DIM, VALUE_DIM, V_PER_K, RMS_EPS,
    chunk_gated_delta_rule_torch,
    rms_norm_gated,
)


def report(name, lynn, ref):
    diff = (lynn - ref).float().abs()
    rel = diff.max().item() / max(ref.float().abs().mean().item(), 1e-8) * 100
    print(f"  {name:24}  max_diff={diff.max().item():.3e}  "
          f"mean_diff={diff.mean().item():.3e}  ref_mag={ref.float().abs().mean().item():.3f}  "
          f"rel={rel:.3f}%")


def main():
    device = "cuda"
    dtype = torch.bfloat16
    seq_len = 128

    weights, _ = load_qwen36_layer("/models/Qwen3.6-35B-A3B-FP8", 0,
                                   device=device, dequant_dtype=dtype)

    # Build HF module
    from types import SimpleNamespace
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
    )
    cfg = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        linear_num_value_heads=NUM_V_HEADS,
        linear_num_key_heads=NUM_K_HEADS,
        linear_key_head_dim=HEAD_K_DIM,
        linear_value_head_dim=HEAD_V_DIM,
        linear_conv_kernel_dim=CONV_KERNEL,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=dtype,
    )
    hf = Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0).to(device=device, dtype=dtype)
    hf.eval()
    sd = hf.state_dict()
    for hkey, lkey in [
        ("in_proj_qkv.weight", "linear_attn.in_proj_qkv.weight"),
        ("in_proj_z.weight",   "linear_attn.in_proj_z.weight"),
        ("in_proj_b.weight",   "linear_attn.in_proj_b.weight"),
        ("in_proj_a.weight",   "linear_attn.in_proj_a.weight"),
        ("out_proj.weight",    "linear_attn.out_proj.weight"),
        ("A_log",              "linear_attn.A_log"),
        ("dt_bias",            "linear_attn.dt_bias"),
        ("norm.weight",        "linear_attn.norm.weight"),
        ("conv1d.weight",      "linear_attn.conv1d.weight"),
    ]:
        sd[hkey] = weights[lkey].to(sd[hkey].dtype)
    hf.load_state_dict(sd, strict=True)

    torch.manual_seed(42)
    h = torch.randn(1, seq_len, HIDDEN_SIZE, device=device, dtype=dtype)

    print(f"\nInput h: shape={tuple(h.shape)} mag={h.float().abs().mean().item():.3f}")

    # ============== STEP 1: in_proj_qkv ==============
    lynn_qkv = F.linear(h, weights["linear_attn.in_proj_qkv.weight"])
    ref_qkv = hf.in_proj_qkv(h)
    report("1) in_proj_qkv (B,T,conv)", lynn_qkv, ref_qkv)

    # ============== STEP 2: conv1d + silu ==============
    qkv_t = lynn_qkv.transpose(1, 2)  # [B, conv_dim, T]
    pad = CONV_KERNEL - 1
    qkv_padded = F.pad(qkv_t, (pad, 0))
    lynn_conv = F.silu(F.conv1d(qkv_padded, weights["linear_attn.conv1d.weight"],
                                bias=None, padding=0, groups=CONV_DIM))
    lynn_conv = lynn_conv.transpose(1, 2)  # [B, T, conv_dim]

    # HF version: nn.Conv1d with padding=k-1 (both sides), then [:T]
    ref_qkv_t = ref_qkv.transpose(1, 2)
    ref_conv = F.silu(hf.conv1d(ref_qkv_t)[:, :, :seq_len]).transpose(1, 2)
    report("2) conv1d+silu (B,T,conv)", lynn_conv, ref_conv)

    # ============== STEP 3: split q/k/v ==============
    lynn_q, lynn_k, lynn_v = torch.split(lynn_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    lynn_q = lynn_q.reshape(1, seq_len, NUM_K_HEADS, HEAD_K_DIM)
    lynn_k = lynn_k.reshape(1, seq_len, NUM_K_HEADS, HEAD_K_DIM)
    lynn_v = lynn_v.reshape(1, seq_len, NUM_V_HEADS, HEAD_V_DIM)

    ref_q, ref_k, ref_v = torch.split(ref_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    ref_q = ref_q.reshape(1, seq_len, NUM_K_HEADS, HEAD_K_DIM)
    ref_k = ref_k.reshape(1, seq_len, NUM_K_HEADS, HEAD_K_DIM)
    ref_v = ref_v.reshape(1, seq_len, NUM_V_HEADS, HEAD_V_DIM)
    report("3a) q after split", lynn_q, ref_q)
    report("3b) k after split", lynn_k, ref_k)
    report("3c) v after split", lynn_v, ref_v)

    # ============== STEP 4-6: z, b, a, beta, g ==============
    lynn_z = F.linear(h, weights["linear_attn.in_proj_z.weight"]).reshape(1, seq_len, NUM_V_HEADS, HEAD_V_DIM)
    ref_z = hf.in_proj_z(h).reshape(1, seq_len, NUM_V_HEADS, HEAD_V_DIM)
    report("4) z", lynn_z, ref_z)

    lynn_b = F.linear(h, weights["linear_attn.in_proj_b.weight"])
    ref_b = hf.in_proj_b(h)
    report("5) b", lynn_b, ref_b)

    lynn_beta = lynn_b.sigmoid()
    ref_beta = ref_b.sigmoid()
    report("5b) beta=sigmoid(b)", lynn_beta, ref_beta)

    lynn_a = F.linear(h, weights["linear_attn.in_proj_a.weight"])
    ref_a = hf.in_proj_a(h)
    report("6) a", lynn_a, ref_a)

    lynn_g = -weights["linear_attn.A_log"].float().exp() * F.softplus(
        lynn_a.float() + weights["linear_attn.dt_bias"].float()
    )
    ref_g = -hf.A_log.float().exp() * F.softplus(ref_a.float() + hf.dt_bias)
    report("6b) g", lynn_g, ref_g)

    # ============== STEP 7: repeat q, k by V_PER_K ==============
    lynn_q_rep = lynn_q.repeat_interleave(V_PER_K, dim=2)
    lynn_k_rep = lynn_k.repeat_interleave(V_PER_K, dim=2)
    ref_q_rep = ref_q.repeat_interleave(V_PER_K, dim=2)
    ref_k_rep = ref_k.repeat_interleave(V_PER_K, dim=2)
    report("7a) q_rep", lynn_q_rep, ref_q_rep)
    report("7b) k_rep", lynn_k_rep, ref_k_rep)

    # ============== STEP 8: chunk_gated_delta_rule ==============
    lynn_core, _ = chunk_gated_delta_rule_torch(
        lynn_q_rep, lynn_k_rep, lynn_v, lynn_g, lynn_beta,
        chunk_size=64, use_qk_l2norm=True,
    )
    ref_core, _ = hf.chunk_gated_delta_rule(
        ref_q_rep, ref_k_rep, ref_v, g=ref_g, beta=ref_beta,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    report("8) core_attn_out", lynn_core, ref_core)

    # ============== STEP 9: RMSNormGated ==============
    flat_lynn = lynn_core.reshape(-1, HEAD_V_DIM)
    flat_lynn_z = lynn_z.reshape(-1, HEAD_V_DIM)
    lynn_normed = rms_norm_gated(flat_lynn, weights["linear_attn.norm.weight"], flat_lynn_z, eps=RMS_EPS)
    lynn_normed = lynn_normed.reshape(1, seq_len, -1)

    flat_ref = ref_core.reshape(-1, HEAD_V_DIM)
    flat_ref_z = ref_z.reshape(-1, HEAD_V_DIM)
    ref_normed = hf.norm(flat_ref, flat_ref_z).reshape(1, seq_len, -1)
    report("9) RMSNormGated", lynn_normed, ref_normed)

    # ============== STEP 10: out_proj ==============
    lynn_out = F.linear(lynn_normed, weights["linear_attn.out_proj.weight"])
    ref_out = hf.out_proj(ref_normed)
    report("10) out_proj (final)", lynn_out, ref_out)


if __name__ == "__main__":
    main()
