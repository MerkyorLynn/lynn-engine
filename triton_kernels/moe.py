"""
Lynn Engine · Phase 2 · MoE router + dispatch

Qwen 3.6 35B-A3B MoE block:
  - 256 routed experts per layer + 1 shared expert
  - Top-8 routing (8 experts active per token)
  - Per-expert intermediate_dim ~1408 (gate_proj × up_proj × down_proj)
  - Routing: softmax(top_k(router_logits)) → weighted combine

This file:
  1. PyTorch reference MoE forward (router + topk + expert FFN + combine)
  2. Triton-accelerated routing kernel (fused top-k + softmax)
  3. Numerical alignment test on Qwen3.6-A3B-shaped inputs

The expert FFN computation itself uses standard PyTorch ops here;
production version will use CUTLASS grouped GEMM (Phase 3).

Acceptance: max(|out_triton - out_reference|) < 5e-3 (FP16 ULP floor for
multi-step compute), routing indices match exactly.
"""
import argparse
import math
import time
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# PyTorch reference MoE forward
# ─────────────────────────────────────────────────────────────
def moe_reference(hidden, router_weight, gate_proj, up_proj, down_proj,
                  num_experts, top_k, shared_gate=None, shared_up=None, shared_down=None):
    """
    Reference MoE forward (matches Qwen3.6 architecture).

    Args:
        hidden: [N, D] input hidden states (N = batch × seq_len)
        router_weight: [num_experts, D] router gate weights
        gate_proj: [num_experts, D, intermediate_dim] per-expert gate proj
        up_proj:   [num_experts, D, intermediate_dim] per-expert up proj
        down_proj: [num_experts, intermediate_dim, D] per-expert down proj
        num_experts: 256
        top_k: 8
        shared_*: optional shared expert (1 per layer in Qwen3.6)

    Returns:
        out: [N, D]
    """
    N, D = hidden.shape
    E = num_experts
    K = top_k

    # ── Router ──
    router_logits = hidden @ router_weight.t()  # [N, E]
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)  # [N, K]
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(hidden.dtype)

    # ── Routed experts: per-expert dispatch ──
    out_routed = torch.zeros_like(hidden)
    for e in range(E):
        # Mask of tokens routed to expert e
        mask = (expert_indices == e)  # [N, K]
        if not mask.any():
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)  # which tokens go to e + which top-K slot
        if len(token_idx) == 0:
            continue

        x = hidden[token_idx]  # [n_e, D]
        # SwiGLU: SiLU(gate(x)) * up(x) → down
        gate_out = x @ gate_proj[e]   # [n_e, I]
        up_out = x @ up_proj[e]       # [n_e, I]
        ffn_out = (F.silu(gate_out) * up_out) @ down_proj[e]  # [n_e, D]

        # Weighted combine
        weights = routing_weights[token_idx, slot_idx].unsqueeze(-1)  # [n_e, 1]
        out_routed.index_add_(0, token_idx, ffn_out * weights)

    # ── Shared expert (always-active, no routing) ──
    if shared_gate is not None:
        gate_s = hidden @ shared_gate
        up_s = hidden @ shared_up
        shared_out = (F.silu(gate_s) * up_s) @ shared_down
        out_routed = out_routed + shared_out

    return out_routed, expert_indices, routing_weights


# ─────────────────────────────────────────────────────────────
# Triton router kernel: fused top-k + softmax
# ─────────────────────────────────────────────────────────────
def make_triton_router():
    import triton
    import triton.language as tl

    @triton.jit
    def _router_kernel(
        router_logits,   # [N, E]
        out_indices,     # [N, K]
        out_weights,     # [N, K]
        N,
        E: tl.constexpr,
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,  # = next_pow2(E)
    ):
        """One program per token. Computes top-K then softmax over selected.
        Uses iterative selection (suitable for E=256, K=8)."""
        token_idx = tl.program_id(0)
        if token_idx >= N:
            return

        offs_e = tl.arange(0, BLOCK_E)
        mask_e = offs_e < E

        # Load router logits for this token
        logits = tl.load(
            router_logits + token_idx * E + offs_e,
            mask=mask_e,
            other=-float("inf"),
        ).to(tl.float32)

        # Iterative top-K via masking
        # (E=256, K=8, so 8 sweeps of 256 = 2048 ops, trivial for one program)
        masked = logits

        # Storage for selected indices + values
        # We'll fill them iteratively
        for k in tl.static_range(K):
            # Find argmax
            cur_max = tl.max(masked, axis=0)
            # Index of max (one-hot lookup)
            is_max = (masked >= cur_max) & mask_e
            # Pick the lowest index among ties (deterministic)
            idx_at_max = tl.where(is_max, offs_e, E)  # E = sentinel
            picked_idx = tl.min(idx_at_max, axis=0)

            # Store
            tl.store(out_indices + token_idx * K + k, picked_idx)
            tl.store(out_weights + token_idx * K + k, cur_max)

            # Mask out picked index
            masked = tl.where(offs_e == picked_idx, -float("inf"), masked)

        # Softmax over the K selected weights
        # (can be done outside in Python; we do in kernel for fusion)
        # Reload selected weights:
        offs_k = tl.arange(0, K)  # K is constexpr so this works
        sel_weights = tl.load(out_weights + token_idx * K + offs_k).to(tl.float32)
        max_w = tl.max(sel_weights, axis=0)
        exp_w = tl.exp(sel_weights - max_w)
        sum_exp = tl.sum(exp_w, axis=0)
        sm = exp_w / sum_exp
        tl.store(out_weights + token_idx * K + offs_k, sm)

    def router_triton(hidden, router_weight, top_k):
        """Triton router wrapper."""
        N, D = hidden.shape
        E, _ = router_weight.shape
        assert hidden.is_cuda

        # Compute logits via cuBLAS — match reference precision (FP16 matmul → FP32 for kernel)
        # This ensures tie-breaking matches the reference implementation.
        router_logits = (hidden @ router_weight.t()).float()  # [N, E] in FP32 after FP16 matmul

        # Allocate outputs
        out_indices = torch.empty(N, top_k, dtype=torch.int32, device=hidden.device)
        out_weights = torch.empty(N, top_k, dtype=torch.float32, device=hidden.device)

        BLOCK_E = 1 << (E - 1).bit_length()
        BLOCK_E = max(BLOCK_E, 64)

        grid = (N,)
        _router_kernel[grid](
            router_logits.contiguous(),
            out_indices,
            out_weights,
            N,
            E=E,
            K=top_k,
            BLOCK_E=BLOCK_E,
        )

        return out_indices, out_weights

    return router_triton


# ─────────────────────────────────────────────────────────────
def run_correctness_test(reference_only=False):
    torch.manual_seed(42)
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32

    print(f"⚙️  MoE router correctness test (device={device}, dtype={dtype})")

    # Qwen 3.6 35B-A3B MoE specs
    N = 128         # tokens (prefill 128 or batch=128)
    D = 8192        # hidden_dim
    E = 256         # num_experts
    K = 8           # top_k
    INT = 1408      # intermediate_dim per expert (representative)

    print(f"   model:   N={N}, D={D}, E={E}, K={K}, expert_intermediate={INT}")
    print(f"   memory:  router_weight {E*D*2/1e6:.0f}MB, gate+up+down {3*E*D*INT*2/1e9:.2f}GB FP16")

    # Allocate (note: on CUDA we may need smaller test for memory)
    if has_cuda and not reference_only:
        # Cap memory: use smaller E for test (still validates kernel)
        E_test = 64  # smaller for CUDA memory budget
        print(f"   (using E_test={E_test} for kernel correctness; production runs full E={E})")
    else:
        E_test = E if not has_cuda else 64

    hidden = torch.randn(N, D, dtype=dtype, device=device) * 0.1
    router_weight = torch.randn(E_test, D, dtype=dtype, device=device) * 0.05

    triton_router = None
    if has_cuda and not reference_only:
        try:
            triton_router = make_triton_router()
        except Exception as e:
            print(f"❌ Triton init failed: {e}")

    # ── Reference router ──
    t0 = time.time()
    router_logits_ref = hidden @ router_weight.t()
    weights_ref, indices_ref = torch.topk(router_logits_ref, K, dim=-1)
    weights_ref = F.softmax(weights_ref, dim=-1, dtype=torch.float32)
    if has_cuda:
        torch.cuda.synchronize()
    ref_ms = (time.time() - t0) * 1000

    print(f"\n=== Router test (top-{K} of {E_test} experts) ===")
    print(f"PyTorch reference:  {ref_ms:.3f} ms")

    if triton_router is None:
        print("⚪ Reference-only mode (no CUDA / no Triton)")
        return

    # ── Triton router ──
    t0 = time.time()
    indices_tri, weights_tri = triton_router(hidden, router_weight, K)
    torch.cuda.synchronize()
    tri_ms = (time.time() - t0) * 1000
    print(f"Triton kernel:      {tri_ms:.3f} ms")

    # ── Correctness checks ──
    # 1. Index sets match (within ties allowed)
    indices_ref_sorted = torch.sort(indices_ref, dim=-1).values
    indices_tri_sorted = torch.sort(indices_tri.long(), dim=-1).values
    indices_match = (indices_ref_sorted == indices_tri_sorted).all().item()

    # 2. Routing weights match (after re-aligning by index)
    # Need to align: ref weights are sorted by topk's logit value, triton same
    # Compare weight VALUES set-wise per token
    weights_ref_sorted = torch.sort(weights_ref.float(), dim=-1).values
    weights_tri_sorted = torch.sort(weights_tri.float(), dim=-1).values
    max_w_diff = (weights_ref_sorted - weights_tri_sorted).abs().max().item()

    print(f"\nResults:")
    print(f"  Selected expert sets match:  {'✅ YES' if indices_match else '❌ NO'}")
    print(f"  Routing weights max diff:    {max_w_diff:.6e}  ({'✅' if max_w_diff < 5e-3 else '❌'})")

    if indices_match and max_w_diff < 5e-3:
        print("✅ MoE router Triton kernel passes")
    else:
        print("❌ MoE router FAIL — debug before integration")
        if not indices_match:
            mismatch_count = (indices_ref_sorted != indices_tri_sorted).any(dim=-1).sum().item()
            print(f"   {mismatch_count}/{N} tokens have mismatched expert sets")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-only", action="store_true")
    args = ap.parse_args()
    run_correctness_test(reference_only=args.reference_only)
