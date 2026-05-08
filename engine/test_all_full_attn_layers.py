"""
Lynn Engine · Phase 2 · P1.1 — Sequential alignment on all 10 full_attention layers.

Qwen 3.6 35B-A3B has 40 layers, 10 of which are full_attention (indices
3, 7, 11, 15, 19, 23, 27, 31, 35, 39). The other 30 are linear_attention
(Mamba/GLA-style) which our kernels don't cover yet.

This test loads each full_attention layer's real Qwen 3.6 weights and runs
our integration test (lynn vs reference) sequentially. Goal: confirm our 4
Triton kernels work on every full_attention layer position, not just layer 3.

Memory budget per layer: ~1.7 GB BF16. Total never exceeds ~3 GB at peak.
Doesn't disturb running vLLM Qwen 3.6 / ELYZA / voice services.
"""
import sys, json, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.loader import load_qwen36_layer
from engine.qwen36_block import qwen36_reference, qwen36_lynn


def main():
    model_dir = "/models/Qwen3.6-35B-A3B-FP8"
    full_attention_layers = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    M = 64

    # Load config from model
    with open(f"{model_dir}/config.json") as f:
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

    # Triton kernels (load once)
    print(f"⚙️  Loading Triton kernels...", flush=True)
    from triton_kernels.attention import make_triton_attention
    from triton_kernels.rope import make_triton_rope
    from triton_kernels.rmsnorm import make_triton_rmsnorm
    from triton_kernels.moe import make_triton_router
    rmsnorm_fn = make_triton_rmsnorm()
    rope_fn = make_triton_rope()
    attn_fn = make_triton_attention()
    router_fn = make_triton_router()
    print(f"   kernels loaded ✅", flush=True)

    # Test inputs (same for all layers — different real weights each time)
    torch.manual_seed(42)
    D = cfg["hidden_size"]
    hidden_template = torch.randn(1, M, D, dtype=torch.bfloat16, device="cuda") * 0.1
    position_ids = torch.arange(M, device="cuda", dtype=torch.long).unsqueeze(0).expand(1, M)

    print(f"\n{'Layer':<6} {'load(s)':<8} {'ref(ms)':<10} {'lynn(ms)':<10} {'max diff':<14} {'mean diff':<14} {'rel diff':<14} {'status'}", flush=True)
    print("-" * 110, flush=True)

    rows = []
    all_pass = True
    PASS = 5e-2

    for layer_idx in full_attention_layers:
        # Load this layer
        t0 = time.time()
        weights, _ = load_qwen36_layer(model_dir, layer_idx, num_experts=cfg["num_experts"], device="cuda")
        load_s = time.time() - t0

        # Use same input for all layers for fair comparison
        hidden = hidden_template.clone()

        # Reference
        torch.cuda.synchronize()
        t0 = time.time()
        out_ref = qwen36_reference(hidden, position_ids, weights, cfg)
        torch.cuda.synchronize()
        ref_ms = (time.time() - t0) * 1000

        # Lynn
        t0 = time.time()
        out_lynn = qwen36_lynn(hidden, position_ids, weights, cfg,
                              rmsnorm_fn, rope_fn, attn_fn, router_fn)
        torch.cuda.synchronize()
        lynn_ms = (time.time() - t0) * 1000

        # Compare
        max_diff = (out_ref.float() - out_lynn.float()).abs().max().item()
        mean_diff = (out_ref.float() - out_lynn.float()).abs().mean().item()
        ref_norm = out_ref.float().abs().mean().item()
        rel_diff = mean_diff / ref_norm if ref_norm > 0 else 0
        status = "✅" if max_diff < PASS else "❌"
        if max_diff >= PASS:
            all_pass = False

        print(f"L{layer_idx:<3}   {load_s:<7.1f} {ref_ms:<10.1f} {lynn_ms:<10.1f} {max_diff:<14.6e} {mean_diff:<14.6e} {rel_diff:<14.6e} {status}", flush=True)
        rows.append({
            "layer": layer_idx, "load_s": load_s,
            "ref_ms": ref_ms, "lynn_ms": lynn_ms,
            "max_diff": max_diff, "mean_diff": mean_diff, "rel_diff": rel_diff,
            "ref_norm": ref_norm,
        })

        # Free this layer's weights before loading next
        del weights, out_ref, out_lynn
        torch.cuda.empty_cache()

    print("-" * 110, flush=True)
    if all_pass:
        avg_rel = sum(r["rel_diff"] for r in rows) / len(rows)
        avg_max = sum(r["max_diff"] for r in rows) / len(rows)
        print(f"\n✅ ALL 10 full_attention layers PASS")
        print(f"   Avg max diff:  {avg_max:.4e}")
        print(f"   Avg rel diff:  {avg_rel:.4e}")
        print(f"   PASS threshold: {PASS}")
        print(f"\nP1.1 milestone: Lynn Engine 4 Triton kernels validated on all 10 Qwen 3.6")
        print(f"full_attention layers (3, 7, 11, ..., 39) with REAL learned weights.")
        print(f"Engine MVP path validated layer-by-layer; ready for P1.2 (full 40-layer assembly).")
    else:
        fails = [r["layer"] for r in rows if r["max_diff"] >= PASS]
        print(f"\n❌ {len(fails)} layers FAIL: {fails}")

    # Save
    out = Path(__file__).parent.parent / f"benchmarks/results/p1_1_all_full_attn_layers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"layers": rows, "all_pass": all_pass, "threshold": PASS}, ensure_ascii=False, indent=2))
    print(f"\n💾 saved → {out}")


if __name__ == "__main__":
    main()
