"""
Lynn Engine · linear_attention block alignment test.

Validates `lynn_linear_attn_forward` against HF `Qwen3_5MoeGatedDeltaNet`
on real layer 0 weights with synthetic input. Mirrors the strategy of
test_all_full_attn_layers.py (P1.1) but for the 30 linear_attention layers.

Strategy (memory-bounded, doesn't need vLLM stopped):
  1. Load layer 0 linear_attn weights via Lynn loader (~600 MB BF16)
  2. Instantiate a stand-alone HF Qwen3_5MoeGatedDeltaNet on cuda:0
  3. Copy Lynn-loaded weights into the HF module
  4. Run identical synthetic h_in through both, compare outputs
  5. Repeat across the 8 sampled linear_attention layers (0,4,8,...,32)

Pass criterion: rel_diff < 5% (same as P1.1).

Why this is a strong validation:
  - 30 of 40 layers are linear_attention. P1.1 only covered 10 full_attn
    layers. Together = full architecture coverage.
  - Math is non-trivial: gated delta rule with chunk-wise update + l2norm
    + decay accumulation. Mistranscribing any of it shows up here.
  - Real weights catch overflow / scale issues that synthetic-weight tests
    would miss.

Usage (inside vllm container):
    python3 engine/test_lynn_linear_attn.py \
        --model /models/Qwen3.6-35B-A3B-FP8 \
        --layers 0,4,8,12,16,20,24,28,32,36 \
        --seq-len 64
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


def make_hf_module(num_v_heads, num_k_heads, head_k_dim, head_v_dim,
                   conv_kernel, hidden_size, dtype, device):
    """Instantiate HF's Qwen3_5MoeGatedDeltaNet without a config dependency.

    The HF class takes a Qwen3_5MoeConfig object. We construct a minimal
    object satisfying only the fields the constructor reads.
    """
    from types import SimpleNamespace
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
    )

    cfg = SimpleNamespace(
        hidden_size=hidden_size,
        linear_num_value_heads=num_v_heads,
        linear_num_key_heads=num_k_heads,
        linear_key_head_dim=head_k_dim,
        linear_value_head_dim=head_v_dim,
        linear_conv_kernel_dim=conv_kernel,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=dtype,
    )
    mod = Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0).to(device=device, dtype=dtype)
    mod.eval()
    return mod


def copy_weights_into_hf(hf_mod, lynn_weights, prefix="linear_attn"):
    """Copy Lynn-loaded weights into the HF GatedDeltaNet module."""
    sd = hf_mod.state_dict()

    # Map our keys → HF keys
    pairs = [
        (f"{prefix}.in_proj_qkv.weight", "in_proj_qkv.weight"),
        (f"{prefix}.in_proj_z.weight",   "in_proj_z.weight"),
        (f"{prefix}.in_proj_b.weight",   "in_proj_b.weight"),
        (f"{prefix}.in_proj_a.weight",   "in_proj_a.weight"),
        (f"{prefix}.out_proj.weight",    "out_proj.weight"),
        (f"{prefix}.A_log",              "A_log"),
        (f"{prefix}.dt_bias",            "dt_bias"),
        (f"{prefix}.norm.weight",        "norm.weight"),
        (f"{prefix}.conv1d.weight",      "conv1d.weight"),
    ]
    new_sd = {}
    for lkey, hkey in pairs:
        if lkey not in lynn_weights:
            raise KeyError(f"Lynn weights missing {lkey}")
        if hkey not in sd:
            raise KeyError(f"HF state_dict missing {hkey}")
        if lynn_weights[lkey].shape != sd[hkey].shape:
            raise ValueError(
                f"Shape mismatch {lkey} ({tuple(lynn_weights[lkey].shape)}) "
                f"vs HF {hkey} ({tuple(sd[hkey].shape)})"
            )
        new_sd[hkey] = lynn_weights[lkey].to(sd[hkey].dtype).to(sd[hkey].device)

    # Some keys may not be in Lynn weights (e.g. conv1d.bias = None case).
    # Fill rest from current state_dict.
    for k in sd:
        if k not in new_sd:
            new_sd[k] = sd[k]

    hf_mod.load_state_dict(new_sd, strict=True)


def run_one_layer(model_dir: str, layer_idx: int, seq_len: int, device: str,
                  dtype=torch.bfloat16):
    sys.path.insert(0, "/work")
    from engine.loader import load_qwen36_layer
    from engine.qwen36_linear_attn_block import (
        lynn_linear_attn_forward,
        HIDDEN_SIZE, NUM_V_HEADS, NUM_K_HEADS, HEAD_K_DIM, HEAD_V_DIM, CONV_KERNEL,
    )

    print(f"\n--- Layer {layer_idx} (linear_attention) ---", flush=True)
    t0 = time.time()
    weights, _ = load_qwen36_layer(model_dir, layer_idx, device=device,
                                   dequant_dtype=dtype)
    print(f"  loader: {time.time()-t0:.1f}s, {len(weights)} tensors", flush=True)

    # HF reference
    hf = make_hf_module(NUM_V_HEADS, NUM_K_HEADS, HEAD_K_DIM, HEAD_V_DIM,
                        CONV_KERNEL, HIDDEN_SIZE, dtype, device)
    copy_weights_into_hf(hf, weights, prefix="linear_attn")

    torch.manual_seed(42 + layer_idx)
    # Use unit-variance input — closer to what real residual streams look like
    # after RMSNorm and a few accumulated layers. Scale 0.02 (~init scale)
    # produces near-zero gated-delta-net output and saturates rel% on
    # noise floor.
    h_in = torch.randn(1, seq_len, HIDDEN_SIZE, device=device, dtype=dtype)

    t0 = time.time()
    with torch.no_grad():
        ref = hf(h_in)
    t_ref = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        lynn = lynn_linear_attn_forward(h_in, weights)
    t_lynn = time.time() - t0

    diff = (lynn - ref).float().abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    ref_mag = ref.float().abs().mean().item()
    rel = max_d / max(ref_mag, 1e-8) * 100
    status = "✅ PASS" if rel < 5.0 else "❌ FAIL"
    print(
        f"  L{layer_idx:2}  shape_lynn={tuple(lynn.shape)} shape_ref={tuple(ref.shape)}\n"
        f"        max={max_d:.3e}  mean={mean_d:.3e}  ref_mag={ref_mag:.3f}  rel={rel:.3f}%\n"
        f"        ref={t_ref*1000:.0f}ms  lynn={t_lynn*1000:.0f}ms  {status}",
        flush=True,
    )

    del weights, hf
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {"layer": layer_idx, "rel_diff_pct": rel, "passed": rel < 5.0,
            "t_ref_ms": t_ref * 1000, "t_lynn_ms": t_lynn * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3.6-35B-A3B-FP8",
                    help="Qwen3.6-35B-A3B-FP8 dir (FP8 dequanted on the fly by Lynn loader)")
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,32,36",
                    help="comma-sep linear_attention layer indices")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--single-layer", type=int, default=None)
    args = ap.parse_args()

    layers = ([args.single_layer] if args.single_layer is not None
              else [int(s) for s in args.layers.split(",")])

    results = []
    for li in layers:
        try:
            r = run_one_layer(args.model, li, args.seq_len, args.device)
            results.append(r)
        except Exception as e:
            print(f"L{li} ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results.append({"layer": li, "error": str(e)})

    print("\n" + "=" * 60)
    print(f"Lynn linear_attention validation — {len(results)} layers")
    print("=" * 60)
    passed = [r for r in results if r.get("passed")]
    failed = [r for r in results if not r.get("passed")]
    print(f"Passed: {len(passed)}/{len(results)}")
    if passed:
        rels = [r["rel_diff_pct"] for r in passed]
        print(f"  rel_diff: avg={sum(rels)/len(rels):.3f}%  max={max(rels):.3f}%")
    for r in failed:
        print(f"  L{r['layer']}: {r.get('rel_diff_pct', '?')}%  err={r.get('error', '')}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
