"""
Lynn Engine · Phase 2 · P1.3 — Lynn full_attention block validated on REAL
hidden-state distributions from HF transformers BF16 forward.

This closes the validation loop:
  P1.1: synthetic random input → full_attention layer  ✅ done (10/10 layers passed)
  P1.3: REAL hidden states from real prompt → full_attention layer  ← this script

Why this matters more than P1.1:
  - P1.1 used randn input. Real hidden states have very different distribution
    after RMSNorm + RoPE + attention residual streams accumulate over 1-3 layers.
  - L1-L2 (linear_attention) running before each full_attention layer can produce
    pathological activations that a naïve impl would mishandle.
  - This is the strongest single-block validation we can do without going
    end-to-end (which needs our own linear_attention kernel — Phase 3 work).

Strategy:
  1. Load HF Qwen3_5_MoE in BF16 (35B model, ~70 GB BF16 on disk)
  2. Tokenize a test prompt + run forward with output_hidden_states=True
  3. For each full_attention layer i in {3, 7, 11, ..., 39}:
       h_in   = hidden_states[i]      (HF says this is the input to layer i)
       h_ref  = hidden_states[i+1]    (HF says this is the output of layer i)
       h_lynn = lynn_block.forward(h_in, weights_layer_i)
       max_diff = (h_ref - h_lynn).abs().max()
       rel_diff = max_diff / h_ref.abs().mean()
  4. Pass criterion: rel_diff < 5%

Pre-requisite: vLLM Qwen on port 18002 must be stopped to free ~60 GB unified mem
(70 GB BF16 model + activations + tokenizer ≈ 75 GB total).

Usage (inside a fresh container):
    pip install --user transformers==5.8.0 accelerate
    python3 engine/test_p1_3_real_hidden_states.py \
        --model /models/Qwen3.6-35B-A3B-BF16 \
        --prompt "The capital of France is"
"""
import argparse
import sys
import time

import torch


def load_hf_model(path: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HF Qwen3_5_MoE BF16 from {path} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    # low_cpu_mem_usage=True avoids transient peaks during shard assembly.
    # device_map="auto" with explicit caps spreads weights GPU↔CPU on Spark
    # unified memory (earlyoom kicks in at 10% available; pure GPU load
    # peaks past 100 GB transient and triggers it).
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={0: "60GiB", "cpu": "20GiB"},
        trust_remote_code=False,
    )
    model.eval()
    print(f"HF loaded in {time.time()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)", flush=True)
    return model, tok


def run_hf_with_hidden_states(model, tok, prompt: str, device: str = "cuda"):
    inputs = tok(prompt, return_tensors="pt").to(device)
    print(f"  prompt: {prompt!r}  tokens: {inputs.input_ids.shape}", flush=True)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    # hidden_states is a tuple of length (num_layers + 1):
    #   hidden_states[0]  = embeddings (input to layer 0)
    #   hidden_states[i]  = output of layer i-1 = input to layer i (for i >= 1)
    #   hidden_states[N]  = output of layer N-1 (final layer)
    return out.logits, out.hidden_states


def validate_full_attn_layer(layer_idx: int, h_in: torch.Tensor, h_ref: torch.Tensor,
                             model_dir: str, device: str):
    """Run our Lynn full_attention block on the same h_in and compare to h_ref."""
    from engine.loader import load_qwen36_layer
    from engine.qwen36_block import qwen36_lynn_forward

    print(f"\n  === Layer {layer_idx} (full_attention) ===", flush=True)
    weights, config = load_qwen36_layer(model_dir, layer_idx, device=device)
    print(f"  weights loaded: {len(weights)} tensors", flush=True)

    # Lynn forward expects shape [batch, seq, hidden]
    h_in_3d = h_in if h_in.dim() == 3 else h_in.unsqueeze(0)
    t0 = time.time()
    h_lynn = qwen36_lynn_forward(h_in_3d, weights, config)
    elapsed = time.time() - t0

    # Match shape
    h_ref_3d = h_ref if h_ref.dim() == 3 else h_ref.unsqueeze(0)
    diff = (h_lynn - h_ref_3d).float().abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_mag = h_ref_3d.float().abs().mean().item()
    rel_diff = max_diff / max(ref_mag, 1e-8) * 100

    status = "✅ PASS" if rel_diff < 5.0 else "❌ FAIL"
    print(f"  L{layer_idx:2}  max_diff={max_diff:.3e}  mean_diff={mean_diff:.3e}  "
          f"ref_mean_mag={ref_mag:.3f}  rel={rel_diff:.3f}%  {status}  ({elapsed*1000:.0f}ms)",
          flush=True)

    # Free
    del weights
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    return {
        "layer": layer_idx,
        "max_diff": max_diff,
        "rel_diff_pct": rel_diff,
        "passed": rel_diff < 5.0,
        "elapsed_ms": elapsed * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Qwen3.6-35B-A3B-BF16 model dir")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full-attn-layers", default="3,7,11,15,19,23,27,31,35,39",
                    help="comma-sep list of full_attention layer indices")
    ap.add_argument("--single-layer", type=int, default=None,
                    help="run only this single layer (debug)")
    args = ap.parse_args()

    sys.path.insert(0, "/work")  # so `from engine.* import` works inside container

    full_attn_layers = (
        [args.single_layer] if args.single_layer is not None
        else [int(s) for s in args.full_attn_layers.split(",")]
    )

    # Step 1: HF forward
    model, tok = load_hf_model(args.model, args.device)
    print("\nRunning HF reference forward with hidden_states ...", flush=True)
    t0 = time.time()
    ref_logits, hidden_states = run_hf_with_hidden_states(model, tok, args.prompt, args.device)
    print(f"HF forward done in {time.time()-t0:.1f}s, {len(hidden_states)} hidden states "
          f"(num_layers + 1)", flush=True)

    top_token_ref = ref_logits[0, -1].argmax().item()
    print(f"  HF top-1 next token: {top_token_ref} ({tok.decode([top_token_ref])!r})", flush=True)

    # Move hidden_states to CPU to free GPU mem; reload per-layer weights to GPU
    hidden_states_cpu = [h.cpu() for h in hidden_states]
    del model
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    print("  HF model unloaded to free GPU memory", flush=True)

    # Step 2: per-layer Lynn validation
    print(f"\nValidating Lynn block on {len(full_attn_layers)} full_attention layers ...",
          flush=True)
    results = []
    for layer_idx in full_attn_layers:
        h_in = hidden_states_cpu[layer_idx].to(args.device)
        h_ref = hidden_states_cpu[layer_idx + 1].to(args.device)
        try:
            result = validate_full_attn_layer(layer_idx, h_in, h_ref, args.model, args.device)
            results.append(result)
        except Exception as e:
            print(f"  L{layer_idx} ERROR: {type(e).__name__}: {e}", flush=True)
            results.append({"layer": layer_idx, "error": str(e)})

    # Step 3: summary
    print("\n" + "=" * 60)
    print("P1.3 SUMMARY — Lynn block on REAL hidden states")
    print("=" * 60)
    passed = [r for r in results if r.get("passed")]
    failed = [r for r in results if not r.get("passed")]
    print(f"Passed: {len(passed)}/{len(results)}")
    if failed:
        print("Failed:")
        for r in failed:
            print(f"  L{r['layer']}: rel={r.get('rel_diff_pct', '?'):.2f}%  err={r.get('error', '')}")
    if passed:
        avg_rel = sum(r["rel_diff_pct"] for r in passed) / len(passed)
        max_rel = max(r["rel_diff_pct"] for r in passed)
        print(f"Avg rel_diff: {avg_rel:.3f}%  (max: {max_rel:.3f}%)")

    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
