"""
Lynn Engine · P1.2 + P1.3 — Full 40-layer forward via HF transformers,
                              capture full_attention layer I/O, replay with Lynn block.

Strategy:
  1. Load HF Qwen 3.6 35B-A3B-FP8 (auto-dequant FP8→BF16)
  2. Register forward hooks on the 10 full_attention layers
     (indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
  3. Run forward on test prompt → capture (input_hidden, output_hidden) per layer
  4. For each captured pair, replay our Lynn engine qwen36_lynn block with
     same input → verify output matches
  5. This validates our block on REAL forward activations (not synthetic random),
     including linear_attention state from preceding layers

Memory: HF model in BF16 = ~70 GB peak. Plus activations + captures = ~75 GB.
Spark has ~88 GB free with Qwen vLLM stopped. Within 0.85 ceiling.
"""
import sys, json, time, gc
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.qwen36_block import qwen36_lynn


def main():
    model_dir = "/models/Qwen3.6-35B-A3B-FP8"

    # Read config
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
    full_attn_layers = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

    print(f"⚙️  P1.2/P1.3 — full forward + Lynn block validation")
    print(f"   model:  {model_dir}")
    print(f"   config: {cfg}", flush=True)

    # ── Load Triton kernels first (they JIT compile lazily) ──
    print(f"\n📦 Loading Triton kernels...", flush=True)
    from triton_kernels.attention import make_triton_attention
    from triton_kernels.rope import make_triton_rope
    from triton_kernels.rmsnorm import make_triton_rmsnorm
    from triton_kernels.moe import make_triton_router
    rmsnorm_fn = make_triton_rmsnorm()
    rope_fn = make_triton_rope()
    attn_fn = make_triton_attention()
    router_fn = make_triton_router()
    print(f"   kernels ready ✅", flush=True)

    # ── Load HF model ──
    print(f"\n📥 Loading HF Qwen 3.6 35B-A3B-FP8 (will dequant to BF16, ~70 GB)...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",  # disable flash for hooks to capture cleanly
    )
    model.eval()
    print(f"   loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"   model dtype: {next(model.parameters()).dtype}", flush=True)
    print(f"   num params: {sum(p.numel() for p in model.parameters())/1e9:.2f} B", flush=True)
    torch.cuda.empty_cache()
    print(f"   GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # ── Find actual layer module path ──
    text_model = None
    for name in ["model.language_model", "language_model", "model"]:
        try:
            obj = model
            for part in name.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "layers"):
                text_model = obj
                print(f"\n   text_model path: {name}", flush=True)
                break
        except AttributeError:
            continue
    if text_model is None:
        print(f"❌ Could not find layers attribute on model")
        return
    layers = text_model.layers
    print(f"   total layers: {len(layers)}", flush=True)

    # ── Register hooks on full_attention layers ──
    captures = {}  # layer_idx → {input_hidden, output_hidden}

    def make_hook(layer_idx):
        def pre_hook(module, args, kwargs):
            # First arg is hidden_states for HF DecoderLayer convention
            if args:
                captures.setdefault(layer_idx, {})["input_hidden"] = args[0].detach().clone()
            elif "hidden_states" in kwargs:
                captures.setdefault(layer_idx, {})["input_hidden"] = kwargs["hidden_states"].detach().clone()

        def post_hook(module, args, output):
            # output is typically (hidden_states,) tuple or just hidden_states
            if isinstance(output, tuple):
                captures.setdefault(layer_idx, {})["output_hidden"] = output[0].detach().clone()
            else:
                captures.setdefault(layer_idx, {})["output_hidden"] = output.detach().clone()
        return pre_hook, post_hook

    handles = []
    for li in full_attn_layers:
        pre, post = make_hook(li)
        h1 = layers[li].register_forward_pre_hook(pre, with_kwargs=True)
        h2 = layers[li].register_forward_hook(post)
        handles.extend([h1, h2])
    print(f"   registered hooks on layers {full_attn_layers}", flush=True)

    # ── Test prompt ──
    prompt = "用一句话解释 Transformer 的 attention 机制。"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    print(f"\n🧪 Running forward on prompt: {prompt!r}", flush=True)
    print(f"   input_ids shape: {inputs.input_ids.shape}", flush=True)

    t0 = time.time()
    with torch.inference_mode():
        out = model(**inputs)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) * 1000
    print(f"   forward: {fwd_ms:.0f} ms", flush=True)
    print(f"   logits shape: {out.logits.shape}", flush=True)

    # Top-1 token
    next_token = out.logits[0, -1].argmax().item()
    next_text = tokenizer.decode([next_token])
    print(f"   top-1 next token: {next_token} ({next_text!r})", flush=True)

    # Remove hooks
    for h in handles:
        h.remove()
    print(f"\n   captured {len(captures)} layer I/O pairs", flush=True)

    # Save reference logits for P1.3 vLLM comparison later
    ref_logits = out.logits[0, -1].float().cpu()
    ref_top10 = torch.topk(ref_logits, 10)
    print(f"\n   HF top-10 next-token logits:")
    for tok, lp in zip(ref_top10.indices.tolist(), ref_top10.values.tolist()):
        print(f"     token {tok:>6} ({tokenizer.decode([tok])!r:<10}): {lp:.4f}")

    # Free model BEFORE running Lynn blocks (we have captures, we don't need HF model anymore)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n   freed HF model. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # ── Replay each full_attention layer with Lynn block ──
    # We need the LAYER's weights, so re-load via our loader
    from engine.loader import load_qwen36_layer

    print(f"\n🔬 Replaying full_attention layers with Lynn block...", flush=True)
    print(f"\n{'Layer':<6} {'M':<5} {'capture max diff':<18} {'lynn max diff':<16} {'rel diff':<14} {'status'}", flush=True)
    print("-" * 95, flush=True)

    M = inputs.input_ids.shape[1]
    position_ids = torch.arange(M, device="cuda", dtype=torch.long).unsqueeze(0).expand(1, M)
    PASS = 5e-2
    rows = []
    all_pass = True

    for li in full_attn_layers:
        if li not in captures:
            print(f"L{li:<3}   skipped (no capture)", flush=True)
            continue

        cap = captures[li]
        h_in = cap["input_hidden"]      # [B, M, D]
        h_out_ref = cap["output_hidden"] # [B, M, D]

        # Load this layer's weights
        weights, _ = load_qwen36_layer(model_dir, li, num_experts=cfg["num_experts"], device="cuda")

        # Run Lynn block
        with torch.inference_mode():
            h_out_lynn = qwen36_lynn(h_in, position_ids, weights, cfg,
                                      rmsnorm_fn, rope_fn, attn_fn, router_fn)

        # Compare lynn output to captured HF output
        max_diff = (h_out_ref.float() - h_out_lynn.float()).abs().max().item()
        mean_diff = (h_out_ref.float() - h_out_lynn.float()).abs().mean().item()
        ref_norm = h_out_ref.float().abs().mean().item()
        rel_diff = mean_diff / ref_norm if ref_norm > 0 else 0
        status = "✅" if max_diff < PASS else "❌"
        if max_diff >= PASS:
            all_pass = False

        print(f"L{li:<3}   M={M:<3} _                 {max_diff:<16.4e} {rel_diff:<14.4e} {status}", flush=True)
        rows.append({"layer": li, "max_diff": max_diff, "mean_diff": mean_diff, "rel_diff": rel_diff})

        del weights, h_out_lynn
        gc.collect()
        torch.cuda.empty_cache()

    print("-" * 95, flush=True)
    if all_pass:
        avg_max = sum(r["max_diff"] for r in rows) / len(rows)
        avg_rel = sum(r["rel_diff"] for r in rows) / len(rows)
        print(f"\n✅ P1.2 PASSED — All {len(rows)} full_attention layers match HF reference under REAL forward activations")
        print(f"   Avg max diff: {avg_max:.4e}")
        print(f"   Avg rel diff: {avg_rel:.4e}")
        print(f"\nThis is a stronger result than P1.1 (synthetic random input) because:")
        print(f"  - input_hidden comes from REAL Qwen 3.6 forward including linear_attention layers")
        print(f"  - reference output comes from HF transformers' own attention implementation")
        print(f"  - 4 Triton kernels reproduce HF's full_attention block to {avg_rel:.3%} relative error")
    else:
        fails = [r["layer"] for r in rows if r["max_diff"] >= PASS]
        print(f"\n❌ {len(fails)} layers FAIL: {fails}")

    # Save
    out_path = Path(__file__).parent.parent / "benchmarks/results/p1_2_full_forward.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "prompt": prompt,
        "M": M,
        "next_token": next_token,
        "next_text": next_text,
        "ref_top10": [{"token": t, "text": tokenizer.decode([t]), "logit": l} for t, l in zip(ref_top10.indices.tolist(), ref_top10.values.tolist())],
        "layer_alignment": rows,
        "all_pass": all_pass,
        "threshold": PASS,
    }, ensure_ascii=False, indent=2))
    print(f"\n💾 saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
