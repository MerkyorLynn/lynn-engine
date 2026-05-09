"""
Lynn Engine · multi-prompt logits validation.

Phase 1 (vLLM running): query vLLM with N test prompts, save top-10 logprobs
                        baseline to /tmp/lynn_expected.json
Phase 2 (vLLM stopped): load Lynn weights resident (all 40 layers + outside),
                        run forward on same prompts, compare top-K agreement
Phase 3:                restart vLLM (manual)

Usage:
    # Phase 1 — with vllm-qwen35a3b running
    python3 engine/multi_prompt_validation.py --phase capture

    # Stop vLLM, then run Phase 2
    python3 engine/multi_prompt_validation.py --phase verify
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# torch is only needed in Phase 2 — imported lazily so capture phase
# can run on the host without GPU dependencies.


# Diverse prompts: factual, math, code-y, multi-lingual hint, completion, narrative.
TEST_PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "The largest planet in our solar system is",
    "Albert Einstein was famous for developing",
    "Two plus two equals",
    "def fibonacci(n):",
    "Once upon a time in a small village,",
    "The chemical formula for water is",
]


VLLM_URL = "http://127.0.0.1:18002/v1/completions"
VLLM_MODEL = "Qwen3.6-35B-A3B-FP8"
# Default to /work (the mount visible in our container) so capture (host)
# and verify (container) phases share the file. Override with --baseline.
DEFAULT_BASELINE = (
    Path("/work/lynn_expected.json") if Path("/work").is_dir() and Path("/work/.").is_dir()
    else Path("/tmp/lynn-engine/lynn_expected.json")
)


def query_vllm(prompt: str, top_k: int = 10):
    body = json.dumps({
        "model": VLLM_MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "logprobs": top_k,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        VLLM_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    c = d["choices"][0]
    return {
        "top_text": c["text"],
        "top_logprobs": c["logprobs"]["top_logprobs"][0],
    }


def capture_baseline(prompts):
    print(f"Querying vLLM at {VLLM_URL} for {len(prompts)} prompts ...")
    expected = {}
    for i, p in enumerate(prompts):
        try:
            expected[p] = query_vllm(p)
            top_text = expected[p]["top_text"]
            print(f"  [{i+1}/{len(prompts)}] {p!r:55} -> {top_text!r}")
        except Exception as e:
            print(f"  [{i+1}/{len(prompts)}] {p!r}: ERROR {e}")
            return False
    out = (Path("/tmp/lynn-engine/lynn_expected.json")
           if Path("/tmp/lynn-engine").is_dir() else DEFAULT_BASELINE)
    out.write_text(json.dumps(expected, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(expected)} baselines → {out}")
    return True


def load_all_layers(model_dir: str, n_layers: int, num_experts: int,
                    device: str, dtype) -> list:
    """Load all 40 layers' weights into GPU memory at once.

    With vLLM stopped (frees 60 GB) on Spark we have ~110 GB available;
    67 GB resident BF16 weights for the model fit comfortably.
    """
    from engine.loader import load_qwen36_layer

    print(f"Loading all {n_layers} layers (resident on {device}) ...")
    weights_per_layer = []
    t_start = time.time()
    for i in range(n_layers):
        t0 = time.time()
        w, _ = load_qwen36_layer(model_dir, i, num_experts=num_experts,
                                 device=device, dequant_dtype=dtype)
        weights_per_layer.append(w)
        print(f"  L{i:2} loaded in {time.time()-t0:.1f}s "
              f"(running total: {(time.time()-t_start):.1f}s)", flush=True)
    print(f"\nAll {n_layers} layers resident ({time.time()-t_start:.1f}s total).")
    return weights_per_layer


def run_lynn_forward(prompts, model_dir, device="cuda", dtype=None):
    """Phase 2: load Lynn weights resident, forward all prompts, return top-K."""
    import torch
    import torch.nn.functional as F
    if dtype is None:
        dtype = torch.bfloat16
    sys.path.insert(0, "/work")
    from engine.full_forward import _layer_forward, _rms_norm, load_outside_weights

    # Config
    with open(Path(model_dir) / "config.json") as f:
        full_cfg = json.load(f)
    tc = full_cfg["text_config"]
    rope_p = tc.get("rope_parameters", {})
    cfg = {
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "num_experts": tc["num_experts"],
        "num_experts_per_tok": tc["num_experts_per_tok"],
        "rope_theta": rope_p.get("rope_theta", tc.get("rope_theta", 1e6)),
        "partial_rotary_factor": rope_p.get("partial_rotary_factor", 1.0),
    }
    layer_types = tc["layer_types"]
    n_layers = tc["num_hidden_layers"]

    # Tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)

    # Outside weights (embed + lm_head + final norm)
    print(f"Loading outside weights ...")
    outside = load_outside_weights(model_dir, device, dtype)

    # All 40 layers resident
    layer_weights = load_all_layers(model_dir, n_layers, cfg["num_experts"], device, dtype)

    # Forward each prompt
    results = {}
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        B, T = ids.shape
        pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)

        t0 = time.time()
        h = F.embedding(ids, outside["model.language_model.embed_tokens.weight"])
        for i in range(n_layers):
            h = _layer_forward(h, pos, layer_types[i], layer_weights[i], cfg)
        h = _rms_norm(h, outside["model.language_model.norm.weight"])
        last_h = h[:, -1, :]
        logits = F.linear(last_h, outside["lm_head.weight"])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        topv, topi = torch.topk(logits[0], 10)
        results[p] = {
            "top_text": tok.decode([topi[0].item()]),
            "top_10": [
                {"text": tok.decode([i.item()]), "logit": v.item(), "id": i.item()}
                for i, v in zip(topi, topv)
            ],
            "elapsed_s": elapsed,
            "T": T,
        }
        print(f"  {p!r:55} -> {results[p]['top_text']!r}  ({elapsed:.2f}s, T={T})",
              flush=True)
    return results


def compare_results(expected: dict, actual: dict):
    """Compare Lynn top-1 + top-K text overlap to vLLM baseline."""
    print("\n" + "=" * 60)
    print(f"{'Prompt':<55} {'Lynn':<12} {'vLLM':<12} {'Match':>6} {'TopK∩':>5}")
    print("=" * 60)
    n_top1 = n_total = 0
    overlap_sum = 0
    for p in actual:
        if p not in expected:
            print(f"  ⚠️  {p!r} no vLLM baseline, skipping")
            continue
        n_total += 1
        lynn_top = actual[p]["top_text"]
        vllm_top = expected[p]["top_text"]
        ok = lynn_top == vllm_top
        if ok:
            n_top1 += 1
        # K∩: how many of Lynn's top-10 appear in vLLM's top-10
        lynn_set = {item["text"] for item in actual[p]["top_10"]}
        vllm_set = set(expected[p]["top_logprobs"].keys())
        k_overlap = len(lynn_set & vllm_set)
        overlap_sum += k_overlap
        print(f"  {p!r:<55} {lynn_top!r:<12} {vllm_top!r:<12} "
              f"{('✅' if ok else '❌'):>6} {k_overlap}/10")
    print("=" * 60)
    print(f"Top-1 match:  {n_top1}/{n_total}  ({100.0*n_top1/n_total:.1f}%)"
          if n_total else "no comparable prompts")
    if n_total:
        print(f"Top-K avg ∩:  {overlap_sum/n_total:.1f}/10")
    return n_top1 == n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["capture", "verify"], required=True,
                    help="capture: query vLLM (it must be running). "
                         "verify: run Lynn forward (vLLM may be stopped).")
    ap.add_argument("--model", default="/models/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompts", default=None,
                    help="optional path to JSON list of prompts (override defaults)")
    ap.add_argument("--baseline", default=None,
                    help="path to baseline JSON file (default: see DEFAULT_BASELINE)")
    args = ap.parse_args()

    if args.prompts:
        prompts = json.loads(Path(args.prompts).read_text())
    else:
        prompts = TEST_PROMPTS

    baseline_path = Path(args.baseline) if args.baseline else DEFAULT_BASELINE
    if args.phase == "capture":
        ok = capture_baseline(prompts)
        sys.exit(0 if ok else 1)
    else:
        if not baseline_path.exists():
            print(f"ERROR: no baseline at {baseline_path}. Run --phase capture first.")
            sys.exit(2)
        expected = json.loads(baseline_path.read_text())
        actual = run_lynn_forward(prompts, args.model, args.device)
        all_match = compare_results(expected, actual)
        sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
