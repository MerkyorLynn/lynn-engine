"""
Lynn-27B-A3B pruning · Phase 1 Week 1 — Activation profiler.

Loads Lynn engine + Qwen 3.6 35B-A3B baseline, runs every prompt in
`calibration_set_v1.1.jsonl` through forward, captures per-layer per-token
top-K=8 expert routing decisions, and writes them to JSONL (one row per prompt).

Output schema (per-prompt JSON line):
{
  "id": "keep/coding/001",
  "cat": "coding",
  "lang": "en",
  "n_tokens": 47,
  "routing": {           # layer_idx → [num_tokens] of [top_K=8 expert_ids]
      "0": [[12, 47, 99, 145, 178, 199, 221, 250], ...],   # 47 tokens × 8 experts
      "1": [...],
      ...
      "39": [...]
  }
}

Memory profile per prompt: ~ 40 layers × ~50 tokens × 8 expert_ids × 2 bytes ≈ 32 KB.
1436 prompts × 32 KB ≈ 46 MB total output JSONL.

Run on DGX (needs Lynn engine + 67 GB BF16 model resident):
    docker run --rm --gpus all --ipc=host --user 1000:1000 \
        -v /home/merkyor/models:/models \
        -v /tmp/lynn-engine:/work -w /work \
        -e PYTHONPATH=/work \
        nvcr.io/nvidia/vllm:26.03.post1-py3 \
        bash -c "pip install -q --user transformers==5.8.0 && \
                 python3 pruning/profile_activations.py \
                     --calibration pruning/calibration/calibration_set_v1.1.jsonl \
                     --out /home/merkyor/pruning/activation_profile_35B.jsonl"

Estimated wall time on Spark: 1436 prompts × ~3 s/prompt ≈ 75 min.
With Phase 3.2/3.3 optimizations applied: ~25 min.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def profile_forward(model_dir: str, calibration_path: Path, out_path: Path,
                    device: str = "cuda", dtype_str: str = "bfloat16",
                    max_tokens_per_prompt: int = 256, limit: int = 0,
                    skip_done: bool = True):
    """Walk every prompt, forward through 40 layers, log per-layer expert_indices.

    skip_done: if out_path exists, skip prompt ids already logged (resume support).
    """
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from engine.loader import load_qwen36_layer
    from engine.full_forward import _full_attn_forward, _rms_norm, _moe_forward, load_outside_weights
    from engine.qwen36_linear_attn_block import lynn_linear_attn_forward
    from engine.inference_state import LAYER_TYPES

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]

    # Resume support — collect already-logged prompt ids
    done_ids = set()
    if skip_done and out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
        print(f"[resume] skipping {len(done_ids)} prompts already in {out_path}",
              flush=True)

    # Read calibration set
    prompts = []
    with open(calibration_path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("id") not in done_ids:
                    prompts.append(d)
    if limit:
        prompts = prompts[:limit]
    print(f"[load] {len(prompts)} prompts pending", flush=True)

    # Config
    cfg_full = json.loads((Path(model_dir) / "config.json").read_text())
    tc = cfg_full["text_config"]
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
    n_layers = tc["num_hidden_layers"]
    layer_types = LAYER_TYPES
    K = cfg["num_experts_per_tok"]

    # Tokenizer + outside
    print(f"[load] tokenizer + outside weights", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    outside = load_outside_weights(model_dir, device, dtype)

    # All 40 layers resident
    print(f"[load] {n_layers} layer weights ...", flush=True)
    weights_per_layer = []
    t_start = time.time()
    for i in range(n_layers):
        w, _ = load_qwen36_layer(model_dir, i, num_experts=cfg["num_experts"],
                                 device=device, dequant_dtype=dtype)
        weights_per_layer.append(w)
        if (i + 1) % 5 == 0 or i == n_layers - 1:
            print(f"[load]   L{i:2}  cum {time.time()-t_start:.1f}s", flush=True)
    print(f"[load] all weights resident in {time.time()-t_start:.1f}s\n", flush=True)

    # Open output JSONL in append mode
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "a", buffering=1)   # line-buffered

    def custom_layer_forward(h, position_ids, layer_type, w, cfg, layer_idx,
                             routing_log: dict):
        """Forward one DecoderLayer, but capture expert_indices in routing_log[layer_idx]."""
        residual = h
        h_norm = _rms_norm(h, w["input_layernorm.weight"])

        if layer_type == "linear_attention":
            attn_out = lynn_linear_attn_forward(h_norm, w)
        else:
            attn_out = _full_attn_forward(h_norm, position_ids, w, cfg)
        h = residual + attn_out

        # Post-attention norm + MoE — capture routing
        residual = h
        h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])

        # Inline MoE forward with capture
        B, T, D = h_norm.shape
        E = cfg["num_experts"]

        h_flat = h_norm.view(B * T, D)
        router_logits = F.linear(h_flat, w["mlp.gate.weight"])
        routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)

        # CAPTURE: expert_indices is [B*T, K] — record as list-of-lists
        routing_log[layer_idx] = expert_indices.tolist()

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

        if "mlp.shared_expert.gate_proj.weight" in w:
            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
            shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
            if "mlp.shared_expert_gate.weight" in w:
                shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
                shared_ffn = shared_ffn * shared_gate
            moe_out = moe_out + shared_ffn

        return residual + moe_out.view(B, T, D)

    t_total_start = time.time()
    for i, p in enumerate(prompts):
        ids = tok(p["text"], return_tensors="pt", truncation=True,
                  max_length=max_tokens_per_prompt).input_ids.to(device)
        T = ids.shape[1]

        h = F.embedding(ids, outside["model.language_model.embed_tokens.weight"])
        pos = __import__("torch").arange(T, device=device, dtype=__import__("torch").long).unsqueeze(0)

        routing_log: dict = {}

        t0 = time.time()
        with __import__("torch").no_grad():
            for layer_idx in range(n_layers):
                h = custom_layer_forward(
                    h, pos, layer_types[layer_idx],
                    weights_per_layer[layer_idx], cfg, layer_idx, routing_log,
                )
        if device.startswith("cuda"):
            __import__("torch").cuda.synchronize()
        elapsed = time.time() - t0

        # Write profile
        rec = {
            "id": p["id"],
            "cat": p["cat"],
            "lang": p.get("lang", "?"),
            "n_tokens": T,
            # Convert layer_idx to string keys (JSON friendly)
            "routing": {str(k): v for k, v in sorted(routing_log.items())},
            "elapsed_s": elapsed,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0 or i == len(prompts) - 1:
            avg = (time.time() - t_total_start) / (i + 1)
            eta = avg * (len(prompts) - i - 1)
            print(f"[{i+1}/{len(prompts)}] {p['id']:<35} T={T:3} {elapsed*1000:5.0f}ms  "
                  f"avg {avg*1000:.0f}ms  ETA {eta/60:.1f}min", flush=True)

    fout.close()
    total = time.time() - t_total_start
    print(f"\nDone. {len(prompts)} prompts profiled in {total/60:.1f} min "
          f"({total/len(prompts):.1f}s/prompt avg)", flush=True)
    print(f"Output: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--calibration", required=True,
                    help="path to calibration_set_v1.1.jsonl")
    ap.add_argument("--out", default="activation_profile_35B.jsonl",
                    help="output JSONL path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="truncate prompts to this many tokens (default 256)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only first N prompts (0 = all)")
    args = ap.parse_args()

    profile_forward(
        args.model, Path(args.calibration), Path(args.out),
        device=args.device, dtype_str=args.dtype,
        max_tokens_per_prompt=args.max_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
