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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    fused_experts = (
        "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w
    )
    for e in range(E):
        mask = (expert_indices == e)
        if not mask.any():
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]
        if fused_experts:
            gate_up = F.linear(x_e, w["mlp.experts.gate_up_proj"][e])
            gate_e, up_e = gate_up.chunk(2, dim=-1)
            ffn_e = F.linear(F.silu(gate_e) * up_e, w["mlp.experts.down_proj"][e])
        else:
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
    """Load embeddings + lm_head + final norm.

    Lynn's early internal checkpoints store these tensors in `outside.safetensors`.
    Public HF-style BF16 and NVFP4 artifacts keep them in regular model shards
    (`model.safetensors.index.json`) or a single `model.safetensors`. Support
    both layouts so the same forward code can run on released checkpoints.
    """
    from safetensors import safe_open

    model_path = Path(model_dir)
    keys = [
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.language_model.norm.weight",
    ]

    outside_path = model_path / "outside.safetensors"
    if outside_path.exists():
        weight_map = {k: outside_path.name for k in keys}
    else:
        index_path = model_path / "model.safetensors.index.json"
        single_path = model_path / "model.safetensors"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            weight_map = index["weight_map"]
        elif single_path.exists():
            weight_map = {k: single_path.name for k in keys}
        else:
            raise FileNotFoundError(
                f"No outside.safetensors, model.safetensors.index.json, or "
                f"model.safetensors found under {model_path}"
            )

    file_to_keys: dict[str, list[str]] = {}
    for k in keys:
        if k not in weight_map:
            raise KeyError(f"Outside tensor {k!r} not found in {model_path}")
        file_to_keys.setdefault(weight_map[k], []).append(k)

    out = {}
    for file_name, file_keys in file_to_keys.items():
        with safe_open(model_path / file_name, framework="pt", device=device) as f:
            for k in file_keys:
                out[k] = f.get_tensor(k).to(dtype)
    return out


# ----------------- incremental greedy decode (Phase 3.1) -----------------

def _prefill_layer(h, position_ids, layer_type, w, cfg, state, layer_idx):
    """Forward one DecoderLayer in prefill mode + populate cache."""
    from engine.incremental_decode import prefill_full_attn, prefill_linear_attn

    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])
    if layer_type == "linear_attention":
        attn_out, last_state, last_conv = prefill_linear_attn(h_norm, w)
        state.update_linear_attn_state(layer_idx, last_state, last_conv)
    else:  # full_attention
        attn_out, K, V = prefill_full_attn(h_norm, position_ids, w, cfg)
        state.update_full_attn_kv(layer_idx, K, V, position_start=0)
    h = residual + attn_out

    residual = h
    h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])
    moe_out = _moe_forward(h_norm, w, cfg)
    return residual + moe_out


def _decode_layer(h_new, position_id, layer_type, w, cfg, state, layer_idx):
    """Forward one DecoderLayer in decode mode (T=1) using cached state.

    LYNN_MOE_IMPL env var selects MoE implementation:
      optimized    Phase 3.2.1 active-experts loop (default)
      bmm          Phase 3.2.2 batched matmul
      indexed_bmm  Phase 3.2.2.5 pre-stacked grouped indexed_bmm
    """
    import os
    from engine.incremental_decode import decode_full_attn, decode_linear_attn

    residual = h_new
    h_norm = _rms_norm(h_new, w["input_layernorm.weight"])
    if layer_type == "linear_attention":
        attn_out, new_state, new_conv = decode_linear_attn(
            h_norm, w,
            state.recurrent_state[layer_idx],
            state.conv_state[layer_idx],
        )
        state.update_linear_attn_state(layer_idx, new_state, new_conv)
    else:
        K, V = state.kv_cache[layer_idx]
        attn_out = decode_full_attn(
            h_norm, position_id, w, cfg, K, V,
            cached_seq_len=state.seq_len,
        )
    h = residual + attn_out

    residual = h
    h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])
    impl = os.environ.get("LYNN_MOE_IMPL", "optimized")
    if impl == "optimized":
        from engine.moe_optimized import moe_forward_decode_optimized as _moe
    elif impl == "bmm":
        from engine.moe_optimized import moe_forward_decode_bmm as _moe
    elif impl == "indexed_bmm":
        from triton_kernels.moe_expert_ffn import moe_forward_decode_indexed_bmm as _moe
    else:
        raise ValueError(f"Unknown LYNN_MOE_IMPL: {impl}")
    moe_out = _moe(h_norm, w, cfg)
    return residual + moe_out


def generate_incremental(model_dir, prompt, max_new=5, device="cuda",
                         dtype=torch.bfloat16, verbose=True, max_seq_len=2048):
    """Phase 3.1 incremental decode: prefill once, then 1-token-per-step decode.

    Compared to generate_greedy (brute-force), this should be ~10x faster
    on Spark for short generations and scale O(1) per token (vs O(T) brute).
    """
    from engine.loader import load_qwen36_layer
    from engine.inference_state import LynnInferenceState, LAYER_TYPES

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
    n_layers = tc["num_hidden_layers"]
    assert LAYER_TYPES == tc["layer_types"], "layer_types config mismatch"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    T = ids.shape[1]

    if verbose:
        print(f"prompt: {prompt!r}, T={T}, max_new={max_new}", flush=True)

    # Outside weights (resident)
    if verbose:
        print("Loading outside weights ...", flush=True)
    outside = load_outside_weights(model_dir, device, dtype)

    # All 40 layers resident
    if verbose:
        print(f"Loading all {n_layers} layers (resident) ...", flush=True)
    layer_weights = []
    t_start = time.time()
    for i in range(n_layers):
        w, _ = load_qwen36_layer(model_dir, i, num_experts=cfg["num_experts"],
                                 device=device, dequant_dtype=dtype)
        layer_weights.append(w)
        if verbose and (i % 5 == 4 or i == n_layers - 1):
            print(f"  L{i:2}: cumulative {time.time()-t_start:.1f}s", flush=True)
    if verbose:
        print(f"All weights resident in {time.time()-t_start:.1f}s\n", flush=True)

    # Phase 3.2 implementation selector. Indexed decode needs original expert
    # weights for prefill, so stacking happens after prefill completes.
    import os as _os
    _impl = _os.environ.get("LYNN_MOE_IMPL", "optimized")

    # Allocate inference state
    state = LynnInferenceState(batch=1, max_seq_len=max_seq_len, device=device, dtype=dtype)

    # === PREFILL ===
    if verbose:
        print(f"Prefill T={T} ...", flush=True)
    t0 = time.time()
    h = F.embedding(ids, outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)
    for i in range(n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], layer_weights[i], cfg, state, i)
    state.seq_len = T

    # First token from prefill last position
    h_final = _rms_norm(h, outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], outside["lm_head.weight"])
    next_id = int(logits[0].argmax().item())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    if verbose:
        print(f"  prefill done in {time.time()-t0:.2f}s, "
              f"next={next_id} {tok.decode([next_id])!r}", flush=True)
    new_ids = [next_id]

    # Phase 3.2 indexed decode preparation. Prefill above uses the baseline MoE
    # path, so only now can we replace per-expert weights with stacked tensors.
    if _impl in ("indexed_bmm", "triton"):
        from triton_kernels.moe_expert_ffn import stack_expert_weights
        if verbose:
            print(f"Pre-stacking expert weights for {_impl} after prefill ...", flush=True)
        _t0 = time.time()
        for _i in range(n_layers):
            stack_expert_weights(layer_weights[_i], num_experts=cfg["num_experts"])
            for _e in range(cfg["num_experts"]):
                for _proj in ("gate_proj", "up_proj", "down_proj"):
                    _key = f"mlp.experts.{_e}.{_proj}.weight"
                    layer_weights[_i].pop(_key, None)
            if verbose and (_i % 10 == 9 or _i == n_layers - 1):
                torch.cuda.empty_cache()
                _free = torch.cuda.mem_get_info()[0] / 1e9
                print(f"  pre-stack L{_i:2}: {time.time()-_t0:.1f}s GPU free {_free:.1f} GB", flush=True)
        torch.cuda.empty_cache()
        if verbose:
            _free = torch.cuda.mem_get_info()[0] / 1e9
            print(f"Pre-stack done {time.time()-_t0:.1f}s, GPU free {_free:.1f} GB\n", flush=True)

    # === DECODE ===
    for step in range(1, max_new):
        t0 = time.time()
        new_token_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
        h = F.embedding(new_token_tensor, outside["model.language_model.embed_tokens.weight"])
        pos = state.seq_len   # new token position

        for i in range(n_layers):
            h = _decode_layer(h, pos, LAYER_TYPES[i], layer_weights[i], cfg, state, i)

        state.seq_len += 1
        h_final = _rms_norm(h, outside["model.language_model.norm.weight"])
        logits = F.linear(h_final[:, -1, :], outside["lm_head.weight"])
        next_id = int(logits[0].argmax().item())
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if verbose:
            print(f"  decode step {step+1}/{max_new}  pos={state.seq_len}  "
                  f"{(time.time()-t0)*1000:.0f}ms  next={next_id} "
                  f"{tok.decode([next_id])!r}", flush=True)
        new_ids.append(next_id)

    full_ids = ids[0].tolist() + new_ids
    full_text = tok.decode(full_ids)
    return full_text, new_ids


# ----------------- multi-token greedy decode (brute-force, Phase 2) ---

def generate_greedy(model_dir: str, prompt: str, max_new: int = 5,
                    device: str = "cuda", dtype=torch.bfloat16, verbose: bool = True):
    """Brute-force greedy generation: re-prefill at every step.

    No KV cache — slower than a proper engine, but unambiguously correct.
    Loads all 40 layers resident once, then runs N successive full forwards.
    Per-step cost grows ~quadratic in (prompt_len + new_tokens) since the
    full forward sees every position each time.

    Returns:
        text: prompt + generated continuation
        new_token_ids: list of generated token ids
    """
    from engine.loader import load_qwen36_layer

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

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    if verbose:
        print(f"prompt: {prompt!r}  ids[:10]={ids[0].tolist()[:10]}", flush=True)

    # Outside + all 40 layers resident
    if verbose:
        print("Loading outside weights ...", flush=True)
    outside = load_outside_weights(model_dir, device, dtype)
    embed = outside["model.language_model.embed_tokens.weight"]
    lm_head_w = outside["lm_head.weight"]
    final_norm_w = outside["model.language_model.norm.weight"]

    if verbose:
        print(f"Loading all {n_layers} layers (resident) ...", flush=True)
    weights_per_layer = []
    t_start = time.time()
    for i in range(n_layers):
        w, _ = load_qwen36_layer(model_dir, i, num_experts=cfg["num_experts"],
                                 device=device, dequant_dtype=dtype)
        weights_per_layer.append(w)
        if verbose and (i % 5 == 4 or i == n_layers - 1):
            print(f"  L{i:2}: cumulative {time.time()-t_start:.1f}s", flush=True)
    if verbose:
        print(f"All weights resident in {time.time()-t_start:.1f}s\n", flush=True)

    new_ids = []
    for step in range(max_new):
        T = ids.shape[1]
        pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)

        t0 = time.time()
        h = F.embedding(ids, embed)
        for i in range(n_layers):
            h = _layer_forward(h, pos, layer_types[i], weights_per_layer[i], cfg)
        h = _rms_norm(h, final_norm_w)
        last_h = h[:, -1, :]
        logits = F.linear(last_h, lm_head_w)
        next_id = int(logits[0].argmax().item())
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        new_ids.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if verbose:
            text = tok.decode([next_id]).replace("\n", "\\n")
            print(f"  step {step+1}/{max_new}  T={T}->{T+1}  "
                  f"{elapsed:.2f}s  next={next_id} {text!r}", flush=True)

    full_text = tok.decode(ids[0].tolist())
    return full_text, new_ids


# ----------------- end-to-end forward (single token, original) ---------

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
    ap.add_argument("--max-new", type=int, default=1,
                    help="number of tokens to greedy-decode (1 = next-token only)")
    ap.add_argument("--mode", choices=["prefill", "brute", "incremental"],
                    default="incremental",
                    help="prefill = single-token next-logit only; "
                         "brute = brute-force re-prefill per token; "
                         "incremental = KV cache + recurrent state cache (Phase 3.1)")
    args = ap.parse_args()

    sys.path.insert(0, "/work")

    if args.mode == "prefill" or args.max_new <= 1:
        out = run_forward(args.model, args.prompt, device=args.device)
        print(f"\n=== Lynn Engine top-1 next token: "
              f"{out['top_token_id']} ({out['top_token_text']!r}) ===")
    elif args.mode == "brute":
        text, new_ids = generate_greedy(args.model, args.prompt,
                                        max_new=args.max_new, device=args.device)
        print(f"\n=== Lynn Engine brute-force ({args.max_new} new tokens) ===")
        print(text)
        print(f"\nnew_ids: {new_ids}")
    else:  # incremental
        text, new_ids = generate_incremental(args.model, args.prompt,
                                             max_new=args.max_new, device=args.device)
        print(f"\n=== Lynn Engine incremental ({args.max_new} new tokens) ===")
        print(text)
        print(f"\nnew_ids: {new_ids}")


if __name__ == "__main__":
    main()
