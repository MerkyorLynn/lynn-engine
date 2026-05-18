"""
Lynn Engine · Phase 3.1 · LynnInferenceState

Per-request state holder for incremental decode:
  - full_attention layers (10 of 40): standard KV cache
  - linear_attention layers (30 of 40): recurrent state + conv state

Memory profile (max_T=32K, B=1, BF16/FP32 mixed):
  full_attn KV:   10 layers × 2 tensors (K,V) × [2, 32768, 256] × 2 B = 640 MB
  recurrent_state: 30 layers × [32, 128, 128] × 4 B (FP32) = 60 MB
  conv_state:     30 layers × [8192, 3] × 2 B = 1.5 MB
  TOTAL:          ~700 MB per request at max_T=32K

For shorter contexts, cache is pre-allocated max_T but only first seq_len positions used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch


# Qwen 3.6 35B-A3B fixed dims (default; overridden by from_config for other models)
HIDDEN = 2048
NUM_KV_HEADS = 2
HEAD_DIM = 256
LINEAR_NUM_V_HEADS = 32
LINEAR_HEAD_K_DIM = 128
LINEAR_HEAD_V_DIM = 128
LINEAR_CONV_KERNEL = 4
LINEAR_CONV_DIM = 8192   # 2*key_dim + value_dim = 2*2048 + 4096
NUM_LAYERS = 40

# Layer types per index (Qwen 3.6 fixed pattern)
LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 10
assert len(LAYER_TYPES) == NUM_LAYERS

FULL_ATTN_INDICES = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]
LINEAR_ATTN_INDICES = [i for i, t in enumerate(LAYER_TYPES) if t == "linear_attention"]


def _infer_layer_types(tc: dict) -> list[str]:
    """Infer layer_types from config, with fallback for dense (all-full_attention) models."""
    if "layer_types" in tc:
        return list(tc["layer_types"])
    n = tc.get("num_hidden_layers", 40)
    # Dense models (e.g. Qwen3.5-9B) have no linear_attention layers
    return ["full_attention"] * n


@dataclass
class LynnInferenceState:
    """Per-request KV cache + recurrent state for incremental decode."""

    batch: int = 1
    max_seq_len: int = 32768
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    seq_len: int = 0   # current populated length (incl. prompt)

    # Architecture dimensions (set by from_config or defaults)
    hidden_size: int = HIDDEN
    num_kv_heads: int = NUM_KV_HEADS
    head_dim: int = HEAD_DIM
    layer_types: list[str] = field(default_factory=lambda: list(LAYER_TYPES))

    # full_attention KV cache
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    # linear_attention recurrent state + conv state
    recurrent_state: dict[int, torch.Tensor] = field(default_factory=dict)
    conv_state: dict[int, torch.Tensor] = field(default_factory=dict)

    # Linear attention dims (only used if layer has linear_attention)
    linear_num_v_heads: int = LINEAR_NUM_V_HEADS
    linear_head_k_dim: int = LINEAR_HEAD_K_DIM
    linear_head_v_dim: int = LINEAR_HEAD_V_DIM
    linear_conv_kernel: int = LINEAR_CONV_KERNEL
    linear_conv_dim: int = LINEAR_CONV_DIM

    @classmethod
    def from_config(
        cls,
        tc: dict,
        *,
        batch: int = 1,
        max_seq_len: int = 32768,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "LynnInferenceState":
        """Create state from a text_config dict (config.json['text_config'])."""
        layer_types = _infer_layer_types(tc)
        return cls(
            batch=batch,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            hidden_size=tc.get("hidden_size", HIDDEN),
            num_kv_heads=tc.get("num_key_value_heads", NUM_KV_HEADS),
            head_dim=tc.get("head_dim", HEAD_DIM),
            layer_types=layer_types,
        )

    def __post_init__(self):
        if not self.kv_cache:
            self._allocate()

    def _allocate(self):
        """Pre-allocate all caches at max_seq_len. Called once on init."""
        B, T = self.batch, self.max_seq_len
        D = self.head_dim
        full_attn = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        linear_attn = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

        # full_attn KV: per-layer K and V
        for i in full_attn:
            K = torch.zeros(B, self.num_kv_heads, T, D, device=self.device, dtype=self.dtype)
            V = torch.zeros(B, self.num_kv_heads, T, D, device=self.device, dtype=self.dtype)
            self.kv_cache[i] = (K, V)

        # linear_attn recurrent state (FP32 for accumulator stability)
        for i in linear_attn:
            S = torch.zeros(B, self.linear_num_v_heads, self.linear_head_k_dim,
                            self.linear_head_v_dim,
                            device=self.device, dtype=torch.float32)
            self.recurrent_state[i] = S

        # linear_attn conv1d left context (kernel - 1 = 3 tokens)
        for i in linear_attn:
            C = torch.zeros(B, self.linear_conv_dim, self.linear_conv_kernel - 1,
                            device=self.device, dtype=self.dtype)
            self.conv_state[i] = C

    def reset(self):
        """Clear all cache contents in place. Re-use for new prompt without re-alloc."""
        for K, V in self.kv_cache.values():
            K.zero_()
            V.zero_()
        for S in self.recurrent_state.values():
            S.zero_()
        for C in self.conv_state.values():
            C.zero_()
        self.seq_len = 0

    def memory_bytes(self) -> int:
        """Total bytes occupied by all cache tensors."""
        total = 0
        for K, V in self.kv_cache.values():
            total += K.element_size() * K.numel() + V.element_size() * V.numel()
        for S in self.recurrent_state.values():
            total += S.element_size() * S.numel()
        for C in self.conv_state.values():
            total += C.element_size() * C.numel()
        return total

    def update_full_attn_kv(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
                            position_start: int):
        """Write k_new/v_new into KV cache at positions [position_start, position_start+T_new].

        k_new/v_new shape: [B, NUM_KV_HEADS, T_new, HEAD_DIM]
        """
        K, V = self.kv_cache[layer_idx]
        T_new = k_new.shape[2]
        K[:, :, position_start:position_start + T_new, :] = k_new
        V[:, :, position_start:position_start + T_new, :] = v_new

    def get_full_attn_kv(self, layer_idx: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return slice of K, V cache up to `length` (exclusive).

        Returns: (K[:, :, :length, :], V[:, :, :length, :])
        """
        K, V = self.kv_cache[layer_idx]
        return K[:, :, :length, :], V[:, :, :length, :]

    def update_linear_attn_state(self, layer_idx: int, new_recurrent: torch.Tensor,
                                  new_conv: torch.Tensor):
        """Replace recurrent state + conv state for one linear_attn layer."""
        self.recurrent_state[layer_idx] = new_recurrent
        self.conv_state[layer_idx] = new_conv


def summarize_memory(state: LynnInferenceState) -> str:
    """Pretty-print breakdown of cache memory usage."""
    kv_bytes = sum(K.element_size() * K.numel() + V.element_size() * V.numel()
                   for K, V in state.kv_cache.values())
    rec_bytes = sum(S.element_size() * S.numel() for S in state.recurrent_state.values())
    conv_bytes = sum(C.element_size() * C.numel() for C in state.conv_state.values())
    total = kv_bytes + rec_bytes + conv_bytes
    return (
        f"LynnInferenceState memory @ B={state.batch}, max_T={state.max_seq_len}, "
        f"dtype={state.dtype}\n"
        f"  KV cache (10 full_attn layers):       {kv_bytes/1e6:7.1f} MB\n"
        f"  Recurrent state (30 linear_attn):     {rec_bytes/1e6:7.1f} MB\n"
        f"  Conv state (30 linear_attn):          {conv_bytes/1e6:7.1f} MB\n"
        f"  TOTAL:                                {total/1e6:7.1f} MB"
    )


if __name__ == "__main__":
    # Smoke test on CPU (no GPU needed for state allocation correctness)
    s = LynnInferenceState(batch=1, max_seq_len=32768, device="cpu",
                           dtype=torch.bfloat16)
    print(summarize_memory(s))
    print(f"\nFull attn layer indices: {FULL_ATTN_INDICES}")
    print(f"Linear attn layer indices count: {len(LINEAR_ATTN_INDICES)}")
    print(f"\nReset works: ", end="")
    s.seq_len = 100
    s.reset()
    assert s.seq_len == 0
    print("✓")
