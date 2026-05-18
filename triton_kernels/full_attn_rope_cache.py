"""Owned RoPE cosine/sine table cache for full-attention decode.

Stream B Task 1 of ``docs/QWEN36_W4A16_KERNEL_REFACTOR_PLAN_20260518.md``.

The previous implementation lived inline in ``engine/incremental_decode.py``
as a module-level ``dict`` keyed by
``(device, dtype, rotary_dim, theta, max_seq)`` plus a free function
``_build_rope_cos_sin_cached``. It worked, but:

* the table was built lazily on first ``lookup`` call, so the first decode
  step in a serving session paid a one-shot allocation latency;
* the cache lived as a private module-level dict — there was no single place
  to inspect, reset, or bound it;
* every refactor that added a new caller risked silently forking the table
  contract.

This module replaces that with an explicit class:

* ``FullAttnRoPECache.prewarm`` lets serving startup pre-allocate the table
  so the first token is not slower than the rest;
* ``FullAttnRoPECache.lookup`` is the single hot path used by prefill and
  decode (mirrors the old function signature);
* ``FullAttnRoPECache.reset`` clears the table (used by tests and parity
  probes);
* ``FullAttnRoPECache.info`` reports inventory so the serving banner can log
  resident table sizes.

A module-level singleton ``_GLOBAL_CACHE`` plus a free function
``build_rope_cos_sin_cached`` preserve the previous import surface so
``engine/incremental_decode.py`` migrates with a single import swap.

Behaviour is byte-equivalent to the previous inline implementation; this
refactor does not flip the ``LYNN_FULL_ATTN_ROPE_CACHE`` default — that
flip waits on the P123 / promotion-gate parity evidence the refactor plan
requires before any default change ships.
"""
from __future__ import annotations

import os
from typing import Any

import torch


_DEFAULT_MAX_SEQ_ENV = "LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ"
_DEFAULT_MAX_SEQ_FALLBACK = 65536

CacheKey = tuple[str, str, int, float, int]


class FullAttnRoPECache:
    """Pre-built (cos, sin) table for full-attention RoPE on decode.

    The cache is keyed by ``(device, dtype, rotary_dim, theta, max_seq)``.
    Tables are torch tensors resident on the same device as the model
    weights; lookup does a single ``index_select`` on the cached cos/sin
    tables.

    The class is intentionally torch-only (no Triton kernel here) so the
    contract stays trivial to validate. Faster kernels can later replace
    ``lookup`` while preserving the public surface.
    """

    __slots__ = ("_tables",)

    def __init__(self) -> None:
        self._tables: dict[CacheKey, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _key(device: Any, dtype: Any, rotary_dim: int, theta: float, max_seq: int) -> CacheKey:
        return (str(torch.device(device)), str(dtype), int(rotary_dim), float(theta), int(max_seq))

    def _build(
        self,
        device: Any,
        dtype: torch.dtype,
        rotary_dim: int,
        theta: float,
        max_seq: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                / rotary_dim
            )
        )
        seq = torch.arange(max_seq, device=device, dtype=torch.float32)
        freqs = seq[:, None] * inv_freq[None, :]
        return (
            freqs.cos().to(dtype).contiguous(),
            freqs.sin().to(dtype).contiguous(),
        )

    def prewarm(
        self,
        device: Any,
        dtype: torch.dtype,
        rotary_dim: int,
        theta: float,
        max_seq: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ensure the cos/sin table is allocated for the given config.

        Returns the (cos, sin) tables. Idempotent; subsequent calls with
        the same config hit the cache.
        """
        if max_seq is None:
            max_seq = int(os.environ.get(_DEFAULT_MAX_SEQ_ENV, str(_DEFAULT_MAX_SEQ_FALLBACK)))
        key = self._key(device, dtype, rotary_dim, theta, max_seq)
        cached = self._tables.get(key)
        if cached is None:
            cached = self._build(device, dtype, rotary_dim, theta, max_seq)
            self._tables[key] = cached
        return cached

    def lookup(
        self,
        positions: torch.Tensor,
        rotary_dim: int,
        theta: float,
        device: Any,
        dtype: torch.dtype,
        max_seq: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) for the requested positions.

        ``positions`` is a long tensor of any shape ``[*, M]``. The output
        is shaped ``[*, 1, M, rotary_dim/2]`` matching the previous module
        function so the caller does not change.
        """
        cos_table, sin_table = self.prewarm(device, dtype, rotary_dim, theta, max_seq)
        half = rotary_dim // 2
        flat = positions.reshape(-1).to(device=device, dtype=torch.long)
        cos = cos_table.index_select(0, flat).reshape(*positions.shape, half).unsqueeze(1)
        sin = sin_table.index_select(0, flat).reshape(*positions.shape, half).unsqueeze(1)
        return cos, sin

    def reset(self) -> None:
        """Drop all cached tables. Mostly for tests / parity probes."""
        self._tables.clear()

    def info(self) -> list[dict[str, Any]]:
        """Return resident-table inventory for serving banner + diagnostics."""
        out: list[dict[str, Any]] = []
        for key, (cos, sin) in self._tables.items():
            out.append(
                {
                    "device": key[0],
                    "dtype": key[1],
                    "rotary_dim": key[2],
                    "theta": key[3],
                    "max_seq": key[4],
                    "cos_shape": list(cos.shape),
                    "sin_shape": list(sin.shape),
                    "bytes": cos.element_size() * cos.numel() + sin.element_size() * sin.numel(),
                }
            )
        return out


# Module-level singleton. Mirrors the previous module-level dict so the
# import surface in engine/incremental_decode.py changes minimally.
_GLOBAL_CACHE = FullAttnRoPECache()


def build_rope_cos_sin_cached(
    positions: torch.Tensor,
    rotary_dim: int,
    theta: float,
    device: Any,
    dtype: torch.dtype,
    max_seq: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Free-function shim. Drop-in replacement for the previous
    ``_build_rope_cos_sin_cached`` in ``engine/incremental_decode.py``."""
    return _GLOBAL_CACHE.lookup(positions, rotary_dim, theta, device, dtype, max_seq)


def get_global_cache() -> FullAttnRoPECache:
    """Hook for serving start / dev tooling that wants prewarm + inspect."""
    return _GLOBAL_CACHE


__all__ = [
    "FullAttnRoPECache",
    "build_rope_cos_sin_cached",
    "get_global_cache",
]
