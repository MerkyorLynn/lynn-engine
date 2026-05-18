"""Owned causal-mask cache scaffold for future explicit-mask attention.

Stream B Task (roadmap step 2) of
``docs/QWEN36_W4A16_KERNEL_REFACTOR_PLAN_20260518.md``.

Today's full-attention decode path goes through
``torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)``,
which generates the causal mask inside the cuDNN/Flash kernel without
materialising it on the user side. There is **no need** to allocate an
explicit mask in the safe-default route, so this module is intentionally
**scaffold-only**.

The module exists for the cases below — none of which are on by default:

* a future candidate kernel that needs a buffered ``[max_seq, max_seq]``
  causal mask (e.g. a manual GQA path that beats SDPA at long context);
* batched decode with prefix-cache where the explicit attention mask
  varies per request and a per-call ``torch.triu`` would be wasteful;
* a sliding-window candidate that wants a fixed mask buffer to slice
  views off.

The default ``decode_full_attn`` keeps ``is_causal=True``. Wiring this
module into a serving path requires:

1. a new ``LYNN_FULL_ATTN_MASK_CACHE`` env toggle, default ``"0"``;
2. P37 3/3 exact parity vs the SDPA reference;
3. structured 40/40 + P25 512 ≥ 108 TPS DEFAULT gate.

Until then the module is read-only by ``benchmarks/p126_workspace_mask_cache_probe.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


_DEFAULT_MAX_SEQ_ENV = "LYNN_FULL_ATTN_MASK_CACHE_MAX_SEQ"
_DEFAULT_MAX_SEQ_FALLBACK = 65536

MaskKey = tuple[str, str, int]


@dataclass(slots=True)
class MaskInfo:
    device: str
    dtype: str
    max_seq: int
    bytes: int


class FullAttnMaskCache:
    """Pre-built causal mask buffer for explicit-mask attention candidates.

    Stores one upper-triangular ``[max_seq, max_seq]`` boolean tensor per
    ``(device, dtype, max_seq)`` key. ``lookup(seq_len)`` returns a view
    sliced down to ``[seq_len, seq_len]`` so callers do not allocate.

    The boolean layout matches PyTorch SDPA's expected mask shape: ``True``
    where attention is allowed, ``False`` where blocked. Adjust per kernel
    if a different convention is required.
    """

    __slots__ = ("_tables",)

    def __init__(self) -> None:
        self._tables: dict[MaskKey, torch.Tensor] = {}

    @staticmethod
    def _key(device: Any, dtype: Any, max_seq: int) -> MaskKey:
        return (str(torch.device(device)), str(dtype), int(max_seq))

    def _build(self, device: Any, dtype: torch.dtype, max_seq: int) -> torch.Tensor:
        # Allowed-region mask: lower-triangle (current token can attend to
        # itself + past) True. The complement (future positions) False.
        return torch.tril(
            torch.ones(max_seq, max_seq, device=device, dtype=torch.bool)
        ).contiguous().to(dtype if dtype != torch.bool else torch.bool)

    def prewarm(
        self,
        device: Any,
        dtype: torch.dtype = torch.bool,
        max_seq: int | None = None,
    ) -> torch.Tensor:
        if max_seq is None:
            max_seq = int(os.environ.get(_DEFAULT_MAX_SEQ_ENV, str(_DEFAULT_MAX_SEQ_FALLBACK)))
        key = self._key(device, dtype, max_seq)
        cached = self._tables.get(key)
        if cached is None:
            cached = self._build(device, dtype, max_seq)
            self._tables[key] = cached
        return cached

    def lookup(
        self,
        seq_len: int,
        device: Any,
        dtype: torch.dtype = torch.bool,
        max_seq: int | None = None,
    ) -> torch.Tensor:
        """Return a ``[seq_len, seq_len]`` view of the cached causal mask."""
        if seq_len < 0:
            raise ValueError(f"seq_len must be >= 0, got {seq_len}")
        table = self.prewarm(device, dtype, max_seq)
        if seq_len > table.shape[0]:
            raise ValueError(
                f"seq_len {seq_len} exceeds table max_seq {table.shape[0]}; "
                f"raise {_DEFAULT_MAX_SEQ_ENV} or call prewarm with a larger max_seq"
            )
        return table[:seq_len, :seq_len]

    def reset(self) -> None:
        self._tables.clear()

    def info(self) -> list[MaskInfo]:
        out: list[MaskInfo] = []
        for key, table in self._tables.items():
            out.append(
                MaskInfo(
                    device=key[0],
                    dtype=key[1],
                    max_seq=key[2],
                    bytes=table.element_size() * table.numel(),
                )
            )
        return out


_GLOBAL_MASK_CACHE = FullAttnMaskCache()


def get_global_mask_cache() -> FullAttnMaskCache:
    """Hook for serving start / dev tooling that wants prewarm + inspect."""
    return _GLOBAL_MASK_CACHE


def prewarm_global_mask(
    device: Any,
    dtype: torch.dtype = torch.bool,
    max_seq: int | None = None,
) -> torch.Tensor:
    """Free-function shim for serving start."""
    return _GLOBAL_MASK_CACHE.prewarm(device, dtype, max_seq)


__all__ = [
    "FullAttnMaskCache",
    "MaskInfo",
    "get_global_mask_cache",
    "prewarm_global_mask",
]
