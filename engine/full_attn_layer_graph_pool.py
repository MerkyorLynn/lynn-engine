"""Stream B — full-attention layer CUDA graph reuse pool (V2, bucket-aligned).

Wraps the dynamic-position graph capture in
``LynnIncrementalRunner._capture_full_attn_layer_graph_slot_v2`` (in turn
backed by ``decode_full_attn_graphable`` in ``incremental_decode.py``)
into a per-layer per-bucket reuse pool.

Slot key = ``(layer_idx, bucket_idx)`` where
``bucket_idx = state.seq_len // bucket_size``. A given slot is valid for
all positions in ``[bucket_idx * bucket_size, (bucket_idx + 1) * bucket_size)``.
Crossing a bucket boundary triggers lazy recapture with a longer
attention mask + cache view, so the SDPA compute / memory bandwidth
scales with the bucket, not with ``max_seq_len``.

Spec: ``docs/STREAM_B_FULL_ATTN_LAYER_GRAPH_REUSE_SPEC_20260518.md``.

Env opt-in:

* ``LYNN_FULL_ATTN_LAYER_GRAPH_POOL=1`` — enable the pool. Off by default.
* ``LYNN_FULL_ATTN_LAYER_GRAPH_BUCKET`` — bucket size in tokens
  (default 256). Larger buckets = fewer captures but each replay
  computes SDPA over more positions; smaller buckets = more captures
  with shorter SDPA. 256 balances capture cost vs SDPA waste for
  typical 256–2048 token decode sessions.

Promotion gate (from spec):

* P37 exact-greedy 3/3 required (graph replay is bit-identical to eager
  by construction; any drift is a wrapper bug, not a tolerated artifact).
* P25 512-token decode TPS lift over the linear-block-only baseline.
"""
from __future__ import annotations

import os
import time
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from engine.inference_state import LynnInferenceState
    from engine.resident_runner import (
        FullAttentionLayerGraphSlotV2,
        LynnIncrementalRunner,
    )


DEFAULT_BUCKET = 256


class FullAttnLayerGraphPool:
    """Bucket-aligned per-layer full-attention graph reuse pool (V2).

    Slots are captured lazily on first access. The pool tracks
    ``(layer_idx, bucket_idx)`` keys; a single decode session typically
    uses 1–4 buckets per layer (one per 256-token band of seq_len).
    """

    def __init__(self, runner: "LynnIncrementalRunner", bucket: int = DEFAULT_BUCKET) -> None:
        self.runner = runner
        self.bucket = int(bucket)
        if self.bucket <= 0:
            raise ValueError(f"bucket must be >= 1, got {bucket}")
        self._slots: dict[tuple[int, int], "FullAttentionLayerGraphSlotV2"] = {}
        self.capture_seconds: list[float] = []
        self.captures_total = 0
        self.reuses_total = 0

    def get(
        self,
        state: "LynnInferenceState",
        layer_idx: int,
    ) -> "FullAttentionLayerGraphSlotV2":
        """Return a valid slot for ``(layer_idx, state.seq_len // bucket)``."""
        bucket_idx = int(state.seq_len) // self.bucket
        key = (layer_idx, bucket_idx)
        slot = self._slots.get(key)
        if slot is None:
            t0 = time.time()
            slot = self.runner._capture_full_attn_layer_graph_slot_v2(
                state, layer_idx, bucket_idx, self.bucket,
            )
            self.capture_seconds.append(time.time() - t0)
            self.captures_total += 1
            self._slots[key] = slot
        else:
            self.reuses_total += 1
        return slot

    def invalidate(self) -> None:
        """Drop all cached slots — call on weight reload / max_seq_len change."""
        self._slots.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "captures_total": self.captures_total,
            "reuses_total": self.reuses_total,
            "mean_capture_seconds": (
                sum(self.capture_seconds) / len(self.capture_seconds)
                if self.capture_seconds else None
            ),
            "total_capture_seconds": sum(self.capture_seconds),
        }


def maybe_create_pool(runner: "LynnIncrementalRunner") -> FullAttnLayerGraphPool | None:
    """Construct a pool if ``LYNN_FULL_ATTN_LAYER_GRAPH_POOL=1``."""
    if os.environ.get("LYNN_FULL_ATTN_LAYER_GRAPH_POOL", "0") != "1":
        return None
    bucket_env = os.environ.get("LYNN_FULL_ATTN_LAYER_GRAPH_BUCKET", str(DEFAULT_BUCKET))
    try:
        bucket = int(bucket_env)
    except ValueError as exc:
        raise RuntimeError(
            f"LYNN_FULL_ATTN_LAYER_GRAPH_BUCKET must be int, got {bucket_env!r}"
        ) from exc
    return FullAttnLayerGraphPool(runner, bucket=bucket)
