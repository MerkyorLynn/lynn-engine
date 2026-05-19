"""Stream B — full-attention layer graph reuse pool.

Per-layer CUDA-graph slots for the 10 full-attention layers in Qwen3.6-35B-A3B
(or 8 in Qwen3.5-9B Dense). The default Spark Config D decode loop already
replays per-3-layer linear-attention block graphs but issues 60-80 eager
launches per token across the full-attention layers. Wrapping each
full-attention layer in its own CUDA graph closes ~all of that launch
overhead while keeping per-request KV state mutable.

Spec: ``docs/STREAM_B_FULL_ATTN_LAYER_GRAPH_REUSE_SPEC_20260518.md``.

Lifecycle:

* The pool holds one ``FullAttentionLayerGraphSlot`` per full-attn layer
  index. Each slot is captured at a specific ``state.seq_len`` and is
  valid while ``state.seq_len`` stays within the same bucket
  (``bucket = LYNN_FULL_ATTN_LAYER_GRAPH_BUCKET``, default 256).
* When ``state.seq_len`` crosses a bucket boundary, the slot is
  invalidated lazily and re-captured on next access.
* KV cache positions written during graph replay are correctly indexed
  into ``state.kv_cache`` because the cache tensor identity (and
  ``state.seq_len``) is owned by ``state``, not by the slot.

Promotion gate (from spec):

* P37 exact-greedy 3/3 required (byte-identical kernel sequence; any drift
  is a wrapper bug, not a tolerated artifact).
* P25 512-token decode TPS ≥ 108 (DEFAULT) / ≥ 118 (AMBER, opt-in).
"""
from __future__ import annotations

import os
import time
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from engine.inference_state import LynnInferenceState
    from engine.resident_runner import FullAttentionLayerGraphSlot, LynnIncrementalRunner


DEFAULT_BUCKET = 256


class FullAttnLayerGraphPool:
    """Bucket-aligned per-layer full-attention graph reuse pool.

    The pool is constructed once per ``LynnIncrementalRunner`` lifetime
    (typically when ``LYNN_FULL_ATTN_LAYER_GRAPH_POOL=1`` is set and the
    decode loop enters the safe-default branch). Each ``get()`` call
    returns a captured slot keyed by ``(layer_idx, bucket)`` — if the
    cached slot's bucket no longer matches ``state.seq_len // bucket``,
    the slot is re-captured.

    The wrapping decode loop is responsible for copying the per-token
    input into ``slot.input_buf`` before calling ``slot.graph.replay()``,
    then reading ``slot.output_buf``. This pool only manages capture +
    invalidation.
    """

    def __init__(self, runner: "LynnIncrementalRunner", bucket: int = DEFAULT_BUCKET) -> None:
        self.runner = runner
        self.bucket = int(bucket)
        if self.bucket <= 0:
            raise ValueError(f"bucket must be >= 1, got {bucket}")
        self._slots: dict[int, "FullAttentionLayerGraphSlot"] = {}
        self._slot_bucket: dict[int, int] = {}
        self.capture_seconds: list[float] = []
        self.captures_total = 0
        self.reuses_total = 0

    def get(
        self,
        state: "LynnInferenceState",
        h_seed: torch.Tensor,
        pos_tensor: torch.Tensor,
        layer_idx: int,
    ) -> "FullAttentionLayerGraphSlot":
        """Return a valid slot for ``(layer_idx, state.seq_len // bucket)``.

        Captures lazily if the cached slot is missing or its bucket no
        longer matches. ``h_seed`` and ``pos_tensor`` shapes must match
        what the eager ``_decode_layer_fast`` for this layer would
        receive (``[1, 1, hidden]`` and ``[[seq_len]]``).
        """
        cur_bucket = int(state.seq_len) // self.bucket
        slot = self._slots.get(layer_idx)
        slot_bucket = self._slot_bucket.get(layer_idx)
        if slot is None or slot_bucket != cur_bucket:
            t0 = time.time()
            slot = self.runner._capture_full_attn_layer_graph_slot(
                state, h_seed, pos_tensor, layer_idx,
            )
            self.capture_seconds.append(time.time() - t0)
            self.captures_total += 1
            self._slots[layer_idx] = slot
            self._slot_bucket[layer_idx] = cur_bucket
        else:
            self.reuses_total += 1
        return slot

    def invalidate(self) -> None:
        """Drop all cached slots — call on weight reload / max_seq_len change."""
        self._slots.clear()
        self._slot_bucket.clear()

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
    """Construct a pool if env opt-in is set; return ``None`` otherwise.

    Reads:
    * ``LYNN_FULL_ATTN_LAYER_GRAPH_POOL=1`` to enable.
    * ``LYNN_FULL_ATTN_LAYER_GRAPH_BUCKET`` (default 256).
    """
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
