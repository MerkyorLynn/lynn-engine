"""Owned decode-step scratch tensor workspace for full-attention path.

Stream B Task (roadmap step 3) of
``docs/QWEN36_W4A16_KERNEL_REFACTOR_PLAN_20260518.md``.

The serving decode loop (``engine/resident_runner.py`` lines 1263-1264)
already pre-allocates ``new_token_tensor`` + ``pos_tensor`` once and
reuses them via ``.fill_()`` each step. But several other callers
(``engine/incremental_decode.py:348`` fallback,
``engine/resident_runner.py:739`` and ``:1093``) still allocate fresh
tensors per call.

This module makes the scratch buffers an explicit owned object so:

* one place to ``prewarm`` at serving start;
* one place to inspect resident scratch state from the banner / probe;
* the non-graph fallback path can adopt the same buffer set without
  re-allocating each token.

The class is **scaffold-only this commit**:

* not wired into the safe-default decode path;
* not wired into the graph-capture path either;
* only the new benchmark (``benchmarks/p126_workspace_mask_cache_probe.py``)
  reads the module.

Wiring is a separate commit that must pass the P37 3/3 + structured 40/40
+ P25 512 ≥ 108 TPS DEFAULT gate before changing serving behaviour
(2026-05-18 promotion bar, see Stream B roadmap doc).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class WorkspaceInfo:
    """Snapshot for serving banner / parity probe."""

    device: str
    dtype: str
    pos_tensor_shape: tuple[int, ...] | None
    new_token_tensor_shape: tuple[int, ...] | None
    pos_tensor_filled_value: int | None
    new_token_tensor_filled_value: int | None


class FullAttnDecodeWorkspace:
    """Owns the 1×1 long scratch buffers for the single-stream decode loop.

    Lifecycle: build once at serving start via ``prewarm(device, dtype)``,
    then per-token call ``set_token(id)`` / ``set_position(pos)`` to fill
    the same buffer. The buffer references stay stable across decode
    steps so any CUDA-graph capture that pins them does not have to
    re-capture.

    The default decode dtype for both buffers is ``torch.long`` because
    they hold token ids and position indices. Override only for tests.
    """

    __slots__ = ("device", "dtype", "_pos", "_new_token")

    def __init__(self, device: Any, dtype: torch.dtype = torch.long) -> None:
        self.device = device
        self.dtype = dtype
        self._pos: torch.Tensor | None = None
        self._new_token: torch.Tensor | None = None

    def prewarm(self) -> "FullAttnDecodeWorkspace":
        """Allocate scratch buffers if missing. Idempotent."""
        if self._pos is None:
            self._pos = torch.empty((1, 1), device=self.device, dtype=self.dtype)
        if self._new_token is None:
            self._new_token = torch.empty((1, 1), device=self.device, dtype=self.dtype)
        return self

    def get_pos_tensor(self) -> torch.Tensor:
        if self._pos is None:
            self.prewarm()
        return self._pos  # type: ignore[return-value]

    def get_new_token_tensor(self) -> torch.Tensor:
        if self._new_token is None:
            self.prewarm()
        return self._new_token  # type: ignore[return-value]

    def set_position(self, pos: int) -> torch.Tensor:
        """Fill the position buffer in place; returns the buffer view."""
        buf = self.get_pos_tensor()
        buf.fill_(int(pos))
        return buf

    def set_token(self, token_id: int) -> torch.Tensor:
        """Fill the new-token buffer in place; returns the buffer view."""
        buf = self.get_new_token_tensor()
        buf.fill_(int(token_id))
        return buf

    def reset(self) -> None:
        """Drop scratch buffers. Mostly for tests / parity probes."""
        self._pos = None
        self._new_token = None

    def info(self) -> WorkspaceInfo:
        return WorkspaceInfo(
            device=str(self.device),
            dtype=str(self.dtype),
            pos_tensor_shape=tuple(self._pos.shape) if self._pos is not None else None,
            new_token_tensor_shape=(
                tuple(self._new_token.shape) if self._new_token is not None else None
            ),
            pos_tensor_filled_value=(
                int(self._pos.item()) if self._pos is not None else None
            ),
            new_token_tensor_filled_value=(
                int(self._new_token.item()) if self._new_token is not None else None
            ),
        )


# Module-level singleton mirrors the RoPE-cache pattern in
# ``triton_kernels/full_attn_rope_cache.py``. Callers that want the
# default scratch space can fetch this without threading state.
_GLOBAL_WORKSPACE: FullAttnDecodeWorkspace | None = None


def get_global_workspace(
    device: Any = None, dtype: torch.dtype = torch.long
) -> FullAttnDecodeWorkspace:
    """Return the process-wide workspace, allocating it on first call.

    If the workspace was previously created on a different device, a new
    one is constructed for the requested device so tests can switch
    between cuda / cpu without bleeding state.
    """
    global _GLOBAL_WORKSPACE
    if _GLOBAL_WORKSPACE is None or (
        device is not None and str(_GLOBAL_WORKSPACE.device) != str(device)
    ):
        _GLOBAL_WORKSPACE = FullAttnDecodeWorkspace(device=device, dtype=dtype).prewarm()
    return _GLOBAL_WORKSPACE


def reset_global_workspace() -> None:
    """Drop the global singleton. Tests + parity probes only."""
    global _GLOBAL_WORKSPACE
    _GLOBAL_WORKSPACE = None


__all__ = [
    "FullAttnDecodeWorkspace",
    "WorkspaceInfo",
    "get_global_workspace",
    "reset_global_workspace",
]
