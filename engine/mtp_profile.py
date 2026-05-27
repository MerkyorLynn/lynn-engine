"""Tiny opt-in profiler for MTP verifier ROI work."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

import torch

_SECTIONS: dict[str, dict[str, float | int]] = {}


def enabled() -> bool:
    return os.environ.get("LYNN_MTP_PROFILE", "0") == "1"


def _sync() -> None:
    if os.environ.get("LYNN_MTP_PROFILE_SYNC", "1") == "1" and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset() -> None:
    _SECTIONS.clear()


def record(name: str, seconds: float) -> None:
    if not enabled():
        return
    row = _SECTIONS.setdefault(name, {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0})
    row["count"] = int(row["count"]) + 1
    row["total_seconds"] = float(row["total_seconds"]) + float(seconds)
    row["max_seconds"] = max(float(row["max_seconds"]), float(seconds))


@contextmanager
def section(name: str) -> Iterator[None]:
    if not enabled():
        yield
        return
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        record(name, time.perf_counter() - t0)


def snapshot() -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for name, row in sorted(_SECTIONS.items()):
        count = int(row["count"])
        total = float(row["total_seconds"])
        out[name] = {
            "count": count,
            "total_seconds": total,
            "mean_ms": (total / count * 1000.0) if count else None,
            "max_ms": float(row["max_seconds"]) * 1000.0,
        }
    return out
