"""Runtime contract helpers for W4A8-folded artifacts.

Folded W4A8 Recovery artifacts adjust down-projection weights so the model is
numerically aligned with FP8-activation active-MoE inference.  They are not a
safe BF16 fallback artifact.  This module is deliberately small and import-free
from the production loader until a W4A8 backend explicitly opts in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


W4A8_ALPHA_FOLD_MANIFEST = "lynn_w4a8_alpha_fold_manifest.json"
W4A8_BACKEND_MARKERS = ("w4a8", "fp8", "split16", "spark_fp8")


def load_w4a8_contract(model_dir: str | Path) -> dict[str, Any]:
    """Load the optional W4A8 fold manifest.

    Non-folded artifacts do not carry this file, so callers can use an empty
    dict as "no special contract".
    """

    manifest_path = Path(model_dir) / W4A8_ALPHA_FOLD_MANIFEST
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def is_w4a8_backend(active_backend: str | None) -> bool:
    """Return whether a backend name claims W4A8/FP8-active semantics."""

    backend = (active_backend or "").lower()
    return any(marker in backend for marker in W4A8_BACKEND_MARKERS)


def assert_w4a8_runtime_contract(
    model_dir: str | Path,
    *,
    active_backend: str | None,
    allow_unsafe_env: str = "LYNN_ALLOW_UNSAFE_BF16_FALLBACK",
) -> None:
    """Fail loudly if a W4A8-folded artifact is launched on a BF16 path.

    The override exists for explicit research probes only.  It should not be set
    in production or eval gates.
    """

    manifest = load_w4a8_contract(model_dir)
    if manifest.get("inference_path_required") != "w4a8":
        return
    if os.environ.get(allow_unsafe_env) == "1":
        return
    if is_w4a8_backend(active_backend):
        return

    drift = manifest.get("bf16_fallback_drift_estimate", "unknown")
    required = manifest.get("inference_path_required", "w4a8")
    raise RuntimeError(
        "folded artifact requires W4A8 inference path; "
        f"active_backend={active_backend!r} does not satisfy required={required!r}. "
        f"BF16 fallback would silently degrade quality (estimated drift={drift}). "
        "Use a compatible W4A8/FP8-active runtime or the original BF16 artifact. "
        f"Set {allow_unsafe_env}=1 only for explicit research probes."
    )


def describe_w4a8_contract(model_dir: str | Path) -> str:
    """Return a one-line human-readable summary for logs and reports."""

    manifest = load_w4a8_contract(model_dir)
    if not manifest:
        return "no W4A8 fold manifest"
    required = manifest.get("inference_path_required", "unspecified")
    fallback = manifest.get("fallback_path_allowed", "unspecified")
    w4a8_drift = manifest.get("w4a8_drift_vs_bf16_reference", "unknown")
    bf16_drift = manifest.get("bf16_fallback_drift_estimate", "unknown")
    return (
        f"W4A8 fold manifest: required={required}, "
        f"fallback_allowed={fallback}, "
        f"w4a8_drift={w4a8_drift}, bf16_fallback_drift={bf16_drift}"
    )
