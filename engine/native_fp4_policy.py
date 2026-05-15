"""Opt-in policy helpers for native FP4 decode experiments.

P5-C is deliberately a policy gate, not a default engine switch. P5-B proved
that selected single linear-attention projections can pass numeric gates, while
also showing the current native path is not yet faster than the scalar bridge.

This module keeps that contract explicit:

- default backend remains scalar_bridge;
- native_scaled_mm may be enabled only for allow-listed `(layer, projection)`;
- policies generated from benchmark reports carry correctness and timing data.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class NativeFP4Projection:
    layer: int
    projection: str

    @property
    def key(self) -> str:
        return f"layer{self.layer}:{self.projection}"


@dataclass(slots=True)
class NativeFP4Policy:
    """Runtime opt-in policy for native FP4 decode projections."""

    enabled: bool
    allowlist: set[str]
    require_speedup: bool = False
    backend: str = "native_scaled_mm"

    @classmethod
    def disabled(cls) -> "NativeFP4Policy":
        return cls(enabled=False, allowlist=set())

    @classmethod
    def from_json(cls, path: str | Path) -> "NativeFP4Policy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            enabled=bool(data.get("enabled", False)),
            allowlist=set(data.get("allowlist", [])),
            require_speedup=bool(data.get("require_speedup", False)),
            backend=str(data.get("backend", "native_scaled_mm")),
        )

    def backend_for(self, *, layer: int, projection: str) -> str:
        if not self.enabled:
            return "scalar_bridge"
        item = NativeFP4Projection(layer=layer, projection=projection)
        return self.backend if item.key in self.allowlist else "scalar_bridge"


def build_policy_from_p5_reports(
    report_dir: str | Path,
    *,
    min_cosine: float = 0.98,
    max_rel_l2: float = 0.25,
    require_speedup: bool = False,
) -> dict[str, Any]:
    """Build an auditable policy document from P5-B report JSON files."""
    report_dir = Path(report_dir)
    allowlist: list[str] = []
    rejected: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []

    for path in sorted(report_dir.glob("p5_layer*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        layer = data.get("layer")
        projection = data.get("replace_mode")
        cmp = (
            data.get("comparisons", {})
            .get("native_vs_scalar_bridge", {})
            .get("decode_output", {})
        )
        timing = data.get("timing_ms", {})
        speed_ratio = timing.get("native_vs_scalar_speed_ratio")
        ok = (
            data.get("verdict") == "PASS"
            and cmp.get("cosine", 0.0) >= min_cosine
            and cmp.get("rel_l2", 999.0) <= max_rel_l2
            and (not require_speedup or (speed_ratio is not None and speed_ratio > 1.0))
        )
        rec = {
            "file": str(path),
            "layer": layer,
            "projection": projection,
            "cosine": cmp.get("cosine"),
            "rel_l2": cmp.get("rel_l2"),
            "speed_ratio": speed_ratio,
            "verdict": data.get("verdict"),
        }
        if ok:
            key = NativeFP4Projection(layer=int(layer), projection=str(projection)).key
            allowlist.append(key)
            accepted.append(rec)
        else:
            rejected.append(rec)

    return {
        "schema_version": "lynn-engine-native-fp4-policy-v1",
        "enabled": bool(allowlist),
        "require_speedup": require_speedup,
        "thresholds": {
            "min_cosine": min_cosine,
            "max_rel_l2": max_rel_l2,
        },
        "allowlist": allowlist,
        "accepted": accepted,
        "rejected": rejected,
        "default_backend": "scalar_bridge",
        "notes": [
            "Policy is opt-in only; engine defaults must remain scalar_bridge.",
            "Current P5-C policy is a correctness gate. Do not treat it as a production speedup unless require_speedup=true also passes.",
        ],
    }


def build_policy_from_p5d_fastpath_reports(
    report_dir: str | Path,
    *,
    min_cosine: float = 0.999,
    max_rel_l2: float = 0.01,
    min_speed_ratio: float = 1.0,
    backend: str = "native_fast_2d",
) -> dict[str, Any]:
    """Build a speed-gated policy from P5-D fastpath reports.

    P5-D reports compare the specialized native fast path against both generic
    native and scalar bridge. This policy is stricter than P5-C: it only accepts
    projections that are numerically equivalent to generic native and faster
    than scalar bridge.
    """
    report_dir = Path(report_dir)
    allowlist: list[str] = []
    rejected: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []

    for path in sorted(report_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        layer = int(data.get("layer"))
        projection = str(data.get("weight", "")).removesuffix(".weight")
        cmp = data.get("comparisons", {}).get("fast_vs_native", {})
        derived = data.get("derived", {})
        speed_ratio = derived.get("fast_vs_scalar_ratio")
        ok = (
            cmp.get("cosine", 0.0) >= min_cosine
            and cmp.get("rel_l2", 999.0) <= max_rel_l2
            and speed_ratio is not None
            and speed_ratio > min_speed_ratio
        )
        rec = {
            "file": str(path),
            "layer": layer,
            "projection": projection,
            "cosine": cmp.get("cosine"),
            "rel_l2": cmp.get("rel_l2"),
            "speed_ratio": speed_ratio,
            "latency_ms": data.get("latency_ms", {}),
        }
        if ok:
            key = NativeFP4Projection(layer=layer, projection=projection).key
            allowlist.append(key)
            accepted.append(rec)
        else:
            rejected.append(rec)

    return {
        "schema_version": "lynn-engine-native-fp4-fastpath-policy-v1",
        "enabled": bool(allowlist),
        "backend": backend,
        "require_speedup": True,
        "thresholds": {
            "min_cosine": min_cosine,
            "max_rel_l2": max_rel_l2,
            "min_speed_ratio": min_speed_ratio,
        },
        "allowlist": allowlist,
        "accepted": accepted,
        "rejected": rejected,
        "default_backend": "scalar_bridge",
        "notes": [
            "P5-D fastpath policy is projection-scoped and speed-gated.",
            "Accepted projections may use native_fast_2d; rejected projections must stay scalar_bridge.",
            "This is still a microbench policy until decode-loop integration passes end-to-end TPS gates.",
        ],
    }
