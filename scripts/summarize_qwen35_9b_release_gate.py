#!/usr/bin/env python3
"""Build the Qwen3.5-9B release gate summary.

This gate is intentionally report-first and GPU-free.  It aggregates the
existing BF16, Q4_K_M, and Lynn-native NVFP4 benchmark summaries, then layers on
filesystem artifact checks when run on R6000.  Missing data is represented as
PENDING or BLOCKED rather than omitted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_SCRIPT = SCRIPT_DIR / "summarize_qwen35_9b_release_matrix.py"


def _load_matrix_builder():
    spec = importlib.util.spec_from_file_location("qwen35_release_matrix", MATRIX_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {MATRIX_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_matrix_json


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _json_load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _display_path(path: Path) -> str:
    """Return a repo-relative path when possible, otherwise the absolute path."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _latest(base: Path, pattern: str) -> Path | None:
    items = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime)
    return items[-1] if items else None


def _bytes_path(path: Path) -> int | None:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return None
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 3)


def _entry_by_variant(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("variant")): item for item in matrix.get("entries", [])}


def _artifact(
    *,
    name: str,
    candidates: list[Path],
    required_children: list[str] | None = None,
    reported_size_gib: float | None = None,
    evidence_reports: list[Path] | None = None,
) -> dict[str, Any]:
    required_children = required_children or []
    evidence_reports = [p for p in (evidence_reports or []) if p is not None and p.exists()]

    checked = []
    selected: Path | None = None
    selected_bytes: int | None = None
    missing_required: list[str] = []
    for path in candidates:
        size = _bytes_path(path)
        exists = size is not None
        req_missing: list[str] = []
        if exists and path.is_dir():
            for child in required_children:
                if not (path / child).exists():
                    req_missing.append(child)
        checked.append(
            {
                "path": str(path),
                "exists": exists,
                "bytes": size,
                "size_gib": _gib(size),
                "missing_required": req_missing,
            }
        )
        if exists and selected is None and not req_missing:
            selected = path
            selected_bytes = size
            missing_required = req_missing

    if selected is not None:
        status = "PRESENT"
    elif evidence_reports:
        status = "REPORTED_PRESENT"
    else:
        status = "PENDING_ARTIFACT_CHECK"

    return {
        "name": name,
        "status": status,
        "selected_path": str(selected) if selected is not None else None,
        "bytes": selected_bytes,
        "size_gib": _gib(selected_bytes) if selected_bytes is not None else reported_size_gib,
        "reported_size_gib": reported_size_gib,
        "checked_paths": checked,
        "required_children": required_children,
        "missing_required": missing_required,
        "evidence_reports": [_display_path(p) for p in evidence_reports],
    }


def _score(metric: dict[str, Any]) -> str:
    if metric.get("score") is None:
        return "PENDING" if metric.get("status") != "BLOCKED" else "BLOCKED"
    return f"{metric['score'] * 100:.2f}%"


def _metric_status(metric: dict[str, Any]) -> str:
    return str(metric.get("status") or "PENDING")


def _tps(entry: dict[str, Any], group: str, key: str) -> float | None:
    value = entry.get(group, {}).get(key)
    return float(value) if value is not None else None


def _runtime_smoke(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "PENDING", "report": None}
    payload = _json_load(path)
    if not payload:
        return {"status": "PENDING", "report": _display_path(path)}
    return {
        "status": payload.get("status") or "UNKNOWN",
        "report": _display_path(path),
        "completion_text": payload.get("completion_text"),
        "decode_tps": payload.get("timings", {}).get("decode_tps"),
        "new_token_ids": payload.get("new_token_ids"),
    }


def _variant_gate(
    variant: str,
    entry: dict[str, Any],
    artifact: dict[str, Any],
    *,
    runtime_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mmlu_done = _metric_status(entry.get("mmlu", {})) == "DONE"
    gpqa_done = _metric_status(entry.get("gpqa", {})) == "DONE"
    quality_done = mmlu_done and gpqa_done
    artifact_ok = artifact["status"] in {"PRESENT", "REPORTED_PRESENT"}

    if variant == "BF16":
        if quality_done and artifact_ok:
            decision = "PASS_QUALITY_CEILING"
            blocker = "TPS not required for BF16 release ceiling"
        elif quality_done:
            decision = "PENDING_ARTIFACT_CHECK"
            blocker = "BF16 quality is present, but artifact path was not found on this host"
        else:
            decision = "PENDING_QUALITY"
            blocker = "BF16 quality reports missing"
    elif variant == "Q4_K_M":
        has_speed = _tps(entry, "single_tps", "512") is not None and _tps(entry, "concurrent_tps", "8") is not None
        has_long = _tps(entry, "long_context", "32k") is not None
        if quality_done and has_speed and has_long and artifact_ok:
            decision = "PASS_RELEASE"
            blocker = None
        else:
            missing = []
            if not quality_done:
                missing.append("quality")
            if not has_speed:
                missing.append("single/concurrent TPS")
            if not has_long:
                missing.append("32k long-context TPS")
            if not artifact_ok:
                missing.append("GGUF artifact check")
            decision = "PENDING"
            blocker = ", ".join(missing)
    elif variant == "NVFP4":
        smoke_pass = (runtime_smoke or {}).get("status") == "GENERATION_PASS"
        if smoke_pass:
            decision = "PENDING_QUALITY_TPS"
            blocker = "NVFP4 dense resident generation smoke passes; MMLU/GPQA and serving TPS gates still pending"
            if not artifact_ok:
                blocker += "; artifact presence still needs R6000 filesystem check"
        else:
            decision = "BLOCKED_RUNTIME"
            blocker = "Lynn-native W4A16 NVFP4 pack exists/reported, but Qwen3.5 dense resident runtime has not passed generation smoke yet"
            if not artifact_ok:
                blocker += "; artifact presence still needs R6000 filesystem check"
    else:
        decision = "PENDING"
        blocker = "unknown variant"

    quality = {
        "mmlu": dict(entry.get("mmlu", {})),
        "gpqa": dict(entry.get("gpqa", {})),
    }
    if variant == "NVFP4" and (runtime_smoke or {}).get("status") == "GENERATION_PASS":
        for metric in quality.values():
            if metric.get("score") is None and metric.get("status") == "BLOCKED":
                metric["status"] = "PENDING"
                metric["blocker"] = "runtime smoke passes; quality gate not run yet"

    return {
        "variant": variant,
        "artifact": artifact,
        "quality": quality,
        "speed": {
            "single_tps": entry.get("single_tps", {}),
            "concurrent_tps": entry.get("concurrent_tps", {}),
            "long_context": entry.get("long_context", {}),
        },
        "matrix_status": entry.get("status"),
        "runtime_smoke": runtime_smoke or {"status": "PENDING", "report": None},
        "decision": decision,
        "blocker": blocker,
    }


def build_gate(args: argparse.Namespace) -> dict[str, Any]:
    build_matrix_json = _load_matrix_builder()
    report_dir = args.report_dir
    matrix = build_matrix_json(report_dir)
    entries = _entry_by_variant(matrix)

    model_root = args.model_root
    bf16_path = Path(os.environ.get("BF16_MODEL", str(model_root / "Qwen3.5-9B-BF16")))
    q4km_env = os.environ.get("Q4KM_GGUF")
    q4km_candidates = [
        Path(q4km_env) if q4km_env else None,
        model_root / "Qwen3.5-9B-Q4_K_M.gguf",
        model_root / "Qwen3.5-9B-GGUF" / "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
    ]
    nvfp4_path = Path(os.environ.get("NVFP4_MODEL", str(model_root / "Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0")))

    latest_bf16_quality = _latest(report_dir, "bf16_*_quality_summary.json")
    latest_q4_speed = _latest(report_dir, "r6000_qwen35_9b_q4km_baseline_*.json")
    latest_q4_quality = _latest(report_dir, "q4km_llamacpp*_quality_summary.json")
    latest_nvfp4_pack = _latest(report_dir, "r6000_qwen35_9b*_w4a16_pack_summary_*.json")
    if latest_nvfp4_pack is None:
        latest_nvfp4_pack = _latest(report_dir, "*w4a16_pack_summary*.json")
    latest_bf16_smoke = _latest(report_dir, "r6000_qwen35_9b_dense_runtime_smoke*.json")
    latest_nvfp4_smoke = _latest(report_dir, "r6000_qwen35_9b_nvfp4_dense_runtime_smoke*.json")

    artifacts = {
        "BF16": _artifact(
            name="BF16",
            candidates=[bf16_path],
            required_children=["config.json", "model.safetensors.index.json"],
            reported_size_gib=entries["BF16"].get("size_gib"),
            evidence_reports=[p for p in [latest_bf16_quality] if p],
        ),
        "Q4_K_M": _artifact(
            name="Q4_K_M",
            candidates=[p for p in q4km_candidates if p is not None],
            reported_size_gib=entries["Q4_K_M"].get("size_gib"),
            evidence_reports=[p for p in [latest_q4_speed, latest_q4_quality] if p],
        ),
        "NVFP4": _artifact(
            name="NVFP4",
            candidates=[nvfp4_path],
            required_children=["config.json", "lynn_quant_manifest.json"],
            reported_size_gib=entries["NVFP4"].get("size_gib"),
            evidence_reports=[p for p in [latest_nvfp4_pack, latest_nvfp4_smoke, args.release_matrix_json if args.release_matrix_json.exists() else None] if p],
        ),
    }

    variants = [
        _variant_gate("BF16", entries["BF16"], artifacts["BF16"], runtime_smoke=_runtime_smoke(latest_bf16_smoke)),
        _variant_gate("Q4_K_M", entries["Q4_K_M"], artifacts["Q4_K_M"]),
        _variant_gate("NVFP4", entries["NVFP4"], artifacts["NVFP4"], runtime_smoke=_runtime_smoke(latest_nvfp4_smoke)),
    ]

    decisions = {item["variant"]: item["decision"] for item in variants}
    if decisions["Q4_K_M"] == "PASS_RELEASE" and decisions["NVFP4"] == "BLOCKED_RUNTIME":
        overall = "PARTIAL_RELEASE_Q4KM_READY_NVFP4_BLOCKED"
    elif decisions["Q4_K_M"] == "PASS_RELEASE" and decisions["NVFP4"] == "PENDING_QUALITY_TPS":
        overall = "PARTIAL_RELEASE_Q4KM_READY_NVFP4_SMOKE_PASS"
    elif all(v.startswith("PASS") for v in decisions.values()):
        overall = "PASS_RELEASE"
    else:
        overall = "PENDING"

    return {
        "schema": "lynn-qwen35-9b-release-gate-v1",
        "created": _iso_now(),
        "host": socket.gethostname(),
        "model_id": "Qwen/Qwen3.5-9B",
        "model_root": str(model_root),
        "report_dir": str(report_dir),
        "overall_decision": overall,
        "variants": variants,
        "source_matrix": matrix,
    }


def render_markdown(gate: dict[str, Any]) -> str:
    by_variant = {item["variant"]: item for item in gate["variants"]}

    def status(variant: str) -> str:
        return by_variant[variant]["decision"]

    def art(variant: str) -> str:
        a = by_variant[variant]["artifact"]
        size = a.get("size_gib")
        size_text = f"{size:.1f} GiB" if isinstance(size, (int, float)) else "PENDING"
        return f"{a['status']} ({size_text})"

    def quality(variant: str, metric: str) -> str:
        return _score(by_variant[variant]["quality"][metric])

    def tps(variant: str, group: str, key: str) -> str:
        value = by_variant[variant]["speed"].get(group, {}).get(key)
        return f"{value:.1f}" if isinstance(value, (int, float)) else "PENDING"

    lines = [
        "# Qwen3.5-9B Release Status",
        "",
        f"**Generated:** {gate['created']}  ",
        f"**Overall decision:** `{gate['overall_decision']}`  ",
        "",
        "## Compact Gate Matrix",
        "",
        "| Variant | Artifact | MMLU | GPQA | Single 512 TPS | Concurrent x8 TPS | 32k TPS | Gate | Blocker |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for variant in ("BF16", "Q4_K_M", "NVFP4"):
        item = by_variant[variant]
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    art(variant),
                    quality(variant, "mmlu"),
                    quality(variant, "gpqa"),
                    tps(variant, "single_tps", "512"),
                    tps(variant, "concurrent_tps", "8"),
                    tps(variant, "long_context", "32k"),
                    f"`{status(variant)}`",
                    item.get("blocker") or "-",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Explicit Blockers",
            "",
            "- BF16 is the quality ceiling path. It has MMLU/GPQA data, but no serving TPS gate is required for release.",
            "- Q4_K_M is the current ready release path for llama.cpp/Mac and CUDA fallback users.",
            "- Lynn-native W4A16 NVFP4 now has a dense resident generation smoke when the gate can find a `GENERATION_PASS` smoke report; full MMLU/GPQA and serving TPS gates remain pending until run.",
            "",
            "## Artifact Check Notes",
            "",
            "- `PRESENT` means the artifact exists on the host that ran the gate.",
            "- `REPORTED_PRESENT` means a prior benchmark/report proves the artifact existed, but the current host could not inspect that path.",
            "- `PENDING_ARTIFACT_CHECK` means neither filesystem nor report evidence is available.",
            "",
            "## Source Reports",
            "",
        ]
    )
    for item in gate["variants"]:
        artifact = item["artifact"]
        evidence = artifact.get("evidence_reports", [])
        if evidence:
            lines.append(f"- **{item['variant']}**: " + ", ".join(f"`{p}`" for p in evidence))
        else:
            lines.append(f"- **{item['variant']}**: PENDING")
    lines.append("")
    lines.append("*Generated by `scripts/r6000_qwen35_9b_release_gate.sh`.*")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/qwen35_9b"))
    parser.add_argument("--model-root", type=Path, default=Path("/root/autodl-tmp/models"))
    parser.add_argument(
        "--release-matrix-json",
        type=Path,
        default=Path("reports/qwen35_9b/qwen35_9b_release_matrix.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/qwen35_9b/qwen35_9b_release_gate_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/QWEN35_9B_RELEASE_STATUS_20260519.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    gate = build_gate(args)
    args.output_json.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(gate), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "overall_decision": gate["overall_decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
