#!/usr/bin/env python3
"""
summarize_qwen35_9b_release_matrix.py
Read Qwen3.5-9B benchmark reports from ``reports/qwen35_9b/`` and emit a combined
JSON + Markdown release matrix covering BF16, Q4_K_M, and NVFP4 variants.

Auto-discovers the latest report files by glob pattern + mtime so the script
works regardless of exact timestamp in the filename.

Usage:
    python3 summarize_qwen35_9b_release_matrix.py \\
        --report-dir reports/qwen35_9b \\
        --output-json reports/qwen35_9b/qwen35_9b_release_matrix.json \\
        --output-md reports/qwen35_9b/qwen35_9b_release_matrix.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning *None* on any I/O or parse error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Failed to load {path}: {exc}", file=sys.stderr)
        return None


def _latest(glob_pattern: str, base: Path) -> Path | None:
    """Return the most-recently-modified file matching *glob_pattern* under *base*."""
    candidates = sorted(base.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Per-variant data extraction
# ---------------------------------------------------------------------------

def _extract_bf16(report_dir: Path) -> dict[str, Any]:
    """Extract BF16 variant data from quality summary and individual MMLU/GPQA reports."""
    entry: dict[str, Any] = {
        "variant": "BF16",
        "size_gib": 18.0,
        "stack": "transformers-direct",
        "mmlu": {"score": None, "correct": None, "total": None, "status": "PENDING"},
        "gpqa": {"score": None, "correct": None, "total": None, "status": "PENDING"},
        "single_tps": {"128": None, "256": None, "512": None},
        "concurrent_tps": {"2": None, "4": None, "8": None},
        "long_context": {"4k": None, "16k": None, "32k": None},
        "status": "PENDING",
        "blocker": None,
    }

    # --- Try combined quality summary first ---
    quality_path = _latest("bf16_*_quality_summary.json", report_dir)
    if quality_path:
        q = _load_json(quality_path)
        if q:
            # MMLU: top-level .mmlu
            mmlu = q.get("mmlu", {})
            if mmlu:
                acc = mmlu.get("accuracy") or mmlu.get("score")
                correct = mmlu.get("correct")
                total = mmlu.get("n") or mmlu.get("total")
                if acc is not None:
                    entry["mmlu"] = {
                        "score": round(acc, 4),
                        "correct": correct,
                        "total": total,
                        "status": "DONE",
                    }
            # GPQA: top-level .gpqa
            gpqa = q.get("gpqa", {})
            if gpqa:
                acc = gpqa.get("accuracy") or gpqa.get("score")
                correct = gpqa.get("correct")
                total = gpqa.get("n") or gpqa.get("total")
                if acc is not None:
                    entry["gpqa"] = {
                        "score": round(acc, 4),
                        "correct": correct,
                        "total": total,
                        "status": "DONE",
                    }

    # --- Fallback: individual MMLU summary ---
    if entry["mmlu"]["status"] != "DONE":
        mmlu_path = _latest("bf16_*_mmlu*.summary.json", report_dir)
        if mmlu_path:
            m = _load_json(mmlu_path)
            if m:
                acc = m.get("accuracy") or m.get("score")
                if acc is not None:
                    entry["mmlu"] = {
                        "score": round(acc, 4),
                        "correct": m.get("correct"),
                        "total": m.get("n") or m.get("total"),
                        "status": "DONE",
                    }

    # --- Fallback: individual GPQA summary ---
    if entry["gpqa"]["status"] != "DONE":
        gpqa_path = _latest("bf16_*_gpqa*.summary.json", report_dir)
        if gpqa_path:
            g = _load_json(gpqa_path)
            if g:
                acc = g.get("accuracy") or g.get("score")
                if acc is not None:
                    entry["gpqa"] = {
                        "score": round(acc, 4),
                        "correct": g.get("correct"),
                        "total": g.get("n") or g.get("total"),
                        "status": "DONE",
                    }

    # --- Derive overall status ---
    has_quality = entry["mmlu"]["status"] == "DONE" or entry["gpqa"]["status"] == "DONE"
    has_tps = any(v is not None for v in entry["single_tps"].values())
    if has_quality and has_tps:
        entry["status"] = "DONE"
    elif has_quality:
        entry["status"] = "PARTIAL"
        entry["blocker"] = "No TPS benchmarks yet"
    else:
        entry["status"] = "PENDING"
        entry["blocker"] = "No quality reports found"

    return entry


def _extract_q4km(report_dir: Path) -> dict[str, Any]:
    """Extract Q4_K_M variant data from llama.cpp baseline reports."""
    entry: dict[str, Any] = {
        "variant": "Q4_K_M",
        "size_gib": 5.5,
        "stack": "llama.cpp",
        "mmlu": {"score": None, "correct": None, "total": None, "status": "PENDING"},
        "gpqa": {"score": None, "correct": None, "total": None, "status": "PENDING"},
        "single_tps": {"128": None, "256": None, "512": None},
        "concurrent_tps": {"2": None, "4": None, "8": None},
        "long_context": {"4k": None, "16k": None, "32k": None},
        "status": "PENDING",
        "blocker": "GGUF download pending",
    }

    baseline_path = _latest("r6000_qwen35_9b_q4km_baseline_*.json", report_dir)
    if not baseline_path:
        return entry

    report = _load_json(baseline_path)
    if report is None:
        return entry

    file_status = report.get("status")
    if file_status == "PENDING_DOWNLOAD":
        entry["status"] = "PENDING"
        entry["blocker"] = "GGUF download pending"
        return entry

    # --- Single TPS ---
    if report.get("schema_version") == "openai-serving-matrix-probe-v1":
        for row in report.get("single", {}).get("rows", []):
            key = str(row.get("max_tokens"))
            if key in entry["single_tps"] and row.get("ok"):
                entry["single_tps"][key] = round(row.get("wall_tps", 0.0), 1)
    else:
        single = report.get("single_tps", {})
        for key in ("128", "256", "512"):
            e = single.get(key)
            if e and e.get("ok"):
                entry["single_tps"][key] = round(e.get("wall_tps", 0.0), 1)

    # --- Concurrent TPS ---
    if report.get("schema_version") == "openai-serving-matrix-probe-v1":
        for row in report.get("concurrency", {}).get("rows", []):
            key = str(row.get("concurrency"))
            if key in entry["concurrent_tps"] and row.get("ok"):
                entry["concurrent_tps"][key] = round(row.get("batch_wall_tps", 0.0), 1)
    else:
        concurrent = report.get("concurrent_tps", {})
        for key in ("2", "4", "8"):
            e = concurrent.get(key)
            if e and e.get("ok"):
                entry["concurrent_tps"][key] = round(e.get("batch_wall_tps", 0.0), 1)

    # --- Long Context ---
    lc_key_map = {"4k": "4096", "16k": "16384", "32k": "32768"}
    if report.get("schema_version") == "openai-serving-matrix-probe-v1":
        by_chars = {
            str(row.get("target_prompt_chars")): row
            for row in report.get("long_context", {}).get("rows", [])
        }
        for label, lk in lc_key_map.items():
            e = by_chars.get(lk)
            if e and e.get("ok"):
                entry["long_context"][label] = round(e.get("wall_tps", 0.0), 1)
    else:
        lc = report.get("long_context", {})
        for label, lk in lc_key_map.items():
            e = lc.get(lk)
            if e and e.get("ok"):
                entry["long_context"][label] = round(e.get("wall_tps", 0.0), 1)
    long32_path = _latest("q4km_long32k_parallel1_*.json", report_dir)
    if long32_path:
        long32 = _load_json(long32_path)
        if long32 and long32.get("ok"):
            entry["long_context"]["32k"] = round(long32.get("wall_tps", 0.0), 1)

    # --- Quality (MMLU/GPQA) from Q4_K_M reports if present ---
    mmlu_data = report.get("mmlu", {})
    if mmlu_data:
        acc = mmlu_data.get("accuracy") or mmlu_data.get("score")
        if acc is not None:
            entry["mmlu"] = {
                "score": round(acc, 4),
                "correct": mmlu_data.get("correct"),
                "total": mmlu_data.get("n") or mmlu_data.get("total"),
                "status": "DONE",
            }

    gpqa_data = report.get("gpqa", {})
    if gpqa_data:
        acc = gpqa_data.get("accuracy") or gpqa_data.get("score")
        if acc is not None:
            entry["gpqa"] = {
                "score": round(acc, 4),
                "correct": gpqa_data.get("correct"),
                "total": gpqa_data.get("n") or gpqa_data.get("total"),
                "status": "DONE",
            }

    quality_path = _latest("q4km_llamacpp*_quality_summary.json", report_dir)
    if quality_path:
        quality = _load_json(quality_path)
        if quality:
            mmlu_data = quality.get("mmlu_500_5shot", {})
            if mmlu_data:
                acc = mmlu_data.get("accuracy") or mmlu_data.get("score")
                if acc is not None:
                    entry["mmlu"] = {
                        "score": round(acc, 4),
                        "correct": mmlu_data.get("correct"),
                        "total": mmlu_data.get("total") or mmlu_data.get("n"),
                        "status": "DONE",
                    }
            gpqa_data = quality.get("gpqa_diamond", {})
            if gpqa_data:
                acc = gpqa_data.get("accuracy") or gpqa_data.get("score")
                if acc is not None:
                    entry["gpqa"] = {
                        "score": round(acc, 4),
                        "correct": gpqa_data.get("correct"),
                        "total": gpqa_data.get("total") or gpqa_data.get("n"),
                        "status": "DONE",
                    }

    if entry["mmlu"]["status"] != "DONE":
        mmlu_path = _latest("q4km*_mmlu*.summary.json", report_dir)
        if mmlu_path:
            mmlu = _load_json(mmlu_path)
            if mmlu:
                acc = mmlu.get("accuracy") or mmlu.get("score")
                if acc is not None:
                    entry["mmlu"] = {
                        "score": round(acc, 4),
                        "correct": mmlu.get("correct"),
                        "total": mmlu.get("n") or mmlu.get("total"),
                        "status": "DONE",
                    }
    if entry["gpqa"]["status"] != "DONE":
        gpqa_path = _latest("q4km*_gpqa*.summary.json", report_dir)
        if gpqa_path:
            gpqa = _load_json(gpqa_path)
            if gpqa:
                acc = gpqa.get("accuracy") or gpqa.get("score")
                if acc is not None:
                    entry["gpqa"] = {
                        "score": round(acc, 4),
                        "correct": gpqa.get("correct"),
                        "total": gpqa.get("n") or gpqa.get("total"),
                        "status": "DONE",
                    }

    # --- Size ---
    size = report.get("size_gib")
    if size is not None:
        entry["size_gib"] = round(size, 1)

    # --- Derive overall status ---
    has_quality = entry["mmlu"]["status"] == "DONE" or entry["gpqa"]["status"] == "DONE"
    has_tps = any(v is not None for v in entry["single_tps"].values())
    has_conc = any(v is not None for v in entry["concurrent_tps"].values())

    if has_quality and has_tps:
        entry["status"] = "DONE"
        entry["blocker"] = None
    elif has_tps or has_conc:
        entry["status"] = "PARTIAL"
        entry["blocker"] = "Quality benchmarks pending"
    else:
        errors = report.get("errors", [])
        if errors:
            entry["blocker"] = errors[0]
        entry["status"] = "PENDING"

    return entry

    return entry


def _extract_nvfp4(report_dir: Path) -> dict[str, Any]:
    """Extract NVFP4 variant data from matrix/watch reports."""
    entry: dict[str, Any] = {
        "variant": "NVFP4",
        "size_gib": 8.3,
        "stack": "Lynn Engine (CUDA)",
        "mmlu": {"score": None, "correct": None, "total": None, "status": "PENDING"},
        "gpqa": {"score": None, "correct": None, "total": None, "status": "PENDING"},
        "single_tps": {"128": None, "256": None, "512": None},
        "concurrent_tps": {"2": None, "4": None, "8": None},
        "long_context": {"4k": None, "16k": None, "32k": None},
        "status": "PENDING",
        "blocker": "NVFP4 quality/TPS benchmarks pending",
    }

    smoke_path = _latest("r6000_qwen35_9b_nvfp4_dense_runtime_smoke*.json", report_dir)
    if smoke_path:
        smoke = _load_json(smoke_path)
        if smoke and smoke.get("status") == "GENERATION_PASS":
            entry["status"] = "PARTIAL"
            entry["blocker"] = "Generation smoke passes; quality/TPS benchmarks pending"

    # --- Quality from standalone OpenAI evaluator summaries ---
    mmlu_path = _latest("nvfp4*_mmlu*.summary.json", report_dir)
    if mmlu_path:
        mmlu = _load_json(mmlu_path)
        if mmlu:
            acc = mmlu.get("accuracy") or mmlu.get("score")
            if acc is not None:
                entry["mmlu"] = {
                    "score": round(acc, 4),
                    "correct": mmlu.get("correct"),
                    "total": mmlu.get("n") or mmlu.get("total"),
                    "status": "DONE",
                }

    gpqa_path = _latest("nvfp4*_gpqa*.summary.json", report_dir)
    if gpqa_path:
        gpqa = _load_json(gpqa_path)
        if gpqa:
            acc = gpqa.get("accuracy") or gpqa.get("score")
            if acc is not None:
                entry["gpqa"] = {
                    "score": round(acc, 4),
                    "correct": gpqa.get("correct"),
                    "total": gpqa.get("n") or gpqa.get("total"),
                    "status": "DONE",
                }

    # Search for any nvfp4 matrix/watch reports
    nvfp4_paths = list(report_dir.glob("*nvfp4*matrix*.json")) + list(
        report_dir.glob("*nvfp4*watch*.json")
    )
    if nvfp4_paths:
        # Pick the latest
        nvfp4_path = max(nvfp4_paths, key=lambda p: p.stat().st_mtime)
        report = _load_json(nvfp4_path)
        if report is not None:
            # Try to extract data from report if it exists
            file_status = report.get("status", "")
            if file_status == "BLOCKED":
                entry["blocker"] = report.get("blocked_reason", entry["blocker"])
                return entry

            # Extract quality if present
            for metric_name in ("mmlu", "gpqa", "mmlu_500_5shot", "gpqa_diamond"):
                metric_data = report.get(metric_name, {})
                if metric_data:
                    acc = metric_data.get("accuracy") or metric_data.get("score")
                    key = "mmlu" if "mmlu" in metric_name else "gpqa"
                    if acc is not None and entry[key]["status"] != "DONE":
                        entry[key] = {
                            "score": round(acc, 4),
                            "correct": metric_data.get("correct"),
                            "total": metric_data.get("n") or metric_data.get("total"),
                            "status": "DONE",
                        }

            # Extract TPS if present. Lynn server reports use the same generic OpenAI
            # matrix schema as the Q4_K_M llama.cpp baseline.
            if report.get("schema_version") == "openai-serving-matrix-probe-v1":
                for row in report.get("single", {}).get("rows", []):
                    key = str(row.get("max_tokens"))
                    if key in entry["single_tps"] and row.get("ok"):
                        entry["single_tps"][key] = round(row.get("wall_tps", 0.0), 1)
                for row in report.get("concurrency", {}).get("rows", []):
                    key = str(row.get("concurrency"))
                    if key in entry["concurrent_tps"] and row.get("ok"):
                        entry["concurrent_tps"][key] = round(row.get("batch_wall_tps", 0.0), 1)
                by_chars = {
                    str(row.get("target_prompt_chars")): row
                    for row in report.get("long_context", {}).get("rows", [])
                }
                for label, chars in {"4k": "4096", "16k": "16384", "32k": "32768"}.items():
                    row = by_chars.get(chars)
                    if row and row.get("ok"):
                        entry["long_context"][label] = round(row.get("wall_tps", 0.0), 1)
            else:
                single = report.get("single_tps", {})
                if single:
                    for key in ("128", "256", "512"):
                        e = single.get(key) or single.get(f"tps_{key}")
                        if isinstance(e, dict) and e.get("ok"):
                            entry["single_tps"][key] = round(e.get("wall_tps", 0.0), 1)
                        elif isinstance(e, (int, float)):
                            entry["single_tps"][key] = round(float(e), 1)

    # The promoted 9B NVFP4 path is the linear-block graph profile.  Its P150
    # single-stream gate reports decode TPS, while P151 reports wall TPS plus
    # concurrency and long-context smoke.  These files do not use the older
    # `*_openai_matrix*` names, so keep them as first-class release inputs.
    p151_summary_path = _latest("p151_qwen35_9b_nvfp4_linear_graph_matrix_summary_*.json", report_dir)
    if p151_summary_path:
        p151 = _load_json(p151_summary_path)
        if p151:
            for key, value in (p151.get("single_wall_tps") or {}).items():
                if key in entry["single_tps"] and isinstance(value, (int, float)):
                    entry["single_tps"][key] = round(float(value), 1)
            for key, value in (p151.get("concurrent_total_tps") or {}).items():
                if key in entry["concurrent_tps"] and isinstance(value, (int, float)):
                    entry["concurrent_tps"][key] = round(float(value), 1)
            long_map = {"4096": "4k", "16384": "16k", "32768": "32k"}
            for key, value in (p151.get("long_context_wall_tps") or {}).items():
                label = long_map.get(str(key), str(key))
                if label in entry["long_context"] and isinstance(value, (int, float)):
                    entry["long_context"][label] = round(float(value), 1)

    p150_summary_path = _latest("p150_qwen35_9b_nvfp4_linear_graph_summary_*.json", report_dir)
    if p150_summary_path:
        p150 = _load_json(p150_summary_path)
        if p150:
            # Prefer the newer P150 decode-TPS single-stream gate when present;
            # it is the stable service-line number we quote for Lynn-native
            # NVFP4.  P151 remains the source for concurrency and long context.
            for key, value in (p150.get("decode_tps") or {}).items():
                if key in entry["single_tps"] and isinstance(value, (int, float)):
                    entry["single_tps"][key] = round(float(value), 1)

    # Derive status
    has_quality = entry["mmlu"]["status"] == "DONE" or entry["gpqa"]["status"] == "DONE"
    has_tps = any(v is not None for v in entry["single_tps"].values())
    has_conc = any(v is not None for v in entry["concurrent_tps"].values())
    if has_quality and has_tps:
        entry["status"] = "DONE"
        entry["blocker"] = None
    elif has_quality or has_tps or has_conc:
        entry["status"] = "PARTIAL"
        entry["blocker"] = "Quality benchmarks pending" if not has_quality else "Partial NVFP4 data"

    return entry


# ---------------------------------------------------------------------------
# Previous matrix comparison
# ---------------------------------------------------------------------------

def _load_previous_matrix(report_dir: Path) -> dict[str, Any] | None:
    """Load any existing release_matrix JSON for comparison."""
    prev = _latest("*release_matrix*.json", report_dir)
    if prev is None:
        return None
    return _load_json(prev)


def _diff_entries(prev: dict[str, Any] | None, current: list[dict[str, Any]]) -> list[str]:
    """Return human-readable diff lines between previous and current entries."""
    if prev is None:
        return []
    prev_entries = {e["variant"]: e for e in prev.get("entries", [])}
    diffs: list[str] = []
    for entry in current:
        v = entry["variant"]
        old = prev_entries.get(v)
        if old is None:
            diffs.append(f"- **{v}**: NEW variant (not in previous matrix)")
            continue
        # Check status change
        if old.get("status") != entry["status"]:
            diffs.append(f"- **{v}**: status {old['status']} -> {entry['status']}")
        # Check quality changes
        for metric in ("mmlu", "gpqa"):
            old_score = old.get(metric, {}).get("score")
            new_score = entry.get(metric, {}).get("score")
            if old_score != new_score:
                diffs.append(f"- **{v}** {metric.upper()}: {old_score} -> {new_score}")
    return diffs


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_matrix_json(report_dir: Path) -> dict[str, Any]:
    """Build the complete release matrix JSON."""
    entries = [
        _extract_bf16(report_dir),
        _extract_q4km(report_dir),
        _extract_nvfp4(report_dir),
    ]
    return {
        "schema": "lynn-qwen35-9b-release-matrix-v1",
        "created": _iso_now(),
        "model_id": "Qwen3.5-9B",
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_STATUS_ICON = {
    "DONE": "\U0001f7e2 DONE",        # green circle
    "PARTIAL": "\U0001f7e1 PARTIAL",   # yellow circle
    "PENDING": "\u23f3 PENDING",       # hourglass
    "BLOCKED": "\U0001f534 BLOCKED",   # red circle
}


def _status_icon(status: str) -> str:
    return _STATUS_ICON.get(status, status)


def _score_or_dash(metric: dict[str, Any]) -> str:
    """Render a metric's score as a formatted string or dash."""
    if metric.get("status") == "DONE" and metric.get("score") is not None:
        return f"{metric['score']:.3f}"
    return "\u2014"


def _tps_summary(tps: dict[str, Any | None]) -> str:
    """Render TPS dict as a compact summary string."""
    parts: list[str] = []
    for key in sorted(tps.keys(), key=lambda x: int(x)):
        val = tps[key]
        if val is not None:
            parts.append(f"{key}t={val:.1f}")
    return " / ".join(parts) if parts else "\u2014"


def _concurrent_summary(tps: dict[str, Any | None]) -> str:
    """Render concurrent TPS dict as a compact summary string."""
    parts: list[str] = []
    for key in sorted(tps.keys(), key=lambda x: int(x)):
        val = tps[key]
        if val is not None:
            parts.append(f"x{key}={val:.1f}")
    return " / ".join(parts) if parts else "\u2014"


def _lc_summary(lc: dict[str, Any | None]) -> str:
    """Render long-context dict as a compact summary string."""
    parts: list[str] = []
    for key in ("4k", "16k", "32k"):
        val = lc.get(key)
        if val is not None:
            parts.append(f"{key}={val:.1f}")
    return " / ".join(parts) if parts else "\u2014"


def render_markdown(matrix: dict[str, Any]) -> str:
    """Render the release matrix as a Markdown document."""
    entries = matrix.get("entries", [])
    created = matrix.get("created", "unknown")
    model_id = matrix.get("model_id", "Qwen3.5-9B")

    lines: list[str] = []
    lines.append(f"# {model_id} Release Matrix")
    lines.append("")
    lines.append(f"**Generated:** {created}  ")
    lines.append(f"**Schema:** `{matrix.get('schema', 'unknown')}`  ")
    lines.append("")
    lines.append(
        "> Cross-variant benchmark matrix for BF16 (quality ceiling), "
        "Q4_K_M (llama.cpp / Mac), and NVFP4 (Lynn Engine / NVIDIA Blackwell)."
    )
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary Matrix")
    lines.append("")
    lines.append(
        "| Variant | Size (GiB) | Stack | Status | MMLU | GPQA | Single TPS | Concurrent TPS | Long Context | Blocker |"
    )
    lines.append(
        "|---------|-----------|-------|--------|------|------|------------|----------------|--------------|---------|"
    )
    for e in entries:
        variant = e["variant"]
        size = e.get("size_gib", "\u2014")
        stack = e.get("stack", "\u2014")
        status = _status_icon(e.get("status", "PENDING"))
        mmlu = _score_or_dash(e.get("mmlu", {}))
        gpqa = _score_or_dash(e.get("gpqa", {}))
        stps = _tps_summary(e.get("single_tps", {}))
        ctps = _concurrent_summary(e.get("concurrent_tps", {}))
        lc = _lc_summary(e.get("long_context", {}))
        blocker = e.get("blocker") or "\u2014"
        lines.append(
            f"| **{variant}** | {size} | {stack} | {status} | {mmlu} | {gpqa} "
            f"| {stps} | {ctps} | {lc} | {blocker} |"
        )
    lines.append("")

    # --- Per-variant detail sections ---
    for e in entries:
        variant = e["variant"]
        lines.append(f"## {variant} Details")
        lines.append("")
        stack_value = e.get("stack", "—")
        size_value = e.get("size_gib", "—")
        lines.append(f"- **Stack:** {stack_value}")
        lines.append(f"- **Size:** {size_value} GiB")
        lines.append(f"- **Status:** {_status_icon(e.get('status', 'PENDING'))}")
        if e.get("blocker"):
            lines.append(f"- **Blocker:** {e['blocker']}")
        lines.append("")

        # Quality sub-table
        lines.append("| Metric | Score | Correct | Total | Status |")
        lines.append("|--------|-------|---------|-------|--------|")
        for metric_name in ("mmlu", "gpqa"):
            m = e.get(metric_name, {})
            score = f"{m['score']:.4f}" if m.get("score") is not None else "\u2014"
            correct = m.get("correct") if m.get("correct") is not None else "\u2014"
            total = m.get("total") if m.get("total") is not None else "\u2014"
            mstatus = _status_icon(m.get("status", "PENDING"))
            lines.append(f"| {metric_name.upper()} | {score} | {correct} | {total} | {mstatus} |")
        lines.append("")

        # TPS sub-table
        lines.append("| TPS Type | 128 tok / 2 concurrency / 4k ctx | 256 tok / 4 concurrency / 16k ctx | 512 tok / 8 concurrency / 32k ctx |")
        lines.append("|----------|-----------------------------------|-----------------------------------|-----------------------------------|")

        stps = e.get("single_tps", {})
        ctps = e.get("concurrent_tps", {})
        lc = e.get("long_context", {})

        def _val(d: dict, k: str) -> str:
            v = d.get(k)
            return f"{v:.1f} TPS" if v is not None else "\u2014"

        lines.append(
            f"| Single TPS | {_val(stps, '128')} | {_val(stps, '256')} | {_val(stps, '512')} |"
        )
        lines.append(
            f"| Concurrent TPS | {_val(ctps, '2')} | {_val(ctps, '4')} | {_val(ctps, '8')} |"
        )
        lines.append(
            f"| Long Context | {_val(lc, '4k')} | {_val(lc, '16k')} | {_val(lc, '32k')} |"
        )
        lines.append("")

    # --- Legend ---
    lines.append("## Status Legend")
    lines.append("")
    lines.append("| Icon | Meaning |")
    lines.append("|------|---------|")
    lines.append("| \U0001f7e2 DONE | All benchmarks complete |")
    lines.append("| \U0001f7e1 PARTIAL | Some benchmarks complete, more pending |")
    lines.append("| \u23f3 PENDING | No benchmark data yet |")
    lines.append("| \U0001f534 BLOCKED | Blocked by known issue |")
    lines.append("")

    # --- Source files ---
    lines.append("## Source Report Files")
    lines.append("")
    lines.append("This matrix was auto-generated from the following report patterns:")
    lines.append("")
    lines.append("- `bf16_*_quality_summary.json` / `bf16_*_mmlu*.summary.json` / `bf16_*_gpqa*.summary.json`")
    lines.append("- `r6000_qwen35_9b_q4km_baseline_*.json`")
    lines.append("- `*nvfp4*matrix*.json` / `*nvfp4*watch*.json` / `nvfp4*_mmlu*.summary.json` / `nvfp4*_gpqa*.summary.json`")
    lines.append("- `*release_matrix*.json` (previous run, for diff)")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by `scripts/summarize_qwen35_9b_release_matrix.py`*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Qwen3.5-9B benchmark reports into a combined release matrix (JSON + Markdown)."
    )
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Directory containing Qwen3.5-9B benchmark reports (e.g. reports/qwen35_9b)",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output path for the release matrix JSON",
    )
    parser.add_argument(
        "--output-md",
        required=True,
        help="Output path for the release matrix Markdown",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    json_out = Path(args.output_json)
    md_out = Path(args.output_md)

    if not report_dir.is_dir():
        print(f"[ERR] Report directory not found: {report_dir}", file=sys.stderr)
        return 1

    # --- Build matrix ---
    matrix = build_matrix_json(report_dir)

    # --- Diff against previous ---
    prev = _load_previous_matrix(report_dir)
    diffs = _diff_entries(prev, matrix["entries"])
    if diffs:
        print("[INFO] Changes since previous matrix:", file=sys.stderr)
        for d in diffs:
            print(f"  {d}", file=sys.stderr)

    # --- Write JSON ---
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[matrix] Wrote JSON: {json_out}")

    # --- Write Markdown ---
    md = render_markdown(matrix)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md, encoding="utf-8")
    print(f"[matrix] Wrote Markdown: {md_out}")

    # --- Summary ---
    for entry in matrix["entries"]:
        v = entry["variant"]
        s = entry["status"]
        blocker = entry.get("blocker") or ""
        blk_str = f" — {blocker}" if blocker else ""
        print(f"  {v}: {s}{blk_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
