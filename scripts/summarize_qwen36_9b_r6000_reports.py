#!/usr/bin/env python3
"""
summarize_qwen36_9b_r6000_reports.py
Merge R6000 benchmark JSON reports into the qwen36_9b_dense_matrix_schema_v1 format
and emit a preview Markdown report.

Supports empty/missing report directories — outputs a scaffold preview with
PENDING status for all stages.

Usage:
    python3 summarize_qwen36_9b_r6000_reports.py \
        --schema reports/qwen36_9b/qwen36_9b_dense_matrix_schema_v1.json \
        --reports ./r6000_9b_reports \
        --out docs/QWEN36_9B_R6000_PIPELINE_20260518.md
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional


def load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Failed to load {path}: {exc}", file=sys.stderr)
        return None


def extract_metrics(report: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Best-effort metric extraction from arbitrary report JSON."""
    if report is None:
        return {}
    out: dict[str, Any] = {}
    # Flatten common nested keys
    flat: dict[str, Any] = {}
    for k, v in report.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}.{sk}"] = sv
        flat[k] = v

    # MMLU
    for key in ("mmlu", "results.mmlu", "mmlu_score", "mmlu.accuracy", "mmlu.mean"):
        if key in flat and flat[key] is not None:
            out["mmlu"] = float(flat[key])
            break

    # GPQA
    for key in ("gpqa", "results.gpqa", "gpqa_score", "gpqa.accuracy", "gpqa.mean"):
        if key in flat and flat[key] is not None:
            out["gpqa"] = float(flat[key])
            break

    # Single TPS
    for key in ("single_tps", "tps.single", "tps", "throughput.tokens_per_sec",
                "bench.single_tps", "results.single_tps"):
        if key in flat and flat[key] is not None:
            out["single_tps"] = float(flat[key])
            break

    # Concurrent TPS
    for key in ("concurrent_tps", "tps.concurrent", "bench.concurrent_tps",
                "results.concurrent_tps"):
        if key in flat and flat[key] is not None:
            out["concurrent_tps"] = float(flat[key])
            break

    # Size
    for key in ("size_gb", "model.size_gb", "metadata.size_gb"):
        if key in flat and flat[key] is not None:
            out["size_gb"] = float(flat[key])
            break

    return out


def merge_into_schema(schema: dict, reports_dir: Path) -> dict:
    """Merge report JSONs into schema entries. Returns mutated schema copy."""
    schema = dict(schema)
    entries = [dict(e) for e in schema.get("entries", [])]

    # Map quant -> entry index
    def norm_quant(value: str) -> str:
        return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")

    quant_map: dict[str, int] = {}
    for i, e in enumerate(entries):
        q = norm_quant(e.get("quant", ""))
        quant_map[q] = i

    # Try to load each expected report
    report_files = {
        "bf16": reports_dir / "bf16_report.json",
        "q4_k_m": reports_dir / "q4km_llamacpp_report.json",
        "w4a16_nvfp4": reports_dir / "w4a16_nvfp4_report.json",
    }

    for quant_key, path in report_files.items():
        report = load_json(path)
        metrics = extract_metrics(report)

        entry_idx = quant_map.get(quant_key)
        if entry_idx is None:
            continue

        entry = entries[entry_idx]
        updated = False
        for field in ("mmlu", "gpqa", "single_tps", "concurrent_tps", "size_gb"):
            if field in metrics:
                entry[field] = metrics[field]
                updated = True

        if updated:
            entry["provisional"] = False
            entry["source"] = f"r6000 pipeline {path.name}"
            entry["verdict"] = "R6000_VERIFIED"
        elif report is not None:
            entry["source"] = f"r6000 pipeline {path.name} (no metrics extracted)"
            entry["verdict"] = "PARTIAL"

    schema["entries"] = entries
    schema["r6000_pipeline_run"] = str(Path.cwd() / reports_dir)
    return schema


def fmt_val(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def render_markdown(schema: dict) -> str:
    lines: list[str] = []
    now = schema.get("created", "2026-05-18")
    lines.append("# Qwen3.6-9B-Dense R6000 Pipeline Report")
    lines.append("")
    lines.append(f"**Generated:** {now}  ")
    lines.append(f"**Schema:** {schema.get('schema', 'unknown')}  ")
    lines.append("")
    lines.append("> **Note:** 9B Dense is the endpoint / 16G VRAM branch candidate. ")
    lines.append("> Q4_K_M targets Mac/llama.cpp; NVFP4 targets NVIDIA Blackwell / Lynn Engine.")
    lines.append("")

    # Dense entries only
    dense_entries = [e for e in schema.get("entries", []) if e.get("arch") == "dense"]

    # Table 1: Quality
    lines.append("## Quality Metrics")
    lines.append("")
    lines.append("| Model | Quant | Runtime | MMLU | GPQA | Status |")
    lines.append("|-------|-------|---------|------|------|--------|")
    for e in dense_entries:
        badge = "🔶 PROVISIONAL" if e.get("provisional") else "✅ VERIFIED"
        if e.get("verdict", "").startswith("PENDING"):
            badge = "⏳ PENDING"
        lines.append(
            f"| {e.get('model_id', '')} | {e.get('quant', '')} | {e.get('runtime', '')} | "
            f"{fmt_val(e.get('mmlu'))} | {fmt_val(e.get('gpqa'))} | {badge} |"
        )
    lines.append("")

    # Table 2: Performance
    lines.append("## Performance (R6000)")
    lines.append("")
    lines.append("| Model | Quant | Single TPS | Concurrent TPS | Size (GB) | Device |")
    lines.append("|-------|-------|------------|----------------|-----------|--------|")
    for e in dense_entries:
        lines.append(
            f"| {e.get('model_id', '')} | {e.get('quant', '')} | "
            f"{fmt_val(e.get('single_tps'))} | {fmt_val(e.get('concurrent_tps'))} | "
            f"{fmt_val(e.get('size_gb'))} | {e.get('device_class', '')} |"
        )
    lines.append("")

    # Table 3: Pipeline stages
    lines.append("## Pipeline Stages")
    lines.append("")
    lines.append("| Stage | Quant | Report File | Status |")
    lines.append("|-------|-------|-------------|--------|")
    stage_map = {
        "BF16": "bf16_report.json",
        "Q4_K_M": "q4km_llamacpp_report.json",
        "W4A16 / NVFP4": "w4a16_nvfp4_report.json",
    }
    reports_dir = Path(schema.get("r6000_pipeline_run", "./r6000_9b_reports"))
    for quant, filename in stage_map.items():
        path = reports_dir / filename
        status = "✅ Found" if path.exists() else "⏳ Pending"
        lines.append(f"| {quant} | {quant} | `{filename}` | {status} |")
    lines.append("")

    # Product positioning
    lines.append("## Product Positioning")
    lines.append("")
    lines.append("- **BF16**: Full-precision baseline. Quality ceiling for all quantized variants.")
    lines.append("  Target: MMLU/GPQA within 0.5% of theoretical maximum.")
    lines.append("- **Q4_K_M**: Mac / Apple Silicon endpoint. 5.2 GB, ~81 MMLU (provisional).")
    lines.append("  llama.cpp GGUF path; no NVIDIA dependency.")
    lines.append("- **W4A16 / NVFP4**: NVIDIA Blackwell (SM100+) serving path. ~5.5 GB.")
    lines.append("  Lynn-native packed decode + TensorCore MMA. Target TPS competitive with Q4_K_M on R6000.")
    lines.append("")

    lines.append("---")
    lines.append("*Report generated by `summarize_qwen36_9b_r6000_reports.py`*")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge R6000 reports into schema preview")
    parser.add_argument("--schema", required=True, help="Path to qwen36_9b_dense_matrix_schema_v1.json")
    parser.add_argument("--reports", required=True, help="Directory containing JSON report files")
    parser.add_argument("--out", required=True, help="Output Markdown path")
    args = parser.parse_args()

    schema_path = Path(args.schema)
    reports_dir = Path(args.reports)
    out_path = Path(args.out)

    schema = load_json(schema_path)
    if schema is None:
        log_err = lambda msg: print(f"[ERR] {msg}", file=sys.stderr)
        log_err(f"Schema not found: {schema_path}")
        return 1

    # Merge reports (safe even if reports_dir is empty / missing)
    merged = merge_into_schema(schema, reports_dir)

    # Render markdown
    md = render_markdown(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[matrix] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
