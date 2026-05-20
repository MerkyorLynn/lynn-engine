#!/usr/bin/env python3
"""Spark W4A16 vs W4A8/FP8 end-to-end TPS smoke harness.

Runs the canonical six prompt smoke set against the current Lynn-native W4A16
NVFP4 graph baseline and the upcoming W4A8/FP8 path, at ``max_new`` 64 and
128 by default. The FP8 model dir and resident-runner integration are expected
to be missing until Phase 2 repack + kernel work lands; FP8 config failures are
captured as structured per-config errors so the W4A16 baseline can still run and
write a report.

Usage on Spark::

    /home/merkyor/comfyui/ComfyUI/.venv/bin/python \
        scripts/spark_fp8_e2e_tps_smoke.py

By default reports are written as::

    reports/mtp/spark_fp8_e2e_tps_smoke_<YYYYMMDD_HHMMSS>.json
    reports/mtp/spark_fp8_e2e_tps_smoke_<YYYYMMDD_HHMMSS>.md

JSON schema (``schema_version = lynn-fp8-e2e-tps-smoke-v1``)::

    {
      "schema_version": str,
      "created_at_utc": str,
      "models": {
        "w4a16": str,
        "w4a8_fp8": str
      },
      "max_new_values": [int, ...],
      "prompts": [{"prompt_id": str, "prompt": str}, ...],
      "base_env": {str: str},
      "configs": [
        {
          "label": str,
          "model_dir": str,
          "env_overrides": {str: str | null},
          "runs": [
            {
              "max_new": int,
              "status": "ok" | "error",
              "error": null | {
                "type": str,
                "message": str,
                "traceback_tail": [str, ...]
              },
              "rows": [
                {
                  "prompt_id": str,
                  "prompt": str,
                  "new_ids": [int, ...],
                  "new_ids_head": [int, ...],
                  "completion_head": str,
                  "exact_match": bool | null,
                  "prefix_match_len": int | null,
                  "wall_seconds": float,
                  "prefill_seconds": float | null,
                  "decode_tps": float | null,
                  "decode_step_seconds": [float, ...],
                  "stopped_reason": str
                }, ...
              ],
              "summary": {
                "n_prompts": int,
                "exact_match_count": int | null,
                "exact_match_rate": float | null,
                "mean_prefix_match_len": float | null,
                "mean_decode_tps": float | null,
                "mean_wall_seconds": float | null
              }
            }, ...
          ],
          "aggregate_summary": {
            "ok_runs": int,
            "error_runs": int,
            "mean_decode_tps_by_max_new": {str: float | null}
          }
        }, ...
      ],
      "summary": {
        "tps_lift": [
          {
            "label": str,
            "max_new": int,
            "status": "ok" | "error" | "missing_baseline",
            "baseline_tps": float | null,
            "mean_decode_tps": float | null,
            "tps_lift_ratio": float | null,
            "tps_lift_pct": float | null,
            "exact_match_rate": float | null,
            "mean_prefix_match_len": float | null,
            "error": null | {"type": str, "message": str}
          }, ...
        ]
      },
      "gates": {
        "w4a16_baseline_graph_ok": bool,
        "fp8_errors_graceful": bool,
        "all_completed_non_baseline_runs_exact": bool | null
      }
    }
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


DEFAULT_W4A16_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000"
DEFAULT_W4A8_FP8_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8"

DEFAULT_PROMPTS = [
    "Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.",
    "用一句话解释 speculative decoding 的核心思想。",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "请输出一个 JSON: {\"city\": \"Tokyo\", \"unit\": \"celsius\"}",
    "Summarize the role of the MoE router in one paragraph.",
]

# Spark Config D-style serving env: current production W4A16 graph baseline plus
# the stable native NVFP4 decode knobs used by the existing smoke runner.
BASE_ENV = {
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_MOE_FAST_FIXED": "1",
    "LYNN_NATIVE_DOWN_BACKEND": "triton",
    "LYNN_ROUTER_TOPK_SORTED": "0",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_FULL_TOKEN_GRAPH_SLOT": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _set_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    """Apply env updates, return previous values for restore."""
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _prefix_match_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _error_payload(exc: BaseException) -> dict[str, Any]:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": [line.rstrip("\n") for line in tb[-8:]],
    }


def _summarize_rows(rows: list[dict[str, Any]], *, is_baseline: bool) -> dict[str, Any]:
    n_rows = len(rows)
    exact_values = [row["exact_match"] for row in rows if row["exact_match"] is not None]
    exact_count = sum(1 for value in exact_values if value)
    prefix_lens = [row["prefix_match_len"] for row in rows if row["prefix_match_len"] is not None]
    tps_values = [row["decode_tps"] for row in rows if row["decode_tps"]]
    wall_values = [row["wall_seconds"] for row in rows]
    return {
        "n_prompts": n_rows,
        "exact_match_count": n_rows if is_baseline else (exact_count if exact_values else None),
        "exact_match_rate": 1.0 if is_baseline and n_rows else (
            exact_count / len(exact_values) if exact_values else None
        ),
        "mean_prefix_match_len": _mean([float(x) for x in prefix_lens]),
        "mean_decode_tps": _mean([float(x) for x in tps_values]),
        "mean_wall_seconds": _mean([float(x) for x in wall_values]),
    }


def _run_generation_sweep(
    runner: LynnIncrementalRunner,
    *,
    label: str,
    prompts: list[str],
    max_new_values: list[int],
    baseline_by_max_new: dict[int, list[list[int]]],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    is_baseline = label == "w4a16_baseline_graph"
    for max_new in max_new_values:
        try:
            baseline_ids = baseline_by_max_new.get(max_new)
            rows: list[dict[str, Any]] = []
            for idx, prompt in enumerate(prompts):
                t0 = time.time()
                out = runner.generate(prompt, max_new=max_new)
                if runner.device.startswith("cuda"):
                    torch.cuda.synchronize()
                wall = time.time() - t0
                new_ids = [int(x) for x in out["new_ids"]]
                base_ids = None if baseline_ids is None else baseline_ids[idx]
                timings = out.get("timings", {}) or {}
                row = {
                    "prompt_id": f"prompt_{idx:03d}",
                    "prompt": prompt,
                    "new_ids": new_ids,
                    "new_ids_head": new_ids[:24],
                    "completion_head": str(out.get("completion_text", ""))[:240],
                    "exact_match": True if is_baseline else (None if base_ids is None else new_ids == base_ids),
                    "prefix_match_len": len(new_ids) if is_baseline else (
                        None if base_ids is None else _prefix_match_len(new_ids, base_ids)
                    ),
                    "wall_seconds": wall,
                    "prefill_seconds": timings.get("prefill_seconds"),
                    "decode_tps": timings.get("decode_tps"),
                    "decode_step_seconds": timings.get("decode_step_seconds", []),
                    "stopped_reason": out.get("stopped_reason"),
                }
                rows.append(row)
            if is_baseline:
                baseline_by_max_new[max_new] = [row["new_ids"] for row in rows]
            runs.append({
                "max_new": max_new,
                "status": "ok",
                "error": None,
                "rows": rows,
                "summary": _summarize_rows(rows, is_baseline=is_baseline),
            })
        except Exception as exc:  # keep later configs/max_new values running
            runs.append({
                "max_new": max_new,
                "status": "error",
                "error": _error_payload(exc),
                "rows": [],
                "summary": _summarize_rows([], is_baseline=False),
            })
    return runs


def _run_config(
    config: dict[str, Any],
    *,
    prompts: list[str],
    max_new_values: list[int],
    dtype: torch.dtype,
    device: str,
    baseline_by_max_new: dict[int, list[list[int]]],
) -> dict[str, Any]:
    label = str(config["label"])
    model_dir = str(config["model_dir"])
    env_updates: dict[str, str | None] = dict(BASE_ENV)
    env_updates.update(config["env"])
    previous = _set_env(env_updates)
    runner: LynnIncrementalRunner | None = None
    try:
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"model_dir does not exist yet: {model_dir}. "
                "This is expected for the W4A8/FP8 configs until the Phase 2 "
                "repack V1 artifact is produced."
            )
        runner = LynnIncrementalRunner(
            model_dir,
            device=device,
            dtype=dtype,
            verbose=False,
        )
        runs = _run_generation_sweep(
            runner,
            label=label,
            prompts=prompts,
            max_new_values=max_new_values,
            baseline_by_max_new=baseline_by_max_new,
        )
    except Exception as exc:
        error = _error_payload(exc)
        runs = [
            {
                "max_new": max_new,
                "status": "error",
                "error": error,
                "rows": [],
                "summary": _summarize_rows([], is_baseline=False),
            }
            for max_new in max_new_values
        ]
    finally:
        del runner
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _restore_env(previous)

    ok_runs = sum(1 for run in runs if run["status"] == "ok")
    mean_by_max_new = {
        str(run["max_new"]): run["summary"].get("mean_decode_tps")
        for run in runs
    }
    return {
        "label": label,
        "model_dir": model_dir,
        "env_overrides": config["env"],
        "runs": runs,
        "aggregate_summary": {
            "ok_runs": ok_runs,
            "error_runs": len(runs) - ok_runs,
            "mean_decode_tps_by_max_new": mean_by_max_new,
        },
    }


def _build_configs(w4a16_model: str, fp8_model: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "w4a16_baseline_graph",
            "model_dir": w4a16_model,
            "env": {
                "LYNN_W4A8_FP8_PATH": None,
                "LYNN_LINEAR_BLOCK_GRAPH": "1",
                "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
                "LYNN_MTP_SPECULATIVE": "0",
                "LYNN_MTP_SHADOW_VERIFY": "0",
                "LYNN_PACKED_DECODE": "0",
                "LYNN_PACKED_SHARED_EXPERT": "0",
            },
        },
        {
            "label": "w4a8_fp8_baseline_graph",
            "model_dir": fp8_model,
            "env": {
                "LYNN_W4A8_FP8_PATH": "1",
                # FP8 active-expert iteration uses torch.unique(...).tolist()
                # which is a host sync incompatible with CUDA graph capture.
                # Until the FP8 path becomes graph-capture-friendly (V3 scope),
                # both FP8 configs run in eager mode. Keeping the label
                # "_graph" for output schema parity with future graph-enabled
                # versions.
                "LYNN_LINEAR_BLOCK_GRAPH": "0",
                "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
                "LYNN_MTP_SPECULATIVE": "0",
                "LYNN_MTP_SHADOW_VERIFY": "0",
                # FP8 dir has no NVFP4-format tensors — disable every NVFP4
                # native runtime knob that would otherwise read `packed_key`
                # / `scale_key` / `global_scale_key` from a V1 NVFP4 manifest.
                "LYNN_MOE_IMPL": None,
                "LYNN_MOE_FAST_FIXED": None,
                "LYNN_NATIVE_DOWN_BACKEND": None,
                "LYNN_NATIVE_FP4_LM_HEAD": "0",
                "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "0",
                "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": None,
                "LYNN_PACKED_DECODE_BACKEND": None,
                "LYNN_PACKED_DECODE": None,
                "LYNN_PACKED_SHARED_EXPERT": "0",
                "LYNN_FULL_TOKEN_GRAPH_SLOT": None,
                "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": None,
                "LYNN_ROUTER_TOPK_SORTED": None,
            },
        },
        {
            "label": "w4a8_fp8_baseline_eager",
            "model_dir": fp8_model,
            "env": {
                "LYNN_W4A8_FP8_PATH": "1",
                "LYNN_LINEAR_BLOCK_GRAPH": "0",
                "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
                "LYNN_MTP_SPECULATIVE": "0",
                "LYNN_MTP_SHADOW_VERIFY": "0",
                "LYNN_MOE_IMPL": None,
                "LYNN_MOE_FAST_FIXED": None,
                "LYNN_NATIVE_DOWN_BACKEND": None,
                "LYNN_NATIVE_FP4_LM_HEAD": "0",
                "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "0",
                "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": None,
                "LYNN_PACKED_DECODE_BACKEND": None,
                "LYNN_PACKED_DECODE": None,
                "LYNN_PACKED_SHARED_EXPERT": "0",
                "LYNN_FULL_TOKEN_GRAPH_SLOT": None,
                "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": None,
                "LYNN_ROUTER_TOPK_SORTED": None,
            },
        },
    ]


def _derive_tps_lift(configs: list[dict[str, Any]], max_new_values: list[int]) -> list[dict[str, Any]]:
    baseline = next((case for case in configs if case["label"] == "w4a16_baseline_graph"), None)
    baseline_by_max_new = {}
    if baseline:
        baseline_by_max_new = {run["max_new"]: run for run in baseline["runs"]}

    rows: list[dict[str, Any]] = []
    for case in configs:
        for run in case["runs"]:
            max_new = int(run["max_new"])
            base_run = baseline_by_max_new.get(max_new)
            baseline_tps = None
            if base_run and base_run["status"] == "ok":
                baseline_tps = base_run["summary"].get("mean_decode_tps")
            mean_tps = run["summary"].get("mean_decode_tps") if run["status"] == "ok" else None
            if run["status"] != "ok":
                status = "error"
            elif baseline_tps is None:
                status = "missing_baseline"
            else:
                status = "ok"
            ratio = mean_tps / baseline_tps if (mean_tps and baseline_tps) else None
            error = run.get("error")
            rows.append({
                "label": case["label"],
                "max_new": max_new,
                "status": status,
                "baseline_tps": baseline_tps,
                "mean_decode_tps": mean_tps,
                "tps_lift_ratio": ratio,
                "tps_lift_pct": ((ratio - 1.0) * 100.0) if ratio is not None else None,
                "exact_match_rate": run["summary"].get("exact_match_rate"),
                "mean_prefix_match_len": run["summary"].get("mean_prefix_match_len"),
                "error": None if error is None else {
                    "type": error.get("type"),
                    "message": error.get("message"),
                },
            })
    rows.sort(key=lambda item: (int(item["max_new"]), str(item["label"])))
    expected = {(label, max_new) for label in [case["label"] for case in configs] for max_new in max_new_values}
    seen = {(row["label"], row["max_new"]) for row in rows}
    for label, max_new in sorted(expected - seen):
        rows.append({
            "label": label,
            "max_new": max_new,
            "status": "error",
            "baseline_tps": None,
            "mean_decode_tps": None,
            "tps_lift_ratio": None,
            "tps_lift_pct": None,
            "exact_match_rate": None,
            "mean_prefix_match_len": None,
            "error": {"type": "MissingRun", "message": "config did not produce this max_new run"},
        })
    return rows


def _derive_gates(configs: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((case for case in configs if case["label"] == "w4a16_baseline_graph"), None)
    baseline_ok = bool(baseline) and all(run["status"] == "ok" for run in baseline["runs"])
    fp8_cases = [case for case in configs if case["label"].startswith("w4a8_fp8")]
    fp8_errors_graceful = all(
        run["status"] == "ok" or bool(run.get("error", {}).get("message"))
        for case in fp8_cases
        for run in case["runs"]
    )
    completed_non_baseline = [
        run
        for case in configs
        if case["label"] != "w4a16_baseline_graph"
        for run in case["runs"]
        if run["status"] == "ok"
    ]
    if completed_non_baseline:
        exact_gate: bool | None = all(
            run["summary"].get("exact_match_count") == run["summary"].get("n_prompts")
            for run in completed_non_baseline
        )
    else:
        exact_gate = None
    return {
        "w4a16_baseline_graph_ok": baseline_ok,
        "fp8_errors_graceful": fp8_errors_graceful,
        "all_completed_non_baseline_runs_exact": exact_gate,
    }


def _write_markdown(report: dict[str, Any], md_path: Path) -> None:
    lines = [
        "# Spark FP8 end-to-end TPS smoke",
        "",
        f"- JSON: `{md_path.with_suffix('.json').name}`",
        f"- Schema: `{report['schema_version']}`",
        f"- W4A16 model: `{report['models']['w4a16']}`",
        f"- W4A8/FP8 model: `{report['models']['w4a8_fp8']}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in report["gates"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## TPS lift table",
        "",
        "| max_new | config | status | mean TPS | lift vs W4A16 | exact | prefix | error |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ])
    for row in report["summary"]["tps_lift"]:
        mean_tps = row["mean_decode_tps"]
        lift = row["tps_lift_ratio"]
        exact = row["exact_match_rate"]
        prefix = row["mean_prefix_match_len"]
        error = row["error"]["message"] if row.get("error") else ""
        if len(error) > 120:
            error = error[:117] + "..."
        lines.append(
            "| {max_new} | `{label}` | `{status}` | {mean_tps} | {lift} | {exact} | {prefix} | {error} |".format(
                max_new=row["max_new"],
                label=row["label"],
                status=row["status"],
                mean_tps="" if mean_tps is None else f"{mean_tps:.3f}",
                lift="" if lift is None else f"{lift:.3f}x",
                exact="" if exact is None else f"{exact:.3f}",
                prefix="" if prefix is None else f"{prefix:.1f}",
                error=error.replace("|", "\\|"),
            )
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _default_out_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return ROOT / "reports" / "mtp" / f"spark_fp8_e2e_tps_smoke_{stamp}.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w4a16-model", default=DEFAULT_W4A16_MODEL)
    ap.add_argument("--w4a8-fp8-model", default=DEFAULT_W4A8_FP8_MODEL)
    ap.add_argument("--out", default=None, help="JSON report path; defaults to reports/mtp/spark_fp8_e2e_tps_smoke_<date>.json")
    ap.add_argument("--max-new", type=int, nargs="+", default=[64, 128], help="One or more max_new values")
    ap.add_argument("--prompts-json", default=None, help="Optional JSON list of prompts or {prompt: ...} objects")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_json:
        raw = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        prompts = [str(item["prompt"]) if isinstance(item, dict) else str(item) for item in raw]
    if not prompts:
        raise SystemExit("[fp8-smoke] prompt list is empty")

    max_new_values = [int(x) for x in args.max_new]
    if any(x <= 0 for x in max_new_values):
        raise SystemExit("[fp8-smoke] --max-new values must be positive")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]
    configs = _build_configs(args.w4a16_model, args.w4a8_fp8_model)
    baseline_by_max_new: dict[int, list[list[int]]] = {}

    cases: list[dict[str, Any]] = []
    for config in configs:
        print(f"[fp8-smoke] running {config['label']} from {config['model_dir']}", flush=True)
        case = _run_config(
            config,
            prompts=prompts,
            max_new_values=max_new_values,
            dtype=dtype,
            device=args.device,
            baseline_by_max_new=baseline_by_max_new,
        )
        cases.append(case)
        for run in case["runs"]:
            if run["status"] == "ok":
                tps = run["summary"].get("mean_decode_tps")
                print(f"[fp8-smoke]   max_new={run['max_new']} mean_decode_tps={tps}", flush=True)
            else:
                print(
                    f"[fp8-smoke]   max_new={run['max_new']} ERROR: "
                    f"{run['error']['type']}: {run['error']['message']}",
                    flush=True,
                )

    report = {
        "schema_version": "lynn-fp8-e2e-tps-smoke-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            "w4a16": args.w4a16_model,
            "w4a8_fp8": args.w4a8_fp8_model,
        },
        "max_new_values": max_new_values,
        "prompts": [
            {"prompt_id": f"prompt_{idx:03d}", "prompt": prompt}
            for idx, prompt in enumerate(prompts)
        ],
        "base_env": BASE_ENV,
        "configs": cases,
        "summary": {"tps_lift": _derive_tps_lift(cases, max_new_values)},
        "gates": _derive_gates(cases),
    }

    out_path = Path(args.out) if args.out else _default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    _write_markdown(report, md_path)

    print(f"[fp8-smoke] wrote {out_path}", flush=True)
    print(f"[fp8-smoke] wrote {md_path}", flush=True)
    print(f"[fp8-smoke] gates = {json.dumps(report['gates'], sort_keys=True)}", flush=True)
    return 0 if report["gates"]["w4a16_baseline_graph_ok"] and report["gates"]["fp8_errors_graceful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
