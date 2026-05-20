#!/usr/bin/env python3
"""W4A8 FP8 vs W4A16 NVFP4 quality-regression evaluator for Qwen3.6-35B.

Runs MMLU-100 (5-shot) and GPQA-Diamond-50 through two Lynn-native model
dirs and emits per-question + summary JSON comparing accuracy and generation
agreement.  The W4A8 FP8 leg is a placeholder until the repack V1 artifact
lands; the script is designed to run cleanly with model-A only.

Designed to run on **Spark** via ``ssh dgx-via-n5`` inside the lynn-engine
worktree.  Uses ``LynnIncrementalRunner`` (same path as
``spark_mtp_speculative_smoke.py``) for deterministic greedy generation.

Usage::

    # Baseline only (W4A8 dir not yet available)
    python scripts/spark_w4a8_vs_w4a16_quality_regression.py \\
        --model-a /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \\
        --out-dir reports/mtp/fp8_quality_regression_$(date +%Y%m%d_%H%M%S)

    # Full comparison (once W4A8 dir exists)
    python scripts/spark_w4a8_vs_w4a16_quality_regression.py \\
        --model-a /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \\
        --model-b /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8 \\
        --out-dir reports/mtp/fp8_quality_regression_$(date +%Y%m%d_%H%M%S)

Output JSON schema ``lynn-fp8-quality-regression-v1``
-----------------------------------------------------
::

    {
      "schema_version": "lynn-fp8-quality-regression-v1",
      "timestamp": "ISO-8601",
      "models": { "a": { "label": "w4a16", "model_dir": "..." },
                  "b": { "label": "w4a8", "model_dir": "...", "status": "ok|error|placeholder" } },
      "subsets": {
        "mmlu100": {
          "model_a": { "n", "correct", "accuracy", "parse_fail", "elapsed_sec", "by_subject": { ... } },
          "model_b": { ... }  // absent if placeholder/error
        },
        "gpqa50": {
          "model_a": { "n", "correct", "accuracy", "parse_fail", "elapsed_sec" },
          "model_b": { ... }
        }
      },
      "comparison": {
        "mmlu_accuracy_delta": -0.01,    // a - b  (negative = b better)
        "gpqa_accuracy_delta": 0.02,
        "mean_prefix_agreement": 0.95    // token-level prefix match
      },
      "verdict": {
        "mmlu_within_1pp": true,
        "gpqa_within_2pp": true,
        "pass": true
      }
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHOICES = ("A", "B", "C", "D")

BASE_ENV = {
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


# ── env helpers ──────────────────────────────────────────────────────────────

def _set_env(updates: dict[str, str | None]) -> dict[str, str | None]:
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


# ── answer extraction (matches openai_mmlu_500_5shot_eval / openai_gpqa_diamond_eval) ──

def _extract_answer(text: str) -> str | None:
    text = text.strip()
    if text[:1].upper() in CHOICES:
        return text[:1].upper()
    for pat in (
        r"(?i)^(?:answer\s*[:：]?\s*)?([ABCD])\b",
        r"(?i)\banswer\s*(?:is|:|：)?\s*([ABCD])\b",
        r"(?i)\(([ABCD])\)",
        r"(?i)\b([ABCD])\b",
    ):
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()
    return None


# ── dataset loaders ──────────────────────────────────────────────────────────

def _read_mmlu_csv(path: Path, subject: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) < 6:
                continue
            if idx == 0 and row[0].strip().lower() in {"question", "input"}:
                continue
            ans = row[5].strip().upper()[:1]
            if ans not in CHOICES:
                continue
            rows.append({
                "id": f"{split}:{subject}:{idx}",
                "subject": subject,
                "question": row[0].strip(),
                "choices": [row[i].strip() for i in range(1, 5)],
                "answer": ans,
            })
    return rows


def _find_split_dir(data_dir: Path, split: str) -> Path:
    candidates = [data_dir / split, data_dir / "data" / split]
    for cand in candidates:
        if cand.is_dir() and list(cand.glob("*.csv")):
            return cand
    raise SystemExit(
        f"MMLU {split!r} CSV directory not found under {data_dir}. "
        "Expected e.g. /tmp/datasets/mmlu/dev and /tmp/datasets/mmlu/test."
    )


def _load_mmlu(data_dir: Path, sample: int, seed: int) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    dev_dir = _find_split_dir(data_dir, "dev")
    test_dir = _find_split_dir(data_dir, "test")
    dev_by_subject: dict[str, list[dict[str, Any]]] = {}
    tests: list[dict[str, Any]] = []
    for test_file in sorted(test_dir.glob("*.csv")):
        subject = test_file.stem.removesuffix("_test")
        dev_file = dev_dir / f"{subject}_dev.csv"
        if dev_file.exists():
            dev_by_subject[subject] = _read_mmlu_csv(dev_file, subject, "dev")
        tests.extend(_read_mmlu_csv(test_file, subject, "test"))
    if not tests:
        raise SystemExit(f"No MMLU test examples found in {test_dir}")
    if sample and sample < len(tests):
        tests = random.Random(seed).sample(tests, sample)
    return dev_by_subject, tests


def _mmlu_prompt(subject: str, shots: list[dict[str, Any]], ex: dict[str, Any]) -> str:
    def _fmt(e: dict[str, Any], include_answer: bool) -> str:
        lines = [e["question"]]
        for letter, text in zip(CHOICES, e["choices"]):
            lines.append(f"{letter}. {text}")
        lines.append(f"Answer: {e['answer']}" if include_answer else "Answer:")
        return "\n".join(lines)

    parts = [
        f"The following are multiple choice questions about {subject.replace('_', ' ')}.",
        "Return only one letter: A, B, C, or D.",
        "",
    ]
    for shot in shots:
        parts.extend([_fmt(shot, True), ""])
    parts.append(_fmt(ex, False))
    return "\n".join(parts)


def _load_gpqa(path: Path, sample: int, seed: int) -> list[dict[str, Any]]:
    def _norm(row: dict[str, str], *names: str) -> str:
        lowered = {k.lower().strip(): v for k, v in row.items()}
        for name in names:
            val = lowered.get(name.lower())
            if val is not None:
                return val.strip()
        raise KeyError(names)

    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            try:
                q = _norm(row, "Question", "question")
                correct = _norm(row, "Correct Answer", "correct_answer", "answer")
                wrongs = [
                    _norm(row, "Incorrect Answer 1", "incorrect_answer_1"),
                    _norm(row, "Incorrect Answer 2", "incorrect_answer_2"),
                    _norm(row, "Incorrect Answer 3", "incorrect_answer_3"),
                ]
            except Exception:
                continue
            choices = [correct] + wrongs
            rng = random.Random(seed + idx)
            rng.shuffle(choices)
            rows.append({
                "id": f"gpqa:{idx}",
                "question": q,
                "choices": choices,
                "answer": CHOICES[choices.index(correct)],
            })
    if not rows:
        raise SystemExit(f"No GPQA rows found in {path}")
    if sample and sample < len(rows):
        rows = random.Random(seed).sample(rows, sample)
    return rows


def _gpqa_prompt(ex: dict[str, Any]) -> str:
    lines = [
        "Answer the following graduate-level multiple choice question.",
        "Return only one letter: A, B, C, or D.",
        "",
        ex["question"],
    ]
    for letter, text in zip(CHOICES, ex["choices"]):
        lines.append(f"{letter}. {text}")
    lines.append("Answer:")
    return "\n".join(lines)


# ── scoring ──────────────────────────────────────────────────────────────────

def _score(rows: list[dict[str, Any]], texts: list[str]) -> dict[str, Any]:
    by_subject: dict[str, dict[str, int]] = {}
    correct = 0
    parse_fail = 0
    for row, text in zip(rows, texts):
        pred = _extract_answer(text)
        ok = pred == row["answer"]
        correct += int(ok)
        parse_fail += int(pred is None)
        subj = row.get("subject", "default")
        stat = by_subject.setdefault(subj, {"correct": 0, "n": 0})
        stat["n"] += 1
        stat["correct"] += int(ok)
    n = len(rows)
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "parse_fail": parse_fail,
        "by_subject": {
            k: {"acc": round(v["correct"] / v["n"], 4) if v["n"] else 0.0, "n": v["n"]}
            for k, v in sorted(by_subject.items())
        },
    }


# ── model loading ────────────────────────────────────────────────────────────

def _load_runner(model_dir: str, device: str, dtype: "Any") -> "Any":
    import torch  # noqa: F811 — deferred for --help compatibility
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
    from engine.nvfp4_layout import detect_nvfp4_layout  # noqa: E402

    prev = _set_env(BASE_ENV)
    try:
        layout = detect_nvfp4_layout(model_dir)
        # Accept both Lynn-native NVFP4 and (future) W4A8 FP8 layouts.
        if layout.layout_kind not in ("lynn_native_per16_variable", "lynn_w4a8_fp8"):
            raise SystemExit(
                f"[quality-reg] {model_dir} is layout={layout.layout_kind!r}. "
                "Expected Lynn-native W4A16 NVFP4 or W4A8 FP8 artifact."
            )
        return LynnIncrementalRunner(model_dir, device=device, dtype=dtype, verbose=False)
    finally:
        _restore_env(prev)


# ── single-model eval ────────────────────────────────────────────────────────

def _run_model_eval(
    runner: LynnIncrementalRunner,
    *,
    label: str,
    model_dir: str,
    dev_by_subject: dict[str, list[dict[str, Any]]],
    mmlu_rows: list[dict[str, Any]],
    gpqa_rows: list[dict[str, Any]],
    max_new_mmlu: int,
    max_new_gpqa: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label, "model_dir": model_dir}

    # MMLU
    try:
        mmlu_started = time.time()
        mmlu_texts: list[str] = []
        for ex in mmlu_rows:
            shots = dev_by_subject.get(ex["subject"], [])[:5]
            prompt = _mmlu_prompt(ex["subject"], shots, ex)
            out = runner.generate(prompt, max_new=max_new_mmlu)
            mmlu_texts.append(out["completion_text"])
        mmlu_elapsed = time.time() - mmlu_started
        mmlu_result = _score(mmlu_rows, mmlu_texts)
        mmlu_result["elapsed_sec"] = round(mmlu_elapsed, 3)
        result["mmlu100"] = mmlu_result
        print(f"  [{label}] MMLU-100: {mmlu_result['correct']}/{mmlu_result['n']} "
              f"= {mmlu_result['accuracy']:.4f}  ({mmlu_elapsed:.1f}s)")
    except Exception as e:  # noqa: BLE001
        result["mmlu100"] = {"error": repr(e)}
        print(f"  [{label}] MMLU-100: ERROR — {e}")

    # GPQA
    try:
        gpqa_started = time.time()
        gpqa_texts: list[str] = []
        for ex in gpqa_rows:
            prompt = _gpqa_prompt(ex)
            out = runner.generate(prompt, max_new=max_new_gpqa)
            gpqa_texts.append(out["completion_text"])
        gpqa_elapsed = time.time() - gpqa_started
        gpqa_result = _score(gpqa_rows, gpqa_texts)
        gpqa_result["elapsed_sec"] = round(gpqa_elapsed, 3)
        result["gpqa50"] = gpqa_result
        print(f"  [{label}] GPQA-50:  {gpqa_result['correct']}/{gpqa_result['n']} "
              f"= {gpqa_result['accuracy']:.4f}  ({gpqa_elapsed:.1f}s)")
    except Exception as e:  # noqa: BLE001
        result["gpqa50"] = {"error": repr(e)}
        print(f"  [{label}] GPQA-50:  ERROR — {e}")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", required=True,
                    help="Path to Lynn-native W4A16 NVFP4 model dir (baseline).")
    ap.add_argument("--model-b", default=None,
                    help="Path to Lynn-native W4A8 FP8 model dir (optional; skipped if absent).")
    ap.add_argument("--label-a", default="w4a16")
    ap.add_argument("--label-b", default="w4a8")
    ap.add_argument("--mmlu-data-dir", default="/tmp/datasets/mmlu",
                    help="Root of MMLU CSV dataset (expects dev/ and test/ subdirs).")
    ap.add_argument("--gpqa-csv", default="/tmp/datasets/gpqa/gpqa_diamond.csv",
                    help="Path to GPQA Diamond CSV.")
    ap.add_argument("--out-dir", required=True,
                    help="Directory for JSONL per-question results and summary JSON.")
    ap.add_argument("--mmlu-sample", type=int, default=100)
    ap.add_argument("--gpqa-sample", type=int, default=50)
    ap.add_argument("--max-new-mmlu", type=int, default=16)
    ap.add_argument("--max-new-gpqa", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260519)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load datasets ────────────────────────────────────────────────────────
    print(f"[quality-reg] Loading MMLU from {args.mmlu_data_dir} ...")
    dev_by_subject, mmlu_rows = _load_mmlu(Path(args.mmlu_data_dir), args.mmlu_sample, args.seed)
    print(f"[quality-reg]   {len(mmlu_rows)} MMLU questions, "
          f"{len(dev_by_subject)} subjects with dev examples")

    print(f"[quality-reg] Loading GPQA from {args.gpqa_csv} ...")
    gpqa_rows = _load_gpqa(Path(args.gpqa_csv), args.gpqa_sample, args.seed)
    print(f"[quality-reg]   {len(gpqa_rows)} GPQA Diamond questions")

    # torch + engine deferred here so --help and dataset loading work without GPU
    import torch  # noqa: F811 — deferred for --help compatibility
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    # ── model A (baseline) ──────────────────────────────────────────────────
    print(f"[quality-reg] Loading model A ({args.label_a}): {args.model_a}")
    runner_a = _load_runner(args.model_a, args.device, dtype)
    print("[quality-reg] Running eval on model A ...")
    result_a = _run_model_eval(
        runner_a,
        label=args.label_a,
        model_dir=args.model_a,
        dev_by_subject=dev_by_subject,
        mmlu_rows=mmlu_rows,
        gpqa_rows=gpqa_rows,
        max_new_mmlu=args.max_new_mmlu,
        max_new_gpqa=args.max_new_gpqa,
    )
    del runner_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── model B (W4A8 FP8 — placeholder if dir not yet available) ───────────
    result_b: dict[str, Any] | None = None
    model_b_status = "placeholder"
    if args.model_b:
        model_b_path = Path(args.model_b)
        if not model_b_path.is_dir():
            model_b_status = "placeholder"
            print(f"[quality-reg] Model B ({args.label_b}) dir not found: {args.model_b}")
            print(f"[quality-reg]   Skipping W4A8 leg — will be populated once repack V1 lands.")
        else:
            try:
                print(f"[quality-reg] Loading model B ({args.label_b}): {args.model_b}")
                runner_b = _load_runner(args.model_b, args.device, dtype)
                print("[quality-reg] Running eval on model B ...")
                result_b = _run_model_eval(
                    runner_b,
                    label=args.label_b,
                    model_dir=args.model_b,
                    dev_by_subject=dev_by_subject,
                    mmlu_rows=mmlu_rows,
                    gpqa_rows=gpqa_rows,
                    max_new_mmlu=args.max_new_mmlu,
                    max_new_gpqa=args.max_new_gpqa,
                )
                model_b_status = "ok"
                del runner_b
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                model_b_status = "error"
                print(f"[quality-reg] Model B ({args.label_b}) failed: {e}")
    else:
        print("[quality-reg] No --model-b provided; W4A8 leg skipped.")

    # ── comparison ──────────────────────────────────────────────────────────
    comparison: dict[str, Any] | None = None
    verdict: dict[str, Any] | None = None

    mmlu_a = result_a.get("mmlu100", {})
    gpqa_a = result_a.get("gpqa50", {})
    mmlu_b = (result_b or {}).get("mmlu100", {})
    gpqa_b = (result_b or {}).get("gpqa50", {})

    have_comparison = (
        result_b is not None
        and model_b_status == "ok"
        and isinstance(mmlu_a, dict) and "accuracy" in mmlu_a
        and isinstance(gpqa_a, dict) and "accuracy" in gpqa_a
        and isinstance(mmlu_b, dict) and "accuracy" in mmlu_b
        and isinstance(gpqa_b, dict) and "accuracy" in gpqa_b
    )
    if have_comparison:
        mmlu_delta = round(mmlu_a["accuracy"] - mmlu_b["accuracy"], 4)
        gpqa_delta = round(gpqa_a["accuracy"] - gpqa_b["accuracy"], 4)
        comparison = {
            "mmlu_accuracy_delta": mmlu_delta,
            "gpqa_accuracy_delta": gpqa_delta,
            "mean_prefix_agreement": None,  # TODO: compute token-level prefix match
            "note": "delta = model_a - model_b; negative means model_b scored higher",
        }
        verdict = {
            "mmlu_within_1pp": abs(mmlu_delta) <= 0.01,
            "gpqa_within_2pp": abs(gpqa_delta) <= 0.02,
            "pass": abs(mmlu_delta) <= 0.01 and abs(gpqa_delta) <= 0.02,
        }

    # ── report ──────────────────────────────────────────────────────────────
    report: dict[str, Any] = {
        "schema_version": "lynn-fp8-quality-regression-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {
            "a": {"label": args.label_a, "model_dir": args.model_a},
            "b": {
                "label": args.label_b,
                "model_dir": args.model_b or "(not provided)",
                "status": model_b_status,
            },
        },
        "config": {
            "mmlu_sample": args.mmlu_sample,
            "gpqa_sample": args.gpqa_sample,
            "max_new_mmlu": args.max_new_mmlu,
            "max_new_gpqa": args.max_new_gpqa,
            "seed": args.seed,
            "dtype": args.dtype,
        },
        "subsets": {
            "mmlu100": {
                "model_a": mmlu_a,
                **({"model_b": mmlu_b} if result_b is not None else {}),
            },
            "gpqa50": {
                "model_a": gpqa_a,
                **({"model_b": gpqa_b} if result_b is not None else {}),
            },
        },
        "comparison": comparison,
        "verdict": verdict,
    }

    report_path = out_dir / "quality_regression_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[quality-reg] Report: {report_path}")

    # ── print summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Quality Regression Summary")
    print(f"{'='*60}")
    if isinstance(mmlu_a, dict) and "accuracy" in mmlu_a:
        print(f"  MMLU-100  {args.label_a}: {mmlu_a['accuracy']:.4f}  "
              f"({mmlu_a['correct']}/{mmlu_a['n']})")
    if isinstance(gpqa_a, dict) and "accuracy" in gpqa_a:
        print(f"  GPQA-50   {args.label_a}: {gpqa_a['accuracy']:.4f}  "
              f"({gpqa_a['correct']}/{gpqa_a['n']})")
    if isinstance(mmlu_b, dict) and "accuracy" in mmlu_b:
        print(f"  MMLU-100  {args.label_b}: {mmlu_b['accuracy']:.4f}  "
              f"({mmlu_b['correct']}/{mmlu_b['n']})")
    if isinstance(gpqa_b, dict) and "accuracy" in gpqa_b:
        print(f"  GPQA-50   {args.label_b}: {gpqa_b['accuracy']:.4f}  "
              f"({gpqa_b['correct']}/{gpqa_b['n']})")
    if verdict:
        print(f"\n  Verdict: {'PASS ✓' if verdict['pass'] else 'FAIL ✗'}")
        print(f"    MMLU Δ within 1pp:  {verdict['mmlu_within_1pp']}")
        print(f"    GPQA Δ within 2pp:  {verdict['gpqa_within_2pp']}")
    else:
        print(f"\n  Verdict: PENDING (model B status: {model_b_status})")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
