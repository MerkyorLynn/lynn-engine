#!/usr/bin/env python3
"""Small OpenAI-compatible MMLU 500-sample 5-shot evaluator.

This is intentionally dependency-light for rented inference boxes: it uses only
the Python standard library plus ``requests``.  It expects the original MMLU CSV
layout with ``dev`` and ``test`` directories, where each row is:

    question,A,B,C,D,answer

The script writes per-example JSONL to ``--out`` and a sibling
``*.summary.json`` consumed by the Qwen3.5-9B matrix watcher.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


CHOICES = ("A", "B", "C", "D")


def _find_split_dir(data_dir: Path, split: str) -> Path:
    candidates = [data_dir / split, data_dir / "data" / split]
    for cand in candidates:
        if cand.is_dir() and list(cand.glob("*.csv")):
            return cand
    raise SystemExit(
        f"MMLU {split!r} CSV directory not found under {data_dir}. "
        "Expected e.g. /tmp/datasets/mmlu/dev and /tmp/datasets/mmlu/test."
    )


def _read_subject_csv(path: Path, subject: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) < 6:
                continue
            if idx == 0 and row[0].strip().lower() in {"question", "input"}:
                continue
            answer = row[5].strip().upper()[:1]
            if answer not in CHOICES:
                continue
            rows.append(
                {
                    "id": f"{split}:{subject}:{idx}",
                    "subject": subject,
                    "question": row[0].strip(),
                    "choices": [row[i].strip() for i in range(1, 5)],
                    "answer": answer,
                }
            )
    return rows


def _load_mmlu(data_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    dev_dir = _find_split_dir(data_dir, "dev")
    test_dir = _find_split_dir(data_dir, "test")
    dev_by_subject: dict[str, list[dict[str, Any]]] = {}
    tests: list[dict[str, Any]] = []
    for test_file in sorted(test_dir.glob("*.csv")):
        subject = test_file.stem.removesuffix("_test")
        dev_file = dev_dir / f"{subject}_dev.csv"
        if dev_file.exists():
            dev_by_subject[subject] = _read_subject_csv(dev_file, subject, "dev")
        tests.extend(_read_subject_csv(test_file, subject, "test"))
    if not tests:
        raise SystemExit(f"No MMLU test examples found in {test_dir}")
    return dev_by_subject, tests


def _format_question(ex: dict[str, Any], include_answer: bool) -> str:
    lines = [ex["question"]]
    for letter, text in zip(CHOICES, ex["choices"]):
        lines.append(f"{letter}. {text}")
    if include_answer:
        lines.append(f"Answer: {ex['answer']}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def _prompt(subject: str, shots: list[dict[str, Any]], ex: dict[str, Any], *, append_no_think: bool) -> str:
    pretty = subject.replace("_", " ")
    parts = [
        f"The following are multiple choice questions about {pretty}.",
        "Return only one letter: A, B, C, or D.",
        "",
    ]
    for shot in shots:
        parts.append(_format_question(shot, include_answer=True))
        parts.append("")
    parts.append(_format_question(ex, include_answer=False))
    if append_no_think:
        parts.append("/no_think")
    return "\n".join(parts)


def _extract_answer(text: str) -> str | None:
    text = text.strip()
    if text[:1].upper() in CHOICES:
        return text[:1].upper()
    patterns = [
        r"(?i)^(?:answer\s*[:：]?\s*)?([ABCD])\b",
        r"(?i)\banswer\s*(?:is|:|：)?\s*([ABCD])\b",
        r"(?i)\(([ABCD])\)",
        r"(?i)\b([ABCD])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()
    return None


def _chat(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    *,
    disable_thinking: bool,
) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content, data.get("usage", {}) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260519)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--append-no-think", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    dev_by_subject, tests = _load_mmlu(data_dir)
    rng = random.Random(args.seed)
    if args.sample and args.sample < len(tests):
        tests = rng.sample(tests, args.sample)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")

    started = time.time()

    def run_one(ex: dict[str, Any]) -> dict[str, Any]:
        shots = dev_by_subject.get(ex["subject"], [])[: args.shots]
        prompt = _prompt(ex["subject"], shots, ex, append_no_think=args.append_no_think)
        try:
            text, usage = _chat(
                args.base_url,
                args.model,
                prompt,
                args.timeout,
                disable_thinking=args.disable_thinking,
            )
            pred = _extract_answer(text)
            err = None
        except Exception as e:  # noqa: BLE001 - record evaluator failures per sample
            text, usage, pred, err = "", {}, None, repr(e)
        return {
            "id": ex["id"],
            "subject": ex["subject"],
            "answer": ex["answer"],
            "prediction": pred,
            "correct": pred == ex["answer"],
            "response": text,
            "usage": usage,
            "error": err,
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futs = [pool.submit(run_one, ex) for ex in tests]
        for fut in as_completed(futs):
            rows.append(fut.result())

    rows.sort(key=lambda x: x["id"])
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_subject: dict[str, dict[str, Any]] = {}
    for row in rows:
        stat = by_subject.setdefault(row["subject"], {"correct": 0, "n": 0})
        stat["n"] += 1
        stat["correct"] += int(bool(row["correct"]))
    by_subject_out = {
        k: {"acc": round(v["correct"] / v["n"], 4) if v["n"] else 0.0, "n": v["n"]}
        for k, v in sorted(by_subject.items())
    }
    correct = sum(int(bool(r["correct"])) for r in rows)
    elapsed = time.time() - started
    summary = {
        "model": args.model,
        "endpoint": args.base_url,
        "n": len(rows),
        "subset": "all",
        "shots": args.shots,
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "elapsed_sec": round(elapsed, 3),
        "qps": round(len(rows) / elapsed, 4) if elapsed else None,
        "by_subject": by_subject_out,
        "parse_fail": sum(1 for r in rows if not r.get("prediction")),
        "errors": sum(1 for r in rows if r.get("error")),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
