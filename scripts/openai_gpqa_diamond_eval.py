#!/usr/bin/env python3
"""Dependency-light OpenAI-compatible GPQA Diamond evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


CHOICES = ("A", "B", "C", "D")


def _norm_key(row: dict[str, str], *names: str) -> str:
    lowered = {k.lower().strip(): v for k, v in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()].strip()
    raise KeyError(f"missing any of columns: {names}")


def _load_gpqa(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                q = _norm_key(row, "Question", "question")
                correct = _norm_key(row, "Correct Answer", "correct_answer", "answer")
                wrongs = [
                    _norm_key(row, "Incorrect Answer 1", "incorrect_answer_1"),
                    _norm_key(row, "Incorrect Answer 2", "incorrect_answer_2"),
                    _norm_key(row, "Incorrect Answer 3", "incorrect_answer_3"),
                ]
            except Exception:
                continue
            choices = [correct] + wrongs
            rng = random.Random(20260519 + idx)
            rng.shuffle(choices)
            answer = CHOICES[choices.index(correct)]
            rows.append({"id": f"gpqa:{idx}", "question": q, "choices": choices, "answer": answer})
    if not rows:
        raise SystemExit(f"No GPQA rows found in {path}")
    return rows


def _prompt(ex: dict[str, Any]) -> str:
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


def _extract_answer(text: str) -> str | None:
    text = text.strip()
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


def _chat(base_url: str, model: str, prompt: str, timeout: float) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content, data.get("usage", {}) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--sample", type=int, default=0, help="Optional deterministic subset size; 0 means all rows.")
    ap.add_argument("--seed", type=int, default=20260519)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    rows = _load_gpqa(Path(args.csv))
    if args.sample and args.sample < len(rows):
        rows = random.Random(args.seed).sample(rows, args.sample)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")
    started = time.time()

    def run_one(ex: dict[str, Any]) -> dict[str, Any]:
        try:
            text, usage = _chat(args.base_url, args.model, _prompt(ex), args.timeout)
            pred = _extract_answer(text)
            err = None
        except Exception as e:  # noqa: BLE001
            text, usage, pred, err = "", {}, None, repr(e)
        return {
            "id": ex["id"],
            "answer": ex["answer"],
            "prediction": pred,
            "correct": pred == ex["answer"],
            "response": text,
            "usage": usage,
            "error": err,
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futs = [pool.submit(run_one, ex) for ex in rows]
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda x: x["id"])
    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    correct = sum(int(bool(r["correct"])) for r in results)
    elapsed = time.time() - started
    summary = {
        "model": args.model,
        "endpoint": args.base_url,
        "n": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "elapsed_sec": round(elapsed, 3),
        "qps": round(len(results) / elapsed, 4) if elapsed else None,
        "parse_fail": sum(1 for r in results if not r.get("prediction")),
        "errors": sum(1 for r in results if r.get("error")),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
