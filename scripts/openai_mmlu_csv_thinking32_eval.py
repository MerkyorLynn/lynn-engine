#!/usr/bin/env python3
"""CSV-backed MMLU 32K thinking-on evaluator for OpenAI-compatible endpoints.

This is the MMLU companion to ``openai_mcq_thinking32_eval.py`` for hosts that
only have the original CSV MMLU layout and no ``pyarrow``/parquet stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import requests


LETTERS = ("A", "B", "C", "D")
EXPLICIT_ANSWER_PATTERNS = (
    re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:：]\s*([ABCD])\b"),
    re.compile(r"(?i)\b(?:final\s+)?answer\s*(?:is|:|：)?\s*([ABCD])\b"),
)
FALLBACK_ANSWER_PATTERNS = (
    re.compile(r"(?i)\(([ABCD])\)"),
    re.compile(r"\b([ABCD])\b"),
)


def _post_reasoning(text: str) -> str:
    return text.split("</think>", 1)[1] if "</think>" in text else text


def _extract_answer(text: str) -> str | None:
    if not text:
        return None
    tail = _post_reasoning(text).strip()
    search_space = (tail or text)[-2048:].strip()
    for pat in EXPLICIT_ANSWER_PATTERNS:
        matches = list(pat.finditer(search_space))
        if matches:
            return matches[-1].group(1).upper()
    if search_space[:1].upper() in LETTERS and len(search_space) <= 8:
        return search_space[:1].upper()
    for pat in FALLBACK_ANSWER_PATTERNS:
        matches = list(pat.finditer(search_space))
        if matches:
            return matches[-1].group(1).upper()
    return None


def _find_split_dir(data_dir: Path, split: str) -> Path:
    candidates = [data_dir / split, data_dir / "data" / split]
    for cand in candidates:
        if cand.is_dir() and list(cand.glob("*.csv")):
            return cand
    raise SystemExit(f"MMLU {split!r} CSV directory not found under {data_dir}")


def _read_subject_csv(path: Path, subject: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.reader(f)):
            if len(row) < 6:
                continue
            if idx == 0 and row[0].strip().lower() in {"question", "input"}:
                continue
            answer = row[5].strip().upper()[:1]
            if answer not in LETTERS:
                continue
            rows.append(
                {
                    "id": f"{split}:{subject}:{idx}",
                    "subject": subject,
                    "question": row[0].strip(),
                    "choices": [row[i].strip() for i in range(1, 5)],
                    "gold": answer,
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
        dev_by_subject[subject] = _read_subject_csv(dev_file, subject, "dev") if dev_file.exists() else []
        tests.extend(_read_subject_csv(test_file, subject, "test"))
    if not tests:
        raise SystemExit(f"No MMLU test examples found in {test_dir}")
    return dev_by_subject, tests


def _format_question(ex: dict[str, Any], include_answer: bool) -> str:
    lines = [ex["question"]]
    for letter, text in zip(LETTERS, ex["choices"]):
        lines.append(f"{letter}. {text}")
    lines.append(f"Answer: {ex['gold']}" if include_answer else "Answer:")
    return "\n".join(lines)


def _prompt(ex: dict[str, Any], dev_by_subject: dict[str, list[dict[str, Any]]], shots: int) -> str:
    subject_name = ex["subject"].replace("_", " ")
    parts = [f"The following are multiple choice questions about {subject_name}."]
    for shot in dev_by_subject.get(ex["subject"], [])[:shots]:
        parts.append(_format_question(shot, include_answer=True))
        parts.append("")
    parts.append(_format_question(ex, include_answer=False))
    return "\n".join(parts)


def _chat(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert test-taker. Think carefully if useful. "
                    "After reasoning, output the final answer as exactly one "
                    "line: Answer: A, Answer: B, Answer: C, or Answer: D."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    response = requests.post(base_url.rstrip("/") + "/chat/completions", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or message.get("reasoning_content") or ""
    return content, data.get("usage", {}) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260520)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    dev_by_subject, tests = _load_mmlu(Path(args.data_dir))
    if args.sample and args.sample < len(tests):
        tests = random.Random(args.seed).sample(tests, args.sample)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as f:
        for ex in tests:
            item_started = time.time()
            error = None
            content = ""
            usage: dict[str, Any] = {}
            try:
                content, usage = _chat(
                    args.base_url,
                    args.model,
                    _prompt(ex, dev_by_subject, args.shots),
                    args.max_tokens,
                    args.timeout,
                )
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)
            pred = _extract_answer(content)
            row = {
                "id": ex["id"],
                "subject": ex["subject"],
                "gold": ex["gold"],
                "pred": pred,
                "ok": pred == ex["gold"],
                "raw_head": content[:512],
                "raw_tail": content[-512:] if content else "",
                "raw_chars": len(content),
                "usage": usage,
                "elapsed_sec": round(time.time() - item_started, 3),
                "error": error,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    correct = sum(bool(row.get("ok")) for row in rows)
    parse_fail = sum(1 for row in rows if not row.get("pred"))
    elapsed = time.time() - started
    summary = {
        "schema_version": "lynn-openai-mcq-thinking32-eval-v1",
        "task": "mmlu",
        "model": args.model,
        "endpoint": args.base_url,
        "max_tokens": args.max_tokens,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "parse_fail": parse_fail,
        "accuracy_excluding_parse_fail": correct / (len(rows) - parse_fail) if len(rows) - parse_fail else 0.0,
        "elapsed_sec": round(elapsed, 3),
        "qps": round(len(rows) / elapsed, 4) if elapsed else None,
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
