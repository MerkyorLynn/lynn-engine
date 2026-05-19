#!/usr/bin/env python3
"""Clean 32K thinking-on MCQ evaluator for OpenAI-compatible endpoints.

This script intentionally does not inject `/no_think`. It passes
`chat_template_kwargs.enable_thinking=true` to Lynn/SGLang-compatible servers
and asks the model to place the final letter after its reasoning.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


def _extract_answer(text: str) -> str | None:
    if not text:
        return None
    tail = _post_reasoning(text).strip()
    search_space = (tail[-2048:] if tail else text[-2048:]).strip()
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


def _chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
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
    url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or message.get("reasoning_content") or ""
    return content, data.get("usage", {}) or {}


def _load_gpqa(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            question = (row.get("Question") or "").strip()
            correct = (row.get("Correct Answer") or "").strip()
            wrongs = [
                (row.get("Incorrect Answer 1") or "").strip(),
                (row.get("Incorrect Answer 2") or "").strip(),
                (row.get("Incorrect Answer 3") or "").strip(),
            ]
            if not question or not correct or not all(wrongs):
                continue
            choices = [correct] + wrongs
            random.Random(int(repr(question).encode().hex()[:8], 16) + idx).shuffle(choices)
            rows.append(
                {
                    "id": f"gpqa:{idx}",
                    "subject": row.get("Subdomain") or row.get("High-level domain") or "",
                    "question": question,
                    "choices": choices,
                    "gold": LETTERS[choices.index(correct)],
                }
            )
    return rows


def _load_mmlu(data_dir: Path, *, sample: int, seed: int) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    dev = pq.read_table(data_dir / "dev.parquet").to_pylist()
    test = pq.read_table(data_dir / "test.parquet").to_pylist()
    dev_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev:
        dev_by_subject[row["subject"]].append(row)
    if sample and sample < len(test):
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(test)), sample))
        test = [test[i] for i in idxs]
    return dev_by_subject, test


def _format_choices(question: str, choices: list[str], *, answer: str | None = None) -> str:
    lines = [question.strip()]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append(f"Answer: {answer}" if answer else "Answer:")
    return "\n".join(lines)


def _gpqa_prompt(ex: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "Answer the following graduate-level multiple choice question.",
            _format_choices(ex["question"], ex["choices"]),
        ]
    )


def _mmlu_prompt(ex: dict[str, Any], dev_by_subject: dict[str, list[dict[str, Any]]], shots: int) -> str:
    parts = [
        f"The following are multiple choice questions about {ex['subject'].replace('_', ' ')}.",
    ]
    for shot in dev_by_subject[ex["subject"]][:shots]:
        parts.append(_format_choices(shot["question"], list(shot["choices"]), answer=LETTERS[int(shot["answer"])]))
    parts.append(_format_choices(ex["question"], list(ex["choices"])))
    return "\n\n".join(parts)


def _summarize(rows: list[dict[str, Any]], args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    pred_rows = [row for row in rows if row.get("pred")]
    correct = sum(1 for row in rows if row.get("ok"))
    return {
        "schema_version": "lynn-openai-mcq-thinking32-eval-v1",
        "task": args.task,
        "model": args.model,
        "endpoint": args.base_url,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "parse_fail": len(rows) - len(pred_rows),
        "accuracy_excluding_parse_fail": correct / len(pred_rows) if pred_rows else 0.0,
        "elapsed_sec": round(elapsed, 3),
        "qps": round(len(rows) / elapsed, 4) if elapsed else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gpqa", "mmlu"], required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpqa-csv")
    ap.add_argument("--mmlu-data-dir")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=20260519)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    if args.task == "gpqa":
        if not args.gpqa_csv:
            raise SystemExit("--gpqa-csv is required for --task gpqa")
        examples = _load_gpqa(Path(args.gpqa_csv))
        build_prompt = _gpqa_prompt
    else:
        if not args.mmlu_data_dir:
            raise SystemExit("--mmlu-data-dir is required for --task mmlu")
        dev_by_subject, mmlu_examples = _load_mmlu(Path(args.mmlu_data_dir), sample=args.sample, seed=args.sample_seed)
        examples = [
            {
                "id": f"mmlu:{idx}",
                "subject": row["subject"],
                "question": row["question"],
                "choices": list(row["choices"]),
                "gold": LETTERS[int(row["answer"])],
                "_raw": row,
            }
            for idx, row in enumerate(mmlu_examples)
        ]

        def build_prompt(ex: dict[str, Any]) -> str:
            return _mmlu_prompt(ex["_raw"], dev_by_subject, args.shots)

    if args.limit:
        examples = examples[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def run_one(ex: dict[str, Any]) -> dict[str, Any]:
        error = None
        content = ""
        usage: dict[str, Any] = {}
        item_started = time.time()
        for attempt in range(3):
            try:
                content, usage = _chat(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=build_prompt(ex),
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                break
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
        pred = _extract_answer(content)
        return {
            "id": ex["id"],
            "subject": ex.get("subject", ""),
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

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        future_to_id = {pool.submit(run_one, ex): ex["id"] for ex in examples}
        with out_path.open("w", encoding="utf-8") as f:
            for fut in as_completed(future_to_id):
                row = fut.result()
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    rows.sort(key=lambda row: row["id"])
    out_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = _summarize(rows, args, time.time() - started)
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
