#!/usr/bin/env python3
"""Recover GPQA thinking-mode parse failures with a larger token budget.

This is for the Qwen3.5-9B Q4_K_M thinking-on run where many organic chemistry
questions reached the token cap before producing a final A/B/C/D answer.  It
reads the previous JSONL, reruns only rows whose ``pred`` is null by default,
and emits both recovery rows and a merged summary.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm


LETTERS = ["A", "B", "C", "D"]
ANS_RE = re.compile(r"\b([ABCD])\b")
SYSTEM_PROMPT = (
    "You are an expert test-taker. Answer the following multiple-choice "
    "question with only the letter (A, B, C, or D) of the correct answer. "
    "Do not explain. /no_think"
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("Question") or "").strip()
            correct = (row.get("Correct Answer") or "").strip()
            inc = [
                (row.get("Incorrect Answer 1") or "").strip(),
                (row.get("Incorrect Answer 2") or "").strip(),
                (row.get("Incorrect Answer 3") or "").strip(),
            ]
            if not q or not correct or not all(inc):
                continue
            rows.append(
                {
                    "Question": q,
                    "Correct Answer": correct,
                    "Incorrect Answer 1": inc[0],
                    "Incorrect Answer 2": inc[1],
                    "Incorrect Answer 3": inc[2],
                    "Subdomain": row.get("Subdomain") or row.get("High-level domain") or "",
                }
            )
    return rows


def _shuffle_choices(row: dict[str, str]) -> tuple[list[str], str]:
    correct = row["Correct Answer"]
    options = [
        correct,
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    seed = int(hashlib.sha1(row["Question"].encode()).hexdigest()[:8], 16)
    random.Random(seed).shuffle(options)
    return options, LETTERS[options.index(correct)]


def _build_prompt(row: dict[str, str], options: list[str]) -> str:
    lines = [
        "Answer the following multiple-choice question. "
        "Respond with the letter (A, B, C, or D) of the correct answer.",
        "",
        row["Question"].strip(),
        "",
    ]
    for letter, option in zip(LETTERS, options):
        lines.append(f"{letter}. {option.strip()}")
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def _parse_answer(text: str) -> str | None:
    if not text:
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] in LETTERS:
        return stripped[0]
    match = ANS_RE.search(stripped[:128])
    return match.group(1) if match else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


async def _run_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    *,
    model: str,
    row: dict[str, str],
    idx: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    options, gold = _shuffle_choices(row)
    prompt = _build_prompt(row, options)
    async with sem:
        content = ""
        error = None
        started = time.time()
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                if not content.strip():
                    content = getattr(msg, "reasoning_content", None) or ""
                error = None
                break
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)
                if attempt < 2:
                    await asyncio.sleep(2.0 * (attempt + 1))
        pred = _parse_answer(content)
        return {
            "idx": idx,
            "gold": gold,
            "pred": pred,
            "ok": pred == gold,
            "subject": row.get("Subdomain", ""),
            "raw_head": content[:512],
            "raw_tail": content[-512:] if content else "",
            "raw_chars": len(content),
            "elapsed_sec": round(time.time() - started, 3),
            "error": error,
            "max_tokens": max_tokens,
        }


def _summarize(
    *,
    previous: list[dict[str, Any]],
    recovered: list[dict[str, Any]],
    out_jsonl: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_idx = {int(row["idx"]): dict(row) for row in previous}
    for row in recovered:
        by_idx[int(row["idx"])] = dict(row)
    merged = [by_idx[i] for i in sorted(by_idx)]
    pred_rows = [row for row in merged if row.get("pred")]
    correct = sum(1 for row in merged if row.get("ok"))
    parse_fail = len(merged) - len(pred_rows)
    recovered_pred = sum(1 for row in recovered if row.get("pred"))
    recovered_correct = sum(1 for row in recovered if row.get("ok"))
    return {
        "model": args.model,
        "endpoint": args.base_url,
        "previous_jsonl": str(args.previous_jsonl),
        "recovery_jsonl": str(out_jsonl),
        "target_indices": [int(row["idx"]) for row in recovered],
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "n": len(merged),
        "correct": correct,
        "accuracy": correct / len(merged) if merged else 0.0,
        "parse_fail": parse_fail,
        "accuracy_excluding_parse_fail": correct / len(pred_rows) if pred_rows else 0.0,
        "recovered_n": len(recovered),
        "recovered_pred": recovered_pred,
        "recovered_correct": recovered_correct,
        "recovered_parse_fail": len(recovered) - recovered_pred,
        "subjects_recovered": {
            subject: sum(1 for row in recovered if row.get("subject") == subject)
            for subject in sorted({str(row.get("subject", "")) for row in recovered})
        },
    }


async def _main_async(args: argparse.Namespace) -> int:
    csv_rows = _load_csv(args.csv)
    previous = _read_jsonl(args.previous_jsonl)
    if args.indices:
        target_indices = [int(x) for part in args.indices for x in part.split(",") if x.strip()]
    else:
        target_indices = [int(row["idx"]) for row in previous if not row.get("pred")]
    if args.limit:
        target_indices = target_indices[: args.limit]
    out_jsonl = args.out
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(max(1, args.concurrency))
    started = time.time()
    tasks = [
        _run_one(
            client,
            sem,
            model=args.model,
            row=csv_rows[idx],
            idx=idx,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        for idx in target_indices
    ]
    recovered: list[dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), mininterval=2.0):
            row = await future
            recovered.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    recovered.sort(key=lambda row: int(row["idx"]))
    out_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in recovered),
        encoding="utf-8",
    )
    summary = _summarize(previous=previous, recovered=recovered, out_jsonl=out_jsonl, args=args)
    summary["elapsed_sec"] = round(time.time() - started, 3)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--previous-jsonl", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--indices", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=2400.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
