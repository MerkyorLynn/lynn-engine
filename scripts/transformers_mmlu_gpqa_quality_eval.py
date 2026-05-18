#!/usr/bin/env python3
"""Direct Transformers MMLU/GPQA evaluator for Qwen3.5-9B BF16 baselines.

This is a backstop for dense models before Lynn's OpenAI server path supports
the Qwen3.5 dense runtime.  It emits the same ``*.summary.json`` shape used by
the endpoint runners, so the numbers can be dropped into the 9B matrix report.
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CHOICES = ("A", "B", "C", "D")


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
            rows.append(
                {
                    "id": f"{split}:{subject}:{idx}",
                    "subject": subject,
                    "question": row[0].strip(),
                    "choices": [row[i].strip() for i in range(1, 5)],
                    "answer": ans,
                }
            )
    return rows


def load_mmlu(data_dir: Path, sample: int, seed: int) -> tuple[list[str], list[dict[str, Any]]]:
    root = data_dir / "data" if (data_dir / "data" / "test").is_dir() else data_dir
    dev_dir, test_dir = root / "dev", root / "test"
    if not dev_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit(f"MMLU dev/test dirs not found under {data_dir}")
    dev_by_subject: dict[str, list[dict[str, Any]]] = {}
    tests: list[dict[str, Any]] = []
    for test_file in sorted(test_dir.glob("*_test.csv")):
        subject = test_file.stem.removesuffix("_test")
        dev_file = dev_dir / f"{subject}_dev.csv"
        if dev_file.exists():
            dev_by_subject[subject] = _read_mmlu_csv(dev_file, subject, "dev")
        tests.extend(_read_mmlu_csv(test_file, subject, "test"))
    rng = random.Random(seed)
    if sample and sample < len(tests):
        tests = rng.sample(tests, sample)

    def fmt(ex: dict[str, Any], include_answer: bool) -> str:
        lines = [ex["question"]]
        for letter, text in zip(CHOICES, ex["choices"]):
            lines.append(f"{letter}. {text}")
        lines.append(f"Answer: {ex['answer']}" if include_answer else "Answer:")
        return "\n".join(lines)

    prompts: list[str] = []
    for ex in tests:
        subject = ex["subject"]
        parts = [
            f"The following are multiple choice questions about {subject.replace('_', ' ')}.",
            "Return only one letter: A, B, C, or D.",
            "",
        ]
        for shot in dev_by_subject.get(subject, [])[:5]:
            parts.extend([fmt(shot, True), ""])
        parts.append(fmt(ex, False))
        prompts.append("\n".join(parts))
    return prompts, tests


def _norm(row: dict[str, str], *names: str) -> str:
    lowered = {k.lower().strip(): v for k, v in row.items()}
    for name in names:
        val = lowered.get(name.lower())
        if val is not None:
            return val.strip()
    raise KeyError(names)


def load_gpqa(path: Path, sample: int, seed: int) -> tuple[list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            try:
                q = _norm(row, "Question")
                correct = _norm(row, "Correct Answer")
                wrongs = [_norm(row, "Incorrect Answer 1"), _norm(row, "Incorrect Answer 2"), _norm(row, "Incorrect Answer 3")]
            except Exception:
                continue
            choices = [correct] + wrongs
            random.Random(seed + idx).shuffle(choices)
            rows.append({"id": f"gpqa:{idx}", "question": q, "choices": choices, "answer": CHOICES[choices.index(correct)]})
    if sample and sample < len(rows):
        rows = random.Random(seed).sample(rows, sample)
    prompts = []
    for ex in rows:
        lines = [
            "Answer the following graduate-level multiple choice question.",
            "Return only one letter: A, B, C, or D.",
            "",
            ex["question"],
        ]
        for letter, text in zip(CHOICES, ex["choices"]):
            lines.append(f"{letter}. {text}")
        lines.append("Answer:")
        prompts.append("\n".join(lines))
    return prompts, rows


@torch.inference_mode()
def generate_answers(model: Any, tok: Any, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    outputs: list[str] = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=False).to(model.device)
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id,
        )
        prompt_width = enc["input_ids"].shape[1]
        for i, seq in enumerate(out):
            outputs.append(tok.decode(seq[prompt_width:], skip_special_tokens=True))
    return outputs


def score_rows(rows: list[dict[str, Any]], texts: list[str], out_jsonl: Path) -> dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    parse_fail = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row, text in zip(rows, texts):
            pred = _extract_answer(text)
            ok = pred == row["answer"]
            correct += int(ok)
            parse_fail += int(pred is None)
            rec = {"id": row["id"], "answer": row["answer"], "prediction": pred, "correct": ok, "response": text}
            if "subject" in row:
                rec["subject"] = row["subject"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else 0.0, "parse_fail": parse_fail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--mmlu-data-dir", default="/tmp/datasets/mmlu")
    ap.add_argument("--gpqa-csv", default="/tmp/datasets/gpqa/gpqa_diamond.csv")
    ap.add_argument("--mmlu-sample", type=int, default=500)
    ap.add_argument("--gpqa-sample", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260519)
    args = ap.parse_args()

    started = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()

    prefix = Path(args.out_prefix)
    mmlu_prompts, mmlu_rows = load_mmlu(Path(args.mmlu_data_dir), args.mmlu_sample, args.seed)
    mmlu_started = time.time()
    mmlu_texts = generate_answers(model, tok, mmlu_prompts, args.batch_size, 8)
    mmlu = score_rows(mmlu_rows, mmlu_texts, prefix.with_name(prefix.name + "_mmlu_n500.jsonl"))
    mmlu_summary = {
        "model": args.model,
        "endpoint": "transformers-direct",
        "n": mmlu["n"],
        "subset": "all",
        "shots": 5,
        "correct": mmlu["correct"],
        "accuracy": mmlu["accuracy"],
        "elapsed_sec": round(time.time() - mmlu_started, 3),
        "qps": round(mmlu["n"] / max(time.time() - mmlu_started, 1e-6), 4),
        "parse_fail": mmlu["parse_fail"],
    }
    prefix.with_name(prefix.name + "_mmlu_n500.summary.json").write_text(
        json.dumps(mmlu_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    gpqa_prompts, gpqa_rows = load_gpqa(Path(args.gpqa_csv), args.gpqa_sample, args.seed)
    gpqa_started = time.time()
    gpqa_texts = generate_answers(model, tok, gpqa_prompts, args.batch_size, 8)
    gpqa = score_rows(gpqa_rows, gpqa_texts, prefix.with_name(prefix.name + "_gpqa.jsonl"))
    gpqa_summary = {
        "model": args.model,
        "endpoint": "transformers-direct",
        "n": gpqa["n"],
        "correct": gpqa["correct"],
        "accuracy": gpqa["accuracy"],
        "elapsed_sec": round(time.time() - gpqa_started, 3),
        "qps": round(gpqa["n"] / max(time.time() - gpqa_started, 1e-6), 4),
        "parse_fail": gpqa["parse_fail"],
    }
    prefix.with_name(prefix.name + "_gpqa.summary.json").write_text(
        json.dumps(gpqa_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    final = {"model": args.model, "elapsed_sec": round(time.time() - started, 3), "mmlu": mmlu_summary, "gpqa": gpqa_summary}
    prefix.with_name(prefix.name + "_quality_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
