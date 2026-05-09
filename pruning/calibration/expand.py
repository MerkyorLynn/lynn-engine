"""
Lynn-27B-A3B pruning calibration set expansion.

Reads `seeds.jsonl` (~95 hand-curated seed prompts) and expands each by
calling Lynn brain (when DGX up) to generate N paraphrases per seed.

Target output sizes (per memory `project_lynn_27b_pruning_plan_0509.md`):

  lynn_keep (880 prompts total):
    coding (200), tool_call (150), math_numerical (100), finance (80),
    long_doc_research (70), creative_writing_zh (100), creative_writing_en (50),
    novel_multi_pov (30), social_media_styles (50), casual_chat_zh (50)

  lynn_drop (560 prompts total):
    biology_pure (100), physics_quantum (80), medical_clinical (80),
    law_judicial (60), music_theory (40), sports_analytics (40),
    religious_text (30), quantum_chemistry (30), minor_lang_* (100)

Total ~1440 prompts.

The brain (or any LLM with reasonable instruction-following) is used to
paraphrase each seed without changing the topic / category. This produces
diverse routing patterns per expert during activation profiling.

Usage (when brain at port 8789/8790 is reachable):
    python3 expand.py \
        --seeds seeds.jsonl \
        --out calibration_set_v1.1.jsonl \
        --brain http://127.0.0.1:8790 \
        --target-counts target_counts.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path


# Per-category target counts (sums to ~1440)
TARGET_COUNTS = {
    "coding": 200,
    "tool_call": 150,
    "math_numerical": 100,
    "finance": 80,
    "long_doc_research": 70,
    "creative_writing_zh": 100,
    "creative_writing_en": 50,
    "novel_multi_pov": 30,
    "social_media_styles": 50,
    "casual_chat_zh": 50,
    # drop
    "biology_pure": 100,
    "physics_quantum": 80,
    "medical_clinical": 80,
    "law_judicial": 60,
    "music_theory": 40,
    "sports_analytics": 40,
    "religious_text": 30,
    "quantum_chemistry": 30,
    "minor_lang": 100,
}


PARAPHRASE_PROMPT_ZH = """You are a prompt-paraphrasing assistant. Given a seed prompt, generate N variations that:
1. Stay strictly in the same category and language
2. Vary the topic / scenario / specific entities while keeping the same task type
3. Do NOT include the original wording verbatim
4. Output as a JSON array of strings, one per variation, no explanation

Seed (category={cat}, language={lang}):
{text}

Generate {n} paraphrases. Output ONLY the JSON array, nothing else."""


def call_brain(brain_url: str, prompt: str, max_tokens: int = 2000) -> str:
    """Call Lynn brain (chat completions) for paraphrasing."""
    body = json.dumps({
        "model": "Qwen3.6-35B-A3B-FP8",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,   # diverse paraphrases
    }).encode()
    req = urllib.request.Request(
        f"{brain_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"]


def parse_paraphrases(response: str) -> list[str]:
    """Extract JSON array of strings from brain response."""
    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    try:
        arr = json.loads(text.strip())
        if isinstance(arr, list):
            return [str(s) for s in arr if isinstance(s, str) and s.strip()]
    except json.JSONDecodeError:
        pass
    return []


def expand_seeds(seeds_path: Path, out_path: Path, brain_url: str,
                 target_counts: dict, dry_run: bool = False):
    """Expand each seed via brain to reach target counts per category."""
    # Load seeds
    seeds: list[dict] = []
    with open(seeds_path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))

    # Group seeds by category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for s in seeds:
        by_cat[s["cat"]].append(s)

    print(f"Loaded {len(seeds)} seeds across {len(by_cat)} categories", flush=True)

    # For each category, expand to target count
    out_rows: list[dict] = []
    for cat, target in target_counts.items():
        cat_seeds = by_cat.get(cat, [])
        if not cat_seeds:
            print(f"  ⚠️  no seeds for category {cat!r}, skipping", flush=True)
            continue
        n_per_seed = max(1, (target - len(cat_seeds) + len(cat_seeds) - 1) // len(cat_seeds))
        # Each seed becomes (1 original + n_per_seed paraphrases)
        print(f"  [{cat}] {len(cat_seeds)} seeds × {1 + n_per_seed} = "
              f"{len(cat_seeds) * (1 + n_per_seed)} target {target}",
              flush=True)

        for s in cat_seeds:
            out_rows.append(s)   # keep original

            if dry_run:
                # Insert placeholder paraphrases
                for i in range(n_per_seed):
                    placeholder = dict(s)
                    placeholder["id"] = f"{s['id']}-p{i+1}"
                    placeholder["text"] = f"[paraphrase {i+1} of {s['id']}]"
                    placeholder["seed_id"] = s["id"]
                    out_rows.append(placeholder)
                continue

            # Call brain to paraphrase
            prompt = PARAPHRASE_PROMPT_ZH.format(
                cat=cat, lang=s.get("lang", "?"), text=s["text"], n=n_per_seed,
            )
            try:
                response = call_brain(brain_url, prompt)
                paraphrases = parse_paraphrases(response)
                for i, p in enumerate(paraphrases[:n_per_seed]):
                    out_rows.append({
                        "id": f"{s['id']}-p{i+1}",
                        "cat": s["cat"],
                        "sub": s.get("sub", ""),
                        "lang": s.get("lang", "?"),
                        "text": p,
                        "seed_id": s["id"],
                    })
                print(f"    {s['id']} → {len(paraphrases)} paraphrases", flush=True)
            except Exception as e:
                print(f"    {s['id']} ERROR: {e}", flush=True)

    # Trim to target_counts (in case overshooting)
    by_cat_out: dict[str, list[dict]] = defaultdict(list)
    for r in out_rows:
        by_cat_out[r["cat"]].append(r)

    final_rows: list[dict] = []
    for cat, target in target_counts.items():
        rows = by_cat_out.get(cat, [])[:target]
        final_rows.extend(rows)

    with open(out_path, "w") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(final_rows)} prompts to {out_path}", flush=True)
    print(f"Per-category breakdown:")
    actual = defaultdict(int)
    for r in final_rows:
        actual[r["cat"]] += 1
    for cat, target in target_counts.items():
        n = actual.get(cat, 0)
        flag = "✅" if n >= target else "⚠️"
        print(f"  {flag} {cat:30} {n:4} / {target:4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="seeds.jsonl")
    ap.add_argument("--out", default="calibration_set_v1.1.jsonl")
    ap.add_argument("--brain", default="http://127.0.0.1:8790",
                    help="brain URL (must support /v1/chat/completions)")
    ap.add_argument("--target-counts", default=None,
                    help="optional path to JSON file with per-category target counts")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't call brain, just emit placeholders")
    args = ap.parse_args()

    if args.target_counts:
        target = json.loads(Path(args.target_counts).read_text())
    else:
        target = TARGET_COUNTS

    expand_seeds(Path(args.seeds), Path(args.out), args.brain,
                 target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
