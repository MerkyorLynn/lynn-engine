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


def _read_brain_env(path: str = None) -> dict:
    """Load Lynn brain.env (key=value lines, may be quoted)."""
    if path is None:
        path = os.path.expanduser("~/.lynn/brain.env")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            out[k.strip()] = v
    return out


def call_deepseek(prompt: str, api_key: str,
                  base_url: str = "https://api.deepseek.com",
                  model: str = "deepseek-chat", max_tokens: int = 2000,
                  temperature: float = 0.7) -> str:
    """Call DeepSeek API for paraphrasing.

    Default model: deepseek-chat (V4-Flash equivalent). DEEPSEEK_REASONER_MODEL
    or 'deepseek-reasoner' is overkill for paraphrasing.

    Handles both base URL forms:
      https://api.deepseek.com           → appends /v1/chat/completions
      https://api.deepseek.com/v1        → appends /chat/completions
    """
    # Normalize: strip trailing /v1, then always append /v1/chat/completions
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/v1/chat/completions"

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"]


def call_brain(brain_url: str, prompt: str, max_tokens: int = 2000) -> str:
    """Call Lynn brain (chat completions) for paraphrasing."""
    body = json.dumps({
        "model": "Qwen3.6-35B-A3B-FP8",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
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
    """Extract JSON array of strings from API response.

    Handles three common formats:
      ["a", "b", "c"]                  — bare array
      {"paraphrases": ["a", "b", "c"]} — wrapped in JSON object (DeepSeek json_object mode)
      ```json\n[...]\n```              — markdown-fenced
    """
    text = response.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        parsed = json.loads(text)
        # Bare array
        if isinstance(parsed, list):
            return [str(s) for s in parsed if isinstance(s, str) and s.strip()]
        # Object — find first list-of-strings field
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                    return [s for s in v if s.strip()]
    except json.JSONDecodeError:
        pass
    return []


def expand_seeds(seeds_path: Path, out_path: Path, *,
                 backend: str = "deepseek",
                 brain_url: str = "http://127.0.0.1:8790",
                 deepseek_key: str = None,
                 deepseek_base: str = "https://api.deepseek.com",
                 deepseek_model: str = "deepseek-v4-flash",
                 target_counts: dict = None,
                 dry_run: bool = False,
                 progress_path: Path = None):
    """Expand each seed via brain or DeepSeek to reach target counts per category.

    Supports resume: if progress_path exists, skip seeds already in it.
    """
    if target_counts is None:
        target_counts = TARGET_COUNTS
    backend = backend.lower()
    if backend not in ("deepseek", "brain"):
        raise ValueError(f"backend must be 'deepseek' or 'brain', got {backend!r}")
    if backend == "deepseek" and not deepseek_key and not dry_run:
        raise ValueError("DeepSeek API key required (set --deepseek-key or DEEPSEEK_KEY env)")
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

            # Call backend to paraphrase
            prompt = PARAPHRASE_PROMPT_ZH.format(
                cat=cat, lang=s.get("lang", "?"), text=s["text"], n=n_per_seed,
            )
            try:
                if backend == "deepseek":
                    # DeepSeek json_object mode wants explicit JSON instruction
                    json_prompt = (
                        prompt
                        + '\n\nReturn JSON of the form {"paraphrases": ["...", "..."]}'
                    )
                    response = call_deepseek(
                        json_prompt, api_key=deepseek_key,
                        base_url=deepseek_base, model=deepseek_model,
                    )
                else:
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
                # Progress checkpoint (for resume)
                if progress_path is not None:
                    with open(progress_path, "a") as pf:
                        for r in out_rows[-len(paraphrases):]:
                            pf.write(json.dumps(r, ensure_ascii=False) + "\n")
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
    ap.add_argument("--backend", default="deepseek", choices=["deepseek", "brain"],
                    help="which paraphrasing API to use (default deepseek)")
    ap.add_argument("--brain", default="http://127.0.0.1:8790",
                    help="brain URL (when --backend=brain)")
    ap.add_argument("--deepseek-key", default=None,
                    help="DeepSeek API key (default: read DEEPSEEK_KEY env or ~/.lynn/brain.env)")
    ap.add_argument("--deepseek-base", default=None,
                    help="DeepSeek base URL (default: https://api.deepseek.com or DEEPSEEK_BASE)")
    ap.add_argument("--deepseek-model", default="deepseek-chat",
                    help="DeepSeek model name (default deepseek-chat = V4-Flash equivalent)")
    ap.add_argument("--target-counts", default=None,
                    help="optional path to JSON file with per-category target counts")
    ap.add_argument("--progress", default=None,
                    help="optional progress checkpoint file (for resume)")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't call API, just emit placeholders")
    args = ap.parse_args()

    if args.target_counts:
        target = json.loads(Path(args.target_counts).read_text())
    else:
        target = TARGET_COUNTS

    # Resolve credentials from CLI / env / brain.env
    env = _read_brain_env()
    deepseek_key = (
        args.deepseek_key
        or os.getenv("DEEPSEEK_KEY")
        or env.get("DEEPSEEK_KEY")
    )
    deepseek_base = (
        args.deepseek_base
        or os.getenv("DEEPSEEK_BASE")
        or env.get("DEEPSEEK_BASE")
        or "https://api.deepseek.com"
    )

    expand_seeds(
        Path(args.seeds), Path(args.out),
        backend=args.backend,
        brain_url=args.brain,
        deepseek_key=deepseek_key,
        deepseek_base=deepseek_base,
        deepseek_model=args.deepseek_model,
        target_counts=target,
        dry_run=args.dry_run,
        progress_path=Path(args.progress) if args.progress else None,
    )


if __name__ == "__main__":
    main()
