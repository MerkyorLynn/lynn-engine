# Lynn-27B-A3B Pruning · Calibration Set v1.1

Activation-profiling prompts for Phase 1 of the Lynn-35B-A3B → Lynn-27B-A3B
expert pruning pipeline. Per-expert routing frequencies on these prompts
classify each of the 256 MoE experts as **lynn-keep** / **edge** / **lynn-drop** /
**redundant**.

Strategy: prompts are split into **lynn_keep** categories (must keep — Lynn's
core use cases) and **lynn_drop** categories (safe to prune — niche domains
Lynn users don't engage). Experts that activate heavily on lynn_keep prompts
are PROTECTED. Experts that activate heavily on lynn_drop prompts are
PRUNE CANDIDATES.

## Files

  seeds.jsonl                — 95 hand-curated seed prompts (5-10 per category)
  expand.py                  — script to paraphrase seeds via Lynn brain → 1440 prompts
  calibration_set_v1.1.jsonl — generated full set (after running expand.py)

## Categories + target counts

### lynn_keep (880 prompts total)

| Category | Count | Why keep |
|---|---|---|
| coding | 200 | Lynn's primary user activity |
| tool_call | 150 | Stage 1 + Stage 5 LoRA training target |
| math_numerical | 100 | Implicit support for 30-60% of coding |
| finance | 80 | Lynn's stock/fund/macro analysis flows |
| long_doc_research | 70 | Stage 4 research pipeline |
| creative_writing_zh | 100 | Lynn 写作模式 + 7 social media slash commands |
| creative_writing_en | 50 | English writing + translation |
| novel_multi_pov | 30 | novel-workshop substrate (Stage 7 training) |
| social_media_styles | 50 | /xhs /gzh /weibo /douyin /zhihu slash commands |
| casual_chat_zh | 50 | Daily conversational baseline |

### lynn_drop (560 prompts total)

| Category | Count | Why drop |
|---|---|---|
| biology_pure | 100 | GPQA bio domain — Lynn users never query |
| physics_quantum | 80 | QFT / relativity niche; applied physics overlaps with math |
| medical_clinical | 80 | Diagnostic / pharma — out of scope |
| law_judicial | 60 | Lynn doesn't give legal advice |
| music_theory | 40 | Niche academic domain |
| sports_analytics | 40 | Niche; some overlap with stats |
| religious_text | 30 | Niche academic domain |
| quantum_chemistry | 30 | Niche academic domain |
| minor_lang | 100 | French/German/Russian/Arabic/Dutch/Portuguese (CN/EN sufficient for Lynn) |

## Usage

### Step 1: Verify seeds (no brain needed)

```bash
wc -l seeds.jsonl                               # should be ~95
head -1 seeds.jsonl | python3 -m json.tool      # check JSON format
```

### Step 2: Dry-run expansion (no brain needed)

```bash
python3 expand.py --dry-run --out preview.jsonl
# Verify per-category target counts in output
```

### Step 3: Real expansion (requires Lynn brain at port 8790)

```bash
python3 expand.py \
    --seeds seeds.jsonl \
    --out calibration_set_v1.1.jsonl \
    --brain http://127.0.0.1:8790
# Takes ~15-30 min depending on brain throughput.
# Each seed → ~10-15 paraphrases (varies per category target).
```

### Step 4: Use during Lynn-35B-A3B baseline activation profiling

The full set goes through the model (in inference mode) to record routing
decisions per expert per prompt. See Phase 1 Week 1 of
`memory/project_lynn_27b_pruning_plan_0509.md` for the profiling pipeline.

## JSONL schema

```json
{"id": "keep/coding/001", "cat": "coding", "sub": "algorithm",
 "lang": "en", "text": "Implement a Python function that..."}
```

After expansion, paraphrases get `seed_id` field linking back to original:

```json
{"id": "keep/coding/001-p3", "cat": "coding", "sub": "algorithm",
 "lang": "en", "text": "Write a Python program for finding...",
 "seed_id": "keep/coding/001"}
```

## Versioning

  v1.0 — initial 95 seeds (this commit)
  v1.1 — full 1440 prompts after brain paraphrasing (TBD)

When updating seeds, bump v1.x and re-run expand.py.

## Notes

- Seeds are intentionally diverse within each category to avoid biasing
  the activation profile toward a single sub-topic.
- Paraphrases share the same `cat` but vary `sub` and entities to maximize
  expert routing diversity.
- After Phase 1 ships Lynn-27B-A3B, this calibration set should also be
  used for Lynn-22B-A3B Phase 2 calibration (memory: `project_lynn_27b_pruning_plan_0509.md` Phase 2).
