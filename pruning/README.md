# Lynn-27B-A3B Pruning Pipeline

Complete tooling for the Phase 1 pruning of Lynn-35B-A3B → Lynn-27B-A3B
(memory: `project_lynn_27b_pruning_plan_0509.md`).

## Files

```
calibration/
  seeds.jsonl                    102 hand-curated seed prompts (19 cats)
  expand.py                      DeepSeek/brain paraphrasing → 1440 prompts
  calibration_set_v1.1.jsonl     1436 prompts (DeepSeek-expanded)
  README.md                      Calibration design + categories
profile_activations.py           Phase 1 W1: per-expert routing profiler
decide_pruning.py                Phase 1 W2: drop-30-experts algorithm
```

## End-to-end pipeline

### Step 1 (DGX, ~30-75 min depending on Phase 3 optimizations)

Run activation profiler — load Lynn engine + Qwen 3.6 35B-A3B baseline,
walk every prompt, capture per-layer top-K=8 expert routing decisions:

```bash
docker run --rm --gpus all --ipc=host --user 1000:1000 \
  -v /home/merkyor/models:/models \
  -v /tmp/lynn-engine:/work -w /work \
  -e PYTHONPATH=/work \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  bash -c "pip install -q --user transformers==5.8.0 && \
           python3 pruning/profile_activations.py \
             --calibration pruning/calibration/calibration_set_v1.1.jsonl \
             --out /home/merkyor/pruning/activation_profile_35B.jsonl"
```

Expected output: `activation_profile_35B.jsonl` ≈ 50 MB (1436 lines, one per prompt).
Each row contains the full 40-layer × per-token × top-8 routing tensor.

Resume support: re-running with the same `--out` path skips already-logged ids.

### Step 2 (CPU only, ~1 min)

Aggregate + select 30 drop candidates:

```bash
python3 pruning/decide_pruning.py \
  --profile activation_profile_35B.jsonl \
  --n-drop 30 \
  --max-per-layer 2 \
  --out-candidates drop_candidates_30.json \
  --out-stats per_expert_stats.json \
  --out-rationale decision_rationale.md
```

Outputs:
- `drop_candidates_30.json` — list of 30 (layer_idx, expert_id) pairs
- `per_expert_stats.json` — full aggregation (256×40 ≈ 10240 expert positions)
- `decision_rationale.md` — human-readable per-candidate justification

### Step 3 (DGX, ~10 min) — Physical weight deletion

Manual / scripted: load FP8 safetensors, drop the 30 selected experts'
weights, rewrite + repack. Output: `Qwen3.6-27B-A3B-Lynn-pruned-FP8/`.

### Step 4 (DGX, ~5 hours) — Recovery LoRA training

Per [tutorials/07_lora_on_gated_delta_net.md](../tutorials/07_lora_on_gated_delta_net.md),
LoRA r=384 on the surviving 226 experts + GatedDeltaNet projections + router.
Training data = Stage 1 + 5 + 4 + 6' + rehearsal mix (per memory
`feedback_lora_pipeline_stacking.md`).

### Step 5 (DGX) — Gate test + ship decision

V8/V9 benchmark vs Lynn-35B-A3B baseline. Pass criterion: ≤ 3% degradation.

## Decision algorithm rationale

Score per (layer_idx, expert_id) = `drop_pct - keep_pct`:

```
keep_pct = (# top-K hits on lynn_keep prompts) / (total lynn_keep top-K decisions)
drop_pct = (# top-K hits on lynn_drop prompts) / (total lynn_drop top-K decisions)
score    = drop_pct - keep_pct
```

Higher score = expert is biased toward lynn_drop categories = better drop candidate.

Filters:
- `drop_pct ≥ 0.1‰` (must activate meaningfully on lynn_drop, not idle)
- `keep_pct ≤ 0.15‰` (must NOT activate meaningfully on lynn_keep)
- `unique_drop_prompts ≥ 5` (avoid 1-prompt outliers)
- `≤ 2 experts per layer` (preserve router capacity)

Selected: top 30 by score after filters + per-layer cap.

## Categorization

`lynn_keep` (protected — must not drop, contributes to keep_pct):
  `coding`, `tool_call`, `math_numerical`, `creative_writing_zh`,
  `creative_writing_en`, `novel_multi_pov`, `social_media_styles`,
  `casual_chat_zh`, `finance`, `long_doc_research`

`lynn_drop` (target — droppable, contributes to drop_pct):
  `biology_pure`, `physics_quantum`, `medical_clinical`, `law_judicial`,
  `music_theory`, `sports_analytics`, `religious_text`, `quantum_chemistry`,
  `minor_lang` (fr/de/ru/ar/nl/pt)

## Open questions / future improvements

- **Redundancy detection**: experts e1, e2 that activate together on >80% of
  the same prompts are functionally redundant. Could drop one of each redundant
  pair AS WELL as low-keep experts. Not implemented v1.0; future enhancement.
- **Per-domain quality gates**: instead of ≤3% global degradation, set
  per-category gates (coding ≤1%, math ≤2%, biology ≥30% drop OK).
- **Iterative pruning**: drop 10, retrain, drop 10 more, retrain × 3 — vs
  one-shot 30. Iterative tends to be more robust.
- **Expert merging instead of dropping**: take 2 redundant experts, average
  their weights into 1, free the slot. Preserves capacity better than deletion.

These are Phase 2 (22B) considerations, not blocking v1.0.
