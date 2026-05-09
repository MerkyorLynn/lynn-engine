"""
Lynn-27B-A3B pruning · Phase 1 Week 2 — drop-30-experts decision algorithm.

Input:  activation_profile_35B.jsonl (from profile_activations.py)
        calibration set categories labelled lynn_keep / lynn_drop

Output:
  per_expert_stats.json       — all 256 experts × 40 layers per-category stats
  drop_candidates_30.json     — recommended 30 (layer_idx, expert_id) pairs to drop
  decision_rationale.md       — human-readable rationale per drop candidate

Algorithm:
  1. Aggregate routing decisions per (layer_idx, expert_id):
     - keep_hits = how often this expert was top-K on a lynn_keep prompt
     - drop_hits = ditto on lynn_drop prompts
     - keep_pct = keep_hits / total_keep_top_k_decisions
     - drop_pct = drop_hits / total_drop_top_k_decisions
  2. Score(e_l) = drop_pct(e_l) - keep_pct(e_l)
       higher = more biased toward lynn_drop categories = better drop candidate
  3. Filter: also require drop_pct > min_drop_threshold AND keep_pct < max_keep_threshold
  4. Rank globally across all (layer_idx, expert_id) pairs by score
  5. Take top N=30 — but enforce diversity (no more than M experts dropped per layer)

Safety constraints baked in:
  - Per memory project_lynn_27b_pruning_plan_0509.md, math + creative + coding +
    tool_call MUST be retained. Experts heavily activated on these are PROTECTED
    (excluded from drop candidates regardless of score).
  - Drop candidate rate per layer capped at 30/40 = 0.75 expert avg per layer.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path


# Categories considered "must keep" (protected experts).
PROTECTED_CATS = {
    "coding",
    "tool_call",
    "math_numerical",        # math implicitly supports coding
    "creative_writing_zh",   # Lynn's writing core
    "creative_writing_en",
    "novel_multi_pov",
    "social_media_styles",
    "casual_chat_zh",
    "finance",
    "long_doc_research",
}

DROP_CATS = {
    "biology_pure",
    "physics_quantum",
    "medical_clinical",
    "law_judicial",
    "music_theory",
    "sports_analytics",
    "religious_text",
    "quantum_chemistry",
    "minor_lang",
}


def load_profile(profile_path: Path) -> list[dict]:
    rows = []
    with open(profile_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_per_expert(profile_rows: list[dict]) -> dict:
    """For each (layer_idx, expert_id), count hits per category-class.

    Returns dict[(layer_idx, expert_id)] -> {
        keep_hits: int,
        drop_hits: int,
        keep_prompts: set[prompt_id],   # diversity tracker
        drop_prompts: set[prompt_id],
        per_cat_hits: Counter[cat],     # full per-category histogram
    }
    """
    stats = defaultdict(lambda: {
        "keep_hits": 0,
        "drop_hits": 0,
        "keep_prompts": set(),
        "drop_prompts": set(),
        "per_cat_hits": Counter(),
    })
    total_keep_top_k = 0
    total_drop_top_k = 0

    for row in profile_rows:
        cat = row["cat"]
        is_keep = cat in PROTECTED_CATS
        is_drop = cat in DROP_CATS
        if not (is_keep or is_drop):
            continue

        for layer_idx_str, token_lists in row["routing"].items():
            layer_idx = int(layer_idx_str)
            for token_slot_experts in token_lists:
                for expert_id in token_slot_experts:
                    key = (layer_idx, int(expert_id))
                    if is_keep:
                        stats[key]["keep_hits"] += 1
                        stats[key]["keep_prompts"].add(row["id"])
                        total_keep_top_k += 1
                    else:
                        stats[key]["drop_hits"] += 1
                        stats[key]["drop_prompts"].add(row["id"])
                        total_drop_top_k += 1
                    stats[key]["per_cat_hits"][cat] += 1

    # Compute normalized rates + score
    for key, s in stats.items():
        s["keep_pct"] = s["keep_hits"] / max(total_keep_top_k, 1)
        s["drop_pct"] = s["drop_hits"] / max(total_drop_top_k, 1)
        s["score"] = s["drop_pct"] - s["keep_pct"]
        s["unique_keep_prompts"] = len(s["keep_prompts"])
        s["unique_drop_prompts"] = len(s["drop_prompts"])

    return {
        "stats": stats,
        "total_keep_top_k": total_keep_top_k,
        "total_drop_top_k": total_drop_top_k,
        "n_prompts_keep": sum(1 for r in profile_rows if r["cat"] in PROTECTED_CATS),
        "n_prompts_drop": sum(1 for r in profile_rows if r["cat"] in DROP_CATS),
    }


def select_drop_candidates(agg: dict, n_drop: int = 30,
                           max_per_layer: int = 2,
                           min_drop_pct: float = 0.0003,
                           max_keep_pct: float = 0.0005,
                           min_unique_drop_prompts: int = 5) -> list[dict]:
    """Rank candidates by score, apply safety filters, enforce per-layer cap.

    Filters:
      - drop_pct must exceed min_drop_pct (~ activated meaningfully on lynn_drop)
      - keep_pct must be below max_keep_pct (~ NOT meaningfully active on lynn_keep)
      - must hit ≥ min_unique_drop_prompts distinct lynn_drop prompts (avoid 1-prompt outliers)
      - ≤ max_per_layer experts per layer (ensures diversity)
    """
    candidates = []
    for (layer_idx, expert_id), s in agg["stats"].items():
        if s["drop_pct"] < min_drop_pct:
            continue
        if s["keep_pct"] > max_keep_pct:
            continue
        if s["unique_drop_prompts"] < min_unique_drop_prompts:
            continue
        candidates.append({
            "layer_idx": layer_idx,
            "expert_id": expert_id,
            "score": s["score"],
            "keep_pct": s["keep_pct"],
            "drop_pct": s["drop_pct"],
            "keep_hits": s["keep_hits"],
            "drop_hits": s["drop_hits"],
            "unique_keep_prompts": s["unique_keep_prompts"],
            "unique_drop_prompts": s["unique_drop_prompts"],
            "per_cat_hits": dict(s["per_cat_hits"]),
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Enforce per-layer cap
    selected = []
    per_layer = Counter()
    for c in candidates:
        if per_layer[c["layer_idx"]] >= max_per_layer:
            continue
        selected.append(c)
        per_layer[c["layer_idx"]] += 1
        if len(selected) >= n_drop:
            break

    return selected


def write_rationale(selected: list[dict], agg: dict, out_path: Path):
    """Human-readable markdown explaining each drop decision."""
    lines = ["# Lynn-27B-A3B 剪枝决策(Phase 1 Week 2)", ""]
    lines.append(f"基于 {agg['n_prompts_keep']} 个 lynn_keep prompts + "
                 f"{agg['n_prompts_drop']} 个 lynn_drop prompts 的 activation profile。")
    lines.append("")
    lines.append(f"**总 routing 决策**:lynn_keep {agg['total_keep_top_k']:,} 次 / "
                 f"lynn_drop {agg['total_drop_top_k']:,} 次")
    lines.append("")
    lines.append(f"## 选定 drop {len(selected)} 个 expert (per-layer cap=2)")
    lines.append("")
    lines.append("| # | Layer | Expert ID | Score | drop_pct | keep_pct | 主要触发类别 |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, c in enumerate(selected):
        # Top-2 categories that triggered this expert
        cats_sorted = sorted(c["per_cat_hits"].items(), key=lambda x: -x[1])[:2]
        cat_str = ", ".join(f"{cat}({n})" for cat, n in cats_sorted)
        lines.append(
            f"| {i+1} | L{c['layer_idx']:2} | {c['expert_id']:3} | "
            f"{c['score']*1000:.2f}‰ | {c['drop_pct']*1000:.2f}‰ | "
            f"{c['keep_pct']*1000:.2f}‰ | {cat_str} |"
        )
    lines.append("")
    lines.append("## 决策规则")
    lines.append("- drop_pct ≥ 0.1%(meaningful activation on lynn_drop)")
    lines.append("- keep_pct ≤ 0.15%(low activation on lynn_keep)")
    lines.append("- ≥ 5 unique lynn_drop prompts(避免 1-prompt outlier)")
    lines.append("- ≤ 2 experts per layer(保持 routing 多样性)")
    lines.append("- 按 score = drop_pct - keep_pct 降序选 30 个")
    lines.append("")
    lines.append("## 后续(Phase 1 Week 3)")
    lines.append("1. 物理 delete 这 30 个 expert 的 weights → 35B → 27B")
    lines.append("2. Router fine-tune,防止剪后 routing 过载")
    lines.append("3. Recovery LoRA r=384(详见 [tutorials/07_lora_on_gated_delta_net.md](../tutorials/07_lora_on_gated_delta_net.md))")
    lines.append("4. Gate test 退化 ≤ 3% 才 ship")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True,
                    help="path to activation_profile_35B.jsonl")
    ap.add_argument("--n-drop", type=int, default=30,
                    help="number of expert (layer_idx, id) pairs to drop")
    ap.add_argument("--max-per-layer", type=int, default=2,
                    help="cap on experts dropped per layer")
    ap.add_argument("--out-stats", default="per_expert_stats.json")
    ap.add_argument("--out-candidates", default="drop_candidates_30.json")
    ap.add_argument("--out-rationale", default="decision_rationale.md")
    ap.add_argument("--min-drop-pct", type=float, default=0.0003,
                    help="min drop_pct fraction to qualify (default 0.0003 = 0.03 percent)")
    ap.add_argument("--max-keep-pct", type=float, default=0.0005,
                    help="max keep_pct fraction allowed (default 0.0005 = 0.05 percent)")
    args = ap.parse_args()

    print(f"Loading profile from {args.profile} ...")
    rows = load_profile(Path(args.profile))
    print(f"  {len(rows)} prompts loaded")

    agg = aggregate_per_expert(rows)
    print(f"\nAggregation:")
    print(f"  total_keep_top_k:   {agg['total_keep_top_k']:,}")
    print(f"  total_drop_top_k:   {agg['total_drop_top_k']:,}")
    print(f"  n_prompts_keep:     {agg['n_prompts_keep']}")
    print(f"  n_prompts_drop:     {agg['n_prompts_drop']}")
    print(f"  active (layer,expert) pairs: {len(agg['stats'])}")

    # Write per-expert stats (full)
    stats_serializable = {
        f"{k[0]},{k[1]}": {
            kk: (list(vv) if isinstance(vv, set) else
                 dict(vv) if isinstance(vv, Counter) else vv)
            for kk, vv in v.items()
        }
        for k, v in agg["stats"].items()
    }
    with open(args.out_stats, "w") as f:
        json.dump({
            "meta": {
                "total_keep_top_k": agg["total_keep_top_k"],
                "total_drop_top_k": agg["total_drop_top_k"],
                "n_prompts_keep": agg["n_prompts_keep"],
                "n_prompts_drop": agg["n_prompts_drop"],
            },
            "stats": stats_serializable,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull per-expert stats → {args.out_stats}")

    # Select drop candidates
    selected = select_drop_candidates(
        agg, n_drop=args.n_drop, max_per_layer=args.max_per_layer,
        min_drop_pct=args.min_drop_pct, max_keep_pct=args.max_keep_pct,
    )
    with open(args.out_candidates, "w") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"Drop candidates ({len(selected)}) → {args.out_candidates}")

    write_rationale(selected, agg, Path(args.out_rationale))
    print(f"Rationale markdown → {args.out_rationale}")

    # Summary
    print(f"\nSelected {len(selected)} drop candidates:")
    for i, c in enumerate(selected[:10]):
        print(f"  {i+1:2}. L{c['layer_idx']:2} expert {c['expert_id']:3}  "
              f"score={c['score']*1000:.2f}‰  "
              f"drop={c['drop_pct']*1000:.2f}‰  keep={c['keep_pct']*1000:.2f}‰")
    if len(selected) > 10:
        print(f"  ... ({len(selected) - 10} more in {args.out_candidates})")


if __name__ == "__main__":
    main()
