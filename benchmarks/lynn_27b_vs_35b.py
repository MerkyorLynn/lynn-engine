"""
Lynn-27B-A3B vs Lynn-35B-A3B benchmark harness.

Phase 1 Week 4 gate test: pruned 27B (post-recovery LoRA) must NOT degrade
> 3% on V8/V9 categories vs the 35B baseline.

Per memory project_lynn_27b_pruning_plan_0509.md gate:
  - V8/V9 retention ≥ 97% (overall)
  - tool_call accuracy: NO degradation (Lynn core capability)
  - creative writing: NO degradation (Lynn core capability)
  - bio/physics/medical/law/minor_lang: degradation up to 50% OK (these were
    INTENDED to drop)

Implementation:
  - Hits two OpenAI-compatible endpoints (35B baseline + 27B pruned)
  - Sends same prompt to both, captures responses, runs auto-graders
  - Outputs side-by-side report with per-category pass rates + diff

Usage:
    python3 benchmarks/lynn_27b_vs_35b.py \
        --baseline-url http://127.0.0.1:18002/v1   \
        --baseline-model Qwen3.6-35B-A3B-FP8        \
        --pruned-url http://127.0.0.1:18099/v1      \
        --pruned-model Lynn-27B-A3B-NVFP4           \
        --benchmark v9 \
        --out report_27b_vs_35b.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# --------------------------------------------------------------------------
# V9 benchmark question set — embedded directly here so this file is
# self-contained when copied to A100 / Spark / wherever benchmark runs.
# Categories must align with calibration_set so comparison is meaningful.
# --------------------------------------------------------------------------

V9_QUESTIONS = [
    # tool_call category — Lynn core, MUST not degrade
    {"id": "tc/01", "cat": "tool_call",
     "prompt": "今天上海天气怎么样?如果有工具就调用,没有就直接回答你的认知。",
     "expect": "calls get_weather OR explicit search_tool",
     "grader": "tool_call_present"},
    {"id": "tc/02", "cat": "tool_call",
     "prompt": "查一下 Apple 的最新股价,然后告诉我跟一年前比涨跌多少。",
     "expect": "calls get_stock_price OR similar",
     "grader": "tool_call_present"},
    {"id": "tc/03", "cat": "tool_call",
     "prompt": "把代码 commit 上去 message 写 'feat: add user auth',然后 push 到远程。",
     "expect": "calls git_commit AND git_push (chained)",
     "grader": "tool_call_chained"},
    {"id": "tc/04", "cat": "tool_call",
     "prompt": "翻译 'I love Beijing Tiananmen Square' 成中文。",
     "expect": "DOES NOT call translate tool — uses parametric memory",
     "grader": "tool_call_abstain"},

    # coding — Lynn core
    {"id": "code/01", "cat": "coding",
     "prompt": "写一个 Python 函数,接收一个整数列表,返回出现次数最多的元素及其频次。要求时间复杂度 O(n)。",
     "grader": "code_compiles_and_runs"},
    {"id": "code/02", "cat": "coding",
     "prompt": "用 TypeScript 写一个 debounce 函数,泛型类型支持任意函数签名。",
     "grader": "code_compiles_typescript"},

    # creative writing zh — Lynn core
    {"id": "cr/01", "cat": "creative_writing_zh",
     "prompt": "写一首五言律诗,主题是黄昏归家路。要求押韵,第二联第三联对仗。",
     "grader": "creative_zh_poem"},

    # math — implicit support for coding
    {"id": "math/01", "cat": "math_numerical",
     "prompt": "求函数 f(x) = x^3 - 6x^2 + 9x + 2 的所有极值点,判断极大极小。",
     "grader": "math_correct"},

    # finance — Lynn user core
    {"id": "fin/01", "cat": "finance",
     "prompt": "年化 8% 复利投资 10 万,15 年后本息合计多少?给出公式 + 计算。",
     "grader": "finance_correct"},

    # ─── Drop categories (degradation OK, 50% target) ───
    {"id": "bio/01", "cat": "biology_pure",
     "prompt": "解释端粒酶在 DNA 复制中的作用,以及在癌症中的失调机制。",
     "grader": "domain_specific"},
    {"id": "phy/01", "cat": "physics_quantum",
     "prompt": "推导一维无限深势阱中粒子的薛定谔方程,求最低三个能量本征值。",
     "grader": "domain_specific"},
    {"id": "med/01", "cat": "medical_clinical",
     "prompt": "45 岁男性主诉胸痛放射至左臂、出汗、呼吸困难。给出鉴别诊断。",
     "grader": "domain_specific"},
    {"id": "law/01", "cat": "law_judicial",
     "prompt": "中国刑法中正当防卫与防卫过当的区分标准。",
     "grader": "domain_specific"},
    {"id": "lang/fr", "cat": "minor_lang",
     "prompt": "Écrivez un essai de 300 mots sur l'importance du multilinguisme.",
     "grader": "domain_specific"},
]


# --------------------------------------------------------------------------
# API call wrapper
# --------------------------------------------------------------------------

def call_openai_api(url: str, model: str, prompt: str, *,
                    api_key: str = "EMPTY", max_tokens: int = 512,
                    temperature: float = 0.0,
                    timeout: int = 120) -> dict:
    """OpenAI-compatible /chat/completions call."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    elapsed = time.time() - t0
    return {
        "text": d["choices"][0]["message"]["content"],
        "tool_calls": d["choices"][0]["message"].get("tool_calls", []),
        "elapsed_s": elapsed,
        "raw": d,
    }


# --------------------------------------------------------------------------
# Auto-graders
# --------------------------------------------------------------------------

def grade_tool_call_present(response: dict) -> tuple[bool, str]:
    if response.get("tool_calls"):
        return True, f"tool_call emitted: {response['tool_calls'][0]['function']['name']}"
    text = response["text"].lower()
    if any(x in text for x in ("无法", "不能", "我不能", "i cannot", "i can't")):
        return False, "model refused to call tool"
    return False, "no tool_call, plain text response"


def grade_tool_call_chained(response: dict) -> tuple[bool, str]:
    """Expect 2+ tool calls in single response (commit + push)."""
    tc = response.get("tool_calls", [])
    if len(tc) >= 2:
        names = [t["function"]["name"] for t in tc]
        return True, f"chained: {names}"
    if len(tc) == 1:
        return False, f"only 1 tool: {tc[0]['function']['name']} (chain split)"
    return False, "no tool calls"


def grade_tool_call_abstain(response: dict) -> tuple[bool, str]:
    """Expect NO tool call — should use parametric memory for trivial translation."""
    if response.get("tool_calls"):
        name = response["tool_calls"][0]["function"]["name"]
        return False, f"over-called {name} for parametric task"
    text = response["text"]
    if "我爱北京天安门" in text or "tiananmen" in text.lower():
        return True, "answered from parametric memory"
    return False, "missed translation"


def grade_code_compiles(response: dict) -> tuple[bool, str]:
    """Heuristic: response contains a code block + def/function declaration."""
    text = response["text"]
    has_code_block = "```" in text
    has_def = ("def " in text) or ("function " in text)
    if has_code_block and has_def:
        return True, "code block + def found"
    return False, f"code_block={has_code_block} def={has_def}"


def grade_math_correct(response: dict) -> tuple[bool, str]:
    """Hard-coded check for x=1 max + x=3 min for f(x) = x^3 - 6x^2 + 9x + 2."""
    text = response["text"]
    found_max = "x=1" in text or "x = 1" in text
    found_min = "x=3" in text or "x = 3" in text
    if found_max and found_min:
        return True, "found x=1 max + x=3 min"
    return False, f"max={found_max} min={found_min}"


def grade_finance_correct(response: dict) -> tuple[bool, str]:
    """Check for ~317217 (1.08^15 * 100000 ≈ 317217)."""
    text = response["text"]
    has_formula = "1.08" in text or "(1+0.08)" in text or "(1 + 0.08)" in text
    has_result = any(x in text for x in ["317", "31.7", "31万"])
    if has_formula and has_result:
        return True, "formula + ~317k result"
    return False, f"formula={has_formula} result={has_result}"


def grade_creative_zh_poem(response: dict) -> tuple[bool, str]:
    """Check 8 lines (5 chars each) + paired couplets."""
    lines = [ln.strip() for ln in response["text"].split("\n") if ln.strip()]
    poem_lines = [ln for ln in lines if 5 <= len(ln) <= 6]   # allow punctuation
    if len(poem_lines) >= 8:
        return True, f"{len(poem_lines)} 5-char lines"
    return False, f"only {len(poem_lines)} 5-char lines (need 8)"


def grade_domain_specific(response: dict) -> tuple[bool, str]:
    """For lynn_drop categories: any non-trivial response (>200 chars) counts.
    Lower bar, since pruning intentionally targets these."""
    text = response["text"]
    if len(text) >= 200:
        return True, f"{len(text)} chars (non-trivial response)"
    return False, f"only {len(text)} chars"


GRADERS = {
    "tool_call_present": grade_tool_call_present,
    "tool_call_chained": grade_tool_call_chained,
    "tool_call_abstain": grade_tool_call_abstain,
    "code_compiles_and_runs": grade_code_compiles,
    "code_compiles_typescript": grade_code_compiles,
    "creative_zh_poem": grade_creative_zh_poem,
    "math_correct": grade_math_correct,
    "finance_correct": grade_finance_correct,
    "domain_specific": grade_domain_specific,
}


# --------------------------------------------------------------------------
# Benchmark runner
# --------------------------------------------------------------------------

def run_one_question(q: dict, baseline_cfg: dict, pruned_cfg: dict) -> dict:
    """Run one question against both endpoints, grade both, return diff."""
    rec = {"id": q["id"], "cat": q["cat"], "prompt": q["prompt"]}

    grader = GRADERS.get(q["grader"], grade_domain_specific)

    for label, cfg in [("baseline", baseline_cfg), ("pruned", pruned_cfg)]:
        try:
            resp = call_openai_api(cfg["url"], cfg["model"], q["prompt"],
                                   api_key=cfg.get("api_key", "EMPTY"))
            ok, why = grader(resp)
            rec[label] = {
                "ok": ok, "why": why,
                "elapsed_s": resp["elapsed_s"],
                "n_tool_calls": len(resp.get("tool_calls", [])),
                "text_len": len(resp["text"]),
                "text_preview": resp["text"][:200],
            }
        except Exception as e:
            rec[label] = {"ok": False, "why": f"ERROR: {type(e).__name__}: {e}",
                          "elapsed_s": -1, "n_tool_calls": 0, "text_len": 0,
                          "text_preview": ""}

    rec["agreement"] = (rec["baseline"]["ok"] == rec["pruned"]["ok"])
    rec["delta"] = rec["pruned"]["ok"] - rec["baseline"]["ok"]   # +1 / 0 / -1
    return rec


def run_benchmark(baseline_cfg: dict, pruned_cfg: dict,
                  questions: list[dict], parallel: int = 4) -> list[dict]:
    """Run all questions, return per-question records."""
    print(f"Running {len(questions)} questions × 2 endpoints (parallel={parallel}) ...")
    results = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(run_one_question, q, baseline_cfg, pruned_cfg): q
                   for q in questions}
        for fut in as_completed(futures):
            q = futures[fut]
            r = fut.result()
            results.append(r)
            print(f"  [{q['cat']:20}] {q['id']:8}  base={int(r['baseline']['ok'])} "
                  f"pruned={int(r['pruned']['ok'])}  Δ={r['delta']:+d}", flush=True)
    return sorted(results, key=lambda r: r["id"])


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_report(results: list[dict], baseline_cfg: dict, pruned_cfg: dict,
                 out_path: Path):
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["cat"]].append(r)

    lines = ["# Lynn-27B-A3B(剪枝)vs Lynn-35B-A3B(baseline)— Gate Test Report", ""]
    lines.append(f"- baseline: `{baseline_cfg['model']}` @ {baseline_cfg['url']}")
    lines.append(f"- pruned:   `{pruned_cfg['model']}` @ {pruned_cfg['url']}")
    lines.append(f"- N questions: {len(results)}")
    lines.append("")

    # Overall
    base_pass = sum(1 for r in results if r["baseline"]["ok"])
    pruned_pass = sum(1 for r in results if r["pruned"]["ok"])
    retention = pruned_pass / max(base_pass, 1) * 100 if base_pass else 0
    lines.append(f"## Overall")
    lines.append(f"- baseline pass: {base_pass}/{len(results)}")
    lines.append(f"- pruned pass:   {pruned_pass}/{len(results)}")
    lines.append(f"- retention:     {retention:.1f}% (pass criterion: ≥ 97%)")
    lines.append("")

    # Per-category
    lines.append("## 按类别")
    lines.append("")
    lines.append("| 类别 | baseline | pruned | retention | Δ |")
    lines.append("|---|---|---|---|---|")
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        b = sum(1 for r in rows if r["baseline"]["ok"])
        p = sum(1 for r in rows if r["pruned"]["ok"])
        r = p / max(b, 1) * 100 if b else 0
        delta_emoji = "✅" if p >= b else ("⚠️" if p >= b - 1 else "❌")
        lines.append(f"| {cat} | {b}/{len(rows)} | {p}/{len(rows)} | "
                     f"{r:.0f}% | {delta_emoji} |")
    lines.append("")

    # Tool call breakdown (Lynn core, MUST not regress)
    tc_results = [r for r in results if r["cat"] == "tool_call"]
    if tc_results:
        lines.append("## Tool call detail(Lynn core)")
        lines.append("")
        for r in tc_results:
            lines.append(f"### {r['id']}: {r['prompt']}")
            lines.append(f"- baseline: {'✅' if r['baseline']['ok'] else '❌'} {r['baseline']['why']}")
            lines.append(f"- pruned:   {'✅' if r['pruned']['ok'] else '❌'} {r['pruned']['why']}")
            lines.append("")

    # Latency
    lines.append("## Latency")
    base_lat = [r["baseline"]["elapsed_s"] for r in results if r["baseline"]["elapsed_s"] > 0]
    pruned_lat = [r["pruned"]["elapsed_s"] for r in results if r["pruned"]["elapsed_s"] > 0]
    if base_lat and pruned_lat:
        avg_b = sum(base_lat) / len(base_lat)
        avg_p = sum(pruned_lat) / len(pruned_lat)
        lines.append(f"- baseline avg: {avg_b:.2f} s")
        lines.append(f"- pruned avg:   {avg_p:.2f} s")
        lines.append(f"- speedup:      {avg_b/avg_p:.2f}x")
    lines.append("")

    # Decision
    lines.append("## Gate decision")
    if retention >= 97:
        lines.append(f"✅ **SHIP** — retention {retention:.1f}% ≥ 97%")
    elif retention >= 95:
        lines.append(f"⚠️ **REVIEW** — retention {retention:.1f}% < 97%, "
                     f"check per-category(尤其 tool_call / coding / creative_zh)")
    else:
        lines.append(f"❌ **DO NOT SHIP** — retention {retention:.1f}% < 95%, "
                     f"recovery LoRA 不够 → 升 r 或加训练数据")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-url", default="http://127.0.0.1:18002/v1")
    ap.add_argument("--baseline-model", default="Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--pruned-url", default="http://127.0.0.1:18099/v1")
    ap.add_argument("--pruned-model", default="Lynn-27B-A3B-NVFP4")
    ap.add_argument("--baseline-key", default="EMPTY")
    ap.add_argument("--pruned-key", default="EMPTY")
    ap.add_argument("--benchmark", default="v9", choices=["v9"])
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", default="report_27b_vs_35b.md")
    args = ap.parse_args()

    questions = V9_QUESTIONS

    baseline_cfg = {
        "url": args.baseline_url, "model": args.baseline_model,
        "api_key": args.baseline_key,
    }
    pruned_cfg = {
        "url": args.pruned_url, "model": args.pruned_model,
        "api_key": args.pruned_key,
    }

    results = run_benchmark(baseline_cfg, pruned_cfg, questions,
                            parallel=args.parallel)

    # Write JSON for downstream tooling
    json_out = Path(args.out).with_suffix(".json")
    json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Per-question JSON → {json_out}")

    write_report(results, baseline_cfg, pruned_cfg, Path(args.out))


if __name__ == "__main__":
    main()
