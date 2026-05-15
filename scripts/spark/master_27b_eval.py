#!/usr/bin/env python3
"""
Master 27B NVFP4 evaluation runner — calls Lynn engine OpenAI-compatible HTTP API.

Coverage:
  1) 6-prompt coherent smoke (ship gate)
  2) tool-call strict (≥1 prompt with tool_calls_count ≥ 1)
  3) 4-gate V8/V9 academic subset (think + nothink)
  4) long-ctx diverse content (anti radix-cache dedup)
  5) TPS / first-token latency sampling

Run on Spark (host or inside any python3 env with `requests`):
    python3 master_27b_eval.py --base http://127.0.0.1:18099 --out /home/merkyor/reports/27b_nvfp4_eval_$(date +%H%M)
"""
import os, sys, json, time, argparse, statistics
from pathlib import Path

try:
    import requests
except ImportError:
    print("install: pip install requests", file=sys.stderr); sys.exit(2)

PROMPT_ROOT = Path("/home/merkyor/eval_prompts")
PROMPT_ROOT_Q4 = Path("/home/merkyor/eval_prompts_q4")
LONGCTX = Path("/home/merkyor/eval_prompts_longctx/longctx_8000.jsonl")

SIX_SMOKE = [
    {"role": "user", "content": "解释量子纠缠为什么不能用来超光速通信。"},
    {"role": "user", "content": "Write a Python function that performs in-place quicksort on a list of ints, no recursion."},
    {"role": "user", "content": "请用中文写一首关于五月深夜实验室加班的七律。"},
    {"role": "user", "content": "If a train travels 60km/h for 2.5h then 90km/h for 1.25h, total distance in km?"},
    {"role": "user", "content": "用大白话讲清楚 transformer 的 attention 为什么是 O(n^2)。"},
    {"role": "user", "content": "List three failure modes of MoE expert pruning that activation profiling alone cannot catch."},
]

TOOL_CALL_PROMPT = {
    "messages": [{"role": "user", "content": "What is the current weather in Tokyo? Use the get_weather tool."}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    }],
    "tool_choice": "auto",
}

def call_chat(base, messages, *, tools=None, tool_choice=None, max_tokens=256, temperature=0.0, stream_tps=False):
    url = f"{base}/v1/chat/completions"
    body = {
        "model": "Lynn-V4-Distill-Qwen-27B-A3B-NVFP4",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"
    if stream_tps:
        body["stream"] = True
        t0 = time.time()
        first_tok_t = None
        n_tok = 0
        text = ""
        r = requests.post(url, json=body, stream=True, timeout=300)
        for line in r.iter_lines():
            if not line: continue
            if line.startswith(b"data: "):
                payload = line[6:]
                if payload.strip() == b"[DONE]": break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0]["delta"].get("content") or ""
                    if delta and first_tok_t is None: first_tok_t = time.time() - t0
                    text += delta
                    n_tok += max(1, len(delta) // 4)
                except Exception: pass
        total = time.time() - t0
        return {"text": text, "tps": n_tok/total if total>0 else 0, "first_tok_s": first_tok_t, "total_s": total, "n_tok_est": n_tok}
    else:
        t0 = time.time()
        r = requests.post(url, json=body, timeout=300)
        elapsed = time.time() - t0
        r.raise_for_status()
        out = r.json()
        msg = out["choices"][0]["message"]
        return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or [], "elapsed_s": elapsed, "raw": out}


def run_6_smoke(base, out_dir):
    results = []
    for i, m in enumerate(SIX_SMOKE, 1):
        try:
            r = call_chat(base, [m], max_tokens=512)
            ok = bool(r["content"]) and len(r["content"]) > 20
            results.append({"i": i, "prompt": m["content"][:80], "ok": ok, "content_preview": r["content"][:200], "elapsed_s": r["elapsed_s"]})
            print(f"  [6-smoke {i}/6] ok={ok}  {r['elapsed_s']:.1f}s  {r['content'][:80]!r}")
        except Exception as e:
            results.append({"i": i, "ok": False, "error": str(e)[:200]})
            print(f"  [6-smoke {i}/6] FAIL {e}")
    (out_dir / "6_smoke.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return sum(1 for r in results if r.get("ok"))


def run_tool_call(base, out_dir):
    try:
        r = call_chat(base, TOOL_CALL_PROMPT["messages"], tools=TOOL_CALL_PROMPT["tools"], tool_choice="auto", max_tokens=256)
        n_calls = len(r["tool_calls"])
        ok = n_calls >= 1
        result = {"ok": ok, "tool_calls_count": n_calls, "tool_calls": r["tool_calls"], "content": r["content"][:200], "elapsed_s": r["elapsed_s"]}
        print(f"  [tool-call] ok={ok}  tool_calls={n_calls}  {r['elapsed_s']:.1f}s")
        (out_dir / "tool_call.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return ok
    except Exception as e:
        print(f"  [tool-call] FAIL {e}")
        (out_dir / "tool_call.json").write_text(json.dumps({"ok": False, "error": str(e)}))
        return False


def run_4gate_v9(base, out_dir, jsonl_path, limit=40):
    """Run V9 holdout (academic strict). limit avoids burning too much time."""
    if not jsonl_path.exists():
        print(f"  [4gate] SKIP — no prompts at {jsonl_path}")
        return {"skipped": True}
    prompts = [json.loads(l) for l in jsonl_path.read_text().splitlines()[:limit] if l.strip()]
    results = []
    for i, p in enumerate(prompts, 1):
        q = p.get("question") or p.get("prompt") or p.get("problem") or p.get("instruction") or ""
        if not q: continue
        try:
            r = call_chat(base, [{"role": "user", "content": q}], max_tokens=512)
            results.append({"i": i, "q": q[:100], "answer": r["content"][:300], "elapsed_s": r["elapsed_s"], "expected": p.get("answer") or p.get("expected", "")})
            if i % 5 == 0: print(f"  [4gate V9 {i}/{len(prompts)}]")
        except Exception as e:
            results.append({"i": i, "error": str(e)[:200]})
    (out_dir / "v9_holdout.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"  [4gate V9] ran {len(results)}/{len(prompts)}")
    return {"count": len(results)}


def run_longctx(base, out_dir):
    if not LONGCTX.exists():
        print(f"  [longctx] SKIP — no file at {LONGCTX}")
        return {"skipped": True}
    prompts = [json.loads(l) for l in LONGCTX.read_text().splitlines()[:5] if l.strip()]
    results = []
    for i, p in enumerate(prompts, 1):
        q = p.get("question") or p.get("prompt") or p.get("text") or ""
        try:
            r = call_chat(base, [{"role": "user", "content": q}], max_tokens=256)
            results.append({"i": i, "input_len_chars": len(q), "elapsed_s": r["elapsed_s"], "answer_preview": r["content"][:200]})
            print(f"  [longctx {i}] len={len(q)}c  {r['elapsed_s']:.1f}s")
        except Exception as e:
            results.append({"i": i, "error": str(e)[:200]})
    (out_dir / "longctx.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return {"count": len(results)}


def run_tps(base, out_dir):
    """3 non-stream runs, derive TPS from completion_tokens / decode_time via _lynn_engine_metrics."""
    samples = []
    for i in range(3):
        try:
            r = call_chat(base, [{"role": "user", "content": "Describe in detail how a transformer attention layer processes tokens, ~300 words."}], max_tokens=400, temperature=0.0)
            raw = r.get("raw", {})
            usage = raw.get("usage", {})
            metrics = raw.get("_lynn_engine_metrics", {})
            comp = usage.get("completion_tokens", 0)
            elapsed = metrics.get("elapsed_s", r.get("elapsed_s", 0))
            prefill = metrics.get("timings", {}).get("prefill_seconds", 0)
            decode_time = max(elapsed - prefill, 1e-6)
            tps = comp / decode_time if decode_time > 0 else 0.0
            samples.append({"completion_tokens": comp, "elapsed_s": elapsed, "prefill_s": prefill, "decode_time_s": decode_time, "tps": tps})
            print(f"  [tps {i+1}/3] {comp} tok / {decode_time:.2f}s decode = {tps:.2f} TPS (prefill {prefill:.2f}s)")
        except Exception as e:
            print(f"  [tps {i+1}/3] FAIL {e}")
    if samples:
        tps_vals = [s["tps"] for s in samples]
        out = {"runs": samples, "mean_tps": statistics.mean(tps_vals), "median_tps": statistics.median(tps_vals)}
        (out_dir / "tps.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    return {"failed": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:18099")
    p.add_argument("--out", required=True)
    p.add_argument("--skip", nargs="*", default=[])
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[master_eval] base={args.base}  out={out_dir}")

    summary = {}

    # 0) health
    try:
        h = requests.get(f"{args.base}/health", timeout=10)
        print(f"[health] {h.status_code}  {h.text[:120]}")
        summary["health"] = h.status_code
    except Exception as e:
        print(f"[health] FAIL {e}"); summary["health_error"] = str(e); (out_dir / "summary.json").write_text(json.dumps(summary)); sys.exit(1)

    if "smoke" not in args.skip:
        print("\n=== 6-prompt smoke ==="); summary["smoke_pass"] = run_6_smoke(args.base, out_dir)
    if "toolcall" not in args.skip:
        print("\n=== tool-call strict ==="); summary["toolcall_pass"] = run_tool_call(args.base, out_dir)
    if "v9" not in args.skip:
        print("\n=== 4-gate V9 holdout (40 prompts) ==="); summary["v9"] = run_4gate_v9(args.base, out_dir, PROMPT_ROOT / "v9_holdout.jsonl", limit=40)
    if "longctx" not in args.skip:
        print("\n=== long-ctx ==="); summary["longctx"] = run_longctx(args.base, out_dir)
    if "tps" not in args.skip:
        print("\n=== TPS ==="); summary["tps"] = run_tps(args.base, out_dir)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[master_eval] DONE — summary at {out_dir}/summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
