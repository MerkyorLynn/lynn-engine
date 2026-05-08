#!/usr/bin/env python3
"""
Lynn Engine · Phase 0 throughput baseline benchmark

Measures generation tokens/sec on an OpenAI-compatible endpoint at
various prompt lengths. Works against vLLM, SGLang, llama.cpp server,
or (future) Lynn Engine itself.

Usage:
    # Spark Qwen 3.6 35B-A3B-FP8 vLLM:
    python throughput_bench.py --url http://127.0.0.1:18002/v1 --model Qwen3.6-35B-A3B-FP8

    # Spark Step-3.5-Flash GGUF llama.cpp:
    python throughput_bench.py --url http://127.0.0.1:8088/v1 --model step3p5-flash-q4ks

    # 5090 / RTX PRO 6000 future:
    python throughput_bench.py --url http://127.0.0.1:18099/v1 --model qwen3.6-35b-a3b-fp8

Output: JSON with per-prompt-length throughput + p50/p95 latency.
"""
import argparse
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from statistics import median


# ───── Prompt generators (controlled length) ─────
PROMPTS = {
    "short_128": "用 100 字解释什么是 Transformer 架构，重点说明 attention 机制的关键点。",
    "med_512": "请详细介绍一下中国 2025-2026 年的 AI 芯片产业格局，包括头部厂商（华为昇腾、寒武纪、壁仞、燧原等）的产品定位、技术路线、市场份额和最新进展。请按照以下结构展开：(1) 整体市场规模和增长趋势；(2) 头部厂商对比表；(3) 与英伟达 H100/H200/B200 的性能差距分析；(4) 国内云厂商（阿里、腾讯、字节）的采购偏好；(5) 政策和地缘对供给的影响。请客观分析，给出数据支撑。",
    "long_2048": "我现在要做一个完整的论文调研报告，主题是大语言模型在工具调用（function calling / tool use）维度的能力评估及优化。请按以下大纲展开，每节至少 200 字：\n\n第一部分 · 引言（1）什么是 tool use 能力（2）为什么对 AI Agent 至关重要（3）当前业界主要 benchmark 概览\n\n第二部分 · 已有 benchmark 综述（1）BFCL（Berkeley Function Calling Leaderboard）的设计理念、题型分类、评分机制（2）ToolBench 的多步任务设计（3）API-Bank 的 API 调用真实性（4）MetaTool 是否需要调工具的判断维度（5）这些 benchmark 的共同盲区\n\n第三部分 · 真实 production 场景下的 tool use 失败模式（1）staleness refusal：模型自以为知道而拒调（2）chain splitting：多步动作只调第一步（3）over-call：简单任务过度调用工具（4）action-verb hedging：执行型动词触发安全 hedge\n\n第四部分 · 缓解策略（1）prompt-level：system prompt / few-shot / tool_choice forcing（2）routing-level：confidence calibration / two-pass abstention（3）training-level：DPO 偏好对 / SFT 数据构造\n\n第五部分 · 评估方法论的演进（1）单轮 vs 多轮评估（2）confirmation flow 识别（3）hallucination grep（4）time-anchor stale 检测\n\n第六部分 · 结论\n\n请基于你的训练知识尽量详细展开，需要时可以引用具体研究或工程实践案例。",
}

# How many tokens to generate per prompt
GENERATE_TOKENS = 256


def call_completion(url: str, model: str, prompt: str, max_tokens: int = 256, key: str = ""):
    """Single non-streaming completion. Returns (latency_ms, tokens_generated, prompt_tokens)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key or 'none'}",
        },
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    latency_ms = (time.time() - t0) * 1000
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens") or len(content.split())  # crude fallback
    prompt_tokens = usage.get("prompt_tokens") or len(prompt.split())
    return latency_ms, completion_tokens, prompt_tokens, content


def call_streaming(url: str, model: str, prompt: str, max_tokens: int = 256, key: str = ""):
    """Streaming completion to measure TTFT (time to first token) + per-token rate."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key or 'none'}",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    t0 = time.time()
    ttft_ms = None
    last_token_time = None
    n_tokens = 0
    content_buf = []

    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
            except Exception:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            chunk = delta.get("content") or ""
            if chunk:
                if ttft_ms is None:
                    ttft_ms = (time.time() - t0) * 1000
                content_buf.append(chunk)
                n_tokens += 1  # approximate (not exact since chunk may be > 1 token)
                last_token_time = time.time()

    total_ms = (time.time() - t0) * 1000
    # approximate generation rate from content length / generation time
    content = "".join(content_buf)
    gen_time_s = (last_token_time - (t0 + ttft_ms / 1000)) if ttft_ms and last_token_time else 0
    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "n_tokens_approx": n_tokens,
        "content_len": len(content),
        "gen_time_s": gen_time_s,
        "content_preview": content[:200],
    }


def bench_endpoint(url, model, key, n_trials=3, streaming=True):
    results = {}
    for tag, prompt in PROMPTS.items():
        print(f"\n=== {tag} (prompt {len(prompt)} chars) ===", flush=True)
        trials = []
        for trial in range(n_trials):
            print(f"  trial {trial+1}/{n_trials}...", end="", flush=True)
            try:
                if streaming:
                    r = call_streaming(url, model, prompt, max_tokens=GENERATE_TOKENS, key=key)
                    tps = (r["n_tokens_approx"] / r["gen_time_s"]) if r["gen_time_s"] > 0 else 0
                    print(
                        f" TTFT={r['ttft_ms']:.0f}ms total={r['total_ms']:.0f}ms "
                        f"tokens≈{r['n_tokens_approx']} gen={r['gen_time_s']:.2f}s ≈{tps:.1f} t/s",
                        flush=True,
                    )
                    trials.append(r | {"approx_tps": tps})
                else:
                    latency, comp_tok, prompt_tok, content = call_completion(
                        url, model, prompt, max_tokens=GENERATE_TOKENS, key=key
                    )
                    tps = comp_tok / (latency / 1000) if latency > 0 else 0
                    print(
                        f" latency={latency:.0f}ms prompt_tok={prompt_tok} "
                        f"comp_tok={comp_tok} ≈{tps:.1f} t/s",
                        flush=True,
                    )
                    trials.append(
                        {
                            "latency_ms": latency,
                            "prompt_tokens": prompt_tok,
                            "completion_tokens": comp_tok,
                            "tps": tps,
                            "content_preview": content[:200],
                        }
                    )
            except Exception as e:
                print(f" FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)
                trials.append({"error": f"{type(e).__name__}: {str(e)[:200]}"})

        # Aggregate
        valid = [t for t in trials if not t.get("error")]
        if valid:
            tps_vals = [t.get("approx_tps") or t.get("tps", 0) for t in valid]
            ttft_vals = [t.get("ttft_ms") for t in valid if t.get("ttft_ms")]
            results[tag] = {
                "n_trials": len(valid),
                "n_failed": len(trials) - len(valid),
                "tps_median": median(tps_vals) if tps_vals else 0,
                "tps_max": max(tps_vals) if tps_vals else 0,
                "ttft_median_ms": median(ttft_vals) if ttft_vals else None,
                "trials": trials,
            }
        else:
            results[tag] = {"n_trials": 0, "n_failed": len(trials), "trials": trials}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="OpenAI-compat /v1 base URL")
    ap.add_argument("--model", required=True, help="Model name as known to the endpoint")
    ap.add_argument("--key", default="none", help="API key (or 'none')")
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--no-stream", action="store_true", help="Use non-streaming (less accurate TTFT)")
    ap.add_argument("--label", default=None, help="Label for this run (defaults to model name)")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or args.model.replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"throughput_{label}_{ts}.json"

    print(f"⏱  Lynn Engine baseline benchmark")
    print(f"   url:    {args.url}")
    print(f"   model:  {args.model}")
    print(f"   trials: {args.n_trials} per prompt")
    print(f"   output: {out_path}")

    results = bench_endpoint(
        args.url, args.model, args.key,
        n_trials=args.n_trials, streaming=not args.no_stream,
    )

    # Summary table
    print("\n========== Summary ==========")
    print(f"{'Prompt tag':<14} {'TPS median':<12} {'TPS max':<10} {'TTFT med (ms)':<14} {'Errors'}")
    for tag, r in results.items():
        print(
            f"{tag:<14} "
            f"{r.get('tps_median', 0):<12.2f} "
            f"{r.get('tps_max', 0):<10.2f} "
            f"{r.get('ttft_median_ms') or 'n/a':<14} "
            f"{r.get('n_failed', 0)}"
        )

    out = {
        "ts": datetime.now().isoformat(),
        "url": args.url,
        "model": args.model,
        "label": label,
        "n_trials_per_prompt": args.n_trials,
        "streaming": not args.no_stream,
        "results": results,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n💾 saved → {out_path}")


if __name__ == "__main__":
    main()
