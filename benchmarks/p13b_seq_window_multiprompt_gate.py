#!/usr/bin/env python3
"""P13-B: multi-prompt sequential graph-window parity gate.

The single-prompt window probe showed that an 8-token sequential capture window
can preserve greedy parity, while a 16-token window already drifts. This gate
tests whether the 8-token boundary generalizes across several prompts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_PROMPTS = [
    {"id": "zh_moe", "prompt": "用一句话解释 MoE active parameters"},
    {"id": "python_factorial", "prompt": "Python 写一个递归阶乘函数"},
    {"id": "rope_alibi", "prompt": "比较 RoPE 与 ALiBi 的优缺点"},
    {"id": "english_math", "prompt": "If a train travels 60 mph for 2.5 hours, how far does it travel?"},
    {"id": "tool_json", "prompt": "请用 JSON 调用 get_weather 工具查询北京天气"},
    {"id": "longctx_summary", "prompt": "请用两句话总结: Lynn Engine 使用 NVFP4 packed 权重、Blackwell native FP4、CUDA graph 来优化 27B MoE 推理。"},
]


def _load_prompts(path: str | None) -> list[dict[str, str]]:
    if not path:
        return DEFAULT_PROMPTS
    prompts: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt") or item.get("input") or item.get("question") or item.get("problem")
            if not prompt:
                raise ValueError(f"Prompt record {i} missing prompt/input/question/problem")
            prompts.append({"id": str(item.get("id", f"prompt_{i:03d}")), "prompt": str(prompt)})
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--prompts-jsonl")
    args = ap.parse_args()

    prompts = _load_prompts(args.prompts_jsonl)
    out_dir = Path(args.out).resolve().parent / (Path(args.out).stem + "_rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    script = Path(__file__).resolve().parent / "p9k_sequential_capture_graph_family_greedy.py"
    for item in prompts:
        row_out = out_dir / f"{item['id']}.json"
        cmd = [
            sys.executable,
            str(script),
            "--model",
            args.model,
            "--out",
            str(row_out),
            "--prompt",
            item["prompt"],
            "--max-new",
            str(args.max_new),
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        data = json.loads(row_out.read_text(encoding="utf-8")) if row_out.exists() else {}
        bad_steps = [
            {
                "step": r["step"],
                "graph_next_id": r["graph_next_id"],
                "eager_next_id": r["eager_next_id"],
                "cosine": r["diff"]["cosine"],
                "top10_overlap": r["diff"]["top10_overlap"],
            }
            for r in data.get("rows", [])
            if not r.get("diff", {}).get("top1_match", False)
        ]
        rows.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "returncode": proc.returncode,
            "pass": bool(data.get("pass", False)),
            "greedy_pass": bool(data.get("greedy_pass", False)),
            "avg_graph_replay_tps": data.get("avg_graph_replay_tps"),
            "amortized_tps_including_capture": data.get("amortized_tps_including_capture"),
            "bad_steps": bad_steps,
            "row_out": str(row_out),
        })
    result = {
        "schema_version": "lynn-engine-p13b-seq-window-multiprompt-gate-v1",
        "model": args.model,
        "max_new": args.max_new,
        "prompt_count": len(prompts),
        "rows": rows,
        "pass": all(row["pass"] for row in rows),
        "note": "Runs p9k sequential-capture graph window gate across prompts.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
