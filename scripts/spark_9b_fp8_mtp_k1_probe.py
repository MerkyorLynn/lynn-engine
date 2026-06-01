#!/usr/bin/env python3
"""M4-K1 probe: 9B dense W4A8/FP8 + embedded MTP head, K=1 sequential speculative.

The 9B FP8 artifact carries the MTP head inline (``mtp.*`` keys), so this loads
it via the embedded-MTP path (no separate sidecar file) and compares baseline vs
K=1 speculative decode: decode TPS, MTP accept rate, effective token TPS, and
completion coherence. First end-to-end test of the dense-MTP wiring in
``engine/mtp_sidecar.py`` + ``engine/resident_runner.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/merkyor/models/Qwen3.5-9B-lynn-native-w4a8-fp8")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--out", default="/home/merkyor/reports/qwen35_9b/m4_k1_probe.json")
    args = ap.parse_args()

    import torch

    # FP8 dense path env (mirrors the e2e-smoke FP8 config that ran ~15 TPS).
    # MTP requested at load so the embedded head loads via has_embedded_mtp().
    os.environ.setdefault("LYNN_W4A8_FP8_PATH", "1")
    os.environ.setdefault("LYNN_NATIVE_FP4_LM_HEAD", "0")
    os.environ["LYNN_LINEAR_BLOCK_GRAPH"] = "0"
    os.environ["LYNN_LINEAR_BLOCK_GRAPH_REUSE"] = "0"
    os.environ["LYNN_MTP_SPECULATIVE"] = "1"
    os.environ["LYNN_MTP_SPECULATIVE_BATCHED"] = "0"

    from engine.resident_runner import LynnIncrementalRunner

    prompts = [
        "Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.",
        "Write a Python function that returns the n-th Fibonacci number iteratively.",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
    ]

    t0 = time.time()
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=True)
    load_s = time.time() - t0
    loaded = bool(getattr(runner, "mtp_sidecar_loaded", False))
    print(f"[m4-k1] load={load_s:.1f}s mtp_sidecar_loaded={loaded}", flush=True)
    if not loaded:
        print("[m4-k1] FAIL: embedded MTP head did not load", flush=True)
        return 2
    cfg = runner.mtp_layer_cfg or {}
    print(
        f"[m4-k1] mtp_tensors={len(runner.mtp_sidecar)} "
        f"is_moe={cfg.get('is_moe')} num_experts={cfg.get('num_experts')}",
        flush=True,
    )

    def run(label: str, *, spec: bool, batched: bool = False, k: int | None = None) -> list[dict]:
        os.environ["LYNN_MTP_SPECULATIVE"] = "1" if spec else "0"
        os.environ["LYNN_MTP_SPECULATIVE_BATCHED"] = "1" if batched else "0"
        if k is not None:
            os.environ["LYNN_MTP_SPECULATIVE_K"] = str(k)
        else:
            os.environ.pop("LYNN_MTP_SPECULATIVE_K", None)
        rows = []
        for p in prompts:
            t = time.time()
            out = runner.generate(p, max_new=args.max_new)
            wall = time.time() - t
            sp = out.get("mtp_speculative", {}) or {}
            tim = out.get("timings", {}) or {}
            row = {
                "prompt": p[:48],
                "decode_tps": tim.get("decode_tps"),
                "wall_s": round(wall, 3),
                "n_new": len(out["new_ids"]),
                "spec_active": sp.get("active"),
                "accept_rate": sp.get("accept_rate"),
                "tokens_per_event": sp.get("tokens_per_event"),
                "effective_token_tps": sp.get("effective_token_tps"),
                "head": out["completion_text"][:120],
            }
            rows.append(row)
            print(
                f"[m4-k1] {label} eff_tps={row['effective_token_tps']} "
                f"accept={row['accept_rate']} tok/ev={row['tokens_per_event']} "
                f"wall={row['wall_s']}s decode_tps={row['decode_tps']} "
                f":: {out['completion_text'][:46]!r}",
                flush=True,
            )
        return rows

    import traceback

    results: dict = {"baseline": run("baseline", spec=False)}
    for label, k in [("spec_k1_batched", None), ("spec_k2_batched", 2), ("spec_k4_batched", 4)]:
        try:
            results[label] = run(label, spec=True, batched=True, k=k)
        except Exception as exc:  # noqa: BLE001
            print(f"[m4-k1] {label} EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
            results[label] = [{
                "error": f"{type(exc).__name__}: {exc}",
                "tb": traceback.format_exc()[-1200:],
            }]

    report = {
        "model": args.model,
        "max_new": args.max_new,
        "mtp_loaded": loaded,
        "mtp_layer_cfg": {"is_moe": cfg.get("is_moe"), "num_experts": cfg.get("num_experts")},
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[m4-k1] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
