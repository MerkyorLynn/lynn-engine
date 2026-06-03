#!/usr/bin/env python3
"""Static Stage-6 Phase-0 census for packed-prefill / zero-reload work.

This is not a GPU benchmark. It records checkable source anchors for the current
contract: decode-only shadow release is banked, and the next probe is
`LYNN_PACKED_PREFILL_SLOW=1` no-reload prefill from packed NVFP4 aliases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


ANCHORS = [
    {
        "claim": "full-attention prefill q/k/v can fall back to packed aliases",
        "file": "engine/incremental_decode.py",
        "needle": 'q_full = _linear_prefill(h, _prefill_weight(w, "self_attn.q_proj.weight"))',
    },
    {
        "claim": "full-attention prefill o_proj can fall back to packed aliases",
        "file": "engine/incremental_decode.py",
        "needle": 'out = _linear_prefill(attn_out, _prefill_weight(w, "self_attn.o_proj.weight"))',
    },
    {
        "claim": "linear-attention prefill qkv projection can fall back to packed aliases",
        "file": "engine/incremental_decode.py",
        "needle": 'mixed = _linear_prefill(h, W("linear_attn.in_proj_qkv.weight"))',
    },
    {
        "claim": "linear-attention prefill out projection can fall back to packed aliases",
        "file": "engine/incremental_decode.py",
        "needle": 'out = _linear_prefill(core_attn_out, W("linear_attn.out_proj.weight"))',
    },
    {
        "claim": "packed prefill is default-off behind LYNN_PACKED_PREFILL_SLOW",
        "file": "engine/incremental_decode.py",
        "needle": 'os.environ.get("LYNN_PACKED_PREFILL_SLOW", "0")',
    },
    {
        "claim": "MoE packed-prefill proof path reuses exact T=1 packed decode MoE",
        "file": "engine/full_forward.py",
        "needle": "moe_forward_decode_packed_nvfp4(flat[i : i + 1], w, cfg)",
    },
    {
        "claim": "MoE packed-prefill proof path is gated by grouped packed aliases",
        "file": "engine/full_forward.py",
        "needle": '"mlp.experts._gate_up_packed" in w',
    },
    {
        "claim": "current packed linear kernels are still T=1-only",
        "file": "engine/nvfp4_runtime.py",
        "needle": "PackedNVFP4Linear.forward currently supports one token",
    },
    {
        "claim": "server already supports reload->prefill->release->decode cycle",
        "file": "server/openai_http.py",
        "needle": "reload_decode_bf16_shadows()",
    },
]


def find_line(path: Path, needle: str) -> int | None:
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return lineno
    return None


def build_report() -> dict:
    checks = []
    for anchor in ANCHORS:
        path = ROOT / anchor["file"]
        line = find_line(path, anchor["needle"])
        checks.append({
            "claim": anchor["claim"],
            "file": anchor["file"],
            "line": line,
            "ok": line is not None,
        })
    all_ok = all(item["ok"] for item in checks)
    return {
        "schema": "lynn-stage6-phase0-static-census-v1",
        "repo": str(ROOT),
        "verdict": "READY_FOR_SPARK_NO_RELOAD_SMOKE" if all_ok else "ANCHOR_MISSING",
        "next_gate": (
            "Run a Spark smoke with LYNN_PACKED_PREFILL_SLOW=1 after "
            "release_decode_bf16_shadows(); assert no reload, token-exact output, "
            "resident ~28 GiB before/after prefill, and record prefill latency."
        ),
        "checks": checks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    report = build_report()
    if args.markdown:
        print(f"# Stage 6 Phase-0 Static Census\n\nVerdict: **{report['verdict']}**\n")
        print("| claim | source |")
        print("|---|---|")
        for item in report["checks"]:
            source = f"{item['file']}:{item['line']}" if item["ok"] else "MISSING"
            print(f"| {item['claim']} | `{source}` |")
        print(f"\nNext gate: {report['next_gate']}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
