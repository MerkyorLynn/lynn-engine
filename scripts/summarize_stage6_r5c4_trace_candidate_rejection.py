#!/usr/bin/env python3
"""Summarize Stage 6 R5-C4 trace-derived candidate rejection artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(data: dict[str, Any], result_path: Path) -> str:
    passes = data.get("passes") if isinstance(data.get("passes"), dict) else {}
    lines = [
        "# Stage 6 R5-C4 Trace-Derived Candidate Rejection Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Validator decision | `{data.get('validator_decision')}` |",
        f"| Validator rejected trace candidate | `{passes.get('validator_rejected_trace_candidate')}` |",
        f"| Same-scope false | `{passes.get('same_scope_false')}` |",
        f"| Real model weights false | `{passes.get('real_model_weights_false')}` |",
        f"| Full boundary timed false | `{passes.get('full_boundary_timed_false')}` |",
        f"| Decode/default not banked | `{passes.get('decode_tps_not_banked')}` / `{passes.get('default_not_banked')}` |",
        "",
        "## Boundary",
        "",
        "- This artifact banks only rejection behavior for a trace-derived bad candidate.",
        "- It proves R5-C3A gate/up timing plus R5-C3C host composition parity cannot be promoted as R5-C4 speed.",
        "- It does not bank full active-MoE prefill speed, decode TPS, server/RC behavior, or defaults.",
        "",
    ]
    return "\n".join(lines)


def _strict_ok(data: dict[str, Any]) -> bool:
    passes = data.get("passes") if isinstance(data.get("passes"), dict) else {}
    return bool(
        data.get("decision") == "PASS_R5C4_TRACE_DERIVED_CANDIDATE_REJECTED"
        and data.get("validator_decision") == "FAIL_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB"
        and passes.get("validator_rejected_trace_candidate") is True
        and passes.get("same_scope_false") is True
        and passes.get("real_model_weights_false") is True
        and passes.get("full_boundary_timed_false") is True
        and passes.get("decode_tps_not_banked") is True
        and passes.get("default_not_banked") is True
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_json")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument("--strict-exit", action="store_true")
    args = ap.parse_args()
    result_path = Path(args.result_json)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    md = summarize(data, result_path)
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
    print(md)
    return 2 if args.strict_exit and not _strict_ok(data) else 0


if __name__ == "__main__":
    raise SystemExit(main())
