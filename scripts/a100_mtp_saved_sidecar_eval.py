#!/usr/bin/env python3
"""Evaluate saved MTP sidecars on one shared heldout case set.

The iterative trainer reports in-memory eval metrics before it writes a sidecar.
This script reloads the saved safetensors files and evaluates them against the
same base-greedy heldout cases in one process. It catches save/load drift and
keeps the MTP ladder honest before a sidecar is handed to serving work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from scripts.a100_mtp_fc_calibration_train import _mtp_cfg  # noqa: E402
from scripts.a100_mtp_forward_smoke import _load_sidecar, _mtp_layer_weights  # noqa: E402
from scripts.a100_mtp_iterative_train import _collect_cases, _evaluate, _load_prompts  # noqa: E402


def _parse_sidecar(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, value
    label, path = value.split("=", 1)
    return label, path


def _compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": metric.get("accepted"),
        "case_count": metric.get("case_count"),
        "accept_rate": metric.get("accept_rate"),
        "mean_loss": metric.get("mean_loss"),
        "by_step": metric.get("by_step", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--eval-prompts-file", required=True)
    ap.add_argument("--sidecar", action="append", required=True, help="label=/path/to/mtp.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-new-eval", type=int, default=16)
    ap.add_argument("--first-token-weight", type=float, default=1.0)
    ap.add_argument("--step1-weight", type=float, default=1.0)
    ap.add_argument("--later-token-weight", type=float, default=1.0)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompts(args.eval_prompts_file, [])
    runner = LynnIncrementalRunner(args.base_model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    eval_cases = _collect_cases(
        runner=runner,
        specs=specs,
        use_chat_template=args.use_chat_template,
        max_new=args.max_new_eval,
        first_token_weight=args.first_token_weight,
        step1_weight=args.step1_weight,
        later_token_weight=args.later_token_weight,
    )

    rows: list[dict[str, Any]] = []
    for item in args.sidecar:
        label, sidecar_path = _parse_sidecar(item)
        sidecar, inventory = _load_sidecar(Path(sidecar_path), args.device, dtype)
        mtp_w = _mtp_layer_weights(sidecar)
        cfg = _mtp_cfg(runner, mtp_w)
        metric = _evaluate(runner, sidecar, mtp_w, cfg, eval_cases)
        rows.append(
            {
                "label": label,
                "sidecar_file": sidecar_path,
                "inventory": inventory,
                "eval": _compact_metric(metric),
            }
        )
        del sidecar, mtp_w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(rows, key=lambda row: float(row["eval"]["accept_rate"]), default=None)
    result = {
        "schema_version": "lynn-a100-mtp-saved-sidecar-eval-v1",
        "base_model": args.base_model,
        "eval_prompts_file": args.eval_prompts_file,
        "max_new_eval": args.max_new_eval,
        "case_count": len(eval_cases),
        "rows": rows,
        "best_label": None if best is None else best["label"],
        "best_accept_rate": None if best is None else best["eval"]["accept_rate"],
        "decision": (
            "GREEN: at least one saved sidecar clears 55% heldout accept."
            if best and float(best["eval"]["accept_rate"]) >= 0.55
            else "AMBER: saved sidecars remain below serving multiplier threshold."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "best_label": result["best_label"], "best_accept_rate": result["best_accept_rate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
