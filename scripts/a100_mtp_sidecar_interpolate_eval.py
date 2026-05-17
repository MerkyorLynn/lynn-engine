#!/usr/bin/env python3
"""Interpolate two saved MTP sidecars and evaluate on one heldout case set.

This is a small rescue tool for complementary MTP runs: one sidecar may preserve
early-token stability while another fixes a different token position. A linear
scan can show whether there is a useful midpoint before launching another
training job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from scripts.a100_mtp_fc_calibration_train import _mtp_cfg  # noqa: E402
from scripts.a100_mtp_forward_smoke import _load_sidecar, _mtp_layer_weights  # noqa: E402
from scripts.a100_mtp_iterative_train import _collect_cases, _evaluate, _load_prompts  # noqa: E402


def _compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": metric.get("accepted"),
        "case_count": metric.get("case_count"),
        "accept_rate": metric.get("accept_rate"),
        "mean_loss": metric.get("mean_loss"),
        "by_step": metric.get("by_step", {}),
    }


def _blend_sidecars(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if set(a) != set(b):
        missing_a = sorted(set(b) - set(a))
        missing_b = sorted(set(a) - set(b))
        raise KeyError(f"sidecar keys differ missing_a={missing_a[:5]} missing_b={missing_b[:5]}")
    out: dict[str, torch.Tensor] = {}
    for key in a:
        ta = a[key]
        tb = b[key]
        if ta.shape != tb.shape:
            raise ValueError(f"shape mismatch for {key}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
        if ta.is_floating_point():
            out[key] = torch.lerp(ta, tb.to(dtype=ta.dtype), alpha).contiguous()
        else:
            out[key] = ta.detach().clone()
    return out


def _save_best(
    sidecar: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    alpha: float,
    sidecar_a: str,
    sidecar_b: str,
    metric: dict[str, Any],
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "mtp.safetensors"
    save_file(
        {key: tensor.detach().cpu() for key, tensor in sidecar.items()},
        str(out_file),
        metadata={
            "lynn_mtp_interpolated": "true",
            "sidecar_a": sidecar_a,
            "sidecar_b": sidecar_b,
            "alpha": str(alpha),
            "eval_accept_rate": str(metric.get("accept_rate")),
        },
    )
    return str(out_file)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--eval-prompts-file", required=True)
    ap.add_argument("--sidecar-a", required=True)
    ap.add_argument("--sidecar-b", required=True)
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--alpha", type=float, nargs="+", default=[i / 10 for i in range(11)])
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-sidecar-dir")
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-new-eval", type=int, default=16)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompts(args.eval_prompts_file, [])
    runner = LynnIncrementalRunner(args.base_model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    eval_cases = _collect_cases(
        runner=runner,
        specs=specs,
        use_chat_template=args.use_chat_template,
        force_prefix_from_spec=False,
        skip_forced_prefix_cases=False,
        max_new=args.max_new_eval,
        first_token_weight=1.0,
        step1_weight=1.0,
        later_token_weight=1.0,
    )

    sidecar_a, inv_a = _load_sidecar(Path(args.sidecar_a), args.device, dtype)
    sidecar_b, inv_b = _load_sidecar(Path(args.sidecar_b), args.device, dtype)

    rows: list[dict[str, Any]] = []
    best_sidecar: dict[str, torch.Tensor] | None = None
    best_metric: dict[str, Any] | None = None
    best_alpha: float | None = None
    for alpha in args.alpha:
        blended = _blend_sidecars(sidecar_a, sidecar_b, float(alpha))
        mtp_w = _mtp_layer_weights(blended)
        cfg = _mtp_cfg(runner, mtp_w)
        metric = _evaluate(runner, blended, mtp_w, cfg, eval_cases)
        compact = _compact_metric(metric)
        rows.append({"alpha": float(alpha), "eval": compact})
        if best_metric is None or float(compact["accept_rate"]) > float(best_metric["accept_rate"]):
            best_metric = compact
            best_alpha = float(alpha)
            best_sidecar = {key: tensor.detach().clone() for key, tensor in blended.items()}
        del blended, mtp_w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_file = None
    if args.out_sidecar_dir and best_sidecar is not None and best_metric is not None and best_alpha is not None:
        best_file = _save_best(
            best_sidecar,
            Path(args.out_sidecar_dir),
            alpha=best_alpha,
            sidecar_a=args.sidecar_a,
            sidecar_b=args.sidecar_b,
            metric=best_metric,
        )

    result = {
        "schema_version": "lynn-a100-mtp-sidecar-interpolate-eval-v1",
        "base_model": args.base_model,
        "eval_prompts_file": args.eval_prompts_file,
        "max_new_eval": args.max_new_eval,
        "case_count": len(eval_cases),
        "label_a": args.label_a,
        "sidecar_a": args.sidecar_a,
        "inventory_a": inv_a,
        "label_b": args.label_b,
        "sidecar_b": args.sidecar_b,
        "inventory_b": inv_b,
        "rows": rows,
        "best_alpha": best_alpha,
        "best_accept_rate": None if best_metric is None else best_metric.get("accept_rate"),
        "best_accepted": None if best_metric is None else best_metric.get("accepted"),
        "best_sidecar_file": best_file,
        "decision": (
            "GREEN: interpolated sidecar clears 55% heldout accept."
            if best_metric and float(best_metric["accept_rate"]) >= 0.55
            else "AMBER: interpolation improved/checked but remains below serving multiplier threshold."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "best_alpha": result["best_alpha"],
                "best_accepted": result["best_accepted"],
                "case_count": result["case_count"],
                "best_accept_rate": result["best_accept_rate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
