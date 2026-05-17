#!/usr/bin/env python3
"""Evaluate subset-wise combinations of two MTP sidecars.

When a later run fixes one token position but breaks another, full interpolation
can be too blunt. This grid keeps sidecar A as the base and swaps selected
trainable tensor groups from sidecar B to find which surface carries the useful
change.
"""

from __future__ import annotations

import argparse
import itertools
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


GROUP_KEYS = {
    "fc": ["mtp.fc.weight"],
    "pre_hidden": ["mtp.pre_fc_norm_hidden.weight"],
    "pre_embed": ["mtp.pre_fc_norm_embedding.weight"],
    "norm": ["mtp.norm.weight"],
}


def _compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": metric.get("accepted"),
        "case_count": metric.get("case_count"),
        "accept_rate": metric.get("accept_rate"),
        "mean_loss": metric.get("mean_loss"),
        "by_step": metric.get("by_step", {}),
    }


def _build_subset(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    groups: list[str],
) -> dict[str, torch.Tensor]:
    selected = {key for group in groups for key in GROUP_KEYS[group]}
    out: dict[str, torch.Tensor] = {}
    for key, tensor in a.items():
        source = b[key] if key in selected else tensor
        if source.shape != tensor.shape:
            raise ValueError(f"shape mismatch for {key}: {tuple(source.shape)} vs {tuple(tensor.shape)}")
        out[key] = source.detach().clone().contiguous()
    return out


def _save_best(
    sidecar: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    groups: list[str],
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
            "lynn_mtp_subset_grid": "true",
            "sidecar_a": sidecar_a,
            "sidecar_b": sidecar_b,
            "groups_from_b": ",".join(groups),
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
    ap.add_argument("--groups", nargs="+", default=list(GROUP_KEYS), choices=sorted(GROUP_KEYS))
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
        max_new=args.max_new_eval,
        first_token_weight=1.0,
        step1_weight=1.0,
        later_token_weight=1.0,
    )

    sidecar_a, inv_a = _load_sidecar(Path(args.sidecar_a), args.device, dtype)
    sidecar_b, inv_b = _load_sidecar(Path(args.sidecar_b), args.device, dtype)
    if set(sidecar_a) != set(sidecar_b):
        raise KeyError("sidecar key sets differ")

    rows: list[dict[str, Any]] = []
    best_sidecar: dict[str, torch.Tensor] | None = None
    best_metric: dict[str, Any] | None = None
    best_groups: list[str] | None = None
    groups = list(args.groups)
    for width in range(len(groups) + 1):
        for subset in itertools.combinations(groups, width):
            subset_groups = list(subset)
            candidate = _build_subset(sidecar_a, sidecar_b, subset_groups)
            mtp_w = _mtp_layer_weights(candidate)
            cfg = _mtp_cfg(runner, mtp_w)
            metric = _evaluate(runner, candidate, mtp_w, cfg, eval_cases)
            compact = _compact_metric(metric)
            rows.append(
                {
                    "label": "a" if not subset_groups else "b:" + "+".join(subset_groups),
                    "groups_from_b": subset_groups,
                    "eval": compact,
                }
            )
            if best_metric is None or float(compact["accept_rate"]) > float(best_metric["accept_rate"]):
                best_metric = compact
                best_groups = subset_groups
                best_sidecar = {key: tensor.detach().clone() for key, tensor in candidate.items()}
            del candidate, mtp_w
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    best_file = None
    if args.out_sidecar_dir and best_sidecar is not None and best_metric is not None and best_groups is not None:
        best_file = _save_best(
            best_sidecar,
            Path(args.out_sidecar_dir),
            groups=best_groups,
            sidecar_a=args.sidecar_a,
            sidecar_b=args.sidecar_b,
            metric=best_metric,
        )

    result = {
        "schema_version": "lynn-a100-mtp-sidecar-subset-grid-eval-v1",
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
        "groups": groups,
        "rows": rows,
        "best_groups_from_b": best_groups,
        "best_accept_rate": None if best_metric is None else best_metric.get("accept_rate"),
        "best_accepted": None if best_metric is None else best_metric.get("accepted"),
        "best_sidecar_file": best_file,
        "decision": (
            "GREEN: subset sidecar clears 55% heldout accept."
            if best_metric and float(best_metric["accept_rate"]) >= 0.55
            else "AMBER: subset grid checked but remains below serving multiplier threshold."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "best_groups_from_b": result["best_groups_from_b"],
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
