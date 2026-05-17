#!/usr/bin/env python3
"""P112: profile MTP draft component latency on collected decode states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _layer_forward, _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from scripts.a100_mtp_fc_calibration_train import _mtp_cfg  # noqa: E402
from scripts.a100_mtp_forward_smoke import _load_sidecar, _mtp_layer_weights  # noqa: E402
from scripts.a100_mtp_iterative_train import _collect_cases, _load_prompts  # noqa: E402


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _timed(device: str, fn: Any) -> tuple[Any, float]:
    _sync(device)
    t0 = time.time()
    out = fn()
    _sync(device)
    return out, time.time() - t0


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p90_ms": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))] * 1000.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar-file", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--max-cases", type=int, default=64)
    ap.add_argument("--force-prefix-from-spec", action="store_true")
    ap.add_argument("--skip-forced-prefix-cases", action="store_true")
    ap.add_argument("--lm-head", choices=["current", "native_fp4", "bf16"], default="current")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompts(args.prompts_file, [])
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    if args.lm_head == "native_fp4":
        runner._prepare_native_fp4_lm_head()
    elif args.lm_head == "bf16":
        runner.native_fp4_lm_head_enabled = False
    sidecar, inventory = _load_sidecar(Path(args.sidecar_file), args.device, dtype)
    mtp_w = _mtp_layer_weights(sidecar)
    cfg = _mtp_cfg(runner, mtp_w)
    cases = _collect_cases(
        runner=runner,
        specs=specs,
        use_chat_template=False,
        force_prefix_from_spec=args.force_prefix_from_spec,
        skip_forced_prefix_cases=args.skip_forced_prefix_cases,
        max_new=args.max_new,
        first_token_weight=1.0,
        step1_weight=1.0,
        later_token_weight=1.0,
    )[: args.max_cases]

    timings = {key: [] for key in ("pre_fc_norms", "fc", "mtp_layer", "mtp_norm", "lm_head", "total")}
    accepted = 0
    with torch.no_grad():
        for case in cases:
            _sync(args.device)
            total_t0 = time.time()
            hidden_part, dt = _timed(
                args.device,
                lambda: _rms_norm(case["base_hidden"], sidecar["mtp.pre_fc_norm_hidden.weight"]),
            )
            embed_part, dt_embed = _timed(
                args.device,
                lambda: _rms_norm(case["input_embed"], sidecar["mtp.pre_fc_norm_embedding.weight"]),
            )
            timings["pre_fc_norms"].append(dt + dt_embed)
            mtp_hidden, dt = _timed(
                args.device,
                lambda: F.linear(torch.cat([hidden_part, embed_part], dim=-1), sidecar["mtp.fc.weight"]),
            )
            timings["fc"].append(dt)
            pos = torch.tensor([[int(case["current_pos"])]], device=args.device, dtype=torch.long)
            mtp_out, dt = _timed(
                args.device,
                lambda: _layer_forward(mtp_hidden, pos, "full_attention", mtp_w, cfg),
            )
            timings["mtp_layer"].append(dt)
            mtp_normed, dt = _timed(args.device, lambda: _rms_norm(mtp_out, sidecar["mtp.norm.weight"]))
            timings["mtp_norm"].append(dt)
            logits, dt = _timed(args.device, lambda: runner._lm_head_logits(mtp_normed))
            timings["lm_head"].append(dt)
            _sync(args.device)
            timings["total"].append(time.time() - total_t0)
            accepted += int(int(logits[0].argmax().item()) == int(case["label_id"]))

    summary = {key: _stats(values) for key, values in timings.items()}
    result = {
        "schema_version": "lynn-p112-mtp-draft-component-profile-v1",
        "decision": "AMBER: MTP draft component profile collected.",
        "model": args.model,
        "sidecar_file": args.sidecar_file,
        "prompts_file": args.prompts_file,
        "lm_head": args.lm_head,
        "case_count": len(cases),
        "accepted": accepted,
        "accept_rate": accepted / len(cases) if cases else None,
        "summary": summary,
        "sidecar_tensor_count": len(inventory.get("tensors", {})),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
