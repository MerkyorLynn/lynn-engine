#!/usr/bin/env python3
"""P10-T runner gate: full-token graph slot parity.

`p9j_single_position_graph_after_prefix.py` proved the idea in a standalone
probe. This gate verifies the production runner helper
`LynnIncrementalRunner._capture_full_token_graph_slot()` against eager decode
after several real prefixes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _decode_layer, _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _prefill(runner: LynnIncrementalRunner, prompt: str) -> tuple[int, LynnInferenceState]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = ids.shape[1]
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_final)
    return int(logits[0].argmax().item()), state


def _decode_one(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    token_id: int,
    position: int,
) -> torch.Tensor:
    token = torch.tensor([[token_id]], device=runner.device, dtype=torch.long)
    pos_tensor = torch.tensor([[position]], device=runner.device, dtype=torch.long)
    h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    for i in range(runner.n_layers):
        h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len += 1
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    return runner._lm_head_logits(h_final)


def _logit_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    a_top = torch.topk(af, 10).indices.tolist()
    b_top = torch.topk(bf, 10).indices.tolist()
    return {
        "max_abs": float((af - bf).abs().max().item()),
        "mean_abs": float((af - bf).abs().mean().item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
        "top1_match": int(a_top[0]) == int(b_top[0]),
        "top10_overlap": len(set(int(x) for x in a_top) & set(int(x) for x in b_top)),
    }


def _run_one(runner: LynnIncrementalRunner, prompt: str, prefix_new: int) -> dict[str, Any]:
    token_id, state = _prefill(runner, prompt)
    generated_prefix: list[int] = []
    for _ in range(prefix_new):
        logits = _decode_one(runner, state, token_id, int(state.seq_len))
        token_id = int(logits[0].argmax().item())
        generated_prefix.append(token_id)

    base = runner._snapshot_state(state)
    position = int(state.seq_len)
    slot = runner._capture_full_token_graph_slot(state, token_id)

    runner._restore_state(state, base)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph_logits = slot.replay(token_id)
    end.record()
    torch.cuda.synchronize()
    graph_ms = float(start.elapsed_time(end))
    graph_logits = graph_logits.clone()
    graph_next = int(graph_logits[0].argmax().item())

    runner._restore_state(state, base)
    eager_logits = _decode_one(runner, state, token_id, position)
    eager_next = int(eager_logits[0].argmax().item())
    diff = _logit_diff(graph_logits, eager_logits)
    return {
        "prefix_new": prefix_new,
        "generated_prefix": generated_prefix,
        "position": position,
        "input_token_id": token_id,
        "graph_next_id": graph_next,
        "eager_next_id": eager_next,
        "graph_ms": graph_ms,
        "graph_tps": 1000.0 / graph_ms,
        "diff": diff,
        "pass": graph_next == eager_next and diff["top1_match"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--prefix-new", type=int, nargs="+", default=[8, 16, 32])
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    rows = [_run_one(runner, args.prompt, prefix_new) for prefix_new in args.prefix_new]
    result = {
        "schema_version": "lynn-engine-p10t-runner-graph-slot-gate-v1",
        "model": args.model,
        "prompt": args.prompt,
        "native_fp4_lm_head_enabled": runner.native_fp4_lm_head_enabled,
        "rows": rows,
        "pass": all(row["pass"] for row in rows),
        "note": "Verifies LynnIncrementalRunner._capture_full_token_graph_slot against eager decode.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
