"""Resident incremental runner for Lynn engine correctness/perf gates.

`engine.full_forward.generate_incremental` is intentionally simple: load all
weights, run one prompt, exit. That is ideal for isolated correctness checks,
but it hides the engine's steady-state behavior behind repeated weight loads.

This module keeps outside weights and all 40 layer weights resident, then runs
multiple prompts through the same BF16 or NVFP4 model instance. It is still the
slow-dequant correctness path, not native FP4 GEMM, but it is the first shape of
an actual reusable Lynn engine session.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from engine.full_forward import (
    _decode_layer,
    _prefill_layer,
    _rms_norm,
    load_outside_weights,
)
from engine.inference_state import LAYER_TYPES, LynnInferenceState
from engine.loader import load_qwen36_layer


def _logit_topk(logits: torch.Tensor, k: int) -> dict[str, Any]:
    """Return a compact top-k view for one-token logits."""
    values, indices = torch.topk(logits[0].float(), k=k)
    return {
        "ids": [int(x) for x in indices.tolist()],
        "values": [float(x) for x in values.tolist()],
        "top1_margin": (
            float(values[0].item() - values[1].item()) if k >= 2 else None
        ),
    }


def _runtime_config(model_dir: str) -> tuple[dict[str, Any], int]:
    with open(Path(model_dir) / "config.json", encoding="utf-8") as f:
        full_cfg = json.load(f)
    tc = full_cfg["text_config"]
    rope_p = tc.get("rope_parameters", {})
    cfg = {
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "num_experts": tc["num_experts"],
        "num_experts_per_tok": tc["num_experts_per_tok"],
        "rope_theta": rope_p.get("rope_theta", tc.get("rope_theta", 1e6)),
        "partial_rotary_factor": rope_p.get("partial_rotary_factor", 1.0),
    }
    if LAYER_TYPES != tc["layer_types"]:
        raise ValueError("layer_types config mismatch")
    return cfg, tc["num_hidden_layers"]


class LynnIncrementalRunner:
    """Single-model resident incremental decode runner."""

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 2048,
        verbose: bool = True,
    ) -> None:
        impl = os.environ.get("LYNN_MOE_IMPL", "optimized")
        if impl in {"indexed_bmm", "triton"}:
            raise ValueError(
                "LynnIncrementalRunner supports reusable prompts only with "
                "LYNN_MOE_IMPL=optimized or bmm. indexed_bmm mutates weights "
                "after prefill and is single-prompt only today."
            )
        self.model_dir = str(model_dir)
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.verbose = verbose
        self.cfg, self.n_layers = _runtime_config(self.model_dir)

        from transformers import AutoTokenizer

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        if verbose:
            print(f"[resident] loading outside weights from {self.model_dir}", flush=True)
        self.outside = load_outside_weights(self.model_dir, device, dtype)

        if verbose:
            print(f"[resident] loading {self.n_layers} layers", flush=True)
        self.layer_weights = []
        for i in range(self.n_layers):
            w, _ = load_qwen36_layer(
                self.model_dir,
                i,
                num_experts=self.cfg["num_experts"],
                device=device,
                dequant_dtype=dtype,
            )
            self.layer_weights.append(w)
            if verbose and (i % 5 == 4 or i == self.n_layers - 1):
                print(f"  [resident] L{i:02}: {time.time() - t0:.1f}s", flush=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        self.load_seconds = time.time() - t0

    def generate(self, prompt: str, *, max_new: int = 4, top_k: int = 0) -> dict[str, Any]:
        tok = self.tokenizer
        ids = tok(prompt, return_tensors="pt").input_ids.to(self.device)
        T = ids.shape[1]
        state = LynnInferenceState(
            batch=1,
            max_seq_len=self.max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        if self.verbose:
            print(f"[resident] prompt={prompt!r} T={T} max_new={max_new}", flush=True)

        prefill_t0 = time.time()
        h = F.embedding(ids, self.outside["model.language_model.embed_tokens.weight"])
        pos = torch.arange(T, device=self.device, dtype=torch.long).unsqueeze(0)
        for i in range(self.n_layers):
            h = _prefill_layer(h, pos, LAYER_TYPES[i], self.layer_weights[i], self.cfg, state, i)
        state.seq_len = T
        h_final = _rms_norm(h, self.outside["model.language_model.norm.weight"])
        logits = F.linear(h_final[:, -1, :], self.outside["lm_head.weight"])
        next_id = int(logits[0].argmax().item())
        topk_trace = []
        if top_k > 0:
            topk_trace.append(_logit_topk(logits, top_k))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        prefill_seconds = time.time() - prefill_t0
        if self.verbose:
            print(
                f"  [resident] prefill {prefill_seconds:.3f}s "
                f"next={next_id} {tok.decode([next_id])!r}",
                flush=True,
            )
        new_ids = [next_id]
        decode_seconds = []

        for step in range(1, max_new):
            step_t0 = time.time()
            new_token_tensor = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
            h = F.embedding(new_token_tensor, self.outside["model.language_model.embed_tokens.weight"])
            pos_id = state.seq_len
            for i in range(self.n_layers):
                h = _decode_layer(h, pos_id, LAYER_TYPES[i], self.layer_weights[i], self.cfg, state, i)
            state.seq_len += 1
            h_final = _rms_norm(h, self.outside["model.language_model.norm.weight"])
            logits = F.linear(h_final[:, -1, :], self.outside["lm_head.weight"])
            next_id = int(logits[0].argmax().item())
            if top_k > 0:
                topk_trace.append(_logit_topk(logits, top_k))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - step_t0
            decode_seconds.append(elapsed)
            if self.verbose:
                print(
                    f"  [resident] decode {step + 1}/{max_new} "
                    f"{elapsed * 1000:.0f}ms next={next_id} {tok.decode([next_id])!r}",
                    flush=True,
                )
            new_ids.append(next_id)

        full_text = tok.decode(ids[0].tolist() + new_ids)
        completion_text = tok.decode(new_ids)
        result = {
            "text": full_text,
            "completion_text": completion_text,
            "new_ids": [int(x) for x in new_ids],
            "timings": {
                "prefill_seconds": prefill_seconds,
                "decode_step_seconds": decode_seconds,
                "decode_tps": (len(decode_seconds) / sum(decode_seconds)) if decode_seconds else None,
            },
        }
        if top_k > 0:
            result["topk_trace"] = topk_trace
        return result
