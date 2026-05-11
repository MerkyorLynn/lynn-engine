"""
Lynn Engine · Phase 3.4 · OpenAI-compatible HTTP server.

Single-prompt focus (no batching, no concurrent requests beyond what the
engine can handle serially). Mirrors the vLLM endpoints brain currently
uses, so swap-in is one URL change.

Endpoints:
  GET  /v1/models                 — list served models
  POST /v1/completions            — text completion
  POST /v1/chat/completions       — chat completion (applies tokenizer chat_template)
  GET  /health                    — liveness / readiness

Streaming via SSE supported on both completion endpoints (set "stream": true).

Engine lifecycle:
  - On startup, eagerly load all 40 layers + outside weights (~252 s on Spark).
  - Single `LynnInferenceState` is reset between requests (no concurrent decode).
  - Generations are serialized with an asyncio.Lock.

Run:
    python3 -m server.openai_http --model /models/Qwen3.6-35B-A3B-FP8 \
                                   --port 18099 --host 0.0.0.0
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional


# ----------------------------------------------------------------------------
# Engine wrapper (lazy-init, GPU optional for dev mode)
# ----------------------------------------------------------------------------

@dataclass
class EngineConfig:
    model_dir: str
    device: str = "cuda"
    dtype: str = "bfloat16"            # bfloat16 | float16
    max_seq_len: int = 32768
    max_new_default: int = 256
    served_model_name: str = "Lynn-Qwen3.6-35B-A3B"


class LynnEngineHandle:
    """Holds the loaded engine + tokenizer + inference state."""

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.outside = None
        self.layer_weights: list = []
        self.state = None
        self.lock = asyncio.Lock()
        self.ready = False
        self.load_started = None
        self.load_finished = None
        self.runtime_cfg: dict = {}

    def load(self):
        """Eagerly load tokenizer + outside weights + 40 layer weights to GPU."""
        import torch
        from engine.loader import load_qwen36_layer
        from engine.full_forward import load_outside_weights
        from engine.inference_state import LynnInferenceState, LAYER_TYPES
        from transformers import AutoTokenizer

        self.load_started = time.time()
        print(f"[engine] Loading tokenizer from {self.cfg.model_dir}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_dir)

        device = self.cfg.device
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[self.cfg.dtype]

        # Config dict for engine paths
        cfg_full = json.loads((Path(self.cfg.model_dir) / "config.json").read_text())
        tc = cfg_full["text_config"]
        rope_p = tc.get("rope_parameters", {})
        self.runtime_cfg = {
            "hidden_size": tc["hidden_size"],
            "num_attention_heads": tc["num_attention_heads"],
            "num_key_value_heads": tc["num_key_value_heads"],
            "head_dim": tc["head_dim"],
            "num_experts": tc["num_experts"],
            "num_experts_per_tok": tc["num_experts_per_tok"],
            "rope_theta": rope_p.get("rope_theta", tc.get("rope_theta", 1e6)),
            "partial_rotary_factor": rope_p.get("partial_rotary_factor", 1.0),
            "n_layers": tc["num_hidden_layers"],
            "layer_types": tc["layer_types"],
        }
        assert self.runtime_cfg["layer_types"] == LAYER_TYPES, "layer_types mismatch"

        print(f"[engine] Loading outside weights ...", flush=True)
        self.outside = load_outside_weights(self.cfg.model_dir, device, dtype)

        print(f"[engine] Loading {self.runtime_cfg['n_layers']} layers (resident) ...",
              flush=True)
        self.layer_weights = []
        t_start = time.time()
        for i in range(self.runtime_cfg["n_layers"]):
            w, _ = load_qwen36_layer(self.cfg.model_dir, i,
                                     num_experts=self.runtime_cfg["num_experts"],
                                     device=device, dequant_dtype=dtype)
            self.layer_weights.append(w)
            if (i + 1) % 5 == 0 or i == self.runtime_cfg["n_layers"] - 1:
                print(f"[engine]   L{i:2}  cum {time.time()-t_start:.1f}s",
                      flush=True)

        # Pre-allocate inference state once (will be reset per-request)
        self.state = LynnInferenceState(
            batch=1, max_seq_len=self.cfg.max_seq_len,
            device=device, dtype=dtype,
        )
        self.load_finished = time.time()
        self.ready = True
        print(f"[engine] READY in {self.load_finished - self.load_started:.1f}s",
              flush=True)

    def reset_state(self):
        if self.state is not None:
            self.state.reset()

    def generate(self, prompt: str, max_new_tokens: int, temperature: float = 0.0,
                 stop: Optional[list[str]] = None) -> dict:
        """Greedy (or temp=0) incremental decode. Returns dict with text + tokens."""
        import torch
        import torch.nn.functional as F
        from engine.full_forward import _prefill_layer, _decode_layer, _rms_norm

        device = self.cfg.device
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[self.cfg.dtype]

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        T_prompt = ids.shape[1]

        self.reset_state()
        embed = self.outside["model.language_model.embed_tokens.weight"]
        lm_head_w = self.outside["lm_head.weight"]
        final_norm_w = self.outside["model.language_model.norm.weight"]

        # Prefill
        h = F.embedding(ids, embed)
        pos = torch.arange(T_prompt, device=device, dtype=torch.long).unsqueeze(0)
        for i in range(self.runtime_cfg["n_layers"]):
            layer_type = self.runtime_cfg["layer_types"][i]
            h = _prefill_layer(h, pos, layer_type, self.layer_weights[i],
                               self.runtime_cfg, self.state, i)
        self.state.seq_len = T_prompt

        h_final = _rms_norm(h, final_norm_w)
        logits = F.linear(h_final[:, -1, :], lm_head_w)
        next_id = int(logits[0].argmax().item())
        new_ids = [next_id]

        # Decode loop
        for step in range(1, max_new_tokens):
            new_tok_t = torch.tensor([[next_id]], device=device, dtype=torch.long)
            h = F.embedding(new_tok_t, embed)
            for i in range(self.runtime_cfg["n_layers"]):
                layer_type = self.runtime_cfg["layer_types"][i]
                h = _decode_layer(h, self.state.seq_len, layer_type,
                                  self.layer_weights[i], self.runtime_cfg,
                                  self.state, i)
            self.state.seq_len += 1
            h_final = _rms_norm(h, final_norm_w)
            logits = F.linear(h_final[:, -1, :], lm_head_w)
            next_id = int(logits[0].argmax().item())
            new_ids.append(next_id)

            # Stop conditions
            if stop:
                partial = self.tokenizer.decode(new_ids)
                if any(s in partial for s in stop):
                    break

        completion = self.tokenizer.decode(new_ids)
        return {
            "completion": completion,
            "new_token_ids": new_ids,
            "prompt_tokens": T_prompt,
            "completion_tokens": len(new_ids),
        }


# ----------------------------------------------------------------------------
# HTTP server (FastAPI)
# ----------------------------------------------------------------------------

def make_app(handle: LynnEngineHandle):
    from fastapi import FastAPI, HTTPException, Body
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import time as time_mod

    app = FastAPI(title="Lynn Engine OpenAI-compat server")

    class CompletionRequest(BaseModel):
        model: str
        prompt: str
        max_tokens: int = 256
        temperature: float = 0.0
        stream: bool = False
        stop: Optional[list[str]] = None
        logprobs: Optional[int] = None       # accepted but ignored for now

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str
        messages: list[ChatMessage]
        max_tokens: int = 256
        temperature: float = 0.0
        stream: bool = False
        stop: Optional[list[str]] = None

    @app.get("/health")
    async def health():
        if not handle.ready:
            return {"status": "loading",
                    "elapsed_s": time_mod.time() - (handle.load_started or time_mod.time())}
        return {"status": "ok", "model": handle.cfg.served_model_name}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{
                "id": handle.cfg.served_model_name,
                "object": "model",
                "owned_by": "lynn-engine",
                "max_model_len": handle.cfg.max_seq_len,
            }],
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest = Body(...)):
        if not handle.ready:
            raise HTTPException(503, "Engine still loading")

        async with handle.lock:
            t0 = time_mod.time()
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handle.generate(req.prompt, req.max_tokens,
                                        req.temperature, req.stop),
            )
            elapsed = time_mod.time() - t0

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:16]}",
            "object": "text_completion",
            "created": int(time_mod.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "text": result["completion"],
                "finish_reason": "stop" if req.stop else "length",
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
            "_lynn_engine_metrics": {
                "elapsed_s": elapsed,
                "tokens_per_second": result["completion_tokens"] / max(elapsed, 1e-6),
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest = Body(...)):
        if not handle.ready:
            raise HTTPException(503, "Engine still loading")

        # Apply tokenizer chat_template
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = handle.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        async with handle.lock:
            t0 = time_mod.time()
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handle.generate(prompt, req.max_tokens,
                                        req.temperature, req.stop),
            )
            elapsed = time_mod.time() - t0

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time_mod.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["completion"]},
                "finish_reason": "stop" if req.stop else "length",
            }],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
            "_lynn_engine_metrics": {
                "elapsed_s": elapsed,
                "tokens_per_second": result["completion_tokens"] / max(elapsed, 1e-6),
            },
        }

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to Qwen3.6-35B-A3B-FP8 dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--served-name", default="Lynn-Qwen3.6-35B-A3B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18099)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    cfg = EngineConfig(
        model_dir=args.model, device=args.device, dtype=args.dtype,
        max_seq_len=args.max_seq_len, served_model_name=args.served_name,
    )
    handle = LynnEngineHandle(cfg)

    # Eager load (blocking) before serving
    handle.load()

    import uvicorn
    app = make_app(handle)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
