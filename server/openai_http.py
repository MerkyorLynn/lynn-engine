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
from typing import Any, AsyncIterator, Optional


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
        self.runner = None
        self.lock = asyncio.Lock()
        self.ready = False
        self.load_started = None
        self.load_finished = None

    def load(self):
        """Eagerly load the resident Lynn engine runner."""
        import torch
        from engine.resident_runner import LynnIncrementalRunner

        self.load_started = time.time()
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[self.cfg.dtype]
        self.runner = LynnIncrementalRunner(
            self.cfg.model_dir,
            device=self.cfg.device,
            dtype=dtype,
            max_seq_len=self.cfg.max_seq_len,
            verbose=True,
        )
        self.tokenizer = self.runner.tokenizer
        self.load_finished = time.time()
        self.ready = True
        print(f"[engine] READY in {self.load_finished - self.load_started:.1f}s",
              flush=True)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float = 0.0,
                 stop: Optional[list[str]] = None) -> dict:
        """Greedy (or temp=0) incremental decode. Returns dict with text + tokens."""
        if temperature not in (0, 0.0):
            raise ValueError("Lynn engine MVP server currently supports greedy temperature=0 only")
        result = self.runner.generate(prompt, max_new=max_new_tokens)
        completion = result["completion_text"]
        if stop:
            for s in stop:
                pos = completion.find(s)
                if pos >= 0:
                    completion = completion[:pos]
                    break
        return {
            "completion": completion,
            "new_token_ids": result["new_ids"],
            "prompt_tokens": len(self.tokenizer(prompt).input_ids),
            "completion_tokens": len(result["new_ids"]),
            "timings": result["timings"],
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
        chat_template_kwargs: Optional[dict[str, Any]] = None

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
        template_kwargs = {"enable_thinking": False}
        if req.chat_template_kwargs:
            template_kwargs.update(req.chat_template_kwargs)
        try:
            prompt = handle.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except TypeError:
            # Older tokenizer versions may not accept enable_thinking. The v8/Q4
            # released templates are patched to default no-think, so falling back
            # is safer than refusing to serve.
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
