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
    LYNN_PREFILL_WARMUP=1 \
    LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare \
    LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1 \
    LYNN_MOE_IMPL=packed_nvfp4 \
    LYNN_QK_NORM_ROPE_BACKEND=triton_pair \
    LYNN_RMSNORM_GATED_BACKEND=triton \
    LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1 \
    LYNN_LINEAR_BLOCK_GRAPH=1 \
    LYNN_LINEAR_BLOCK_GRAPH_REUSE=1 \
    LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1 \
    LYNN_NATIVE_FP4_LM_HEAD=1 \
    LYNN_LINEAR_STATE_UPDATE=inplace \
    LYNN_PACKED_DECODE=0 \
    LYNN_PACKED_SHARED_EXPERT=0 \
      python3 -m server.openai_http --model /models/lynn-27b-variable-recovery-step5000-nvfp4-final \
                                     --port 18099 --host 0.0.0.0
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional


# ----------------------------------------------------------------------------
# Engine wrapper (lazy-init, GPU optional for dev mode)
# ----------------------------------------------------------------------------

def _trim_before(text: str, needles: list[str]) -> str:
    hits = [text.find(item) for item in needles if item and text.find(item) >= 0]
    return text if not hits else text[: min(hits)].rstrip()


def _balanced_json(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[: idx + 1]
    return text


def _closed_code_fence(text: str) -> str:
    stripped = text.lstrip()
    offset = len(text) - len(stripped)
    if not stripped.startswith("```"):
        return text
    first_newline = text.find("\n", offset)
    if first_newline < 0:
        return text
    close = text.find("```", first_newline + 1)
    return text if close < 0 else text[: close + 3]


def _first_bullets(text: str, count: int) -> str:
    lines = text.splitlines()
    out: list[str] = []
    bullet_count = 0
    started = False
    for line in lines:
        if line.lstrip().startswith("-"):
            started = True
            bullet_count += 1
        elif started and bullet_count >= count:
            break
        if started:
            out.append(line)
        if bullet_count >= count:
            break
    return "\n".join(out).rstrip() if out else text


def resolve_format_guard(
    *,
    lynn_format_guard: Optional[dict[str, Any]],
    response_format: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if lynn_format_guard:
        return dict(lynn_format_guard)
    if response_format and response_format.get("type") == "json_object":
        return {
            "forced_prefix": "{\n  \"",
            "stop_after": "balanced_json",
            "stop_before": ["<think>", "</think>", "```"],
            "strip": True,
        }
    return None


def apply_format_guard(text: str, guard: Optional[dict[str, Any]]) -> tuple[str, bool]:
    if not guard:
        return text, False
    served = text
    stop_before = guard.get("stop_before") or []
    if stop_before:
        served = _trim_before(served, [str(item) for item in stop_before])
    stop_after = guard.get("stop_after")
    if stop_after == "balanced_json":
        served = _balanced_json(served)
    elif stop_after == "code_fence":
        served = _closed_code_fence(served)
    elif stop_after == "bullet_count":
        served = _first_bullets(served, int(guard.get("bullet_count", 3)))
    if guard.get("strip", True):
        served = served.strip()
    return served, served != text


def finish_reason_from_result(result: dict[str, Any], *, has_stop: bool, tool_calls: bool = False) -> str:
    if tool_calls:
        return "tool_calls"
    if has_stop or result["format_guard"]["stopped"]:
        return "stop"
    if result.get("stopped_reason") == "stop_token":
        return "stop"
    return "length"


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
        self.release_decode_shadows_after_prefill = (
            os.environ.get("LYNN_RELEASE_DECODE_SHADOWS_AFTER_PREFILL", "0") == "1"
        )
        self.release_decode_shadows_consumed = False

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
                 stop: Optional[list[str]] = None,
                 format_guard: Optional[dict[str, Any]] = None) -> dict:
        """Greedy (or temp=0) incremental decode. Returns dict with text + tokens."""
        if temperature not in (0, 0.0):
            raise ValueError("Lynn engine MVP server currently supports greedy temperature=0 only")
        if self.release_decode_shadows_after_prefill and self.release_decode_shadows_consumed:
            raise RuntimeError(
                "LYNN_RELEASE_DECODE_SHADOWS_AFTER_PREFILL=1 is a one-shot/session-scoped mode. "
                "BF16 shadows were already released; restart the server for another prefill request."
            )
        result = self.runner.generate(
            prompt,
            max_new=max_new_tokens,
            forced_prefix_text=None if not format_guard else format_guard.get("forced_prefix"),
            release_decode_shadows_after_prefill=self.release_decode_shadows_after_prefill,
        )
        if self.release_decode_shadows_after_prefill:
            self.release_decode_shadows_consumed = True
        completion = result["completion_text"]
        completion, format_guard_stopped = apply_format_guard(completion, format_guard)
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
            "mtp_shadow": result.get("mtp_shadow"),
            "format_guard": {
                "enabled": bool(format_guard),
                "stopped": format_guard_stopped,
                "forced_prefix": result.get("forced_prefix"),
                "stop_after": None if not format_guard else format_guard.get("stop_after"),
            },
            "stopped_reason": result.get("stopped_reason"),
            "release_decode_shadows_after_prefill": self.release_decode_shadows_after_prefill,
            "release_decode_shadows_consumed": self.release_decode_shadows_consumed,
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
        response_format: Optional[dict[str, Any]] = None
        lynn_format_guard: Optional[dict[str, Any]] = None

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
        tools: Optional[list[dict[str, Any]]] = None
        tool_choice: Optional[Any] = None       # accepted for OpenAI compatibility; model decides today
        chat_template_kwargs: Optional[dict[str, Any]] = None
        response_format: Optional[dict[str, Any]] = None
        lynn_format_guard: Optional[dict[str, Any]] = None

    def parse_qwen_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
        """Parse Qwen-style XML tool calls into OpenAI `tool_calls`.

        Qwen chat templates render tool calls as:

          <tool_call>
          <function=get_weather>
          <parameter=location>
          北京
          </parameter>
          </function>
          </tool_call>

        Keep this parser deliberately small and strict-ish: if parsing fails,
        return the original text as normal assistant content rather than
        fabricating a tool call.
        """
        calls: list[dict[str, Any]] = []
        for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.S):
            m = re.search(r"<function=([^>]+)>(.*?)</function>", block, flags=re.S)
            if not m:
                continue
            name = m.group(1).strip()
            body = m.group(2)
            args: dict[str, str] = {}
            for pm in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, flags=re.S):
                args[pm.group(1).strip()] = pm.group(2).strip()
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })
        if not calls:
            return text, []
        content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
        return content, calls

    @app.get("/health")
    async def health():
        if not handle.ready:
            return {"status": "loading",
                    "elapsed_s": time_mod.time() - (handle.load_started or time_mod.time())}
        runtime = {}
        if handle.runner is not None:
            runtime = {
                "moe_impl": getattr(handle.runner, "moe_impl", None),
                "packed_nvfp4_moe_aliases_attached": getattr(
                    handle.runner, "packed_nvfp4_moe_aliases_attached", None
                ),
                "moe_repack_sidecar_dir": getattr(handle.runner, "moe_repack_sidecar_dir", None),
                "moe_repack_sidecar_layers_attached": getattr(
                    handle.runner, "moe_repack_sidecar_layers_attached", None
                ),
                "moe_repack_sidecar_load_seconds": getattr(
                    handle.runner, "moe_repack_sidecar_load_seconds", None
                ),
                "moe_active_scratch_attached": getattr(
                    handle.runner, "moe_active_scratch_attached", None
                ),
                "packed_decode_backend": getattr(handle.runner, "packed_decode_backend", None),
                "packed_decode_aliases_attached": getattr(
                    handle.runner, "packed_decode_aliases_attached", None
                ),
                "packed_decode_native_prepared": getattr(
                    handle.runner, "packed_decode_native_prepared", None
                ),
                "packed_decode_aliases_skipped": getattr(
                    handle.runner, "packed_decode_aliases_skipped", None
                ),
                "native_fp4_lm_head_enabled": getattr(
                    handle.runner, "native_fp4_lm_head_enabled", None
                ),
                "mtp_sidecar_loaded": getattr(handle.runner, "mtp_sidecar_loaded", None),
                "mtp_shadow_verify_enabled": getattr(handle.runner, "mtp_shadow_verify_enabled", None),
                "mtp_sidecar_path": getattr(handle.runner, "mtp_sidecar_path", None),
                "mtp_load_seconds": getattr(handle.runner, "mtp_load_seconds", None),
                "runtime_warnings": getattr(handle.runner, "runtime_warnings", []),
            }
        return {
            "status": "ok",
            "model": handle.cfg.served_model_name,
            "release_decode_shadows_after_prefill": handle.release_decode_shadows_after_prefill,
            "release_decode_shadows_consumed": handle.release_decode_shadows_consumed,
            "runtime": runtime,
        }

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
            format_guard = resolve_format_guard(
                lynn_format_guard=req.lynn_format_guard,
                response_format=req.response_format,
            )
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: handle.generate(req.prompt, req.max_tokens,
                                            req.temperature, req.stop, format_guard),
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
            elapsed = time_mod.time() - t0

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:16]}",
            "object": "text_completion",
            "created": int(time_mod.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "text": result["completion"],
                "finish_reason": finish_reason_from_result(result, has_stop=bool(req.stop)),
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
                "timings": result["timings"],
                "mtp_shadow": result.get("mtp_shadow"),
                "format_guard": result["format_guard"],
                "stopped_reason": result.get("stopped_reason"),
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
                tools=req.tools,
                **template_kwargs,
            )
        except TypeError:
            # Older tokenizer versions may not accept enable_thinking. The v8/Q4
            # released templates are patched to default no-think, so falling back
            # is safer than refusing to serve.
            try:
                prompt = handle.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, tools=req.tools,
                )
            except TypeError:
                prompt = handle.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

        async with handle.lock:
            t0 = time_mod.time()
            format_guard = resolve_format_guard(
                lynn_format_guard=req.lynn_format_guard,
                response_format=req.response_format,
            )
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: handle.generate(prompt, req.max_tokens,
                                            req.temperature, req.stop, format_guard),
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
            elapsed = time_mod.time() - t0

        content, tool_calls = parse_qwen_tool_calls(result["completion"])
        message: dict[str, Any] = {"role": "assistant", "content": content}
        finish_reason = finish_reason_from_result(
            result,
            has_stop=bool(req.stop),
            tool_calls=bool(tool_calls),
        )
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time_mod.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
            "_lynn_engine_metrics": {
                "elapsed_s": elapsed,
                "tokens_per_second": result["completion_tokens"] / max(elapsed, 1e-6),
                "timings": result["timings"],
                "mtp_shadow": result.get("mtp_shadow"),
                "format_guard": result["format_guard"],
                "stopped_reason": result.get("stopped_reason"),
            },
        }

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to Lynn/Qwen model dir")
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
