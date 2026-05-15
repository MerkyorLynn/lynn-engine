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
    _with_inferred_layer_config,
    load_outside_weights,
)
from engine.inference_state import LAYER_TYPES, LynnInferenceState
from engine.loader import load_qwen36_layer
from engine.nvfp4_runtime import (
    _compact_scale_to_swizzled_fp8,
    load_grouped_nvfp4_weight,
    load_packed_nvfp4_linear,
)
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native


_E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


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


def _encode_prompt(tokenizer, prompt: str, device: str, *, use_chat_template: bool) -> torch.Tensor:
    """Encode either raw text or a no-think chat turn.

    Qwen chat templates across transformers versions may return a Tensor,
    BatchEncoding, string, or list depending on arguments/template. Normalize
    all variants into a `[1, T]` tensor.
    """
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if isinstance(ids, str):
            ids = tokenizer(ids, return_tensors="pt").input_ids
        elif not isinstance(ids, torch.Tensor):
            ids = torch.tensor([ids], dtype=torch.long)
        return ids.to(device)
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


class FullTokenGraphSlot:
    """One fixed-position full-token CUDA graph slot.

    P10-T showed that a full-token graph is strict when it is captured at the
    same real sequence position it will replay. This tiny wrapper keeps the
    mutable token buffer, logits buffer, and graph object together so later
    serving code can cache slots by `(position, state contract)`.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        token_buf: torch.Tensor,
        logits_buf: torch.Tensor,
        graph: torch.cuda.CUDAGraph,
    ) -> None:
        self.seq_len = int(seq_len)
        self.token_buf = token_buf
        self.logits_buf = logits_buf
        self.graph = graph
        self.replays = 0

    def replay(self, token_id: int) -> torch.Tensor:
        self.token_buf.fill_(int(token_id))
        self.graph.replay()
        self.replays += 1
        return self.logits_buf


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
        if impl == "indexed_bmm":
            raise ValueError(
                "LynnIncrementalRunner supports reusable prompts only with "
                "LYNN_MOE_IMPL=optimized, bmm, triton, or packed_nvfp4. indexed_bmm mutates "
                "weights after prefill and is single-prompt only today."
            )
        self.model_dir = str(model_dir)
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.verbose = verbose
        self.moe_impl = impl
        self.packed_nvfp4_moe_aliases_attached = 0
        self.packed_decode_backend = os.environ.get("LYNN_PACKED_DECODE_BACKEND", "scalar_bridge")
        self.packed_decode_aliases_attached = 0
        self.packed_decode_native_prepared = 0
        self.packed_decode_aliases_skipped = 0
        self.cfg, self.n_layers = _runtime_config(self.model_dir)

        from transformers import AutoTokenizer

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        stop_ids = set()
        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(int(self.tokenizer.eos_token_id))
        for token in ("<|im_end|>", "<|endoftext|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                stop_ids.add(int(token_id))
        self.stop_token_ids = stop_ids
        if verbose:
            print(f"[resident] loading outside weights from {self.model_dir}", flush=True)
        self.outside = load_outside_weights(self.model_dir, device, dtype)

        if verbose:
            print(f"[resident] loading {self.n_layers} layers", flush=True)
        self.layer_weights = []
        self.layer_cfgs = []
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for i in range(self.n_layers):
            w, inferred = load_qwen36_layer(
                self.model_dir,
                i,
                num_experts=self.cfg["num_experts"],
                device=device,
                dequant_dtype=dtype,
            )
            self.layer_weights.append(w)
            self.layer_cfgs.append(_with_inferred_layer_config(self.cfg, inferred))
            if verbose and (i % 5 == 4 or i == self.n_layers - 1):
                print(f"  [resident] L{i:02}: {time.time() - t0:.1f}s", flush=True)
        if impl == "triton":
            self._prepare_triton_moe_layout()
        if impl == "packed_nvfp4":
            self._prepare_packed_nvfp4_moe_layout()
        if (
            os.environ.get("LYNN_PACKED_DECODE", "0") == "1"
            or os.environ.get("LYNN_PACKED_DECODE_LINEAR_ATTN", "0") == "1"
            or os.environ.get("LYNN_PACKED_DECODE_FULL_ATTN", "0") == "1"
        ):
            self._prepare_packed_decode_aliases()
        if os.environ.get("LYNN_LINEAR_ATTN_INPROJ_FUSED", "0") == "1":
            self._prepare_linear_attn_inproj_fused()
        if os.environ.get("LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4", "0") == "1":
            self._prepare_linear_attn_inproj_fused_native_fp4()
        self._prepare_linear_attn_decode_constants()
        self._linear_block_graph_slot: dict[str, Any] | None = None
        self.prefill_warmup_seconds: float | None = None
        self.linear_block_graph_prewarm_seconds: float | None = None
        self.native_fp4_lm_head_enabled = False
        self._native_fp4_lm_head_weight: torch.Tensor | None = None
        self._native_fp4_lm_head_scale_b: torch.Tensor | None = None
        self.native_fp4_lm_head_prepare_seconds: float | None = None
        if os.environ.get("LYNN_NATIVE_FP4_LM_HEAD", "0") == "1":
            self._prepare_native_fp4_lm_head()
        if os.environ.get("LYNN_PREFILL_WARMUP", "0") == "1" and device.startswith("cuda"):
            t_prefill = time.time()
            self._warmup_prefill_kernels()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            self.prefill_warmup_seconds = time.time() - t_prefill
            if verbose:
                print(
                    f"[resident] warmed prefill kernels in {self.prefill_warmup_seconds:.3f}s",
                    flush=True,
                )
        if (
            os.environ.get("LYNN_LINEAR_BLOCK_GRAPH_PREWARM", "0") == "1"
            and os.environ.get("LYNN_LINEAR_BLOCK_GRAPH_REUSE", "0") == "1"
            and device.startswith("cuda")
        ):
            t_graph = time.time()
            self._prewarm_linear_block_graph_slot()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            self.linear_block_graph_prewarm_seconds = time.time() - t_graph
            if verbose:
                print(
                    f"[resident] prewarmed linear block graph slot in "
                    f"{self.linear_block_graph_prewarm_seconds:.3f}s",
                    flush=True,
                )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        self.load_seconds = time.time() - t0
        self.cuda_memory_after_load: dict[str, float] = {}
        if device.startswith("cuda"):
            self.cuda_memory_after_load = {
                "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
                "max_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "max_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            }

    def _prepare_triton_moe_layout(self) -> None:
        """Attach zero-copy / stacked MoE aliases required by Triton decode.

        The 27B variable skeleton already stores fused expert tensors; use
        stride-aware views instead of copying. Full 35B checkpoints need a
        one-time stack because they keep per-expert tensors.
        """
        from triton_kernels.moe_expert_ffn import stack_expert_weights

        for w, cfg in zip(self.layer_weights, self.layer_cfgs):
            if "mlp.experts._gate_stacked" in w:
                continue
            if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
                gate_stacked, up_stacked = w["mlp.experts.gate_up_proj"].chunk(2, dim=1)
                w["mlp.experts._gate_stacked"] = gate_stacked
                w["mlp.experts._up_stacked"] = up_stacked
                w["mlp.experts._down_stacked"] = w["mlp.experts.down_proj"]
                cfg["num_experts"] = int(w["mlp.experts.gate_up_proj"].shape[0])
            else:
                stack_expert_weights(w, num_experts=cfg["num_experts"])

    def _prepare_packed_nvfp4_moe_layout(self) -> None:
        """Attach grouped packed NVFP4 expert tensors for decode MoE."""
        attached = 0
        for layer_idx, w in enumerate(self.layer_weights):
            base = f"model.language_model.layers.{layer_idx}.mlp.experts"
            try:
                gate_up_packed, gate_up_scale, gate_up_global = load_grouped_nvfp4_weight(
                    self.model_dir,
                    f"{base}.gate_up_proj",
                    device=self.device,
                )
                down_packed, down_scale, down_global = load_grouped_nvfp4_weight(
                    self.model_dir,
                    f"{base}.down_proj",
                    device=self.device,
                )
            except (FileNotFoundError, KeyError):
                continue
            w["mlp.experts._gate_up_packed"] = gate_up_packed
            w["mlp.experts._gate_up_scale"] = gate_up_scale
            w["mlp.experts._gate_up_global_scale"] = gate_up_global
            w["mlp.experts._down_packed"] = down_packed
            w["mlp.experts._down_scale"] = down_scale
            w["mlp.experts._down_global_scale"] = down_global
            if os.environ.get("LYNN_PACKED_SHARED_EXPERT", "0") == "1":
                shared_base = f"model.language_model.layers.{layer_idx}.mlp.shared_expert"
                try:
                    w["mlp.shared_expert.gate_proj.weight.packed"] = load_packed_nvfp4_linear(
                        self.model_dir,
                        f"{shared_base}.gate_proj",
                        device=self.device,
                    )
                    w["mlp.shared_expert.up_proj.weight.packed"] = load_packed_nvfp4_linear(
                        self.model_dir,
                        f"{shared_base}.up_proj",
                        device=self.device,
                    )
                    w["mlp.shared_expert.down_proj.weight.packed"] = load_packed_nvfp4_linear(
                        self.model_dir,
                        f"{shared_base}.down_proj",
                        device=self.device,
                    )
                except KeyError:
                    pass
            attached += 1
        self.packed_nvfp4_moe_aliases_attached = attached
        if self.verbose:
            print(f"[resident] packed NVFP4 MoE aliases attached={attached}", flush=True)

    def _prepare_linear_attn_inproj_fused(self) -> None:
        """Attach fused qkv/z/b/a projection weights for linear-attn decode.

        This is an opt-in decode speed path. It trades ~1.5 GB extra resident
        weights on the 27B skeleton for replacing four small GEMMs per
        linear-attention layer with one larger GEMM.
        """
        for layer_type, w in zip(LAYER_TYPES, self.layer_weights):
            if layer_type != "linear_attention":
                continue
            key = "linear_attn._in_proj_qkv_z_b_a.weight"
            if key in w:
                continue
            w[key] = torch.cat(
                [
                    w["linear_attn.in_proj_qkv.weight"],
                    w["linear_attn.in_proj_z.weight"],
                    w["linear_attn.in_proj_b.weight"],
                    w["linear_attn.in_proj_a.weight"],
                ],
                dim=0,
            ).contiguous()

    def _prepare_linear_attn_inproj_fused_native_fp4(self) -> None:
        """Attach packed native-FP4 fused qkv/z/b/a projection weights.

        This is an opt-in packed-resident bridge. It intentionally coexists
        with the BF16 fused in-projection path so benchmarks can compare both
        paths without changing the default production runner.
        """
        from engine.nvfp4_runtime import fuse_packed_nvfp4_linears, load_packed_nvfp4_linear

        for layer_idx, (layer_type, w) in enumerate(zip(LAYER_TYPES, self.layer_weights)):
            if layer_type != "linear_attention":
                continue
            key = "linear_attn._in_proj_qkv_z_b_a.weight"
            base_prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
            fused = fuse_packed_nvfp4_linears(
                f"{base_prefix}._in_proj_qkv_z_b_a",
                [
                    load_packed_nvfp4_linear(self.model_dir, f"{base_prefix}.in_proj_qkv", device=self.device),
                    load_packed_nvfp4_linear(self.model_dir, f"{base_prefix}.in_proj_z", device=self.device),
                    load_packed_nvfp4_linear(self.model_dir, f"{base_prefix}.in_proj_b", device=self.device),
                    load_packed_nvfp4_linear(self.model_dir, f"{base_prefix}.in_proj_a", device=self.device),
                ],
            )
            w[key] = fused

    def _prepare_linear_attn_decode_constants(self) -> None:
        """Precompute tiny static decode constants for linear-attention layers."""
        for layer_type, w in zip(LAYER_TYPES, self.layer_weights):
            if layer_type != "linear_attention":
                continue
            key = "linear_attn._neg_exp_A_log"
            if key not in w:
                w[key] = -w["linear_attn.A_log"].float().exp().contiguous()

    @staticmethod
    def _quantize_weight_to_native_fp4(
        weight: torch.Tensor,
        *,
        chunk_rows: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack a dense BF16/FP16 matrix to torch native E2M1 FP4 layout.

        This is a one-time startup conversion for tensors that are not shipped
        packed yet. It is intentionally simple and deterministic; the runtime
        hot path only sees the packed tensor and swizzled scales.
        """
        if weight.ndim != 2 or weight.shape[1] % 16 != 0:
            raise ValueError(f"expected [N, K] with K divisible by 16, got {tuple(weight.shape)}")
        table = _E2M1_TABLE.to(device=weight.device)
        out_features, in_features = weight.shape
        groups = in_features // 16
        packed = torch.empty((out_features, in_features // 2), device=weight.device, dtype=torch.uint8)
        scale = torch.empty((out_features, groups), device=weight.device, dtype=torch.float32)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            block = weight[start:end].float().reshape(end - start, groups, 16)
            scale_chunk = (block.abs().amax(dim=-1) / float(table[-1])).clamp_min(1e-8)
            normalized = block.abs() / scale_chunk.unsqueeze(-1)
            magnitude = torch.argmin(
                (normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(),
                dim=-1,
            )
            sign = (block < 0).to(torch.uint8) * 8
            codes = (magnitude.to(torch.uint8) | sign).reshape(end - start, in_features)
            packed[start:end] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
            scale[start:end] = scale_chunk
        return packed, scale.contiguous()

    def _prepare_native_fp4_lm_head(self) -> None:
        """Prepare opt-in native FP4 lm_head for greedy serving.

        P10-O/P10-P validated this path as greedy-safe on six prompts. It is
        still opt-in because sampling-heavy production needs a broader parity
        gate, but exposing it through the resident runner lets the HTTP server
        measure a real serving-shaped speedup instead of a benchmark-only one.
        """
        if not self.device.startswith("cuda"):
            raise RuntimeError("LYNN_NATIVE_FP4_LM_HEAD requires CUDA")
        if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("native FP4 lm_head requires torch.float4_e2m1fn_x2 and torch._scaled_mm")
        t0 = time.time()
        lm_head = self.outside["lm_head.weight"].contiguous()
        packed, compact_scale = self._quantize_weight_to_native_fp4(lm_head)
        self._native_fp4_lm_head_weight = packed.view(torch.float4_e2m1fn_x2)
        self._native_fp4_lm_head_scale_b = _compact_scale_to_swizzled_fp8(
            compact_scale,
            outer_dim=lm_head.shape[0],
            k=lm_head.shape[1],
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.native_fp4_lm_head_prepare_seconds = time.time() - t0
        self.native_fp4_lm_head_enabled = True
        if self.verbose:
            print(
                "[resident] native FP4 lm_head prepared "
                f"in {self.native_fp4_lm_head_prepare_seconds:.3f}s",
                flush=True,
            )

    def _lm_head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project final hidden state to logits using BF16 or opt-in native FP4."""
        h2d = hidden[:, -1, :] if hidden.ndim == 3 else hidden
        if not self.native_fp4_lm_head_enabled:
            return F.linear(h2d, self.outside["lm_head.weight"])
        assert self._native_fp4_lm_head_weight is not None
        assert self._native_fp4_lm_head_scale_b is not None
        act_packed, scale_a = quantize_fp4_m1_native(h2d.contiguous())
        return torch._scaled_mm(
            act_packed.view(torch.float4_e2m1fn_x2),
            self._native_fp4_lm_head_weight.t(),
            scale_a=scale_a,
            scale_b=self._native_fp4_lm_head_scale_b,
            out_dtype=torch.float16,
        )

    @staticmethod
    def _snapshot_state(state: LynnInferenceState) -> dict[str, Any]:
        """Clone all mutable decode state.

        This is intentionally heavier than the linear-only snapshot because a
        full-token graph mutates full-attention KV plus linear-attention
        recurrent/conv state. P10-T uses it as a correctness bridge; later
        gates can replace it with cheaper per-slot refresh logic.
        """
        return {
            "seq_len": int(state.seq_len),
            "kv": {i: (kv[0].clone(), kv[1].clone()) for i, kv in state.kv_cache.items()},
            "recurrent": {i: t.clone() for i, t in state.recurrent_state.items()},
            "conv": {i: t.clone() for i, t in state.conv_state.items()},
        }

    @staticmethod
    def _restore_state(state: LynnInferenceState, snap: dict[str, Any]) -> None:
        state.seq_len = int(snap["seq_len"])
        for i, (k, v) in snap["kv"].items():
            state.kv_cache[i][0].copy_(k)
            state.kv_cache[i][1].copy_(v)
        for i, t in snap["recurrent"].items():
            state.recurrent_state[i].copy_(t)
        for i, t in snap["conv"].items():
            state.conv_state[i].copy_(t)

    def _capture_full_token_graph_slot(
        self,
        state: LynnInferenceState,
        token_id: int,
    ) -> FullTokenGraphSlot:
        """Capture one strict full-token graph at the current sequence length.

        This is not yet wired into default `generate()`: future-window graph
        families drift, while current-position graph slots are strict. Keeping
        this as an explicit method lets P10/P11 gates build a lazy graph-slot
        cache without changing production serving behavior prematurely.
        """
        if not self.device.startswith("cuda"):
            raise RuntimeError("full-token graph slots require CUDA")

        seq_len = int(state.seq_len)
        token_buf = torch.empty((1, 1), device=self.device, dtype=torch.long)
        token_buf.fill_(int(token_id))
        pos_tensor = torch.tensor([[seq_len]], device=self.device, dtype=torch.long)
        vocab = int(self.outside["lm_head.weight"].shape[0])
        logits_dtype = torch.float16 if self.native_fp4_lm_head_enabled else self.dtype
        logits_buf = torch.empty((1, vocab), device=self.device, dtype=logits_dtype)
        snap = self._snapshot_state(state)

        def graph_body() -> None:
            state.seq_len = seq_len
            h = F.embedding(token_buf, self.outside["model.language_model.embed_tokens.weight"])
            for i in range(self.n_layers):
                h = _decode_layer(
                    h,
                    pos_tensor,
                    LAYER_TYPES[i],
                    self.layer_weights[i],
                    self.layer_cfgs[i],
                    state,
                    i,
                )
            h_final = _rms_norm(h, self.outside["model.language_model.norm.weight"])
            logits_buf.copy_(self._lm_head_logits(h_final).to(logits_dtype))

        graph_body()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._restore_state(state, snap)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_body()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._restore_state(state, snap)
        return FullTokenGraphSlot(seq_len=seq_len, token_buf=token_buf, logits_buf=logits_buf, graph=graph)

    def release_decode_bf16_shadows(
        self,
        *,
        include_moe_experts: bool = True,
        include_projection_aliases: bool = True,
    ) -> dict[str, Any]:
        """Release BF16 tensors covered by packed decode aliases.

        This is a session-scoped / decode-only primitive. Do not call it before
        prefill in the current runner, and do not use it for multi-request
        serving unless the server can reload shadows or run packed prefill.
        """
        released: list[dict[str, Any]] = []
        released_bytes = 0

        def drop(layer_idx: int, w: dict[str, Any], key: str, reason: str) -> None:
            nonlocal released_bytes
            tensor = w.get(key)
            if not isinstance(tensor, torch.Tensor):
                return
            nbytes = int(tensor.numel() * tensor.element_size())
            del w[key]
            released_bytes += nbytes
            released.append({
                "layer": layer_idx,
                "key": key,
                "bytes": nbytes,
                "reason": reason,
            })

        for layer_idx, w in enumerate(self.layer_weights):
            if include_moe_experts:
                if "mlp.experts._gate_up_packed" in w:
                    drop(layer_idx, w, "mlp.experts.gate_up_proj", "packed_grouped_moe_gate_up")
                if "mlp.experts._down_packed" in w:
                    drop(layer_idx, w, "mlp.experts.down_proj", "packed_grouped_moe_down")

            if include_projection_aliases:
                for key in list(w.keys()):
                    if not key.endswith(".weight"):
                        continue
                    if key + ".packed" not in w:
                        continue
                    drop(layer_idx, w, key, "packed_linear_alias")

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return {
            "released_tensors": len(released),
            "released_bytes": released_bytes,
            "released_gib": released_bytes / (1024**3),
            "items": released[:50],
            "truncated_items": max(0, len(released) - 50),
        }

    def _prepare_packed_decode_aliases(self) -> None:
        """Attach packed NVFP4 decode aliases while keeping BF16 prefill safe.

        The current resident loader slow-dequants NVFP4 into BF16 tensors so
        prefill can use ordinary multi-token GEMMs. Native packed kernels are
        decode-only today, so we attach opt-in `.packed` aliases instead of
        replacing the BF16 weights. This gives P8 a clean integration gate:
        parity and timing can be measured end-to-end before any memory-saving
        resident layout change.
        """
        backend = self.packed_decode_backend
        projections_by_type = {
            "linear_attention": [
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.out_proj.weight",
            ],
            "full_attention": [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
            ],
        }
        attached = 0
        native_prepared = 0
        skipped = 0
        for layer_idx, (layer_type, w) in enumerate(zip(LAYER_TYPES, self.layer_weights)):
            for short_key in projections_by_type.get(layer_type, []):
                alias_key = short_key + ".packed"
                if alias_key in w or short_key not in w:
                    continue
                base_key = f"model.language_model.layers.{layer_idx}.{short_key.removesuffix('.weight')}"
                try:
                    w[alias_key] = load_packed_nvfp4_linear(
                        self.model_dir,
                        base_key,
                        name=base_key,
                        device=self.device,
                        default_backend=backend,
                    )
                    if (
                        backend == "native_fast_2d"
                        and os.environ.get("LYNN_PACKED_DECODE_PREPARE_NATIVE", "0") == "1"
                    ):
                        # Move scale swizzle / native tensor view setup out of
                        # the first user decode token. This is still an opt-in
                        # P9 bridge path; packed-resident memory ownership is
                        # handled by a later gate.
                        w[alias_key]._native_scale_b()
                        w[alias_key]._native_weight_t()
                        native_prepared += 1
                    attached += 1
                except KeyError:
                    skipped += 1
        if self.verbose:
            print(
                f"[resident] packed decode aliases attached={attached} "
                f"skipped={skipped} backend={backend} "
                f"native_prepared={native_prepared}",
                flush=True,
            )
        self.packed_decode_aliases_attached = attached
        self.packed_decode_native_prepared = native_prepared
        self.packed_decode_aliases_skipped = skipped

    @staticmethod
    def _snapshot_linear_state(state: LynnInferenceState) -> dict[str, Any]:
        return {
            "seq_len": state.seq_len,
            "recurrent": {i: t.clone() for i, t in state.recurrent_state.items()},
            "conv": {i: t.clone() for i, t in state.conv_state.items()},
        }

    @staticmethod
    def _restore_linear_state(state: LynnInferenceState, snap: dict[str, Any]) -> None:
        state.seq_len = int(snap["seq_len"])
        for i, t in snap["recurrent"].items():
            state.recurrent_state[i].copy_(t)
        for i, t in snap["conv"].items():
            state.conv_state[i].copy_(t)

    @staticmethod
    def _copy_linear_state(dst: LynnInferenceState, src: LynnInferenceState) -> None:
        """Copy only linear-attention recurrent/conv state into a graph slot."""
        dst.seq_len = int(src.seq_len)
        for i, t in src.recurrent_state.items():
            dst.recurrent_state[i].copy_(t)
        for i, t in src.conv_state.items():
            dst.conv_state[i].copy_(t)

    def _get_reusable_linear_block_graphs(
        self,
        request_state: LynnInferenceState,
        h_seed: torch.Tensor,
        pos_tensor: torch.Tensor,
    ) -> tuple[list[dict[str, Any]], float, bool]:
        """Return graph-captured linear-attn blocks that can be reused.

        P6-S captured the 10 linear-attention block graphs per request. That is
        correct, but short completions pay the capture cost over and over. This
        opt-in slot keeps a separate linear-attn state object and graph buffers
        resident across requests, then copies each request's prefill recurrent
        state into the slot before decode. Full-attention KV remains on the
        per-request state and is still run eagerly between graph blocks.
        """
        if self._linear_block_graph_slot is None:
            graph_state = LynnInferenceState(
                batch=1,
                max_seq_len=self.max_seq_len,
                device=self.device,
                dtype=self.dtype,
            )
            self._copy_linear_state(graph_state, request_state)
            blocks, capture_seconds = self._capture_linear_block_graphs(graph_state, h_seed, pos_tensor)
            self._linear_block_graph_slot = {
                "state": graph_state,
                "blocks": blocks,
                "capture_seconds": capture_seconds,
            }
            return blocks, capture_seconds, True

        slot = self._linear_block_graph_slot
        self._copy_linear_state(slot["state"], request_state)
        return slot["blocks"], 0.0, False

    def _capture_linear_block_graphs(
        self,
        state: LynnInferenceState,
        h_seed: torch.Tensor,
        pos_tensor: torch.Tensor,
    ) -> tuple[list[dict[str, Any]], float]:
        """Capture the 10 repeated 3-layer linear-attn blocks for this request.

        The full-attention layers remain eager because their KV slice length
        changes with `state.seq_len`; linear-attention recurrent/conv state is
        fixed-shape, so it can be graph-replayed safely after restoring the
        state mutated during capture.
        """
        t0 = time.time()
        snap = self._snapshot_linear_state(state)
        blocks: list[dict[str, Any]] = []
        for start_layer in range(0, self.n_layers, 4):
            layers = [start_layer, start_layer + 1, start_layer + 2]
            if any(LAYER_TYPES[i] != "linear_attention" for i in layers):
                raise RuntimeError(f"unexpected linear block layout at {layers}")
            input_buf = torch.empty_like(h_seed)
            output_buf = torch.empty_like(h_seed)

            def block_fn(layers=layers, input_buf=input_buf, output_buf=output_buf):
                h = input_buf
                for i in layers:
                    h = _decode_layer(
                        h,
                        pos_tensor,
                        LAYER_TYPES[i],
                        self.layer_weights[i],
                        self.layer_cfgs[i],
                        state,
                        i,
                    )
                output_buf.copy_(h)

            input_buf.copy_(h_seed)
            block_fn()
            self._restore_linear_state(state, snap)
            input_buf.copy_(h_seed)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                block_fn()
            blocks.append({"start_layer": start_layer, "input": input_buf, "output": output_buf, "graph": graph})
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._restore_linear_state(state, snap)
        return blocks, time.time() - t0

    def _prewarm_linear_block_graph_slot(self) -> None:
        """Capture reusable linear-attn block graphs during runner warmup.

        The graph only depends on tensor addresses/shapes and mutable linear
        recurrent/conv state. Prompt-specific values are copied into the slot
        before each request by `_get_reusable_linear_block_graphs`.
        """
        graph_state = LynnInferenceState(
            batch=1,
            max_seq_len=self.max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        h_seed = torch.zeros(
            (1, 1, int(self.cfg["hidden_size"])),
            device=self.device,
            dtype=self.dtype,
        )
        pos_tensor = torch.zeros((1, 1), device=self.device, dtype=torch.long)
        blocks, capture_seconds = self._capture_linear_block_graphs(graph_state, h_seed, pos_tensor)
        self._linear_block_graph_slot = {
            "state": graph_state,
            "blocks": blocks,
            "capture_seconds": capture_seconds,
        }

    def _warmup_prefill_kernels(self) -> None:
        """Run one tiny prefill to compile kernels before serving requests."""
        ids = self.tokenizer("warmup", return_tensors="pt").input_ids.to(self.device)
        state = LynnInferenceState(
            batch=1,
            max_seq_len=self.max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        h = F.embedding(ids, self.outside["model.language_model.embed_tokens.weight"])
        pos = torch.arange(ids.shape[1], device=self.device, dtype=torch.long).unsqueeze(0)
        for i in range(self.n_layers):
            h = _prefill_layer(h, pos, LAYER_TYPES[i], self.layer_weights[i], self.layer_cfgs[i], state, i)
        h_final = _rms_norm(h, self.outside["model.language_model.norm.weight"])
        _ = self._lm_head_logits(h_final)

    def generate(
        self,
        prompt: str,
        *,
        max_new: int = 4,
        top_k: int = 0,
        use_chat_template: bool = False,
        release_decode_shadows_after_prefill: bool = False,
    ) -> dict[str, Any]:
        tok = self.tokenizer
        ids = _encode_prompt(tok, prompt, self.device, use_chat_template=use_chat_template)
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
            h = _prefill_layer(h, pos, LAYER_TYPES[i], self.layer_weights[i], self.layer_cfgs[i], state, i)
        state.seq_len = T
        h_final = _rms_norm(h, self.outside["model.language_model.norm.weight"])
        logits = self._lm_head_logits(h_final)
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
        release_report = None
        if release_decode_shadows_after_prefill:
            release_report = self.release_decode_bf16_shadows()
            if self.verbose:
                print(
                    "  [resident] released decode BF16 shadows "
                    f"{release_report['released_gib']:.2f} GiB",
                    flush=True,
                )
        new_ids = [next_id]
        decode_seconds = []
        stopped_reason = "max_new"
        new_token_tensor = torch.empty((1, 1), device=self.device, dtype=torch.long)
        pos_tensor = torch.empty((1, 1), device=self.device, dtype=torch.long)
        linear_block_graphs = None
        linear_block_graph_state = state
        graph_capture_seconds = None
        graph_reused = None
        if (
            os.environ.get("LYNN_LINEAR_BLOCK_GRAPH", "0") == "1"
            and self.device.startswith("cuda")
            and max_new > 1
        ):
            new_token_tensor.fill_(next_id)
            h_seed = F.embedding(new_token_tensor, self.outside["model.language_model.embed_tokens.weight"])
            pos_tensor.fill_(state.seq_len)
            if os.environ.get("LYNN_LINEAR_BLOCK_GRAPH_REUSE", "0") == "1":
                linear_block_graphs, graph_capture_seconds, graph_created = self._get_reusable_linear_block_graphs(
                    state,
                    h_seed,
                    pos_tensor,
                )
                graph_reused = not graph_created
                linear_block_graph_state = self._linear_block_graph_slot["state"]
            else:
                linear_block_graphs, graph_capture_seconds = self._capture_linear_block_graphs(state, h_seed, pos_tensor)
                graph_reused = False
            if self.verbose:
                action = "reused" if graph_reused else "captured"
                print(
                    f"  [resident] {action} {len(linear_block_graphs)} linear block graphs "
                    f"in {graph_capture_seconds:.3f}s",
                    flush=True,
                )

        if next_id in self.stop_token_ids:
            stopped_reason = "stop_token"

        for step in range(1, max_new):
            if stopped_reason == "stop_token":
                break
            step_t0 = time.time()
            new_token_tensor.fill_(next_id)
            h = F.embedding(new_token_tensor, self.outside["model.language_model.embed_tokens.weight"])
            pos_id = state.seq_len
            pos_tensor.fill_(pos_id)
            if linear_block_graphs is None:
                for i in range(self.n_layers):
                    h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], self.layer_weights[i], self.layer_cfgs[i], state, i)
            else:
                for bi, block in enumerate(linear_block_graphs):
                    block["input"].copy_(h)
                    block["graph"].replay()
                    h = block["output"]
                    full_layer = bi * 4 + 3
                    h = _decode_layer(
                        h,
                        pos_tensor,
                        LAYER_TYPES[full_layer],
                        self.layer_weights[full_layer],
                        self.layer_cfgs[full_layer],
                        state,
                        full_layer,
                    )
            state.seq_len += 1
            h_final = _rms_norm(h, self.outside["model.language_model.norm.weight"])
            logits = self._lm_head_logits(h_final)
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
            if next_id in self.stop_token_ids:
                stopped_reason = "stop_token"
                break

        full_text = tok.decode(ids[0].tolist() + new_ids)
        completion_text_raw = tok.decode(new_ids)
        completion_text = tok.decode(new_ids, skip_special_tokens=True)
        result = {
            "text": full_text,
            "completion_text": completion_text,
            "completion_text_raw": completion_text_raw,
            "new_ids": [int(x) for x in new_ids],
            "timings": {
                "prefill_seconds": prefill_seconds,
                "decode_step_seconds": decode_seconds,
                "decode_tps": (len(decode_seconds) / sum(decode_seconds)) if decode_seconds else None,
                "linear_block_graph_capture_seconds": graph_capture_seconds,
                "linear_block_graph_reused": graph_reused,
                "prefill_warmup_seconds": self.prefill_warmup_seconds,
                "linear_block_graph_prewarm_seconds": self.linear_block_graph_prewarm_seconds,
                "native_fp4_lm_head_enabled": self.native_fp4_lm_head_enabled,
                "native_fp4_lm_head_prepare_seconds": self.native_fp4_lm_head_prepare_seconds,
                "decode_bf16_shadow_release": release_report,
            },
            "stopped_reason": stopped_reason,
            "stop_token_ids": sorted(self.stop_token_ids),
        }
        if top_k > 0:
            result["topk_trace"] = topk_trace
        return result
