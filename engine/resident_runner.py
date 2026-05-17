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
    _resolve_decode_moe_impl,
    _rms_norm,
    _with_inferred_layer_config,
    load_outside_weights,
)
from engine.inference_state import LAYER_TYPES, LynnInferenceState
from engine.loader import load_qwen36_layer
from engine.mtp_sidecar import load_mtp_sidecar, mtp_layer_config, mtp_layer_weights, mtp_logits
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


def _native_cuda_scalar_can_enter_linear_graph() -> bool:
    """Return whether cuda_scalar MoE may be captured inside linear block graphs.

    P32 showed that `cuda_scalar` inside reusable linear-block CUDA graphs can
    silently replay token-0 / `!` loops. A full-attention-only allowlist is safe
    from that specific graph-capture hazard because full-attention layers are
    executed eagerly between captured linear blocks.
    """
    spec = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_LAYERS")
    if not spec:
        return True
    selected: set[int] = set()
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"full", "full_attention"}:
            selected.update(i for i, t in enumerate(LAYER_TYPES) if t == "full_attention")
        elif item in {"linear", "linear_attention"}:
            selected.update(i for i, t in enumerate(LAYER_TYPES) if t == "linear_attention")
        else:
            selected.add(int(item))
    return any(LAYER_TYPES[i] == "linear_attention" for i in selected)


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


class FullAttentionLayerGraphSlot:
    """One fixed-position full-attention layer CUDA graph slot.

    P9J validates the serving prerequisite for this wrapper: `input_buf` can be
    changed between replays and the graph still matches eager output/KV writes.
    The slot is intentionally not wired into `generate()` yet because the
    position/KV ownership ABI must be explicit before promotion.
    """

    def __init__(
        self,
        *,
        layer_idx: int,
        seq_len: int,
        input_buf: torch.Tensor,
        output_buf: torch.Tensor,
        graph: torch.cuda.CUDAGraph,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.seq_len = int(seq_len)
        self.input_buf = input_buf
        self.output_buf = output_buf
        self.graph = graph
        self.replays = 0

    def replay(self, hidden: torch.Tensor) -> torch.Tensor:
        self.input_buf.copy_(hidden)
        self.graph.replay()
        self.replays += 1
        return self.output_buf


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
        self.decode_moe_fn = _resolve_decode_moe_impl(impl)
        self.decode_recurrent_backend = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch")
        linear_graph_enabled = os.environ.get("LYNN_LINEAR_BLOCK_GRAPH", "0") == "1"
        state_update_default = "inplace" if linear_graph_enabled else "assign"
        self.decode_linear_state_update = os.environ.get("LYNN_LINEAR_STATE_UPDATE", state_update_default)
        self.decode_fast_dispatch = os.environ.get("LYNN_DECODE_FAST_DISPATCH", "1") != "0"
        self.shared_expert_gate_up_fused_attached = 0
        self.packed_nvfp4_moe_aliases_attached = 0
        self.packed_decode_backend = os.environ.get("LYNN_PACKED_DECODE_BACKEND", "scalar_bridge")
        self.packed_decode_aliases_attached = 0
        self.packed_decode_native_prepared = 0
        self.packed_decode_aliases_skipped = 0
        self.runtime_warnings: list[str] = []
        self.cfg, self.n_layers = _runtime_config(self.model_dir)
        if os.environ.get("LYNN_PACKED_DECODE", "0") == "1":
            self.runtime_warnings.append(
                "LYNN_PACKED_DECODE=1 is a diagnostic path, not the current R6000 best "
                "profile. P15 measured it regressing full graph path from ~103.48 tok/s "
                "to ~88.15 tok/s. Prefer LYNN_PACKED_DECODE=0 with "
                "LYNN_MOE_IMPL=packed_nvfp4 and LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1."
            )
        if os.environ.get("LYNN_PACKED_SHARED_EXPERT", "0") == "1":
            self.runtime_warnings.append(
                "LYNN_PACKED_SHARED_EXPERT=1 is slower than the BF16 shared expert path "
                "on R6000 P15 profiles; keep it disabled unless explicitly benchmarking."
            )

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
            for warning in self.runtime_warnings:
                print(f"[resident][warning] {warning}", flush=True)
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
            self.layer_cfgs.append(_with_inferred_layer_config(self.cfg, inferred, i))
            if verbose and (i % 5 == 4 or i == self.n_layers - 1):
                print(f"  [resident] L{i:02}: {time.time() - t0:.1f}s", flush=True)
        if impl == "triton":
            self._prepare_triton_moe_layout()
        if os.environ.get("LYNN_SHARED_EXPERT_GATE_UP_FUSED", "1") != "0":
            self._prepare_shared_expert_gate_up_fused()
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
        self.mtp_sidecar_path = os.environ.get("LYNN_MTP_SIDECAR") or None
        self.mtp_shadow_verify_enabled = (
            os.environ.get("LYNN_MTP_SHADOW_VERIFY", os.environ.get("LYNN_MTP_VERIFY", "0")) == "1"
        )
        self.mtp_sidecar_loaded = False
        self.mtp_load_seconds: float | None = None
        self.mtp_sidecar_inventory: dict[str, Any] | None = None
        self.mtp_sidecar: dict[str, torch.Tensor] | None = None
        self.mtp_layer_w: dict[str, torch.Tensor] | None = None
        self.mtp_layer_cfg: dict[str, Any] | None = None
        if self.mtp_sidecar_path:
            t_mtp = time.time()
            self.mtp_sidecar, inventory = load_mtp_sidecar(
                self.mtp_sidecar_path,
                device=self.device,
                dtype=self.dtype,
            )
            for tensor in self.mtp_sidecar.values():
                tensor.requires_grad_(False)
            self.mtp_layer_w = mtp_layer_weights(self.mtp_sidecar)
            self.mtp_layer_cfg = mtp_layer_config(self.cfg, self.mtp_layer_w)
            self.mtp_load_seconds = time.time() - t_mtp
            self.mtp_sidecar_inventory = {
                "path": self.mtp_sidecar_path,
                "metadata": inventory.get("metadata", {}),
                "tensor_count": len(inventory.get("tensors", {})),
                "tensors": inventory.get("tensors", {}),
            }
            self.mtp_sidecar_loaded = True
            if self.verbose:
                mode = "shadow-verify" if self.mtp_shadow_verify_enabled else "loaded-only"
                print(
                    f"[resident] MTP sidecar {mode} tensors={self.mtp_sidecar_inventory['tensor_count']} "
                    f"load={self.mtp_load_seconds:.3f}s",
                    flush=True,
                )
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

    def _prepare_shared_expert_gate_up_fused(self) -> None:
        """Attach BF16 fused shared-expert gate/up weights.

        The shared expert stays BF16 on R6000 because packed shared paths are
        slower. Fusing gate/up keeps the same BF16 math while replacing two
        small GEMM launches with one larger GEMM.
        """
        attached = 0
        for w in self.layer_weights:
            key = "mlp.shared_expert._gate_up_proj.weight"
            if key in w:
                continue
            if "mlp.shared_expert.gate_proj.weight" not in w or "mlp.shared_expert.up_proj.weight" not in w:
                continue
            w[key] = torch.cat(
                [
                    w["mlp.shared_expert.gate_proj.weight"],
                    w["mlp.shared_expert.up_proj.weight"],
                ],
                dim=0,
            ).contiguous()
            attached += 1
        self.shared_expert_gate_up_fused_attached = attached
        if self.verbose:
            print(f"[resident] shared expert fused gate/up attached={attached}", flush=True)

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

    def _mtp_draft_logits(
        self,
        *,
        base_hidden: torch.Tensor,
        current_token_id: int,
        current_pos: int,
    ) -> torch.Tensor:
        """Run the loaded one-layer MTP sidecar for one draft token."""
        if not self.mtp_sidecar_loaded:
            raise RuntimeError("MTP sidecar is not loaded; set LYNN_MTP_SIDECAR")
        assert self.mtp_sidecar is not None
        assert self.mtp_layer_w is not None
        assert self.mtp_layer_cfg is not None
        return mtp_logits(
            sidecar=self.mtp_sidecar,
            mtp_w=self.mtp_layer_w,
            mtp_cfg=self.mtp_layer_cfg,
            embed_weight=self.outside["model.language_model.embed_tokens.weight"],
            lm_head_fn=self._lm_head_logits,
            base_hidden=base_hidden.contiguous(),
            current_token_id=current_token_id,
            current_pos=current_pos,
            device=self.device,
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

    @staticmethod
    def _snapshot_full_attn_layer_kv(
        state: LynnInferenceState,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = state.kv_cache[layer_idx]
        return k.clone(), v.clone()

    @staticmethod
    def _restore_full_attn_layer_kv(
        state: LynnInferenceState,
        layer_idx: int,
        snap: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        k, v = state.kv_cache[layer_idx]
        k.copy_(snap[0])
        v.copy_(snap[1])

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
                h = self._decode_layer_fast(h, pos_tensor, state, i)
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

    def _capture_full_attn_layer_graph_slot(
        self,
        state: LynnInferenceState,
        h_seed: torch.Tensor,
        pos_tensor: torch.Tensor,
        layer_idx: int,
    ) -> FullAttentionLayerGraphSlot:
        """Capture one fixed-position full-attention layer graph slot.

        This is a scaffold for `LYNN_FULL_ATTN_LAYER_GRAPH`. It deliberately
        captures only one layer and restores only that layer's KV cache, so a
        future graph-family path can compose it with existing linear-block
        graphs without paying the full-token snapshot cost.
        """
        if not self.device.startswith("cuda"):
            raise RuntimeError("full-attention layer graph slots require CUDA")
        if LAYER_TYPES[layer_idx] != "full_attention":
            raise ValueError(f"layer {layer_idx} is {LAYER_TYPES[layer_idx]!r}, expected full_attention")

        seq_len = int(state.seq_len)
        input_buf = torch.empty_like(h_seed)
        output_buf = torch.empty_like(h_seed)
        kv_snap = self._snapshot_full_attn_layer_kv(state, layer_idx)

        def graph_body() -> None:
            state.seq_len = seq_len
            out = self._decode_layer_fast(input_buf, pos_tensor, state, layer_idx)
            output_buf.copy_(out)

        input_buf.copy_(h_seed)
        graph_body()
        torch.cuda.synchronize()
        self._restore_full_attn_layer_kv(state, layer_idx, kv_snap)
        input_buf.copy_(h_seed)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_body()
        torch.cuda.synchronize()
        self._restore_full_attn_layer_kv(state, layer_idx, kv_snap)
        return FullAttentionLayerGraphSlot(
            layer_idx=layer_idx,
            seq_len=seq_len,
            input_buf=input_buf,
            output_buf=output_buf,
            graph=graph,
        )

    def _decode_layer_fast(
        self,
        h: torch.Tensor,
        pos_tensor: torch.Tensor,
        state: LynnInferenceState,
        layer_idx: int,
    ) -> torch.Tensor:
        """Decode one layer with runner-fixed dispatch knobs.

        This keeps the math identical to `_decode_layer` while hoisting env and
        import selection out of the per-layer/per-token hot loop.
        """
        if not self.decode_fast_dispatch:
            return _decode_layer(
                h,
                pos_tensor,
                LAYER_TYPES[layer_idx],
                self.layer_weights[layer_idx],
                self.layer_cfgs[layer_idx],
                state,
                layer_idx,
            )
        return _decode_layer(
            h,
            pos_tensor,
            LAYER_TYPES[layer_idx],
            self.layer_weights[layer_idx],
            self.layer_cfgs[layer_idx],
            state,
            layer_idx,
            moe_fn=self.decode_moe_fn,
            recurrent_backend=self.decode_recurrent_backend,
            linear_state_update=self.decode_linear_state_update,
        )

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
                    h = self._decode_layer_fast(h, pos_tensor, state, i)
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
        forced_prefix_text: str | None = None,
        forced_prefix_ids: list[int] | None = None,
        release_decode_shadows_after_prefill: bool = False,
    ) -> dict[str, Any]:
        tok = self.tokenizer
        ids = _encode_prompt(tok, prompt, self.device, use_chat_template=use_chat_template)
        if forced_prefix_text is not None and forced_prefix_ids is not None:
            raise ValueError("pass either forced_prefix_text or forced_prefix_ids, not both")
        if forced_prefix_text is not None:
            forced_prefix_ids = tok(
                forced_prefix_text,
                add_special_tokens=False,
            ).input_ids
        forced_prefix_ids = [int(x) for x in (forced_prefix_ids or [])]
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
        raw_next_id = int(logits[0].argmax().item())
        mtp_shadow_trace: list[dict[str, Any]] = []
        mtp_shadow_step_seconds: list[float] = []
        mtp_shadow_active = self.mtp_sidecar_loaded and self.mtp_shadow_verify_enabled
        mtp_shadow_disabled_reason = None
        if (
            mtp_shadow_active
            and forced_prefix_ids
            and os.environ.get("LYNN_MTP_DISABLE_FOR_FORMAT_GUARD", "0") == "1"
        ):
            mtp_shadow_active = False
            mtp_shadow_disabled_reason = "format_guard_forced_prefix"

        def record_mtp_shadow(
            *,
            step: int,
            current_token_id: int,
            current_pos: int,
            base_hidden: torch.Tensor,
            base_next_id: int,
        ) -> None:
            if not mtp_shadow_active:
                return
            mtp_t0 = time.time()
            draft_logits = self._mtp_draft_logits(
                base_hidden=base_hidden,
                current_token_id=current_token_id,
                current_pos=current_pos,
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - mtp_t0
            draft_id = int(draft_logits[0].argmax().item())
            base_score = float(draft_logits[0, int(base_next_id)].float().item())
            shadow_topk = max(2, int(os.environ.get("LYNN_MTP_SHADOW_TOPK", "2")))
            draft_topk = _logit_topk(draft_logits, shadow_topk)
            draft_top2 = {
                "ids": draft_topk["ids"][:2],
                "values": draft_topk["values"][:2],
                "top1_margin": draft_topk["top1_margin"],
            }
            mtp_shadow_step_seconds.append(elapsed)
            mtp_shadow_trace.append(
                {
                    "step": int(step),
                    "current_pos": int(current_pos),
                    "current_token_id": int(current_token_id),
                    "current_token_text": tok.decode([int(current_token_id)]),
                    "base_next_id": int(base_next_id),
                    "base_next_text": tok.decode([int(base_next_id)]),
                    "draft_id": draft_id,
                    "draft_text": tok.decode([draft_id]),
                    "accepted": draft_id == int(base_next_id),
                    "draft_seconds": elapsed,
                    "base_score_in_draft_logits": base_score,
                    "draft_top2": draft_top2,
                    "draft_topk": draft_topk,
                }
            )

        record_mtp_shadow(
            step=0,
            current_token_id=int(ids[0, -1].item()),
            current_pos=T - 1,
            base_hidden=h[:, -1:, :],
            base_next_id=raw_next_id,
        )
        forced_trace = []
        if forced_prefix_ids:
            forced_id = forced_prefix_ids[0]
            forced_trace.append(
                {
                    "index": 0,
                    "raw_id": raw_next_id,
                    "raw_text": tok.decode([raw_next_id]),
                    "forced_id": forced_id,
                    "forced_text": tok.decode([forced_id]),
                    "raw_matched": raw_next_id == forced_id,
                }
            )
            next_id = forced_id
        else:
            next_id = raw_next_id
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
        full_token_graph_slot_enabled = (
            os.environ.get("LYNN_FULL_TOKEN_GRAPH_SLOT", "0") == "1"
            and self.device.startswith("cuda")
        )
        if (
            full_token_graph_slot_enabled
            and os.environ.get("LYNN_ROUTER_TOPK_SORTED", "0") != "1"
            and os.environ.get("LYNN_ALLOW_UNSORTED_FULL_TOKEN_GRAPH_SLOT", "0") != "1"
        ):
            raise RuntimeError(
                "LYNN_FULL_TOKEN_GRAPH_SLOT=1 requires LYNN_ROUTER_TOPK_SORTED=1. "
                "P35 showed sorted router top-k restores full-token graph-slot "
                "parity; unsorted top-k is valid for the stable linear-block "
                "graph path but not for graph-slot serving. Set "
                "LYNN_ALLOW_UNSORTED_FULL_TOKEN_GRAPH_SLOT=1 only for diagnostics."
            )
        full_token_graph_slot_capture_seconds: list[float] = []
        full_token_graph_slot_replay_seconds: list[float] = []
        if (
            os.environ.get("LYNN_LINEAR_BLOCK_GRAPH", "0") == "1"
            and self.device.startswith("cuda")
            and max_new > 1
            and not full_token_graph_slot_enabled
        ):
            if (
                os.environ.get("LYNN_NATIVE_ACTIVE_MOE_BACKEND") == "cuda_scalar"
                and _native_cuda_scalar_can_enter_linear_graph()
                and os.environ.get("LYNN_ALLOW_UNSAFE_CUDA_SCALAR_GRAPH", "0") != "1"
            ):
                raise RuntimeError(
                    "LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar is not graph-safe with "
                    "LYNN_LINEAR_BLOCK_GRAPH=1 yet. P32 showed this combination can "
                    "silently replay token-0 / '!' loops. Set "
                    "LYNN_ALLOW_UNSAFE_CUDA_SCALAR_GRAPH=1 only for explicit diagnostics."
                )
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
            decode_input_id = int(next_id)
            new_token_tensor.fill_(next_id)
            pos_id = state.seq_len
            pos_tensor.fill_(pos_id)
            if full_token_graph_slot_enabled:
                capture_t0 = time.time()
                slot = self._capture_full_token_graph_slot(state, next_id)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                full_token_graph_slot_capture_seconds.append(time.time() - capture_t0)
                replay_t0 = time.time()
                logits = slot.replay(next_id).clone()
                state.seq_len += 1
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                full_token_graph_slot_replay_seconds.append(time.time() - replay_t0)
            else:
                h = F.embedding(new_token_tensor, self.outside["model.language_model.embed_tokens.weight"])
            if full_token_graph_slot_enabled:
                pass
            elif linear_block_graphs is None:
                for i in range(self.n_layers):
                    h = self._decode_layer_fast(h, pos_tensor, state, i)
            elif linear_block_graphs is not None:
                for bi, block in enumerate(linear_block_graphs):
                    block["input"].copy_(h)
                    block["graph"].replay()
                    h = block["output"]
                    full_layer = bi * 4 + 3
                    h = self._decode_layer_fast(h, pos_tensor, state, full_layer)
            if not full_token_graph_slot_enabled:
                state.seq_len += 1
                h_final = _rms_norm(h, self.outside["model.language_model.norm.weight"])
                logits = self._lm_head_logits(h_final)
            raw_next_id = int(logits[0].argmax().item())
            if not full_token_graph_slot_enabled:
                record_mtp_shadow(
                    step=step,
                    current_token_id=decode_input_id,
                    current_pos=pos_id,
                    base_hidden=h,
                    base_next_id=raw_next_id,
                )
            if step < len(forced_prefix_ids):
                forced_id = forced_prefix_ids[step]
                forced_trace.append(
                    {
                        "index": step,
                        "raw_id": raw_next_id,
                        "raw_text": tok.decode([raw_next_id]),
                        "forced_id": forced_id,
                        "forced_text": tok.decode([forced_id]),
                        "raw_matched": raw_next_id == forced_id,
                    }
                )
                next_id = forced_id
            else:
                next_id = raw_next_id
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
        mtp_shadow_accepted = sum(1 for row in mtp_shadow_trace if row["accepted"])
        mtp_shadow_summary = {
            "loaded": self.mtp_sidecar_loaded,
            "enabled": mtp_shadow_active,
            "disabled_reason": mtp_shadow_disabled_reason,
            "sidecar_file": self.mtp_sidecar_path,
            "load_seconds": self.mtp_load_seconds,
            "events": len(mtp_shadow_trace),
            "accepted": mtp_shadow_accepted,
            "accept_rate": (
                mtp_shadow_accepted / len(mtp_shadow_trace)
                if mtp_shadow_trace
                else None
            ),
            "max_one_token_speculative_multiplier": (
                (len(mtp_shadow_trace) + mtp_shadow_accepted) / len(mtp_shadow_trace)
                if mtp_shadow_trace
                else None
            ),
            "draft_step_seconds": mtp_shadow_step_seconds,
            "draft_tps": (
                len(mtp_shadow_step_seconds) / sum(mtp_shadow_step_seconds)
                if mtp_shadow_step_seconds
                else None
            ),
            "trace": mtp_shadow_trace,
        }
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
                "full_token_graph_slot_enabled": full_token_graph_slot_enabled,
                "full_token_graph_slot_capture_seconds": full_token_graph_slot_capture_seconds,
                "full_token_graph_slot_replay_seconds": full_token_graph_slot_replay_seconds,
                "full_token_graph_slot_replay_tps": (
                    len(full_token_graph_slot_replay_seconds) / sum(full_token_graph_slot_replay_seconds)
                    if full_token_graph_slot_replay_seconds
                    else None
                ),
                "mtp_shadow": {
                    key: value
                    for key, value in mtp_shadow_summary.items()
                    if key != "trace"
                },
            },
            "mtp_shadow": mtp_shadow_summary,
            "stopped_reason": stopped_reason,
            "stop_token_ids": sorted(self.stop_token_ids),
        }
        if top_k > 0:
            result["topk_trace"] = topk_trace
        if forced_prefix_ids:
            result["forced_prefix"] = {
                "text": forced_prefix_text,
                "ids": forced_prefix_ids,
                "trace": forced_trace,
                "raw_all_matched": all(row["raw_matched"] for row in forced_trace),
            }
        return result
