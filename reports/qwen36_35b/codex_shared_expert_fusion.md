Reading additional input from stdin...
OpenAI Codex v0.136.0-alpha.2
--------
workdir: /Users/lynn/Downloads/Lynn/lynn-engine
model: gpt-5.5
provider: openai
approval: never
sandbox: read-only
reasoning effort: high
reasoning summaries: none
session id: 019e8bab-6188-7543-a1ba-98b76e91a1e7
--------
user
Lynn NVFP4 Qwen3.6-35B-A3B decode, kernel-launch-overhead campaign. We just landed RMSNorm fusion (+8.7% e2e by replacing ~6-8 eager launches/norm with 1 Triton launch). NEXT TARGET: the SHARED EXPERT (~4-5 separate BF16 F.linear launches per MoE layer x ~40 layers = ~160-200 launches/token).

Read: engine/moe_packed_nvfp4.py — shared-expert block in _moe_forward_decode_packed_nvfp4_fixed_triton (~lines 823-835), _finalize_shared_expert_output (~240-248), _apply_shared_expert_gate (~217-230). It does gate_up (fused F.linear OR gate_proj+up_proj), SwiGLU silu(gate)*up, down_proj F.linear, optional sigmoid(shared_expert_gate) scale, residual add.

DELIVER (analysis + CODE, do NOT run): (1) a fused Triton kernel (or minimal 1-2 kernels) for M=1 decode shared-expert doing gate_up->SwiGLU->down (+optional gate+add) in minimal launches, BF16 in/out, matching the F.linear math (token-coherent); (2) exact wiring + the precise edits gated behind LYNN_SHARED_EXPERT_FUSED=1, coexisting with BF16 path; (3) token-exactness risks + keeping cos~1. Cite exact functions/lines, specific to THIS codebase.
codex
I’ll read the MoE packed path and the shared-expert helpers first, then I’ll draft the smallest gated code change that fits the existing Triton style. I’m constrained to read-only filesystem here, so I can provide an exact patch-style implementation but cannot actually edit files.
exec
/bin/zsh -lc "sed -n '1,320p' engine/moe_packed_nvfp4.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
"""Packed NVFP4 MoE decode path."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from engine.nvfp4_runtime import _quantize_activation_to_fp4, dual_scalar_bridge
from triton_kernels.nvfp4_moe import (
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_down_weighted_sum_effective_scale,
    nvfp4_grouped_down_weighted_sum_prepared,
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_gate_up_silu_fast_decode,
    nvfp4_grouped_gate_up_silu_fast_decode_effective_scale,
    nvfp4_grouped_gate_up_silu_fast_decode_prepared,
)
from triton_kernels.shared_expert_gate import (
    HAS_TRITON as HAS_SHARED_EXPERT_GATE_TRITON,
    add_shared_expert_gate_from_scalar_triton,
    apply_shared_expert_gate_triton,
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def _env_first(names: tuple[str, ...]) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw != "":
            return raw
    return None


def _w4a8_fake_quant_mode() -> str:
    mode = os.environ.get("LYNN_W4A8_FAKE_QUANT_ACTIVE", "off").lower()
    if mode in {"0", "false", "no", "off", ""}:
        return "off"
    if mode not in {"gateup", "full"}:
        raise ValueError("LYNN_W4A8_FAKE_QUANT_ACTIVE must be off, gateup, or full")
    return mode


def _use_moe_effective_scale(w: dict) -> bool:
    return (
        _env_bool("LYNN_MOE_EFFECTIVE_SCALE", False)
        and "mlp.experts._gate_up_effective_scale" in w
        and "mlp.experts._down_effective_scale" in w
    )


def _fake_quant_fp8_activation(x: torch.Tensor) -> torch.Tensor:
    """Research-only FP8 activation round-trip for W4A8 quality gates.

    This is intentionally controlled by `LYNN_W4A8_FAKE_QUANT_ACTIVE` and never
    used by default. It mirrors P104's best variant: E4M3, per-16 scaling.
    """
    fmt = os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT", "e4m3").lower()
    granularity = os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY", "per16").lower()
    if fmt == "e4m3":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is required for W4A8 fake quant")
        fp8_dtype = torch.float8_e4m3fn
    elif fmt == "e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError("torch.float8_e5m2 is required for W4A8 fake quant")
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError("LYNN_W4A8_FAKE_QUANT_FORMAT must be e4m3 or e5m2")

    max_fp8 = float(torch.finfo(fp8_dtype).max)
    x32 = x.float()
    if granularity == "tensor":
        scale = (x32.abs().amax() / max_fp8).clamp_min(1e-8)
        return ((x32 / scale).to(fp8_dtype).float() * scale).to(x.dtype)
    if granularity == "row":
        x2 = x32.view(1, -1) if x32.ndim == 1 else x32
        scale = (x2.abs().amax(dim=-1, keepdim=True) / max_fp8).clamp_min(1e-8)
        y = ((x2 / scale).to(fp8_dtype).float() * scale).to(x.dtype)
        return y.view_as(x) if x32.ndim == 1 else y
    if granularity == "per16":
        if x32.shape[-1] % 16 != 0:
            raise ValueError(f"W4A8 per16 fake quant requires last dim divisible by 16, got {tuple(x.shape)}")
        shape = x32.shape
        grouped = x32.reshape(-1, shape[-1] // 16, 16)
        scale = (grouped.abs().amax(dim=-1, keepdim=True) / max_fp8).clamp_min(1e-8)
        return ((grouped / scale).to(fp8_dtype).float() * scale).reshape(shape).to(x.dtype)
    raise ValueError("LYNN_W4A8_FAKE_QUANT_GRANULARITY must be tensor, row, or per16")


def _topk_limit_from_env(top_k: int) -> int:
    raw = _env_first(("LYNN_MOE_TOPK_LIMIT", "LYNN_MOE_PROFILE_TOPK_LIMIT"))
    if raw is None:
        return top_k
    limit = int(raw)
    if not (1 <= limit <= top_k):
        raise ValueError(f"MoE top-k limit must be in [1, {top_k}], got {limit}")
    return limit


def _router_topk(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    sorted: bool,
    scratch_owner: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run router top-k, optionally reusing caller-owned output buffers.

    P163 showed `torch.topk(..., out=...)` is bit-exact for the decode shape and
    saves a small but measurable boundary cost.  Keep it opt-in because the
    scratch tensors are stored on the mutable per-layer weight dict.
    """
    if not _env_bool("LYNN_ROUTER_TOPK_OUT_BUFFER", False):
        return torch.topk(router_logits, top_k, dim=-1, sorted=sorted)
    if router_logits.ndim != 2 or router_logits.shape[0] != 1:
        return torch.topk(router_logits, top_k, dim=-1, sorted=sorted)
    values_key = "mlp.gate._topk_values_scratch"
    indices_key = "mlp.gate._topk_indices_scratch"
    values = scratch_owner.get(values_key)
    indices = scratch_owner.get(indices_key)
    expected_shape = (1, top_k)
    if (
        values is None
        or tuple(values.shape) != expected_shape
        or values.device != router_logits.device
        or values.dtype != router_logits.dtype
    ):
        values = torch.empty(expected_shape, device=router_logits.device, dtype=router_logits.dtype)
        scratch_owner[values_key] = values
    if (
        indices is None
        or tuple(indices.shape) != expected_shape
        or indices.device != router_logits.device
        or indices.dtype != torch.long
    ):
        indices = torch.empty(expected_shape, device=router_logits.device, dtype=torch.long)
        scratch_owner[indices_key] = indices
    torch.topk(router_logits, top_k, dim=-1, sorted=sorted, out=(values, indices))
    return values, indices


def _router_linear(h_flat: torch.Tensor, w: dict) -> torch.Tensor:
    """Run the MoE router projection, optionally with caller-owned output.

    P177 showed `torch.mm(hidden, gate_weight.t(), out=logits_scratch)` is
    bit-exact for the single-token decode router and removes a small allocation
    boundary. The transposed weight and scratch are attached at load time by the
    resident runner.
    """
    if not _env_bool("LYNN_ROUTER_LINEAR_OUT_BUFFER", False):
        return F.linear(h_flat, w["mlp.gate.weight"])
    weight_t = w.get("mlp.gate.weight_t")
    logits = w.get("mlp.gate._logits_scratch")
    if (
        weight_t is None
        or logits is None
        or h_flat.ndim != 2
        or h_flat.shape[0] != 1
        or logits.shape != (1, w["mlp.gate.weight"].shape[0])
        or logits.device != h_flat.device
        or logits.dtype != h_flat.dtype
    ):
        return F.linear(h_flat, w["mlp.gate.weight"])
    torch.mm(h_flat, weight_t, out=logits)
    return logits


def _router_softmax(
    routing_logits: torch.Tensor,
    *,
    scratch_owner: dict,
) -> torch.Tensor:
    """Run router softmax, optionally reusing a caller-owned float32 buffer."""
    if not _env_bool("LYNN_ROUTER_SOFTMAX_OUT_BUFFER", False):
        return F.softmax(routing_logits, dim=-1, dtype=torch.float32)[0].contiguous()
    if routing_logits.ndim != 2 or routing_logits.shape[0] != 1:
        return F.softmax(routing_logits, dim=-1, dtype=torch.float32)[0].contiguous()
    values_key = "mlp.gate._softmax_values_scratch"
    expected_shape = tuple(routing_logits.shape)
    values = scratch_owner.get(values_key)
    if (
        values is None
        or tuple(values.shape) != expected_shape
        or values.device != routing_logits.device
        or values.dtype != torch.float32
    ):
        values = torch.empty(expected_shape, device=routing_logits.device, dtype=torch.float32)
        scratch_owner[values_key] = values
    torch.softmax(routing_logits, dim=-1, dtype=torch.float32, out=values)
    return values[0]


def _skip_shared_from_env() -> bool:
    raw = _env_first(("LYNN_MOE_SKIP_SHARED", "LYNN_MOE_PROFILE_SKIP_SHARED"))
    if raw is None:
        return False
    return raw.lower() not in {"0", "false", "no", "off"}


def _apply_shared_expert_gate(h_flat: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
    if "mlp.shared_expert_gate.weight" not in w:
        return shared
    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
    if backend == "torch":
        return shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    if backend == "torch_inplace":
        shared.mul_(torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"])))
        return shared
    if backend == "triton":
        if not HAS_SHARED_EXPERT_GATE_TRITON:
            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=triton requires Triton")
        return apply_shared_expert_gate_triton(shared, h_flat, w["mlp.shared_expert_gate.weight"])
    raise ValueError("LYNN_SHARED_EXPERT_GATE_BACKEND must be 'torch', 'torch_inplace', or 'triton', got " f"{backend!r}")


def _add_shared_expert_output(moe_out: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    if _env_bool("LYNN_MOE_ADD_SHARED_INPLACE", False):
        moe_out.add_(shared)
        return moe_out
    return moe_out + shared


def _finalize_shared_expert_output(h_flat: torch.Tensor, moe_out: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
    if backend == "torch_scalar_add_triton" and "mlp.shared_expert_gate.weight" in w:
        if not HAS_SHARED_EXPERT_GATE_TRITON:
            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=torch_scalar_add_triton requires Triton")
        gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
        return add_shared_expert_gate_from_scalar_triton(moe_out, shared, gate)
    shared = _apply_shared_expert_gate(h_flat, shared, w)
    return _add_shared_expert_output(moe_out, shared)


def _layer_selected_for_native_cuda(cfg: dict) -> bool:
    spec = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_LAYERS")
    if not spec:
        return True
    layer_idx = cfg.get("layer_idx")
    if layer_idx is None:
        return False
    layer_idx = int(layer_idx)
    from engine.inference_state import LAYER_TYPES

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
    return layer_idx in selected


def _active_moe_native_cuda_scalar(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Opt-in native CUDA scalar contract path.

    This is intentionally slower than the Triton default today. It exists so
    the real grouped native-FP4 kernel can replace the scalar inner loops behind
    the same runtime contract.
    """
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    inter = ext.gate_up_silu_scalar(
        hidden,
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
    )
    return ext.down_weighted_sum_scalar(
        inter,
        expert_ids,
        routing_weights,
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
    )


def _gate_up_native_cuda_tile_inter(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """P55 opt-in tile-inter CUDA scalar gate/up projection."""
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.gate_up_silu_tile_inter_scalar(
        hidden,
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],

exec
/bin/zsh -lc "sed -n '760,880p' engine/moe_packed_nvfp4.py" in /Users/lynn/Downloads/Lynn/lynn-engine
exec
/bin/zsh -lc 'rg -n "shared_expert|_finalize_shared|_apply_shared|triton|@triton|gate_up|NVFP4|bf16|_moe_forward_decode" engine/moe_packed_nvfp4.py' in /Users/lynn/Downloads/Lynn/lynn-engine
 exited 127 in 0ms:
zsh:1: command not found: rg

 succeeded in 0ms:
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )
    else:
        raise ValueError(
            "LYNN_NATIVE_GATEUP_BACKEND must be 'triton', 'triton_fast_decode', 'cuda_tile_inter', "
            "or 'split16_fp4', got "
            f"{gateup_backend!r}"
        )
    if w4a8_mode == "full":
        inter = _fake_quant_fp8_activation(inter)
    down_backend = os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton")
    if down_backend == "cuda_tile" and _layer_selected_for_native_cuda(cfg):
        moe_out = _down_weighted_sum_native_cuda_tile(inter, expert_ids, routing_weights, w).reshape_as(h_flat)
    elif prepared_triton:
        if down_backend != "triton":
            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_DOWN_BACKEND=triton")
        out_scratch = w.get("mlp.experts._active_out_scratch")
        if out_scratch is None:
            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
        moe_out = nvfp4_grouped_down_weighted_sum_prepared(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            out_scratch,
            scale_effective=_use_moe_effective_scale(w),
        ).reshape_as(h_flat)
    elif down_backend == "triton":
        down_fn = (
            nvfp4_grouped_down_weighted_sum_effective_scale
            if _use_moe_effective_scale(w)
            else nvfp4_grouped_down_weighted_sum
        )
        moe_out = down_fn(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
            block_inter=512,
            num_warps=8,
        ).reshape_as(h_flat)
    elif down_backend == "cuda_tile":
        moe_out = nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        ).reshape_as(h_flat)
    else:
        raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")

    if _skip_shared_from_env():
        return moe_out.to(h.dtype).reshape_as(h)

    if "mlp.shared_expert.gate_proj.weight" in w:
        if "mlp.shared_expert._gate_up_proj.weight" in w:
            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
        else:
            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
    return moe_out.to(h.dtype).reshape_as(h)


def moe_forward_verify_smallm_nvfp4(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """Small-M grouped MoE for the MTP spec-verify block (M = K_draft+1).

    Replaces the per-position T=1 MoE loop: routes each of the M positions, then
    for each UNIQUE active expert runs its gate_up+down ONCE over its gathered
    rows (a small-M tile) instead of once per position; the dense shared expert
    is one batched pass over all M rows. Routed dequant uses the canonical
    ``_dequant_nvfp4_slot`` (W4A16 BF16 numerics) so each row matches the T=1
    packed path -> token-exact intent. Eager only (host syncs in grouping).
    Gated by LYNN_MTP_VERIFY_SMALLM=1.
    """
    h_flat = h.reshape(-1, h.shape[-1])
    M, hidden = h_flat.shape
    top_k = int(cfg["num_experts_per_tok"])
    dev = h_flat.device
    out_dtype = h_flat.dtype

    eids = torch.empty(M, top_k, dtype=torch.long, device=dev)
    wts = torch.empty(M, top_k, dtype=torch.float32, device=dev)
    sorted_topk = _env_bool("LYNN_ROUTER_TOPK_SORTED", False)
    for i in range(M):
        rl = _router_linear(h_flat[i : i + 1], w)
        rw, ei = _router_topk(rl, top_k, sorted=sorted_topk, scratch_owner=w)
        rw = _router_softmax(rw, scratch_owner=w)
        eids[i] = ei[0].to(torch.long)
        wts[i] = rw.reshape(-1).to(torch.float32)

    gu_packed = w["mlp.experts._gate_up_packed"]
    gu_scale = w["mlp.experts._gate_up_scale"]
    gu_g = w["mlp.experts._gate_up_global_scale"]
    d_packed = w["mlp.experts._down_packed"]
    d_scale = w["mlp.experts._down_scale"]
    d_g = w["mlp.experts._down_global_scale"]

    x32 = h_flat.to(torch.float32)
    moe_out = torch.zeros(M, hidden, device=dev, dtype=torch.float32)
    for e in torch.unique(eids).tolist():
        sel = eids == e  # [M, top_k]
        rows = sel.any(dim=1).nonzero(as_tuple=False).flatten()
        rw_e = (wts * sel.to(torch.float32)).sum(dim=1)[rows]  # [m_e]
        xr = x32[rows]  # [m_e, hidden]
        gu = _dequant_nvfp4_slot(gu_packed[e], gu_scale[e], gu_g, dev).to(torch.float32)  # [2*inter, hidden]
        inter = xr @ gu.t()  # [m_e, 2*inter]

codex
The shared path is exactly where you described: after active MoE down, it falls back to eager `F.linear` calls and then `_finalize_shared_expert_output`. I’m going to inspect nearby Triton kernel modules now so the proposed kernel matches the repo’s wrapper conventions and launch style.
exec
/bin/zsh -lc "nl -ba engine/moe_packed_nvfp4.py | sed -n '200,260p'" in /Users/lynn/Downloads/Lynn/lynn-engine
exec
/bin/zsh -lc "nl -ba engine/moe_packed_nvfp4.py | sed -n '780,850p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
   200	        or tuple(values.shape) != expected_shape
   201	        or values.device != routing_logits.device
   202	        or values.dtype != torch.float32
   203	    ):
   204	        values = torch.empty(expected_shape, device=routing_logits.device, dtype=torch.float32)
   205	        scratch_owner[values_key] = values
   206	    torch.softmax(routing_logits, dim=-1, dtype=torch.float32, out=values)
   207	    return values[0]
   208	
   209	
   210	def _skip_shared_from_env() -> bool:
   211	    raw = _env_first(("LYNN_MOE_SKIP_SHARED", "LYNN_MOE_PROFILE_SKIP_SHARED"))
   212	    if raw is None:
   213	        return False
   214	    return raw.lower() not in {"0", "false", "no", "off"}
   215	
   216	
   217	def _apply_shared_expert_gate(h_flat: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
   218	    if "mlp.shared_expert_gate.weight" not in w:
   219	        return shared
   220	    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
   221	    if backend == "torch":
   222	        return shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
   223	    if backend == "torch_inplace":
   224	        shared.mul_(torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"])))
   225	        return shared
   226	    if backend == "triton":
   227	        if not HAS_SHARED_EXPERT_GATE_TRITON:
   228	            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=triton requires Triton")
   229	        return apply_shared_expert_gate_triton(shared, h_flat, w["mlp.shared_expert_gate.weight"])
   230	    raise ValueError("LYNN_SHARED_EXPERT_GATE_BACKEND must be 'torch', 'torch_inplace', or 'triton', got " f"{backend!r}")
   231	
   232	
   233	def _add_shared_expert_output(moe_out: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
   234	    if _env_bool("LYNN_MOE_ADD_SHARED_INPLACE", False):
   235	        moe_out.add_(shared)
   236	        return moe_out
   237	    return moe_out + shared
   238	
   239	
   240	def _finalize_shared_expert_output(h_flat: torch.Tensor, moe_out: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
   241	    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
   242	    if backend == "torch_scalar_add_triton" and "mlp.shared_expert_gate.weight" in w:
   243	        if not HAS_SHARED_EXPERT_GATE_TRITON:
   244	            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=torch_scalar_add_triton requires Triton")
   245	        gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
   246	        return add_shared_expert_gate_from_scalar_triton(moe_out, shared, gate)
   247	    shared = _apply_shared_expert_gate(h_flat, shared, w)
   248	    return _add_shared_expert_output(moe_out, shared)
   249	
   250	
   251	def _layer_selected_for_native_cuda(cfg: dict) -> bool:
   252	    spec = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_LAYERS")
   253	    if not spec:
   254	        return True
   255	    layer_idx = cfg.get("layer_idx")
   256	    if layer_idx is None:
   257	        return False
   258	    layer_idx = int(layer_idx)
   259	    from engine.inference_state import LAYER_TYPES
   260	

 succeeded in 0ms:
   780	            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
   781	        moe_out = nvfp4_grouped_down_weighted_sum_prepared(
   782	            inter,
   783	            expert_ids,
   784	            routing_weights,
   785	            w["mlp.experts._down_packed"],
   786	            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
   787	            w["mlp.experts._down_global_scale"],
   788	            out_scratch,
   789	            scale_effective=_use_moe_effective_scale(w),
   790	        ).reshape_as(h_flat)
   791	    elif down_backend == "triton":
   792	        down_fn = (
   793	            nvfp4_grouped_down_weighted_sum_effective_scale
   794	            if _use_moe_effective_scale(w)
   795	            else nvfp4_grouped_down_weighted_sum
   796	        )
   797	        moe_out = down_fn(
   798	            inter,
   799	            expert_ids,
   800	            routing_weights,
   801	            w["mlp.experts._down_packed"],
   802	            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
   803	            w["mlp.experts._down_global_scale"],
   804	            block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
   805	            block_inter=512,
   806	            num_warps=8,
   807	        ).reshape_as(h_flat)
   808	    elif down_backend == "cuda_tile":
   809	        moe_out = nvfp4_grouped_down_weighted_sum(
   810	            inter,
   811	            expert_ids,
   812	            routing_weights,
   813	            w["mlp.experts._down_packed"],
   814	            w["mlp.experts._down_scale"],
   815	            w["mlp.experts._down_global_scale"],
   816	            block_hidden=8,
   817	            block_inter=512,
   818	            num_warps=8,
   819	        ).reshape_as(h_flat)
   820	    else:
   821	        raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")
   822	
   823	    if _skip_shared_from_env():
   824	        return moe_out.to(h.dtype).reshape_as(h)
   825	
   826	    if "mlp.shared_expert.gate_proj.weight" in w:
   827	        if "mlp.shared_expert._gate_up_proj.weight" in w:
   828	            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
   829	            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
   830	        else:
   831	            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
   832	            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
   833	        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
   834	        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
   835	    return moe_out.to(h.dtype).reshape_as(h)
   836	
   837	
   838	def moe_forward_verify_smallm_nvfp4(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
   839	    """Small-M grouped MoE for the MTP spec-verify block (M = K_draft+1).
   840	
   841	    Replaces the per-position T=1 MoE loop: routes each of the M positions, then
   842	    for each UNIQUE active expert runs its gate_up+down ONCE over its gathered
   843	    rows (a small-M tile) instead of once per position; the dense shared expert
   844	    is one batched pass over all M rows. Routed dequant uses the canonical
   845	    ``_dequant_nvfp4_slot`` (W4A16 BF16 numerics) so each row matches the T=1
   846	    packed path -> token-exact intent. Eager only (host syncs in grouping).
   847	    Gated by LYNN_MTP_VERIFY_SMALLM=1.
   848	    """
   849	    h_flat = h.reshape(-1, h.shape[-1])
   850	    M, hidden = h_flat.shape

exec
/bin/zsh -lc 'ls triton_kernels' in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
__init__.py
attention.py
gated_delta.py
linear_conv.py
moe.py
moe_expert_ffn.py
nvfp4_linear.py
nvfp4_moe.py
qk_norm_rope.py
rmsnorm.py
rmsnorm_gated.py
rope.py
router_topk.py
rowwise_attention.py
rowwise_linear.py
shared_expert_gate.py
spark_fp8_gate_up_fused.py
spark_fp8_moe_expert_fused.py

exec
/bin/zsh -lc "sed -n '1,260p' triton_kernels/shared_expert_gate.py" in /Users/lynn/Downloads/Lynn/lynn-engine
exec
/bin/zsh -lc "sed -n '1,260p' triton_kernels/nvfp4_moe.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
"""Triton helpers for the BF16 shared-expert scalar gate."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - import guard for CPU-only dev envs
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for shared expert gate kernels")


if HAS_TRITON:

    @triton.jit
    def _shared_expert_gate_apply_kernel(
        shared_ptr,
        hidden_ptr,
        gate_weight_ptr,
        out_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dot = tl.sum(hidden * weight, axis=0)
        gate = tl.sigmoid(dot)
        shared = tl.load(shared_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, (shared * gate).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _shared_expert_gate_add_from_scalar_kernel(
        moe_ptr,
        shared_ptr,
        gate_ptr,
        out_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        gate = tl.load(gate_ptr).to(tl.float32)
        moe = tl.load(moe_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        shared = tl.load(shared_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, (moe + shared * gate).to(tl.bfloat16), mask=mask)


def apply_shared_expert_gate_triton(
    shared: torch.Tensor,
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply the Qwen shared-expert scalar gate in one Triton kernel.

    This preserves the BF16 shared expert path but replaces the tiny
    `F.linear -> sigmoid -> broadcast multiply` chain with one graph-capturable
    kernel for batch=1 decode.
    """
    _require_triton()
    if shared.ndim != 2 or shared.shape[0] != 1:
        raise ValueError(f"shared must be [1, D], got {tuple(shared.shape)}")
    if hidden.ndim != 2 or hidden.shape[0] != 1 or hidden.shape[1] != shared.shape[1]:
        raise ValueError(f"hidden must be [1, {shared.shape[1]}], got {tuple(hidden.shape)}")
    if gate_weight.ndim != 2 or gate_weight.shape[0] != 1 or gate_weight.shape[1] != shared.shape[1]:
        raise ValueError(f"gate_weight must be [1, {shared.shape[1]}], got {tuple(gate_weight.shape)}")
    d = int(shared.shape[1])
    block_d = 1 << (d - 1).bit_length()
    out = torch.empty_like(shared)
    _shared_expert_gate_apply_kernel[(1,)](
        shared.contiguous(),
        hidden.contiguous(),
        gate_weight.contiguous(),
        out,
        D=d,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out


def add_shared_expert_gate_from_scalar_triton(
    moe_out: torch.Tensor,
    shared: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse `moe_out + shared * gate` when the scalar gate was computed by Torch.

    This is a research backend for decode-only batch=1.  Unlike
    `apply_shared_expert_gate_triton`, it intentionally leaves the scalar gate
    reduction on the default Torch path and only removes the final broadcast
    multiply/add boundary.
    """
    _require_triton()
    if moe_out.ndim != 2 or moe_out.shape[0] != 1:
        raise ValueError(f"moe_out must be [1, D], got {tuple(moe_out.shape)}")
    if shared.shape != moe_out.shape:
        raise ValueError(f"shared must match moe_out, got {tuple(shared.shape)} vs {tuple(moe_out.shape)}")
    if gate.numel() != 1:
        raise ValueError(f"gate must contain one scalar for batch=1 decode, got {tuple(gate.shape)}")
    d = int(moe_out.shape[1])
    block_d = 1 << (d - 1).bit_length()
    out = torch.empty_like(moe_out)
    _shared_expert_gate_add_from_scalar_kernel[(1,)](
        moe_out.contiguous(),
        shared.contiguous(),
        gate.contiguous(),
        out,
        D=d,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out


__all__ = ["HAS_TRITON", "apply_shared_expert_gate_triton", "add_shared_expert_gate_from_scalar_triton"]

 succeeded in 0ms:
"""Packed NVFP4 MoE kernels for Lynn variable-expert decode."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for packed NVFP4 MoE kernels")


if HAS_TRITON:

    @triton.jit
    def _e2m1_from_nibble(nibble):
        mag = nibble & 0x07
        sign = (nibble & 0x08) != 0
        val = tl.where(
            mag == 0,
            0.0,
            tl.where(
                mag == 1,
                0.5,
                tl.where(
                    mag == 2,
                    1.0,
                    tl.where(
                        mag == 3,
                        1.5,
                        tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))),
                    ),
                ),
            ),
        )
        return tl.where(sign, -val, val)

    @triton.jit
    def _e2m1_from_nibble_fast(nibble):
        """Same E2M1 table with a shallower expression tree.

        This keeps the exact compressed-tensors value table:
        [0, .5, 1, 1.5, 2, 3, 4, 6].
        """
        mag = nibble & 0x07
        sign = (nibble & 0x08) != 0
        mag_f = mag.to(tl.float32)
        val = tl.where(
            mag <= 4,
            mag_f * 0.5,
            tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
        )
        return tl.where(sign, -val, val)

    @triton.jit
    def _grouped_gate_up_silu_kernel(
        x_ptr,
        expert_ids_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        inter_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block_i = tl.program_id(1)
        expert = tl.load(expert_ids_ptr + slot)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = tl.arange(0, BLOCK_HIDDEN)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        gate_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)

        for h0 in range(0, HIDDEN, BLOCK_HIDDEN):
            cols = h0 + h_offsets
            col_mask = cols < HIDDEN
            packed_cols = cols // 2
            scale_cols = cols // 16
            x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

            gate_rows = inter_offsets
            up_rows = INTERMEDIATE + inter_offsets
            gate_packed_offsets = (
                expert * PACKED_STRIDE_E
                + gate_rows[:, None] * PACKED_STRIDE_M
                + packed_cols[None, :] * PACKED_STRIDE_N
            )
            up_packed_offsets = (
                expert * PACKED_STRIDE_E
                + up_rows[:, None] * PACKED_STRIDE_M
                + packed_cols[None, :] * PACKED_STRIDE_N
            )
            gate_scale_offsets = (
                expert * SCALE_STRIDE_E
                + gate_rows[:, None] * SCALE_STRIDE_M
                + scale_cols[None, :] * SCALE_STRIDE_G
            )
            up_scale_offsets = (
                expert * SCALE_STRIDE_E
                + up_rows[:, None] * SCALE_STRIDE_M
                + scale_cols[None, :] * SCALE_STRIDE_G
            )

            gate_packed = tl.load(
                gate_up_packed_ptr + gate_packed_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0,
            )
            up_packed = tl.load(
                gate_up_packed_ptr + up_packed_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0,
            )
            gate_nibble = tl.where((cols[None, :] & 1) == 0, gate_packed & 0x0F, (gate_packed >> 4) & 0x0F)
            up_nibble = tl.where((cols[None, :] & 1) == 0, up_packed & 0x0F, (up_packed >> 4) & 0x0F)
            gate_w = _e2m1_from_nibble(gate_nibble)
            up_w = _e2m1_from_nibble(up_nibble)
            gate_scale = tl.load(
                gate_up_scale_ptr + gate_scale_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                gate_up_scale_ptr + up_scale_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_acc += tl.sum(gate_w * (gate_scale / global_scale) * x[None, :], axis=1)
            up_acc += tl.sum(up_w * (up_scale / global_scale) * x[None, :], axis=1)

        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        inter = gate_silu * up_acc
        tl.store(inter_ptr + slot * INTER_STRIDE_K + inter_offsets * INTER_STRIDE_I, inter.to(tl.bfloat16), mask=inter_mask)

    @triton.jit
    def _grouped_gate_up_silu_scale_hoist_kernel(
        x_ptr,
        expert_ids_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        inter_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block_i = tl.program_id(1)
        expert = tl.load(expert_ids_ptr + slot)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        group_offsets = tl.arange(0, 16)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        gate_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)

        for h0 in tl.static_range(0, HIDDEN, BLOCK_HIDDEN):
            for hg in tl.static_range(0, BLOCK_HIDDEN, 16):
                cols = h0 + hg + group_offsets
                col_mask = cols < HIDDEN
                packed_cols = cols // 2
                scale_col = (h0 + hg) // 16
                x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

                gate_rows = inter_offsets
                up_rows = INTERMEDIATE + inter_offsets
                gate_packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + gate_rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                up_packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + up_rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                gate_packed = tl.load(
                    gate_up_packed_ptr + gate_packed_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                up_packed = tl.load(
                    gate_up_packed_ptr + up_packed_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                gate_nibble = tl.where((cols[None, :] & 1) == 0, gate_packed & 0x0F, (gate_packed >> 4) & 0x0F)
                up_nibble = tl.where((cols[None, :] & 1) == 0, up_packed & 0x0F, (up_packed >> 4) & 0x0F)
                gate_w = _e2m1_from_nibble_fast(gate_nibble)
                up_w = _e2m1_from_nibble_fast(up_nibble)
                gate_scale = tl.load(
                    gate_up_scale_ptr
                    + expert * SCALE_STRIDE_E
                    + gate_rows * SCALE_STRIDE_M
                    + scale_col * SCALE_STRIDE_G,
                    mask=inter_mask,
                    other=0.0,
                ).to(tl.float32)
                up_scale = tl.load(
                    gate_up_scale_ptr
                    + expert * SCALE_STRIDE_E
                    + up_rows * SCALE_STRIDE_M
                    + scale_col * SCALE_STRIDE_G,
                    mask=inter_mask,
                    other=0.0,
                ).to(tl.float32)
                gate_acc += tl.sum(gate_w * (gate_scale[:, None] / global_scale) * x[None, :], axis=1)
                up_acc += tl.sum(up_w * (up_scale[:, None] / global_scale) * x[None, :], axis=1)

        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        inter = gate_silu * up_acc
        tl.store(inter_ptr + slot * INTER_STRIDE_K + inter_offsets * INTER_STRIDE_I, inter.to(tl.bfloat16), mask=inter_mask)

    @triton.jit
    def _grouped_gate_up_silu_fast_decode_kernel(
        x_ptr,
        expert_ids_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        inter_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,

codex
The existing gate micro-kernel only fuses the optional scalar gate or final add; it does not touch the dense shared MLP itself. I’m checking tensor shapes and load-time weight preparation next, because the fused path should reuse any prepacked gate_up weight and only activate for the M=1 BF16 decode contract.
exec
/bin/zsh -lc "grep -n \"def nvfp4_grouped_gate_up_silu\\|def nvfp4_grouped_down_weighted_sum\\|prepared\" triton_kernels/nvfp4_moe.py" in /Users/lynn/Downloads/Lynn/lynn-engine
exec
/bin/zsh -lc "sed -n '520,760p' triton_kernels/nvfp4_moe.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
596:def nvfp4_grouped_gate_up_silu(
642:def nvfp4_grouped_gate_up_silu_fast_decode_effective_scale(
707:def nvfp4_grouped_gate_up_silu_scale_hoist(
760:def nvfp4_grouped_gate_up_silu_fast_decode(
818:def nvfp4_grouped_gate_up_silu_fast_decode_prepared(
852:        raise ValueError("prepared gate/up tensors must be contiguous")
884:def nvfp4_grouped_gate_up_silu_merged_topk(
936:def nvfp4_grouped_down_weighted_sum(
996:def nvfp4_grouped_down_weighted_sum_effective_scale(
1057:def nvfp4_grouped_down_weighted_sum_prepared(
1085:        raise ValueError("prepared down tensors must be contiguous")
1118:def nvfp4_grouped_down_weighted_sum_scale_hoist(
1176:    "nvfp4_grouped_down_weighted_sum_prepared",
1181:    "nvfp4_grouped_gate_up_silu_fast_decode_prepared",

 succeeded in 0ms:
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = tl.arange(0, BLOCK_HIDDEN)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        for slot in range(0, TOP_K):
            expert = tl.load(expert_ids_ptr + slot)
            gate_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)

            for h0 in range(0, HIDDEN, BLOCK_HIDDEN):
                cols = h0 + h_offsets
                col_mask = cols < HIDDEN
                packed_cols = cols // 2
                scale_cols = cols // 16
                x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

                gate_rows = inter_offsets
                up_rows = INTERMEDIATE + inter_offsets
                gate_packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + gate_rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                up_packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + up_rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                gate_scale_offsets = (
                    expert * SCALE_STRIDE_E
                    + gate_rows[:, None] * SCALE_STRIDE_M
                    + scale_cols[None, :] * SCALE_STRIDE_G
                )
                up_scale_offsets = (
                    expert * SCALE_STRIDE_E
                    + up_rows[:, None] * SCALE_STRIDE_M
                    + scale_cols[None, :] * SCALE_STRIDE_G
                )

                gate_packed = tl.load(
                    gate_up_packed_ptr + gate_packed_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                up_packed = tl.load(
                    gate_up_packed_ptr + up_packed_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                gate_nibble = tl.where((cols[None, :] & 1) == 0, gate_packed & 0x0F, (gate_packed >> 4) & 0x0F)
                up_nibble = tl.where((cols[None, :] & 1) == 0, up_packed & 0x0F, (up_packed >> 4) & 0x0F)
                gate_w = _e2m1_from_nibble(gate_nibble)
                up_w = _e2m1_from_nibble(up_nibble)
                gate_scale = tl.load(
                    gate_up_scale_ptr + gate_scale_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                up_scale = tl.load(
                    gate_up_scale_ptr + up_scale_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                gate_acc += tl.sum(gate_w * (gate_scale / global_scale) * x[None, :], axis=1)
                up_acc += tl.sum(up_w * (up_scale / global_scale) * x[None, :], axis=1)

            gate_silu = gate_acc * tl.sigmoid(gate_acc)
            inter = gate_silu * up_acc
            tl.store(
                inter_ptr + slot * INTERMEDIATE + inter_offsets * INTER_STRIDE_I,
                inter.to(tl.bfloat16),
                mask=inter_mask,
            )


def nvfp4_grouped_gate_up_silu(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 64,
    block_hidden: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Compute top-k `silu(gate(x))*up(x)` from grouped packed NVFP4 weights."""
    _require_triton()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_kernel[grid](
        x.contiguous(),
        expert_ids,
        gate_up_packed.contiguous(),
        gate_up_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_gate_up_silu_fast_decode_effective_scale(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_effective_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 8,
    block_hidden: int = 256,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fast gate/up path that consumes precomputed `scale / global_scale`.

    This is an opt-in MoE repack probe.  It keeps the same packed weight layout
    and output contract as `nvfp4_grouped_gate_up_silu_fast_decode`, but removes
    the per-element division from the decode kernel when the runner has attached
    effective scale tensors.
    """
    _require_triton()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_effective_scale.ndim != 3:
        raise ValueError(
            "expected grouped 3D tensors, got "
            f"packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_effective_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    if out is None:
        inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    else:
        if out.ndim != 2 or out.shape[0] < expert_ids.numel() or out.shape[1] != INTERMEDIATE_SIZE:
            raise ValueError(
                f"out must be at least [top_k, {INTERMEDIATE_SIZE}], got {tuple(out.shape)} "
                f"for top_k={expert_ids.numel()}"
            )
        if out.device != x.device or out.dtype != torch.bfloat16:
            raise ValueError("out must be a bfloat16 tensor on the same device as x")
        inter = out[: expert_ids.numel()]
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_fast_decode_kernel[grid](
        x.contiguous(),
        expert_ids,
        gate_up_packed.contiguous(),
        gate_up_effective_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_effective_scale.stride(0),
        gate_up_effective_scale.stride(1),
        gate_up_effective_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        SCALE_EFFECTIVE=True,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_gate_up_silu_scale_hoist(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 8,
    block_hidden: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """P53 scale-hoisted gate/up probe.

    This variant loads each per-16 scale once per group instead of repeating it
    across the 16 columns. It is intentionally opt-in until full-generate gates
    prove both speed and numerical stability.
    """
    _require_triton()
    if block_hidden % 16 != 0:
        raise ValueError(f"block_hidden must be divisible by 16, got {block_hidden}")
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_scale_hoist_kernel[grid](
        x.contiguous(),
        expert_ids,
        gate_up_packed.contiguous(),
        gate_up_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_gate_up_silu_fast_decode(

exec
/bin/zsh -lc "sed -n '1,240p' triton_kernels/spark_fp8_gate_up_fused.py" in /Users/lynn/Downloads/Lynn/lynn-engine
exec
/bin/zsh -lc "sed -n '1,260p' triton_kernels/rowwise_linear.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
"""Spark sm_121 FP8 fused gate/up + SwiGLU Triton kernel.

This is Phase 2 step 2 of the Spark TPS plan
(reference_spark_fp8_w4a8_design_strategy_20260519). The kernel takes
a BF16 activation plus pre-repacked FP8 E4M3 gate + up weights with
per-row scales, runs both projections through the FP8 MMA unit
(GB10 sm_121, 162 TFLOPS peak vs 99 BF16), applies per-row weight
scale + per-token activation scale, fuses SwiGLU (``silu(gate) * up``),
and emits the BF16 intermediate ready for ``down_proj``.

The fused boundary collapses what would be three kernel launches +
two activation-cast launches into one launch, which is the heart of
why naive ``torch._scaled_mm`` per-projection landed at only 14 TPS
in the May-19 PoC.

V0 scope:
  * Decode-time M ∈ {1, small batch} hot path (no autotune yet).
  * Weight layout: row-major [N, K] in FP8 E4M3 (matches existing
    Lynn-native storage shape; reuses ``spark_pack_w4a8_fp8`` output).
  * Per-row weight scale [N] (F32), per-token activation scale
    derived inside the kernel from a precomputed [M] F32 tensor.
  * Caller is responsible for computing the activation scale ahead
    of time (a one-shot reduction, cheap).
  * Output: BF16 [M, intermediate].

V1 scope (current — autotune sweep result applied):
  * Default block config = ``(BLOCK_M=16, BLOCK_K=128, BLOCK_N=32)``,
    the universal winner from the 2160-config Spark sm_121 sweep
    (best/near-best for 60%+ of shapes; see
    ``reports/mtp/QWEN36_FP8_AUTOTUNE_SWEEP_RESULT_20260520.md``).
  * Shape-aware override via ``select_block_config(M, K, N)`` helper
    for high-traffic specialised shapes (M=1 N=6144, M=16 K=6144,
    etc.). Callers can opt in via ``auto_block=True``.
  * N ≤ 256 fast-path advisory: caller should fall back to BF16
    ``torch._scaled_mm`` — FP8 cannot beat BF16 at this output size
    (memory-bandwidth bound, ~0.86× best speedup).

V2 scope (next):
  * Concatenated gate+up weight ([2N, K]) for one B-matrix load.
  * Native-owned intermediate buffer (no Python/Torch round-trip
    before down_proj — see P190 finding #2).
  * Optional col-major weight layout for cuBLASLt parity.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for FP8 fused gate/up kernel")


if HAS_TRITON:

    @triton.jit
    def _fp8_gate_up_silu_kernel(
        # Pointers
        x_ptr,                     # [M, K] BF16 activation
        x_scale_ptr,               # [M] F32 per-token activation scale (max_abs / 448)
        w_gate_ptr,                # [N, K] FP8 E4M3 gate weight (row-major)
        w_up_ptr,                  # [N, K] FP8 E4M3 up weight (row-major)
        w_gate_scale_ptr,          # [N] F32 per-row gate weight scale
        w_up_scale_ptr,            # [N] F32 per-row up weight scale
        out_ptr,                   # [M, N] BF16 output = silu(gate*s_g*s_x) * (up*s_u*s_x)
        # Sizes
        M, K, N,
        # Strides
        stride_xm, stride_xk,
        stride_wm, stride_wk,
        stride_om, stride_on,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K-loop: accumulate FP8 × FP8 → F32 MMA.
        for k_block in range(0, K, BLOCK_K):
            k_offs = k_block + offs_k
            mask_k = k_offs < K

            # Load BF16 activation block [BLOCK_M, BLOCK_K]
            x_block = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            # Load per-token activation scale [BLOCK_M]
            x_scale = tl.load(x_scale_ptr + offs_m, mask=mask_m, other=1.0)
            # Quantize activation to FP8: x_fp8 = x_bf16 / x_scale; clipped to FP8 range.
            x_q = x_block.to(tl.float32) / x_scale[:, None]
            x_fp8 = x_q.to(tl.float8e4nv)

            # Load FP8 weight blocks [BLOCK_N, BLOCK_K]
            w_gate_block = tl.load(
                w_gate_ptr + offs_n[:, None] * stride_wm + k_offs[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=tl.zeros((1,), dtype=tl.float8e4nv),
            )
            w_up_block = tl.load(
                w_up_ptr + offs_n[:, None] * stride_wm + k_offs[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=tl.zeros((1,), dtype=tl.float8e4nv),
            )

            # FP8 × FP8 MMA. tl.dot expects [M, K] × [K, N], so transpose weight on read.
            acc_gate += tl.dot(x_fp8, w_gate_block.trans(), out_dtype=tl.float32)
            acc_up += tl.dot(x_fp8, w_up_block.trans(), out_dtype=tl.float32)

        # Apply per-row weight scale + per-token activation scale.
        w_gate_scale = tl.load(w_gate_scale_ptr + offs_n, mask=mask_n, other=1.0)
        w_up_scale = tl.load(w_up_scale_ptr + offs_n, mask=mask_n, other=1.0)
        x_scale = tl.load(x_scale_ptr + offs_m, mask=mask_m, other=1.0)

        scale_combined_g = x_scale[:, None] * w_gate_scale[None, :]
        scale_combined_u = x_scale[:, None] * w_up_scale[None, :]
        gate = acc_gate * scale_combined_g
        up = acc_up * scale_combined_u

        # SwiGLU: silu(gate) * up
        inter = (gate * tl.sigmoid(gate)) * up

        # Store BF16
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            inter.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def select_block_config(m: int, k: int, n: int) -> tuple[int, int, int]:
    """Shape-aware best block config from the Spark sm_121 autotune sweep.

    Returns ``(BLOCK_M, BLOCK_K, BLOCK_N)`` selected from the per-shape
    top-1 entries of ``reports/mtp/QWEN36_FP8_AUTOTUNE_SWEEP_RESULT_20260520.md``.

    For shapes not in the override table this falls back to the universal
    winner ``(16, 128, 32)``. Callers should still avoid this kernel when
    ``n <= 256`` (memory-bound; use BF16 ``_scaled_mm`` instead).
    """
    # High-impact specialised overrides for the Lynn 35B-A3B hot shapes.
    if m == 1:
        if k == 2048 and n == 6144:
            return (16, 64, 32)        # 6.00× — best in sweep
        if k == 4096 and n == 2048:
            return (16, 128, 64)       # 5.76×
        if k == 6144 and n == 2048:
            return (16, 128, 32)       # 4.25×
        if k == 6144 and n == 6144:
            return (64, 128, 32)       # 1.79×
    elif 4 <= m <= 8:
        if k == 2048 and n == 6144:
            return (16, 64, 32)        # 5.67× / 5.38×
        if k == 6144 and n == 6144:
            return (64, 128, 32) if m == 4 else (16, 128, 32)
    elif m >= 16:
        if k == 6144 and n == 6144:
            return (64, 128, 32)       # 2.04×
        if k == 2048 and n == 6144:
            return (32, 64, 64)        # 3.65×
    return (16, 128, 32)               # universal winner


def fp8_gate_up_silu_fused(
    x_bf16: torch.Tensor,             # [M, K]
    w_gate_fp8: torch.Tensor,         # [N, K] FP8 E4M3
    w_up_fp8: torch.Tensor,           # [N, K] FP8 E4M3
    w_gate_scale: torch.Tensor,       # [N] F32
    w_up_scale: torch.Tensor,         # [N] F32
    *,
    block_m: int = 16,
    block_k: int = 128,
    block_n: int = 32,
    auto_block: bool = False,
) -> torch.Tensor:
    """Run fused FP8 gate/up + SwiGLU on Spark sm_121.

    Returns BF16 intermediate [M, N] = silu(gate * scales) * (up * scales)
    ready for the down_proj step.

    The default block config ``(16, 128, 32)`` is the universal winner
    from the 2160-config sweep (see ``select_block_config`` for
    shape-aware overrides). Pass ``auto_block=True`` to dispatch on the
    runtime shape.
    """
    _require_triton()
    if not x_bf16.is_cuda:
        raise ValueError("activation must be CUDA tensor")
    if x_bf16.dtype != torch.bfloat16:
        raise ValueError(f"activation must be BF16, got {x_bf16.dtype}")
    if w_gate_fp8.dtype != torch.float8_e4m3fn or w_up_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("weights must be float8_e4m3fn")
    if w_gate_fp8.shape != w_up_fp8.shape:
        raise ValueError("gate and up weights must have same shape")
    if w_gate_scale.dtype != torch.float32 or w_up_scale.dtype != torch.float32:
        raise ValueError("scales must be float32")

    M, K = x_bf16.shape
    N, K_w = w_gate_fp8.shape
    if K != K_w:
        raise ValueError(f"K mismatch: act K={K}, weight K={K_w}")

    if auto_block:
        block_m, block_k, block_n = select_block_config(M, K, N)

    # Compute per-token activation scale = max_abs / 448 (FP8 E4M3 max).
    x_scale = (x_bf16.abs().amax(dim=-1).clamp_min(1.0e-12) / 448.0).to(torch.float32)

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x_bf16.device)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fp8_gate_up_silu_kernel[grid](
        x_bf16,
        x_scale,
        w_gate_fp8,
        w_up_fp8,
        w_gate_scale,

 succeeded in 0ms:
"""Row-wise Linear kernels for K=2 verifier parity experiments.

These kernels deliberately compute each token row with an independent
accumulator. The goal is not to beat cuBLAS yet; it is to prove a dispatch
shape that can process K=2 rows in one launch while preserving the same
per-row accumulation contract as separate T=1 calls.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - non-GPU dev hosts.
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for rowwise linear kernels")


if HAS_TRITON:

    @triton.jit
    def _rowwise_linear_kernel(
        X,
        W,
        Y,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_ym: tl.constexpr,
        stride_yn: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            k_mask = k_idx < K
            w = tl.load(
                W + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk,
                mask=(offs_n[:, None] < N) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(
                X + k_idx * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            acc0 += tl.sum(w * x0[None, :], axis=1)
            if M == 2:
                x1 = tl.load(
                    X + stride_xm + k_idx * stride_xk,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                acc1 += tl.sum(w * x1[None, :], axis=1)

        tl.store(Y + offs_n * stride_yn, acc0, mask=offs_n < N)
        if M == 2:
            tl.store(Y + stride_ym + offs_n * stride_yn, acc1, mask=offs_n < N)


def rowwise_linear(x: torch.Tensor, weight: torch.Tensor, *, block_n: int = 32, block_k: int = 64) -> torch.Tensor:
    """Compute ``F.linear(x, weight)`` for one or two rows.

    Args:
        x: CUDA tensor ``[M, K]`` where ``M`` is 1 or 2.
        weight: CUDA tensor ``[N, K]``.
    """
    _require_triton()
    if x.ndim != 2:
        raise ValueError(f"x must be [M, K], got {tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be [N, K], got {tuple(weight.shape)}")
    if x.shape[0] not in (1, 2):
        raise ValueError(f"rowwise_linear supports M=1 or M=2, got {x.shape[0]}")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f"K mismatch: x={x.shape[1]} weight={weight.shape[1]}")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")

    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(weight.shape[0], block_n),)
    _rowwise_linear_kernel[grid](
        x,
        weight,
        out,
        x.shape[0],
        x.shape[1],
        weight.shape[0],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


__all__ = ["rowwise_linear"]

exec
/bin/zsh -lc "grep -R \"shared_expert._gate_up_proj\\|shared_expert.gate_proj\\|shared_expert.down_proj\\|shared_expert_gate.weight\" -n . | head -80" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 685ms:
./docs/QWEN36_OFFICIAL_MTP_TO_LYNN_FUSED_CONTRACT_20260519.md:42:    ├── mtp.layers.0.mlp.shared_expert_gate.weight
./docs/QWEN36_OFFICIAL_MTP_TO_LYNN_FUSED_CONTRACT_20260519.md:78:    "mlp.shared_expert.gate_proj.weight",
./docs/QWEN36_OFFICIAL_MTP_TO_LYNN_FUSED_CONTRACT_20260519.md:80:    "mlp.shared_expert.down_proj.weight",
./docs/QWEN36_OFFICIAL_MTP_TO_LYNN_FUSED_CONTRACT_20260519.md:81:    "mlp.shared_expert_gate.weight",
./pruning/training/recovery_lora_qwen36.yaml:50:#   - mlp.shared_expert.gate_proj
./pruning/training/recovery_lora_qwen36.yaml:52:#   - mlp.shared_expert.down_proj
./pruning/profile_activations.py:175:        if "mlp.shared_expert.gate_proj.weight" in w:
./pruning/profile_activations.py:176:            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
./pruning/profile_activations.py:178:            shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
./pruning/profile_activations.py:179:            if "mlp.shared_expert_gate.weight" in w:
./pruning/profile_activations.py:180:                shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
./benchmarks/candidates/native_grouped_per16_nonatomic.py:133:    if "mlp.shared_expert.gate_proj.weight" in layer_weights:
./benchmarks/candidates/native_grouped_per16_nonatomic.py:134:        gate_s = F.linear(h_flat, layer_weights["mlp.shared_expert.gate_proj.weight"])
./benchmarks/candidates/native_grouped_per16_nonatomic.py:136:        shared_ffn = F.linear(F.silu(gate_s) * up_s, layer_weights["mlp.shared_expert.down_proj.weight"])
./benchmarks/candidates/native_grouped_per16_nonatomic.py:137:        if "mlp.shared_expert_gate.weight" in layer_weights:
./benchmarks/candidates/native_grouped_per16_nonatomic.py:139:                F.linear(h_flat, layer_weights["mlp.shared_expert_gate.weight"])
./benchmarks/candidates/native_output_owned_bf16.py:84:    if "mlp.shared_expert.gate_proj.weight" in layer_weights:
./benchmarks/candidates/native_output_owned_bf16.py:85:        gate_s = F.linear(h_flat, layer_weights["mlp.shared_expert.gate_proj.weight"])
./benchmarks/candidates/native_output_owned_bf16.py:87:        shared_ffn = F.linear(F.silu(gate_s) * up_s, layer_weights["mlp.shared_expert.down_proj.weight"])
./benchmarks/candidates/native_output_owned_bf16.py:88:        if "mlp.shared_expert_gate.weight" in layer_weights:
./benchmarks/candidates/native_output_owned_bf16.py:90:                F.linear(h_flat, layer_weights["mlp.shared_expert_gate.weight"])
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:77:    if "mlp.shared_expert._gate_up_proj.weight" in w:
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:78:        gate_up = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:81:        gate = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:83:    return F.linear(F.silu(gate) * up, w["mlp.shared_expert.down_proj.weight"])
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:92:    if "mlp.shared_expert_gate.weight" in w:
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:93:        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:103:    torch.mm(h_flat, w["mlp.shared_expert._gate_up_proj.weight"].t(), out=gate_up_scratch)
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:106:    torch.mm(hidden, w["mlp.shared_expert.down_proj.weight"].t(), out=shared_scratch)
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:117:    torch.mm(h_flat, w["mlp.shared_expert._gate_up_proj.weight"].t(), out=gate_up_scratch)
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:122:    torch.mm(inter_scratch, w["mlp.shared_expert.down_proj.weight"].t(), out=shared_scratch)
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:135:    if "mlp.shared_expert_gate.weight" in w:
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:136:        torch.mm(h_flat, w["mlp.shared_expert_gate.weight"].t(), out=gate_scratch)
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:168:        if "mlp.shared_expert.gate_proj.weight" not in w:
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:170:        if "mlp.shared_expert._gate_up_proj.weight" not in w:
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:171:            w["mlp.shared_expert._gate_up_proj.weight"] = torch.cat(
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:172:                [w["mlp.shared_expert.gate_proj.weight"], w["mlp.shared_expert.up_proj.weight"]],
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:176:        gate_up_scratch = torch.empty((1, w["mlp.shared_expert._gate_up_proj.weight"].shape[0]), device=args.device, dtype=dtype)
./benchmarks/p167_qwen36_shared_expert_prepared_probe.py:177:        inter_scratch = torch.empty((1, w["mlp.shared_expert.down_proj.weight"].shape[1]), device=args.device, dtype=dtype)
./benchmarks/p2_nvfp4_layer0_coverage.py:202:            ("mlp.shared_expert.gate_proj", "mlp.shared_expert.gate_proj.weight"),
./benchmarks/p2_nvfp4_layer0_coverage.py:204:            ("mlp.shared_expert.down_proj", "mlp.shared_expert.down_proj.weight"),
./benchmarks/p127_moe_repack_sidecar_contract.py:132:        "shared_gate": f"{prefix}.mlp.shared_expert.gate_proj.weight",
./benchmarks/p127_moe_repack_sidecar_contract.py:134:        "shared_down": f"{prefix}.mlp.shared_expert.down_proj.weight",
./benchmarks/p127_moe_repack_sidecar_contract.py:135:        "shared_scalar_gate": f"{prefix}.mlp.shared_expert_gate.weight",
./benchmarks/p162_qwen36_shared_finalize_probe.py:91:        "mlp.shared_expert.gate_proj.weight.packed" in w
./benchmarks/p162_qwen36_shared_finalize_probe.py:93:        and "mlp.shared_expert.down_proj.weight.packed" in w
./benchmarks/p162_qwen36_shared_finalize_probe.py:97:            w["mlp.shared_expert.gate_proj.weight.packed"],
./benchmarks/p162_qwen36_shared_finalize_probe.py:100:        return w["mlp.shared_expert.down_proj.weight.packed"](
./benchmarks/p162_qwen36_shared_finalize_probe.py:103:    if "mlp.shared_expert._gate_up_proj.weight" in w:
./benchmarks/p162_qwen36_shared_finalize_probe.py:104:        gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
./benchmarks/p162_qwen36_shared_finalize_probe.py:107:        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
./benchmarks/p162_qwen36_shared_finalize_probe.py:109:    return F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
./benchmarks/p162_qwen36_shared_finalize_probe.py:165:        if "mlp.shared_expert_gate.weight" not in w:
./benchmarks/p162_qwen36_shared_finalize_probe.py:173:            return torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
./benchmarks/p162_qwen36_shared_finalize_probe.py:178:            return active + shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
./benchmarks/p162_qwen36_shared_finalize_probe.py:181:            g = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
./benchmarks/p6g_full_attention_layer_profile.py:226:        gate = _linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
./benchmarks/p6g_full_attention_layer_profile.py:228:        out = _linear(F.silu(gate) * up, w["mlp.shared_expert.down_proj.weight"])
./benchmarks/p6g_full_attention_layer_profile.py:229:        if "mlp.shared_expert_gate.weight" in w:
./benchmarks/p6g_full_attention_layer_profile.py:230:            out = out * torch.sigmoid(_linear(h_flat, w["mlp.shared_expert_gate.weight"]))
./benchmarks/p6g_full_attention_layer_profile.py:314:        gate = _linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
./benchmarks/p6g_full_attention_layer_profile.py:316:        shared = _linear(F.silu(gate) * up, w["mlp.shared_expert.down_proj.weight"])
./benchmarks/p6g_full_attention_layer_profile.py:317:        if "mlp.shared_expert_gate.weight" in w:
./benchmarks/p6g_full_attention_layer_profile.py:318:            shared = shared * torch.sigmoid(_linear(h_flat, w["mlp.shared_expert_gate.weight"]))
./benchmarks/p40_moe_forward_fast_candidate_gate.py:73:    if "mlp.shared_expert._gate_up_proj.weight" in w:
./benchmarks/p40_moe_forward_fast_candidate_gate.py:74:        gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
./benchmarks/p40_moe_forward_fast_candidate_gate.py:76:        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
./benchmarks/p40_moe_forward_fast_candidate_gate.py:78:        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
./benchmarks/p40_moe_forward_fast_candidate_gate.py:80:        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
./benchmarks/p40_moe_forward_fast_candidate_gate.py:81:    if "mlp.shared_expert_gate.weight" in w:
./benchmarks/p40_moe_forward_fast_candidate_gate.py:82:        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
./benchmarks/p134_active_moe_fixture_contract.py:181:    if "mlp.shared_expert.gate_proj.weight" in layer_weights:
./benchmarks/p134_active_moe_fixture_contract.py:182:        gate_s = F.linear(h_flat, layer_weights["mlp.shared_expert.gate_proj.weight"])
./benchmarks/p134_active_moe_fixture_contract.py:184:        shared_ffn = F.linear(F.silu(gate_s) * up_s, layer_weights["mlp.shared_expert.down_proj.weight"])
./benchmarks/p134_active_moe_fixture_contract.py:185:        if "mlp.shared_expert_gate.weight" in layer_weights:
./benchmarks/p134_active_moe_fixture_contract.py:187:                F.linear(h_flat, layer_weights["mlp.shared_expert_gate.weight"])
./benchmarks/p3_nvfp4_layer_decode_packed_probe.py:248:    if "mlp.shared_expert.gate_proj.weight" in resident_w:
./benchmarks/p3_nvfp4_layer_decode_packed_probe.py:250:            gate_s = F.linear(h_flat, resident_w["mlp.shared_expert.gate_proj.weight"])
./benchmarks/p3_nvfp4_layer_decode_packed_probe.py:254:                resident_w["mlp.shared_expert.down_proj.weight"],
./benchmarks/p3_nvfp4_layer_decode_packed_probe.py:260:        if "mlp.shared_expert_gate.weight" in resident_w:
grep: ./.git/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/kimi-qwen35-9b-release-matrix-report-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen35-9b-w4a8-resident-boundary-v2-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-p198-native-fp4-preflight/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/kimi-p192-qwen35-9b-fp4x-fp8-repack/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen35-9b-release-gate-final-report-v2-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-p199-nvfp4-size-audit/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen36-35b-w4a8-route-flip-gate-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen35-9b-release-gate-final-report-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/codex-main-overnight/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/codex-qwen35-9b-r6000-release/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/mimo-moe-packed-stage-admission-20260518/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/mimo-moe-risk-gate-20260518/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen35-9b-verify-release-artifacts-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/kimi-moe-packed-slot-20260518/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/mimo-qwen35-9b-q4km-llamacpp-baseline-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen35-9b-release-gate-summary-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/codebuddy-qwen35-manifest-schema/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/claude-nemotron8b-formal-bench/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/codebuddy-qwen35-app-local-gate/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen35-9b-spark-perf-gate-20260520/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/codebuddy-qwen35-docs-sync/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/codebuddy-qwen35-release-site-copy/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/claude-qwen35-9b-dense-runtime-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/mimo-qwen35-9b-nvfp4-release-gates-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/codebuddy-qwen35-mac-runbook/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/mimo-moe-resident-p37-admission-20260518/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/lynn-engine-main/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/kimi-qwen36-9b-r6000-pipeline-20260518/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/native-moe-kernel-island/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-p197-drift-probe-final/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen35-9b-q4km-realbench-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/claude-moe-packed-resident-abi-v3-20260518/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen35-9b-app-probe-review-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/kimi-qwen36-9b-dense-matrix-20260518/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-p193-qwen35-native-boundary-admission/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen35-9b-q4km-cuda-baseline-20260519/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/qwen-qwen35-9b-qa-manifest-20260520/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/claude-qwen35-9b-convstrict/fsmonitor--daemon.ipc: Operation not supported on socket
grep: ./.git/worktrees/claude-p191-r6000-fp4x-fp8-cute-dense-poc/fsmonitor--daemon.ipc: Operation not supported on socket

codex
There is an old prepared-probe benchmark for this exact area. I’m using it as a sanity check for expected scratch shapes and the current launch count, but the kernel still needs to live as a first-class `triton_kernels` wrapper.
exec
/bin/zsh -lc "sed -n '60,150p' benchmarks/p167_qwen36_shared_expert_prepared_probe.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
    cand_norm = torch.linalg.vector_norm(cf).clamp_min(1e-12)
    max_abs = float(diff.abs().max().item())
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(diff).item() / float(ref_norm.item())),
        "cosine": float(torch.dot(rf, cf).item() / float((ref_norm * cand_norm).item())),
        "exact": 1 if max_abs == 0.0 else 0,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def _shared_default(h_flat: torch.Tensor, w: dict[str, Any]) -> torch.Tensor:
    if "mlp.shared_expert._gate_up_proj.weight" in w:
        gate_up = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
        gate, up = gate_up.chunk(2, dim=-1)
    else:
        gate = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
    return F.linear(F.silu(gate) * up, w["mlp.shared_expert.down_proj.weight"])


def _finalize_default(
    h_flat: torch.Tensor,
    moe_out: torch.Tensor,
    shared: torch.Tensor,
    w: dict[str, Any],
) -> torch.Tensor:
    if "mlp.shared_expert_gate.weight" in w:
        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    return moe_out + shared


def _shared_mm_out(
    h_flat: torch.Tensor,
    w: dict[str, Any],
    gate_up_scratch: torch.Tensor,
    shared_scratch: torch.Tensor,
) -> torch.Tensor:
    torch.mm(h_flat, w["mlp.shared_expert._gate_up_proj.weight"].t(), out=gate_up_scratch)
    gate, up = gate_up_scratch.chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    torch.mm(hidden, w["mlp.shared_expert.down_proj.weight"].t(), out=shared_scratch)
    return shared_scratch


def _shared_mm_out_inplace_silu(
    h_flat: torch.Tensor,
    w: dict[str, Any],
    gate_up_scratch: torch.Tensor,
    inter_scratch: torch.Tensor,
    shared_scratch: torch.Tensor,
) -> torch.Tensor:
    torch.mm(h_flat, w["mlp.shared_expert._gate_up_proj.weight"].t(), out=gate_up_scratch)
    gate, up = gate_up_scratch.chunk(2, dim=-1)
    torch.sigmoid(gate, out=inter_scratch)
    inter_scratch.mul_(gate)
    inter_scratch.mul_(up)
    torch.mm(inter_scratch, w["mlp.shared_expert.down_proj.weight"].t(), out=shared_scratch)
    return shared_scratch


def _finalize_inplace(
    h_flat: torch.Tensor,
    moe_out: torch.Tensor,
    shared: torch.Tensor,
    w: dict[str, Any],
    gate_scratch: torch.Tensor,
    out_scratch: torch.Tensor,
) -> torch.Tensor:
    out_scratch.copy_(moe_out)
    if "mlp.shared_expert_gate.weight" in w:
        torch.mm(h_flat, w["mlp.shared_expert_gate.weight"].t(), out=gate_scratch)
        gate_scratch.sigmoid_()
        shared = shared * gate_scratch
    out_scratch.add_(shared)
    return out_scratch


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe exact prepared BF16 shared-expert boundary.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)

exec
/bin/zsh -lc "sed -n '150,220p' benchmarks/p167_qwen36_shared_expert_prepared_probe.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-fixtures", type=int, default=18)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    t0 = time.time()
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    load_seconds = time.time() - t0

    packed_dir = Path(args.packed_fixtures)
    manifest = json.loads((packed_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in manifest["fixtures"][: args.max_fixtures]:
        layer_id = int(entry["layer_id"])
        data = _load_fixture(packed_dir / entry["fixture_file"], args.device)
        h_flat = data["hidden_in"].to(dtype).view(1, -1).contiguous()
        w = runner.layer_weights[layer_id]
        if "mlp.shared_expert.gate_proj.weight" not in w:
            continue
        if "mlp.shared_expert._gate_up_proj.weight" not in w:
            w["mlp.shared_expert._gate_up_proj.weight"] = torch.cat(
                [w["mlp.shared_expert.gate_proj.weight"], w["mlp.shared_expert.up_proj.weight"]],
                dim=0,
            ).contiguous()
        moe_out = torch.zeros_like(h_flat)
        gate_up_scratch = torch.empty((1, w["mlp.shared_expert._gate_up_proj.weight"].shape[0]), device=args.device, dtype=dtype)
        inter_scratch = torch.empty((1, w["mlp.shared_expert.down_proj.weight"].shape[1]), device=args.device, dtype=dtype)
        shared_scratch = torch.empty_like(h_flat)
        gate_scratch = torch.empty((1, 1), device=args.device, dtype=dtype)
        final_scratch = torch.empty_like(h_flat)

        shared_ref = _shared_default(h_flat, w)
        final_ref = _finalize_default(h_flat, moe_out, shared_ref, w)

        def shared_mm_out_fn() -> torch.Tensor:
            return _shared_mm_out(h_flat, w, gate_up_scratch, shared_scratch)

        def shared_inplace_silu_fn() -> torch.Tensor:
            return _shared_mm_out_inplace_silu(h_flat, w, gate_up_scratch, inter_scratch, shared_scratch)

        shared_mm = shared_mm_out_fn().clone()
        shared_inplace = shared_inplace_silu_fn().clone()

        def final_mm_out_fn() -> torch.Tensor:
            shared = _shared_mm_out(h_flat, w, gate_up_scratch, shared_scratch)
            return _finalize_default(h_flat, moe_out, shared, w)

        def final_prepared_fn() -> torch.Tensor:
            shared = _shared_mm_out(h_flat, w, gate_up_scratch, shared_scratch)
            return _finalize_inplace(h_flat, moe_out, shared, w, gate_scratch, final_scratch)

        final_mm = final_mm_out_fn().clone()
        final_prepared = final_prepared_fn().clone()

        row = {
            "fixture_file": entry["fixture_file"],
            "layer_id": layer_id,
            "prompt_id": int(entry["prompt_id"]),
            "shared_default_ms": _bench_ms(lambda: _shared_default(h_flat, w), warmup=args.warmup, iters=args.iters),
            "shared_mm_out_ms": _bench_ms(shared_mm_out_fn, warmup=args.warmup, iters=args.iters),
            "shared_inplace_silu_ms": _bench_ms(shared_inplace_silu_fn, warmup=args.warmup, iters=args.iters),
            "final_default_ms": _bench_ms(lambda: _finalize_default(h_flat, moe_out, shared_ref, w), warmup=args.warmup, iters=args.iters),
            "final_mm_out_ms": _bench_ms(final_mm_out_fn, warmup=args.warmup, iters=args.iters),
            "final_prepared_ms": _bench_ms(final_prepared_fn, warmup=args.warmup, iters=args.iters),
            "shared_mm_out_vs_default": _metric(shared_ref, shared_mm),
            "shared_inplace_silu_vs_default": _metric(shared_ref, shared_inplace),
            "final_mm_out_vs_default": _metric(final_ref, final_mm),
            "final_prepared_vs_default": _metric(final_ref, final_prepared),
        }
        rows.append(row)

exec
/bin/zsh -lc 'grep -R "gate_up_proj" -n engine | head -80' in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
engine/moe_packed_nvfp4.py:827:        if "mlp.shared_expert._gate_up_proj.weight" in w:
engine/moe_packed_nvfp4.py:828:            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
engine/moe_packed_nvfp4.py:888:        if "mlp.shared_expert._gate_up_proj.weight" in w:
engine/moe_packed_nvfp4.py:889:            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
engine/moe_packed_nvfp4.py:1131:            and "mlp.shared_expert._gate_up_proj.weight" in w
engine/moe_packed_nvfp4.py:1133:            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
engine/resident_runner.py:456:            if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
engine/resident_runner.py:457:                gate_stacked, up_stacked = w["mlp.experts.gate_up_proj"].chunk(2, dim=1)
engine/resident_runner.py:461:                cfg["num_experts"] = int(w["mlp.experts.gate_up_proj"].shape[0])
engine/resident_runner.py:476:                    f"{base}.gate_up_proj",
engine/resident_runner.py:639:            key = "mlp.shared_expert._gate_up_proj.weight"
engine/resident_runner.py:688:            key = "mlp._gate_up_proj.weight"
engine/resident_runner.py:1380:                    drop(layer_idx, w, "mlp.experts.gate_up_proj", "packed_grouped_moe_gate_up")
Binary file engine/__pycache__/moe_optimized.cpython-314.pyc matches
Binary file engine/__pycache__/moe_packed_nvfp4.cpython-314.pyc matches
Binary file engine/__pycache__/resident_runner.cpython-314.pyc matches
Binary file engine/__pycache__/full_forward.cpython-314.pyc matches
Binary file engine/__pycache__/full_forward.cpython-312.pyc matches
Binary file engine/__pycache__/mtp_sidecar.cpython-314.pyc matches
engine/mtp_sidecar.py:183:    is_dense = "mlp.experts.gate_up_proj" not in out and (
engine/mtp_sidecar.py:212:            "mlp.experts.gate_up_proj",
engine/mtp_sidecar.py:229:    if "mlp.experts.gate_up_proj" in mtp_w:
engine/mtp_sidecar.py:230:        cfg["num_experts"] = int(mtp_w["mlp.experts.gate_up_proj"].shape[0])
engine/loader.py:248:    # [out, in]; squared MoE expert weights mirror ``mlp.experts.gate_up_proj``
engine/loader.py:261:    if "mlp.experts.gate_up_proj.weight_fp8" in final:
engine/loader.py:262:        config["expert_intermediate"] = final["mlp.experts.gate_up_proj.weight_fp8"].shape[1] // 2
engine/loader.py:383:    if "mlp.experts.gate_up_proj" in final:
engine/loader.py:384:        config["expert_intermediate"] = final["mlp.experts.gate_up_proj"].shape[1] // 2
engine/loader.py:401:    - `mlp.experts.gate_up_proj`: [E, 2 * intermediate, hidden]
engine/loader.py:483:        final["mlp.experts.gate_up_proj"] = torch.stack(gate_ups, dim=0).to(dequant_dtype)
engine/loader.py:496:    if "mlp.experts.gate_up_proj" in final:
engine/loader.py:497:        config["expert_intermediate"] = final["mlp.experts.gate_up_proj"].shape[1] // 2
engine/loader.py:646:        elif k in ("mlp.experts.down_proj", "mlp.experts.gate_up_proj"):
engine/moe_optimized.py:53:    if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
engine/moe_optimized.py:54:        gate_up = F.linear(x, w["mlp.experts.gate_up_proj"][expert_id])
engine/moe_optimized.py:73:    (``mlp.experts.gate_up_proj.weight_fp8`` etc.) are present in the
engine/moe_optimized.py:97:    w_gate_up_fp8 = w["mlp.experts.gate_up_proj.weight_fp8"]                      # [E, 2I, K]
engine/moe_optimized.py:98:    w_gate_up_scale = w["mlp.experts.gate_up_proj.weight_fp8_scale"]              # [E, 2I]
engine/moe_optimized.py:202:            and "mlp.experts.gate_up_proj.weight_fp8" in w
engine/moe_optimized.py:321:    if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
engine/moe_optimized.py:322:        gate_up_stack = torch.stack([w["mlp.experts.gate_up_proj"][e] for e in expert_ids])
engine/full_forward.py:224:    Phase 2 Spark FP8 path: keyed on ``mlp.experts.gate_up_proj.weight_fp8``
engine/full_forward.py:248:        and "mlp.experts.gate_up_proj.weight_fp8" in w
engine/full_forward.py:255:        # Stacked weight layout: gate_up_proj [E, 2*intermediate, hidden] →
engine/full_forward.py:257:        w_gate_up_fp8 = w["mlp.experts.gate_up_proj.weight_fp8"]   # [E, 2I, K]
engine/full_forward.py:258:        w_gate_up_scale = w["mlp.experts.gate_up_proj.weight_fp8_scale"]  # [E, 2I]
engine/full_forward.py:324:        "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w
engine/full_forward.py:333:            gate_up = F.linear(x_e, w["mlp.experts.gate_up_proj"][e])
engine/full_forward.py:500:            fused = w.get("mlp._gate_up_proj.weight")
engine/full_forward.py:520:    fused = w.get("mlp._gate_up_proj.weight")

exec
/bin/zsh -lc 'grep -R "shared_expert" -n engine | head -120' in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
engine/transformer_block.py:58:      mlp.shared_expert.gate_proj        [shared_inter, D]
engine/transformer_block.py:59:      mlp.shared_expert.up_proj          [shared_inter, D]
engine/transformer_block.py:60:      mlp.shared_expert.down_proj        [D, shared_inter]
engine/transformer_block.py:61:      mlp.shared_expert_gate             [1, D]                   ← scalar gate
engine/transformer_block.py:140:    if "mlp.shared_expert.gate_proj" in weights:
engine/transformer_block.py:141:        gate_s = F.linear(h_flat, weights["mlp.shared_expert.gate_proj"])
engine/transformer_block.py:142:        up_s = F.linear(h_flat, weights["mlp.shared_expert.up_proj"])
engine/transformer_block.py:143:        shared_ffn = F.linear(F.silu(gate_s) * up_s, weights["mlp.shared_expert.down_proj"])
engine/transformer_block.py:144:        if "mlp.shared_expert_gate" in weights:
engine/transformer_block.py:145:            shared_gate_logit = F.linear(h_flat, weights["mlp.shared_expert_gate"])
engine/transformer_block.py:223:    if "mlp.shared_expert.gate_proj" in weights:
engine/transformer_block.py:224:        gate_s = F.linear(h_flat, weights["mlp.shared_expert.gate_proj"])
engine/transformer_block.py:225:        up_s = F.linear(h_flat, weights["mlp.shared_expert.up_proj"])
engine/transformer_block.py:226:        shared_ffn = F.linear(F.silu(gate_s) * up_s, weights["mlp.shared_expert.down_proj"])
engine/transformer_block.py:227:        if "mlp.shared_expert_gate" in weights:
engine/transformer_block.py:228:            shared_gate = torch.sigmoid(F.linear(h_flat, weights["mlp.shared_expert_gate"]))
engine/transformer_block.py:246:    SHARED_INT = config["shared_expert_intermediate_size"]
engine/transformer_block.py:270:        w["mlp.shared_expert.gate_proj"] = torch.randn(SHARED_INT, D, dtype=dtype, device=device) * 0.02
engine/transformer_block.py:271:        w["mlp.shared_expert.up_proj"] = torch.randn(SHARED_INT, D, dtype=dtype, device=device) * 0.02
engine/transformer_block.py:272:        w["mlp.shared_expert.down_proj"] = torch.randn(D, SHARED_INT, dtype=dtype, device=device) * 0.02
engine/transformer_block.py:273:        w["mlp.shared_expert_gate"] = torch.randn(1, D, dtype=dtype, device=device) * 0.02
engine/transformer_block.py:294:        "shared_expert_intermediate_size": 1408,
engine/moe_packed_nvfp4.py:19:from triton_kernels.shared_expert_gate import (
engine/moe_packed_nvfp4.py:21:    add_shared_expert_gate_from_scalar_triton,
engine/moe_packed_nvfp4.py:22:    apply_shared_expert_gate_triton,
engine/moe_packed_nvfp4.py:217:def _apply_shared_expert_gate(h_flat: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
engine/moe_packed_nvfp4.py:218:    if "mlp.shared_expert_gate.weight" not in w:
engine/moe_packed_nvfp4.py:222:        return shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
engine/moe_packed_nvfp4.py:224:        shared.mul_(torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"])))
engine/moe_packed_nvfp4.py:229:        return apply_shared_expert_gate_triton(shared, h_flat, w["mlp.shared_expert_gate.weight"])
engine/moe_packed_nvfp4.py:233:def _add_shared_expert_output(moe_out: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
engine/moe_packed_nvfp4.py:240:def _finalize_shared_expert_output(h_flat: torch.Tensor, moe_out: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
engine/moe_packed_nvfp4.py:242:    if backend == "torch_scalar_add_triton" and "mlp.shared_expert_gate.weight" in w:
engine/moe_packed_nvfp4.py:245:        gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
engine/moe_packed_nvfp4.py:246:        return add_shared_expert_gate_from_scalar_triton(moe_out, shared, gate)
engine/moe_packed_nvfp4.py:247:    shared = _apply_shared_expert_gate(h_flat, shared, w)
engine/moe_packed_nvfp4.py:248:    return _add_shared_expert_output(moe_out, shared)
engine/moe_packed_nvfp4.py:826:    if "mlp.shared_expert.gate_proj.weight" in w:
engine/moe_packed_nvfp4.py:827:        if "mlp.shared_expert._gate_up_proj.weight" in w:
engine/moe_packed_nvfp4.py:828:            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
engine/moe_packed_nvfp4.py:831:            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
engine/moe_packed_nvfp4.py:832:            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
engine/moe_packed_nvfp4.py:833:        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
engine/moe_packed_nvfp4.py:834:        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
engine/moe_packed_nvfp4.py:887:    if not _skip_shared_from_env() and "mlp.shared_expert.gate_proj.weight" in w:
engine/moe_packed_nvfp4.py:888:        if "mlp.shared_expert._gate_up_proj.weight" in w:
engine/moe_packed_nvfp4.py:889:            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
engine/moe_packed_nvfp4.py:892:            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
engine/moe_packed_nvfp4.py:893:            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
engine/moe_packed_nvfp4.py:894:        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
engine/moe_packed_nvfp4.py:895:        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
engine/moe_packed_nvfp4.py:1115:    if "mlp.shared_expert.gate_proj.weight" in w:
engine/moe_packed_nvfp4.py:1117:            "mlp.shared_expert.gate_proj.weight.packed" in w
engine/moe_packed_nvfp4.py:1118:            and "mlp.shared_expert.up_proj.weight.packed" in w
engine/moe_packed_nvfp4.py:1119:            and "mlp.shared_expert.down_proj.weight.packed" in w
engine/moe_packed_nvfp4.py:1123:                w["mlp.shared_expert.gate_proj.weight.packed"],
engine/moe_packed_nvfp4.py:1124:                w["mlp.shared_expert.up_proj.weight.packed"],
engine/moe_packed_nvfp4.py:1126:            shared = w["mlp.shared_expert.down_proj.weight.packed"](
engine/moe_packed_nvfp4.py:1131:            and "mlp.shared_expert._gate_up_proj.weight" in w
engine/moe_packed_nvfp4.py:1133:            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
engine/moe_packed_nvfp4.py:1135:            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
engine/moe_packed_nvfp4.py:1137:            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
engine/moe_packed_nvfp4.py:1138:            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
engine/moe_packed_nvfp4.py:1139:            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
engine/moe_packed_nvfp4.py:1140:        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
engine/resident_runner.py:247:        self.shared_expert_gate_up_fused_attached = 0
engine/resident_runner.py:319:            self._prepare_shared_expert_gate_up_fused()
engine/resident_runner.py:494:                shared_base = f"model.language_model.layers.{layer_idx}.mlp.shared_expert"
engine/resident_runner.py:496:                    w["mlp.shared_expert.gate_proj.weight.packed"] = load_packed_nvfp4_linear(
engine/resident_runner.py:501:                    w["mlp.shared_expert.up_proj.weight.packed"] = load_packed_nvfp4_linear(
engine/resident_runner.py:506:                    w["mlp.shared_expert.down_proj.weight.packed"] = load_packed_nvfp4_linear(
engine/resident_runner.py:539:                shared_base = f"model.language_model.layers.{layer_idx}.mlp.shared_expert"
engine/resident_runner.py:541:                    w["mlp.shared_expert.gate_proj.weight.packed"] = load_packed_nvfp4_linear(
engine/resident_runner.py:546:                    w["mlp.shared_expert.up_proj.weight.packed"] = load_packed_nvfp4_linear(
engine/resident_runner.py:551:                    w["mlp.shared_expert.down_proj.weight.packed"] = load_packed_nvfp4_linear(
engine/resident_runner.py:630:    def _prepare_shared_expert_gate_up_fused(self) -> None:
engine/resident_runner.py:639:            key = "mlp.shared_expert._gate_up_proj.weight"
engine/resident_runner.py:642:            if "mlp.shared_expert.gate_proj.weight" not in w or "mlp.shared_expert.up_proj.weight" not in w:
engine/resident_runner.py:646:                    w["mlp.shared_expert.gate_proj.weight"],
engine/resident_runner.py:647:                    w["mlp.shared_expert.up_proj.weight"],
engine/resident_runner.py:652:        self.shared_expert_gate_up_fused_attached = attached
Binary file engine/__pycache__/moe_optimized.cpython-314.pyc matches
Binary file engine/__pycache__/moe_packed_nvfp4.cpython-314.pyc matches
Binary file engine/__pycache__/resident_runner.cpython-314.pyc matches
Binary file engine/__pycache__/full_forward.cpython-314.pyc matches
Binary file engine/__pycache__/full_forward.cpython-312.pyc matches
Binary file engine/__pycache__/mtp_sidecar.cpython-314.pyc matches
engine/moe_repack_sidecar.py:49:            "mlp.shared_expert.gate_proj.weight": "shared_gate",
engine/moe_repack_sidecar.py:50:            "mlp.shared_expert.up_proj.weight": "shared_up",
engine/moe_repack_sidecar.py:51:            "mlp.shared_expert.down_proj.weight": "shared_down",
engine/moe_repack_sidecar.py:52:            "mlp.shared_expert_gate.weight": "shared_scalar_gate",
engine/qwen36_block.py:128:    if "mlp.shared_expert.gate_proj.weight" in w:
engine/qwen36_block.py:129:        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
engine/qwen36_block.py:130:        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
engine/qwen36_block.py:131:        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
engine/qwen36_block.py:132:        if "mlp.shared_expert_gate.weight" in w:
engine/qwen36_block.py:133:            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
engine/qwen36_block.py:213:    if "mlp.shared_expert.gate_proj.weight" in w:
engine/qwen36_block.py:214:        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
engine/qwen36_block.py:215:        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
engine/qwen36_block.py:216:        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
engine/qwen36_block.py:217:        if "mlp.shared_expert_gate.weight" in w:
engine/qwen36_block.py:218:            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
engine/mtp_sidecar.py:214:            "mlp.shared_expert.gate_proj.weight",
engine/mtp_sidecar.py:215:            "mlp.shared_expert.up_proj.weight",
engine/mtp_sidecar.py:216:            "mlp.shared_expert.down_proj.weight",
engine/mtp_sidecar.py:217:            "mlp.shared_expert_gate.weight",
engine/loader.py:259:    if "mlp.shared_expert.gate_proj.weight_fp8" in final:
engine/loader.py:260:        config["shared_intermediate"] = final["mlp.shared_expert.gate_proj.weight_fp8"].shape[0]
engine/loader.py:381:    if "mlp.shared_expert.gate_proj.weight" in final:
engine/loader.py:382:        config["shared_intermediate"] = final["mlp.shared_expert.gate_proj.weight"].shape[0]
engine/loader.py:494:    if "mlp.shared_expert.gate_proj.weight" in final:
engine/loader.py:495:        config["shared_intermediate"] = final["mlp.shared_expert.gate_proj.weight"].shape[0]
engine/loader.py:668:    if "mlp.shared_expert.gate_proj.weight" in final:
engine/loader.py:669:        config["shared_intermediate"] = final["mlp.shared_expert.gate_proj.weight"].shape[0]
engine/moe_optimized.py:159:    if "mlp.shared_expert.gate_proj.weight_fp8" in w:
engine/moe_optimized.py:162:            w["mlp.shared_expert.gate_proj.weight_fp8"],
engine/moe_optimized.py:163:            w["mlp.shared_expert.up_proj.weight_fp8"],
engine/moe_optimized.py:164:            w["mlp.shared_expert.gate_proj.weight_fp8_scale"].to(torch.float32),
engine/moe_optimized.py:165:            w["mlp.shared_expert.up_proj.weight_fp8_scale"].to(torch.float32),

exec
/bin/zsh -lc 'grep -R "_gate_up_proj.weight" -n engine triton_kernels benchmarks | head -80' in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
engine/moe_packed_nvfp4.py:827:        if "mlp.shared_expert._gate_up_proj.weight" in w:
engine/moe_packed_nvfp4.py:828:            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
engine/moe_packed_nvfp4.py:888:        if "mlp.shared_expert._gate_up_proj.weight" in w:
engine/moe_packed_nvfp4.py:889:            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
engine/moe_packed_nvfp4.py:1131:            and "mlp.shared_expert._gate_up_proj.weight" in w
engine/moe_packed_nvfp4.py:1133:            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
engine/resident_runner.py:639:            key = "mlp.shared_expert._gate_up_proj.weight"
engine/resident_runner.py:688:            key = "mlp._gate_up_proj.weight"
Binary file engine/__pycache__/moe_packed_nvfp4.cpython-314.pyc matches
Binary file engine/__pycache__/resident_runner.cpython-314.pyc matches
Binary file engine/__pycache__/full_forward.cpython-314.pyc matches
engine/full_forward.py:500:            fused = w.get("mlp._gate_up_proj.weight")
engine/full_forward.py:520:    fused = w.get("mlp._gate_up_proj.weight")
benchmarks/p167_qwen36_shared_expert_prepared_probe.py:77:    if "mlp.shared_expert._gate_up_proj.weight" in w:
benchmarks/p167_qwen36_shared_expert_prepared_probe.py:78:        gate_up = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
benchmarks/p167_qwen36_shared_expert_prepared_probe.py:103:    torch.mm(h_flat, w["mlp.shared_expert._gate_up_proj.weight"].t(), out=gate_up_scratch)
benchmarks/p167_qwen36_shared_expert_prepared_probe.py:117:    torch.mm(h_flat, w["mlp.shared_expert._gate_up_proj.weight"].t(), out=gate_up_scratch)
benchmarks/p167_qwen36_shared_expert_prepared_probe.py:170:        if "mlp.shared_expert._gate_up_proj.weight" not in w:
benchmarks/p167_qwen36_shared_expert_prepared_probe.py:171:            w["mlp.shared_expert._gate_up_proj.weight"] = torch.cat(
benchmarks/p167_qwen36_shared_expert_prepared_probe.py:176:        gate_up_scratch = torch.empty((1, w["mlp.shared_expert._gate_up_proj.weight"].shape[0]), device=args.device, dtype=dtype)
benchmarks/p162_qwen36_shared_finalize_probe.py:103:    if "mlp.shared_expert._gate_up_proj.weight" in w:
benchmarks/p162_qwen36_shared_finalize_probe.py:104:        gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
benchmarks/p40_moe_forward_fast_candidate_gate.py:73:    if "mlp.shared_expert._gate_up_proj.weight" in w:
benchmarks/p40_moe_forward_fast_candidate_gate.py:74:        gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
benchmarks/p155_qwen35_9b_dense_ffn_phase_profile.py:231:    fused = w.get("mlp._gate_up_proj.weight")
benchmarks/p23_active_moe_layer_sweep.py:93:        and "mlp.shared_expert._gate_up_proj.weight" in w
benchmarks/p23_active_moe_layer_sweep.py:95:        gate_up = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
benchmarks/p158_qwen36_moe_layer_component_profile.py:79:    if "mlp.shared_expert._gate_up_proj.weight" in w:
benchmarks/p158_qwen36_moe_layer_component_profile.py:80:        gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
benchmarks/p43_shared_expert_inner_profile.py:35:    gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
benchmarks/p43_shared_expert_inner_profile.py:59:    gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
benchmarks/p43_shared_expert_inner_profile.py:67:        "fused_gate_up_ms": _bench(lambda: F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"]), warmup, iters),

codex
There are three shared-expert occurrences in this file: the decode path at 826-834, another packed small-M path around 887-895, and a later packed/quantized branch around 1115-1140. Your request targets `_moe_forward_decode_packed_nvfp4_fixed_triton`, so I’ll keep the wiring scoped there and avoid changing verifier/MTP branches.
exec
/bin/zsh -lc "nl -ba engine/moe_packed_nvfp4.py | sed -n '620,840p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
   620	    gu_packed = w["mlp.experts._gate_up_packed"]       # [256, 1024, 1024]
   621	    gu_scale = w["mlp.experts._gate_up_scale"]         # [256, 1024, 128]
   622	    gu_global = w["mlp.experts._gate_up_global_scale"] # scalar
   623	    d_packed = w["mlp.experts._down_packed"]            # [256, 2048, 256]
   624	    d_scale = w["mlp.experts._down_scale"]              # [256, 2048, 32]
   625	    d_global = w["mlp.experts._down_global_scale"]      # scalar
   626	
   627	    W_fused_T = w["_graphsafe_v31_W_fused_T"]
   628	    W_down_T = w["_graphsafe_v31_W_down_T"]
   629	
   630	    ids = expert_ids.long()
   631	    # Dequant only the 8 selected experts
   632	    slot_gu_packed = gu_packed[ids]   # [8, 1024, 1024]
   633	    slot_gu_scale = gu_scale[ids]     # [8, 1024, 128]
   634	    slot_d_packed = d_packed[ids]     # [8, 2048, 256]
   635	    slot_d_scale = d_scale[ids]       # [8, 2048, 32]
   636	
   637	    slot_gu_bf16 = _dequant_nvfp4_slot(slot_gu_packed, slot_gu_scale, gu_global, device)  # [8, 1024, 2048]
   638	    slot_d_bf16 = _dequant_nvfp4_slot(slot_d_packed, slot_d_scale, d_global, device)      # [8, 2048, 512]
   639	
   640	    # Pretranspose into scratch
   641	    W_fused_T.copy_(slot_gu_bf16.reshape(8 * 1024, 2048).t())
   642	    W_down_T.copy_(slot_d_bf16.transpose(1, 2))
   643	
   644	
   645	def _active_moe_native_grouped_per16_nonatomic_out(
   646	    hidden: torch.Tensor,
   647	    expert_ids: torch.Tensor,
   648	    routing_weights: torch.Tensor,
   649	    w: dict,
   650	) -> torch.Tensor:
   651	    """Caller-owned scratch variant for CUDA graph capture probes."""
   652	    from engine.native_cuda import load_lynn_native_extension
   653	
   654	    inter_scratch = w.get("mlp.experts._active_inter_scratch")
   655	    out_scratch = w.get("mlp.experts._active_out_scratch")
   656	    if inter_scratch is None or out_scratch is None:
   657	        raise RuntimeError(
   658	            "LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16_nonatomic_out requires "
   659	            "LYNN_MOE_ACTIVE_SCRATCH=1 so resident_runner preallocates MoE scratch."
   660	        )
   661	
   662	    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
   663	    return ext.active_moe_grouped_per16_nonatomic_out_reference(
   664	        hidden,
   665	        expert_ids,
   666	        routing_weights,
   667	        w["mlp.experts._gate_up_packed"],
   668	        w["mlp.experts._gate_up_scale"],
   669	        w["mlp.experts._gate_up_global_scale"],
   670	        w["mlp.experts._down_packed"],
   671	        w["mlp.experts._down_scale"],
   672	        w["mlp.experts._down_global_scale"],
   673	        inter_scratch,
   674	        out_scratch,
   675	        _env_int("LYNN_NATIVE_GATEUP_TILE_INTER", 2),
   676	        _env_int("LYNN_NATIVE_DOWN_TILE_HIDDEN", 2),
   677	    )
   678	
   679	
   680	def _moe_forward_decode_packed_nvfp4_fixed_triton(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
   681	    """Fixed-config production fast path for the current R6000 best profile."""
   682	    h_flat = h.reshape(-1, h.shape[-1])
   683	    router_logits = _router_linear(h_flat, w)
   684	    top_k = int(cfg["num_experts_per_tok"])
   685	    routing_weights, expert_indices = _router_topk(
   686	        router_logits,
   687	        top_k,
   688	        sorted=False,
   689	        scratch_owner=w,
   690	    )
   691	    routing_weights = _router_softmax(routing_weights, scratch_owner=w)
   692	    expert_ids = expert_indices[0].to(torch.int32).contiguous()
   693	    limit = _topk_limit_from_env(top_k)
   694	    if limit != top_k:
   695	        expert_ids = expert_ids[:limit].contiguous()
   696	        routing_weights = routing_weights[:limit].contiguous()
   697	        if _env_bool("LYNN_MOE_TOPK_RENORMALIZE", True):
   698	            routing_weights = routing_weights / routing_weights.sum().clamp_min(1e-20)
   699	    hidden = h_flat[0]
   700	    w4a8_mode = _w4a8_fake_quant_mode()
   701	    if w4a8_mode in {"gateup", "full"}:
   702	        hidden = _fake_quant_fp8_activation(hidden)
   703	    gateup_backend = os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton")
   704	    prepared_triton = _env_bool("LYNN_MOE_TRITON_PREPARED", False)
   705	    if prepared_triton:
   706	        if gateup_backend != "triton_fast_decode":
   707	            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode")
   708	        inter_scratch = w.get("mlp.experts._active_inter_scratch")
   709	        if inter_scratch is None:
   710	            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
   711	        inter = nvfp4_grouped_gate_up_silu_fast_decode_prepared(
   712	            hidden,
   713	            expert_ids,
   714	            w["mlp.experts._gate_up_packed"],
   715	            w["mlp.experts._gate_up_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._gate_up_scale"],
   716	            w["mlp.experts._gate_up_global_scale"],
   717	            inter_scratch,
   718	            scale_effective=_use_moe_effective_scale(w),
   719	        )
   720	    elif gateup_backend == "split16_fp4" and _layer_selected_for_native_cuda(cfg):
   721	        inter = _gate_up_native_split16_fp4(hidden, expert_ids, w)
   722	    elif gateup_backend == "cuda_tile_inter" and _layer_selected_for_native_cuda(cfg):
   723	        inter = _gate_up_native_cuda_tile_inter(hidden, expert_ids, w)
   724	    elif gateup_backend == "triton_fast_decode":
   725	        gateup_fn = (
   726	            nvfp4_grouped_gate_up_silu_fast_decode_effective_scale
   727	            if _use_moe_effective_scale(w)
   728	            else nvfp4_grouped_gate_up_silu_fast_decode
   729	        )
   730	        inter = gateup_fn(
   731	            hidden,
   732	            expert_ids,
   733	            w["mlp.experts._gate_up_packed"],
   734	            w["mlp.experts._gate_up_effective_scale"]
   735	            if _use_moe_effective_scale(w)
   736	            else w["mlp.experts._gate_up_scale"],
   737	            w["mlp.experts._gate_up_global_scale"],
   738	            block_inter=8,
   739	            block_hidden=256,
   740	            num_warps=4,
   741	        )
   742	    elif gateup_backend == "triton":
   743	        inter = nvfp4_grouped_gate_up_silu(
   744	            hidden,
   745	            expert_ids,
   746	            w["mlp.experts._gate_up_packed"],
   747	            w["mlp.experts._gate_up_scale"],
   748	            w["mlp.experts._gate_up_global_scale"],
   749	            block_inter=8,
   750	            block_hidden=256,
   751	            num_warps=4,
   752	        )
   753	    elif gateup_backend in {"cuda_tile_inter", "split16_fp4"}:
   754	        inter = nvfp4_grouped_gate_up_silu(
   755	            hidden,
   756	            expert_ids,
   757	            w["mlp.experts._gate_up_packed"],
   758	            w["mlp.experts._gate_up_scale"],
   759	            w["mlp.experts._gate_up_global_scale"],
   760	            block_inter=8,
   761	            block_hidden=256,
   762	            num_warps=4,
   763	        )
   764	    else:
   765	        raise ValueError(
   766	            "LYNN_NATIVE_GATEUP_BACKEND must be 'triton', 'triton_fast_decode', 'cuda_tile_inter', "
   767	            "or 'split16_fp4', got "
   768	            f"{gateup_backend!r}"
   769	        )
   770	    if w4a8_mode == "full":
   771	        inter = _fake_quant_fp8_activation(inter)
   772	    down_backend = os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton")
   773	    if down_backend == "cuda_tile" and _layer_selected_for_native_cuda(cfg):
   774	        moe_out = _down_weighted_sum_native_cuda_tile(inter, expert_ids, routing_weights, w).reshape_as(h_flat)
   775	    elif prepared_triton:
   776	        if down_backend != "triton":
   777	            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_DOWN_BACKEND=triton")
   778	        out_scratch = w.get("mlp.experts._active_out_scratch")
   779	        if out_scratch is None:
   780	            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
   781	        moe_out = nvfp4_grouped_down_weighted_sum_prepared(
   782	            inter,
   783	            expert_ids,
   784	            routing_weights,
   785	            w["mlp.experts._down_packed"],
   786	            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
   787	            w["mlp.experts._down_global_scale"],
   788	            out_scratch,
   789	            scale_effective=_use_moe_effective_scale(w),
   790	        ).reshape_as(h_flat)
   791	    elif down_backend == "triton":
   792	        down_fn = (
   793	            nvfp4_grouped_down_weighted_sum_effective_scale
   794	            if _use_moe_effective_scale(w)
   795	            else nvfp4_grouped_down_weighted_sum
   796	        )
   797	        moe_out = down_fn(
   798	            inter,
   799	            expert_ids,
   800	            routing_weights,
   801	            w["mlp.experts._down_packed"],
   802	            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
   803	            w["mlp.experts._down_global_scale"],
   804	            block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
   805	            block_inter=512,
   806	            num_warps=8,
   807	        ).reshape_as(h_flat)
   808	    elif down_backend == "cuda_tile":
   809	        moe_out = nvfp4_grouped_down_weighted_sum(
   810	            inter,
   811	            expert_ids,
   812	            routing_weights,
   813	            w["mlp.experts._down_packed"],
   814	            w["mlp.experts._down_scale"],
   815	            w["mlp.experts._down_global_scale"],
   816	            block_hidden=8,
   817	            block_inter=512,
   818	            num_warps=8,
   819	        ).reshape_as(h_flat)
   820	    else:
   821	        raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")
   822	
   823	    if _skip_shared_from_env():
   824	        return moe_out.to(h.dtype).reshape_as(h)
   825	
   826	    if "mlp.shared_expert.gate_proj.weight" in w:
   827	        if "mlp.shared_expert._gate_up_proj.weight" in w:
   828	            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
   829	            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
   830	        else:
   831	            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
   832	            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
   833	        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
   834	        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
   835	    return moe_out.to(h.dtype).reshape_as(h)
   836	
   837	
   838	def moe_forward_verify_smallm_nvfp4(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
   839	    """Small-M grouped MoE for the MTP spec-verify block (M = K_draft+1).
   840	

exec
/bin/zsh -lc "nl -ba engine/resident_runner.py | sed -n '220,340p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
   220	        self,
   221	        model_dir: str,
   222	        *,
   223	        device: str = "cuda",
   224	        dtype: torch.dtype = torch.bfloat16,
   225	        max_seq_len: int = 2048,
   226	        verbose: bool = True,
   227	    ) -> None:
   228	        impl = os.environ.get("LYNN_MOE_IMPL", "optimized")
   229	        if impl == "indexed_bmm":
   230	            raise ValueError(
   231	                "LynnIncrementalRunner supports reusable prompts only with "
   232	                "LYNN_MOE_IMPL=optimized, bmm, triton, or packed_nvfp4. indexed_bmm mutates "
   233	                "weights after prefill and is single-prompt only today."
   234	            )
   235	        self.model_dir = str(model_dir)
   236	        self.device = device
   237	        self.dtype = dtype
   238	        self.max_seq_len = max_seq_len
   239	        self.verbose = verbose
   240	        self.moe_impl = impl
   241	        self.decode_moe_fn = _resolve_decode_moe_impl(impl)
   242	        self.decode_recurrent_backend = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch")
   243	        linear_graph_enabled = os.environ.get("LYNN_LINEAR_BLOCK_GRAPH", "0") == "1"
   244	        state_update_default = "inplace" if linear_graph_enabled else "assign"
   245	        self.decode_linear_state_update = os.environ.get("LYNN_LINEAR_STATE_UPDATE", state_update_default)
   246	        self.decode_fast_dispatch = os.environ.get("LYNN_DECODE_FAST_DISPATCH", "1") != "0"
   247	        self.shared_expert_gate_up_fused_attached = 0
   248	        self.dense_ffn_gate_up_fused_attached = 0
   249	        self.dense_ffn_true_fp8_attached = 0
   250	        self.packed_nvfp4_moe_aliases_attached = 0
   251	        self.moe_repack_sidecar_dir = os.environ.get("LYNN_MOE_REPACK_SIDECAR_DIR") or None
   252	        self.moe_repack_sidecar_layers_attached = 0
   253	        self.moe_repack_sidecar_load_seconds: float | None = None
   254	        self.moe_active_scratch_attached = 0
   255	        self.moe_effective_scale_attached = 0
   256	        self.router_linear_out_buffer_attached = 0
   257	        self.packed_decode_backend = os.environ.get("LYNN_PACKED_DECODE_BACKEND", "scalar_bridge")
   258	        self.packed_decode_aliases_attached = 0
   259	        self.packed_decode_native_prepared = 0
   260	        self.packed_decode_aliases_skipped = 0
   261	        self.runtime_warnings: list[str] = []
   262	        self.cfg, self.n_layers = _runtime_config(self.model_dir)
   263	        self.is_moe = bool(self.cfg.get("is_moe", True))
   264	        self.layer_types = list(self.cfg.get("layer_types", LAYER_TYPES))
   265	        if not self.is_moe and verbose:
   266	            print("[resident] dense FFN model detected; MoE decode paths disabled", flush=True)
   267	        if os.environ.get("LYNN_PACKED_DECODE", "0") == "1":
   268	            self.runtime_warnings.append(
   269	                "LYNN_PACKED_DECODE=1 is a diagnostic path, not the current R6000 best "
   270	                "profile. P15 measured it regressing full graph path from ~103.48 tok/s "
   271	                "to ~88.15 tok/s. Prefer LYNN_PACKED_DECODE=0 with "
   272	                "LYNN_MOE_IMPL=packed_nvfp4 and LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1."
   273	            )
   274	        if os.environ.get("LYNN_PACKED_SHARED_EXPERT", "0") == "1":
   275	            self.runtime_warnings.append(
   276	                "LYNN_PACKED_SHARED_EXPERT=1 is slower than the BF16 shared expert path "
   277	                "on R6000 P15 profiles; keep it disabled unless explicitly benchmarking."
   278	            )
   279	
   280	        from transformers import AutoTokenizer
   281	
   282	        t0 = time.time()
   283	        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
   284	        stop_ids = set()
   285	        if self.tokenizer.eos_token_id is not None:
   286	            stop_ids.add(int(self.tokenizer.eos_token_id))
   287	        for token in ("<|im_end|>", "<|endoftext|>"):
   288	            token_id = self.tokenizer.convert_tokens_to_ids(token)
   289	            if token_id is not None and token_id != self.tokenizer.unk_token_id:
   290	                stop_ids.add(int(token_id))
   291	        self.stop_token_ids = stop_ids
   292	        if verbose:
   293	            print(f"[resident] loading outside weights from {self.model_dir}", flush=True)
   294	            for warning in self.runtime_warnings:
   295	                print(f"[resident][warning] {warning}", flush=True)
   296	        self.outside = load_outside_weights(self.model_dir, device, dtype)
   297	
   298	        if verbose:
   299	            print(f"[resident] loading {self.n_layers} layers", flush=True)
   300	        self.layer_weights = []
   301	        self.layer_cfgs = []
   302	        if device.startswith("cuda"):
   303	            torch.cuda.reset_peak_memory_stats()
   304	        for i in range(self.n_layers):
   305	            w, inferred = load_qwen36_layer(
   306	                self.model_dir,
   307	                i,
   308	                num_experts=self.cfg["num_experts"],
   309	                device=device,
   310	                dequant_dtype=dtype,
   311	            )
   312	            self.layer_weights.append(w)
   313	            self.layer_cfgs.append(_with_inferred_layer_config(self.cfg, inferred, i))
   314	            if verbose and (i % 5 == 4 or i == self.n_layers - 1):
   315	                print(f"  [resident] L{i:02}: {time.time() - t0:.1f}s", flush=True)
   316	        if self.is_moe and impl == "triton":
   317	            self._prepare_triton_moe_layout()
   318	        if self.is_moe and os.environ.get("LYNN_SHARED_EXPERT_GATE_UP_FUSED", "1") != "0":
   319	            self._prepare_shared_expert_gate_up_fused()
   320	        if (not self.is_moe) and os.environ.get("LYNN_DENSE_FFN_GATE_UP_FUSED", "0") == "1":
   321	            self._prepare_dense_ffn_gate_up_fused()
   322	        dense_true_fp8_mode = os.environ.get("LYNN_DENSE_FFN_TRUE_FP8", "0").strip().lower()
   323	        if (not self.is_moe) and dense_true_fp8_mode not in {"0", "off", "false", "no"}:
   324	            self._prepare_dense_ffn_true_fp8()
   325	        if self.is_moe and impl == "packed_nvfp4":
   326	            self._prepare_packed_nvfp4_moe_layout()
   327	        if self.is_moe and os.environ.get("LYNN_ROUTER_LINEAR_OUT_BUFFER", "0") == "1":
   328	            self._prepare_router_linear_out_buffer()
   329	        if (
   330	            os.environ.get("LYNN_PACKED_DECODE", "0") == "1"
   331	            or os.environ.get("LYNN_PACKED_DECODE_LINEAR_ATTN", "0") == "1"
   332	            or os.environ.get("LYNN_PACKED_DECODE_FULL_ATTN", "0") == "1"
   333	        ):
   334	            self._prepare_packed_decode_aliases()
   335	        if os.environ.get("LYNN_LINEAR_ATTN_INPROJ_FUSED", "0") == "1":
   336	            self._prepare_linear_attn_inproj_fused()
   337	        if os.environ.get("LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4", "0") == "1":
   338	            self._prepare_linear_attn_inproj_fused_native_fp4()
   339	        if os.environ.get("LYNN_FULL_ATTN_QKV_FUSED", "0") == "1":
   340	            self._prepare_full_attn_qkv_fused()

exec
/bin/zsh -lc "nl -ba engine/moe_packed_nvfp4.py | sed -n '860,920p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
   860	        rw, ei = _router_topk(rl, top_k, sorted=sorted_topk, scratch_owner=w)
   861	        rw = _router_softmax(rw, scratch_owner=w)
   862	        eids[i] = ei[0].to(torch.long)
   863	        wts[i] = rw.reshape(-1).to(torch.float32)
   864	
   865	    gu_packed = w["mlp.experts._gate_up_packed"]
   866	    gu_scale = w["mlp.experts._gate_up_scale"]
   867	    gu_g = w["mlp.experts._gate_up_global_scale"]
   868	    d_packed = w["mlp.experts._down_packed"]
   869	    d_scale = w["mlp.experts._down_scale"]
   870	    d_g = w["mlp.experts._down_global_scale"]
   871	
   872	    x32 = h_flat.to(torch.float32)
   873	    moe_out = torch.zeros(M, hidden, device=dev, dtype=torch.float32)
   874	    for e in torch.unique(eids).tolist():
   875	        sel = eids == e  # [M, top_k]
   876	        rows = sel.any(dim=1).nonzero(as_tuple=False).flatten()
   877	        rw_e = (wts * sel.to(torch.float32)).sum(dim=1)[rows]  # [m_e]
   878	        xr = x32[rows]  # [m_e, hidden]
   879	        gu = _dequant_nvfp4_slot(gu_packed[e], gu_scale[e], gu_g, dev).to(torch.float32)  # [2*inter, hidden]
   880	        inter = xr @ gu.t()  # [m_e, 2*inter]
   881	        half = inter.shape[-1] // 2
   882	        act = F.silu(inter[:, :half]) * inter[:, half:]  # [m_e, inter]
   883	        dn = _dequant_nvfp4_slot(d_packed[e], d_scale[e], d_g, dev).to(torch.float32)  # [hidden, inter]
   884	        moe_out[rows] += (act @ dn.t()) * rw_e[:, None]
   885	
   886	    moe_out = moe_out.to(out_dtype)  # bf16, matching the packed kernel's output
   887	    if not _skip_shared_from_env() and "mlp.shared_expert.gate_proj.weight" in w:
   888	        if "mlp.shared_expert._gate_up_proj.weight" in w:
   889	            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
   890	            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
   891	        else:
   892	            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
   893	            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
   894	        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
   895	        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
   896	    return moe_out.reshape_as(h)
   897	
   898	
   899	def moe_forward_decode_packed_nvfp4(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
   900	    """Decode-only MoE using packed NVFP4 expert weights.
   901	
   902	    Router and shared expert stay on the existing BF16 resident path for now;
   903	    active routed experts consume packed NVFP4 weights directly. This is the
   904	    first production-shaped bridge from P10-H into the resident runner.
   905	    """
   906	    if h.shape[1] != 1:
   907	        raise NotImplementedError("packed NVFP4 MoE path is decode-only")
   908	    required = (
   909	        "mlp.experts._gate_up_packed",
   910	        "mlp.experts._gate_up_scale",
   911	        "mlp.experts._gate_up_global_scale",
   912	        "mlp.experts._down_packed",
   913	        "mlp.experts._down_scale",
   914	        "mlp.experts._down_global_scale",
   915	    )
   916	    missing = [key for key in required if key not in w]
   917	    if missing:
   918	        raise KeyError(f"packed NVFP4 MoE aliases missing: {missing}")
   919	
   920	    h_flat = h.reshape(-1, h.shape[-1])

exec
/bin/zsh -lc "nl -ba engine/resident_runner.py | sed -n '620,660p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
   620	            device=self.device,
   621	            dtype=torch.bfloat16,
   622	        )
   623	        w["mlp.experts._active_out_scratch"] = torch.empty(
   624	            (2048,),
   625	            device=self.device,
   626	            dtype=torch.bfloat16,
   627	        )
   628	        self.moe_active_scratch_attached += 1
   629	
   630	    def _prepare_shared_expert_gate_up_fused(self) -> None:
   631	        """Attach BF16 fused shared-expert gate/up weights.
   632	
   633	        The shared expert stays BF16 on R6000 because packed shared paths are
   634	        slower. Fusing gate/up keeps the same BF16 math while replacing two
   635	        small GEMM launches with one larger GEMM.
   636	        """
   637	        attached = 0
   638	        for w in self.layer_weights:
   639	            key = "mlp.shared_expert._gate_up_proj.weight"
   640	            if key in w:
   641	                continue
   642	            if "mlp.shared_expert.gate_proj.weight" not in w or "mlp.shared_expert.up_proj.weight" not in w:
   643	                continue
   644	            w[key] = torch.cat(
   645	                [
   646	                    w["mlp.shared_expert.gate_proj.weight"],
   647	                    w["mlp.shared_expert.up_proj.weight"],
   648	                ],
   649	                dim=0,
   650	            ).contiguous()
   651	            attached += 1
   652	        self.shared_expert_gate_up_fused_attached = attached
   653	        if self.verbose:
   654	            print(f"[resident] shared expert fused gate/up attached={attached}", flush=True)
   655	
   656	    def _prepare_router_linear_out_buffer(self) -> None:
   657	        """Attach router projection transposed weights and logits scratch.
   658	
   659	        P177 proved the decode-shape router can use `torch.mm(..., out=...)`
   660	        exactly when the transposed weight and output tensor are caller-owned.

exec
/bin/zsh -lc "nl -ba engine/moe_packed_nvfp4.py | sed -n '1090,1150p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
  1090	                    w["mlp.experts._down_packed"],
  1091	                    w["mlp.experts._down_effective_scale"]
  1092	                    if _use_moe_effective_scale(w)
  1093	                    else w["mlp.experts._down_scale"],
  1094	                    w["mlp.experts._down_global_scale"],
  1095	                    block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
  1096	                    block_inter=_env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
  1097	                    num_warps=_env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
  1098	                    out=out_scratch,
  1099	                ).reshape_as(h_flat)
  1100	            else:
  1101	                raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")
  1102	        else:
  1103	            raise ValueError(
  1104	                "LYNN_NATIVE_ACTIVE_MOE_BACKEND must be 'triton', 'cuda_scalar', "
  1105	                "'cuda_scalar_contract', 'grouped_per16', 'grouped_per16_fused', "
  1106	                "'grouped_per16_nonatomic', 'grouped_per16_nonatomic_out', "
  1107	                "'packed_pretransposed_graphsafe_v31', "
  1108	                "'packed_pretransposed_graphsafe_v32_ordered', "
  1109	                f"got {backend!r}"
  1110	            )
  1111	
  1112	    if os.environ.get("LYNN_MOE_PROFILE_SKIP_SHARED", "0") == "1":
  1113	        return moe_out.to(h.dtype).reshape_as(h)
  1114	
  1115	    if "mlp.shared_expert.gate_proj.weight" in w:
  1116	        if (
  1117	            "mlp.shared_expert.gate_proj.weight.packed" in w
  1118	            and "mlp.shared_expert.up_proj.weight.packed" in w
  1119	            and "mlp.shared_expert.down_proj.weight.packed" in w
  1120	        ):
  1121	            gate_s, up_s = dual_scalar_bridge(
  1122	                h_flat[0],
  1123	                w["mlp.shared_expert.gate_proj.weight.packed"],
  1124	                w["mlp.shared_expert.up_proj.weight.packed"],
  1125	            )
  1126	            shared = w["mlp.shared_expert.down_proj.weight.packed"](
  1127	                (F.silu(gate_s) * up_s).to(h.dtype)
  1128	            ).reshape_as(h_flat)
  1129	        elif (
  1130	            _env_bool("LYNN_SHARED_EXPERT_GATE_UP_FUSED", True)
  1131	            and "mlp.shared_expert._gate_up_proj.weight" in w
  1132	        ):
  1133	            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
  1134	            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
  1135	            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
  1136	        else:
  1137	            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
  1138	            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
  1139	            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
  1140	        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
  1141	
  1142	    return moe_out.to(h.dtype).reshape_as(h)

codex
The clean wiring point is a helper before the existing eager block: try the fused BF16 decode path only when `LYNN_SHARED_EXPERT_FUSED=1`, M=1, CUDA, BF16, `_gate_up_proj.weight` is attached, and the down/scalar-gate weights match. Anything else falls straight through to lines 827-834, so the current BF16 behavior and gate backend flags remain intact when the new flag is off.
exec
/bin/zsh -lc "sed -n '1,80p' triton_kernels/__init__.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
"""Local Lynn Engine Triton kernels.

This package marker prevents an installed third-party ``triton_kernels``
package from shadowing the repository-local kernels when benchmarks are run
from conda environments that already ship a package with the same name.
"""

exec
/bin/zsh -lc "grep -R \"_active_inter_scratch\\|_active_out_scratch\" -n engine/resident_runner.py engine/moe_packed_nvfp4.py | head -120" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
engine/resident_runner.py:618:        w["mlp.experts._active_inter_scratch"] = torch.empty(
engine/resident_runner.py:623:        w["mlp.experts._active_out_scratch"] = torch.empty(
engine/moe_packed_nvfp4.py:654:    inter_scratch = w.get("mlp.experts._active_inter_scratch")
engine/moe_packed_nvfp4.py:655:    out_scratch = w.get("mlp.experts._active_out_scratch")
engine/moe_packed_nvfp4.py:708:        inter_scratch = w.get("mlp.experts._active_inter_scratch")
engine/moe_packed_nvfp4.py:778:        out_scratch = w.get("mlp.experts._active_out_scratch")
engine/moe_packed_nvfp4.py:1021:                    w.get("mlp.experts._active_inter_scratch")
engine/moe_packed_nvfp4.py:1077:                    w.get("mlp.experts._active_out_scratch")

exec
/bin/zsh -lc "grep -R \"LYNN_SHARED_EXPERT_GATE_BACKEND\\|LYNN_SHARED_EXPERT_FUSED\\|SHARED_EXPERT_FUSED\" -n . --exclude-dir=.git | head -80" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 1648ms:
./docs/QWEN36_DECODE_LAUNCH_CENSUS_20260519.md:25:PROFILE_ENV='LYNN_SHARED_EXPERT_GATE_BACKEND=triton LYNN_LINEAR_ATTN_CONV_BACKEND=triton_inplace' \
./benchmarks/p37_moe_config_generate_gate.py:62:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./benchmarks/p26_decode_phase_profile.py:91:        "LYNN_SHARED_EXPERT_GATE_BACKEND",
./benchmarks/p143_resident_moe_graphsafe_p37_probe.py:57:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./benchmarks/p144_decode_launch_census.py:30:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./benchmarks/p146_resident_moe_backend_p37_probe.py:59:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./benchmarks/p28_hybrid_block_timing_profile.py:66:        "LYNN_SHARED_EXPERT_GATE_BACKEND",
./benchmarks/p145_resident_moe_v32_ordered_p37_probe.py:64:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./benchmarks/p43_shared_expert_inner_profile.py:140:        "shared_expert_gate_backend": os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch"),
./scripts/r6000_qwen36_candidate_promotion_gate.sh:8:#   CANDIDATE_ENV='LYNN_SHARED_EXPERT_GATE_BACKEND=triton LYNN_LINEAR_ATTN_CONV_BACKEND=triton_inplace' \
./scripts/r6000_qwen36_candidate_promotion_gate.sh:91:  export LYNN_SHARED_EXPERT_GATE_BACKEND=torch
./scripts/qwen36_candidate_env_amber_sharedgate_convinplace.env:11:LYNN_SHARED_EXPERT_GATE_BACKEND=triton
./scripts/r6000_qwen36_linear_core_segment_census.sh:43:export LYNN_SHARED_EXPERT_GATE_BACKEND=torch
./engine/moe_packed_nvfp4.py:220:    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
./engine/moe_packed_nvfp4.py:228:            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=triton requires Triton")
./engine/moe_packed_nvfp4.py:230:    raise ValueError("LYNN_SHARED_EXPERT_GATE_BACKEND must be 'torch', 'torch_inplace', or 'triton', got " f"{backend!r}")
./engine/moe_packed_nvfp4.py:241:    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
./engine/moe_packed_nvfp4.py:244:            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=torch_scalar_add_triton requires Triton")
Binary file ./engine/__pycache__/moe_packed_nvfp4.cpython-314.pyc matches
./reports/qwen36_35b/p37_shared_gate_torch_inplace_20260518_090215.json:6:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_inplace"
./reports/qwen36_35b/p37_shared_gate_torch_inplace_20260518_090215.json:755:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_inplace"
./reports/qwen36_35b/p37_shared_gate_torch_inplace_20260518_090215.json:946:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_inplace"
./reports/qwen36_35b/p37_shared_gate_torch_inplace_20260518_090215.json:1374:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_inplace"
./reports/qwen36_35b/r6000_qwen36_w4a16_moe_repack_sidecar_20260518_164845_moe_repack_runtime_p28.json:21:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./reports/qwen36_35b/codex_fusion_analysis.md:773:    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
./reports/qwen36_35b/codex_fusion_analysis.md:781:            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=triton requires Triton")
./reports/qwen36_35b/codex_fusion_analysis.md:783:    raise ValueError("LYNN_SHARED_EXPERT_GATE_BACKEND must be 'torch', 'torch_inplace', or 'triton', got " f"{backend!r}")
./reports/qwen36_35b/codex_fusion_analysis.md:794:    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
./reports/qwen36_35b/codex_fusion_analysis.md:797:            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=torch_scalar_add_triton requires Triton")
./reports/qwen36_35b/p37_shared_scalar_add_triton_20260518_125634_shared_scalar_add.json:6:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_scalar_add_triton"
./reports/qwen36_35b/p37_shared_scalar_add_triton_20260518_125634_shared_scalar_add.json:755:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_scalar_add_triton"
./reports/qwen36_35b/p37_shared_scalar_add_triton_20260518_125634_shared_scalar_add.json:932:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_scalar_add_triton"
./reports/qwen36_35b/p37_shared_scalar_add_triton_20260518_125634_shared_scalar_add.json:1360:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_scalar_add_triton"
./reports/qwen36_35b/p37_shared_gate_triton_20260518_083325.json:6:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton"
./reports/qwen36_35b/p37_shared_gate_triton_20260518_083325.json:755:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton"
./reports/qwen36_35b/p37_shared_gate_triton_20260518_083325.json:936:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton"
./reports/qwen36_35b/p37_shared_gate_triton_20260518_083325.json:1364:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton"
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_moe_ff0_20260518_134652_p37_fast122_sweep.json:11:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_moe_ff0_20260518_134652_p37_fast122_sweep.json:773:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_moe_ff0_20260518_134652_p37_fast122_sweep.json:958:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_moe_ff0_20260518_134652_p37_fast122_sweep.json:1390:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_sharedgate_convinplace_20260518_131127_promotion_amber_promotion_summary.json:6:    "LYNN_SHARED_EXPERT_GATE_BACKEND=triton",
./reports/qwen36_35b/p26_amber_rope_profile_20260518_124620.json:23:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/p28_amber_rope_profile_20260518_124620.json:21:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_shared_torch_inplace_20260519_0710_shared_torch_inplace_promotion_summary.json:6:    "LYNN_SHARED_EXPERT_GATE_BACKEND=torch_inplace",
./reports/qwen36_35b/p144_safe_fastfixed_20260519_030042.json:12:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_addshared_20260518_1555_amber_addshared_profile_p28.json:21:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_packed_linear_20260518_134652_p37_fast122_sweep.json:11:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_packed_linear_20260518_134652_p37_fast122_sweep.json:774:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_packed_linear_20260518_134652_p37_fast122_sweep.json:960:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_packed_linear_20260518_134652_p37_fast122_sweep.json:1393:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/p144_fastfixed0_20260519_030042.json:12:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./reports/qwen36_35b/r6000_qwen36_w4a16_triton_prepared_20260519_1158_triton_prepared_p28.json:21:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./reports/qwen36_35b/r6000_qwen36_w4a16_shared_scalar_add_triton_20260518_135311_p37_micro122_sweep.json:11:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_scalar_add_triton"
./reports/qwen36_35b/r6000_qwen36_w4a16_shared_scalar_add_triton_20260518_135311_p37_micro122_sweep.json:771:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_scalar_add_triton"
./reports/qwen36_35b/r6000_qwen36_w4a16_shared_scalar_add_triton_20260518_135311_p37_micro122_sweep.json:950:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_scalar_add_triton"
./reports/qwen36_35b/r6000_qwen36_w4a16_shared_scalar_add_triton_20260518_135311_p37_micro122_sweep.json:1380:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_scalar_add_triton"
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_sharedgate_convinplace_70_20260518_1530_amber70_promotion_summary.json:6:    "LYNN_SHARED_EXPERT_GATE_BACKEND=triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_moe_repack_scratch_20260518_165518_moe_repack_scratch_p26.json:23:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./reports/qwen36_35b/p37_moe_inplace_combo_20260518_092445.json:6:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_inplace",
./reports/qwen36_35b/p37_moe_inplace_combo_20260518_092445.json:756:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_inplace",
./reports/qwen36_35b/p37_moe_inplace_combo_20260518_092445.json:948:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_inplace",
./reports/qwen36_35b/p37_moe_inplace_combo_20260518_092445.json:1377:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch_inplace",
./reports/qwen36_35b/r6000_qwen36_w4a16_triton_prepared_20260519_1158_triton_prepared_p26.json:23:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_add_shared_inplace_20260518_135311_p37_micro122_sweep.json:11:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_add_shared_inplace_20260518_135311_p37_micro122_sweep.json:773:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_add_shared_inplace_20260518_135311_p37_micro122_sweep.json:958:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_add_shared_inplace_20260518_135311_p37_micro122_sweep.json:1390:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_sharedgate_convinplace_20260518_131127_promotion_amber_p37.json:11:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_sharedgate_convinplace_20260518_131127_promotion_amber_p37.json:772:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_sharedgate_convinplace_20260518_131127_promotion_amber_p37.json:956:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_sharedgate_convinplace_20260518_131127_promotion_amber_p37.json:1387:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_addshared_20260518_1555_amber_addshared_profile_p26.json:23:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_topk7_20260518_135311_p37_micro122_sweep.json:11:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_topk7_20260518_135311_p37_micro122_sweep.json:774:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_topk7_20260518_135311_p37_micro122_sweep.json:956:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_topk7_20260518_135311_p37_micro122_sweep.json:1389:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_moe_repack_scratch_20260518_165518_moe_repack_scratch_p28.json:21:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_sharedgate_convinplace_addshared_20260518_1535_amber_addshared_combo_p37.json:11:    "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",
./reports/qwen36_35b/r6000_qwen36_w4a16_amber_sharedgate_convinplace_addshared_20260518_1535_amber_addshared_combo_p37.json:773:          "LYNN_SHARED_EXPERT_GATE_BACKEND": "triton",

codex
I can’t apply edits in this session because the workspace is mounted read-only, and I did not run anything beyond file reads. Here is the exact scoped change I’d make.

**Where**
Target is [engine/moe_packed_nvfp4.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/moe_packed_nvfp4.py:680), specifically the eager shared block at lines 826-834. Existing finalize helpers stay at lines 217-248 and remain the fallback path.

**Design**
Use 2 Triton launches, not 1. A true 1-kernel version either recomputes `gate_up` once per output block or needs cross-program synchronization Triton does not provide. The sane launch-minimal version is:

1. `h @ gate_up.T -> chunk -> silu(gate)*up`, stored BF16 `[1, I]`
2. `inter @ down.T`, plus optional `sigmoid(h @ shared_expert_gate.T)` and `moe_out + shared`, stored BF16 into `moe_out`

This replaces the shared expert’s current 4-5+ eager launches per MoE layer with 2 launches under `LYNN_SHARED_EXPERT_FUSED=1`.

```diff
diff --git a/engine/moe_packed_nvfp4.py b/engine/moe_packed_nvfp4.py
@@
 from triton_kernels.shared_expert_gate import (
     HAS_TRITON as HAS_SHARED_EXPERT_GATE_TRITON,
     add_shared_expert_gate_from_scalar_triton,
     apply_shared_expert_gate_triton,
 )
+from triton_kernels.shared_expert_fused import (
+    HAS_TRITON as HAS_SHARED_EXPERT_FUSED_TRITON,
+    shared_expert_decode_fused_triton,
+)
@@
 def _finalize_shared_expert_output(h_flat: torch.Tensor, moe_out: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
@@
     shared = _apply_shared_expert_gate(h_flat, shared, w)
     return _add_shared_expert_output(moe_out, shared)
+
+
+def _try_fused_shared_expert_output(h_flat: torch.Tensor, moe_out: torch.Tensor, w: dict) -> torch.Tensor | None:
+    if not _env_bool("LYNN_SHARED_EXPERT_FUSED", False):
+        return None
+    if not HAS_SHARED_EXPERT_FUSED_TRITON:
+        raise RuntimeError("LYNN_SHARED_EXPERT_FUSED=1 requires Triton")
+    gate_up = w.get("mlp.shared_expert._gate_up_proj.weight")
+    down = w.get("mlp.shared_expert.down_proj.weight")
+    if gate_up is None or down is None:
+        return None
+    if h_flat.ndim != 2 or h_flat.shape[0] != 1 or moe_out.shape != h_flat.shape:
+        return None
+    if not h_flat.is_cuda or h_flat.dtype != torch.bfloat16 or moe_out.dtype != torch.bfloat16:
+        return None
+    if gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
+        return None
+    hidden = h_flat.shape[1]
+    if gate_up.ndim != 2 or gate_up.shape[1] != hidden or gate_up.shape[0] % 2 != 0:
+        return None
+    inter = gate_up.shape[0] // 2
+    if down.ndim != 2 or down.shape != (hidden, inter):
+        return None
+    gate = w.get("mlp.shared_expert_gate.weight")
+    if gate is not None and (gate.ndim != 2 or gate.shape != (1, hidden) or gate.dtype != torch.bfloat16):
+        return None
+    return shared_expert_decode_fused_triton(
+        h_flat,
+        moe_out,
+        gate_up,
+        down,
+        gate,
+        inter=w.get("mlp.shared_expert._inter_scratch"),
+        block_hidden=_env_int("LYNN_SHARED_EXPERT_FUSED_BLOCK_HIDDEN", 128),
+        block_inter=_env_int("LYNN_SHARED_EXPERT_FUSED_BLOCK_INTER", 32),
+        block_out=_env_int("LYNN_SHARED_EXPERT_FUSED_BLOCK_OUT", 32),
+        num_warps=_env_int("LYNN_SHARED_EXPERT_FUSED_NUM_WARPS", 4),
+    )
@@
     if "mlp.shared_expert.gate_proj.weight" in w:
-        if "mlp.shared_expert._gate_up_proj.weight" in w:
+        fused_shared = _try_fused_shared_expert_output(h_flat, moe_out, w)
+        if fused_shared is not None:
+            moe_out = fused_shared
+        elif "mlp.shared_expert._gate_up_proj.weight" in w:
             gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
             gate_s, up_s = gate_up_s.chunk(2, dim=-1)
         else:
             gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
             up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
-        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
-        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
+        if fused_shared is None:
+            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
+            moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
```

Add `triton_kernels/shared_expert_fused.py`:

```python
"""Decode-time fused BF16 shared-expert kernels."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for fused shared expert kernels")


if HAS_TRITON:

    @triton.jit
    def _shared_gate_up_silu_kernel(
        x_ptr, gate_up_ptr, inter_ptr,
        HIDDEN: tl.constexpr, INTER: tl.constexpr,
        stride_xm: tl.constexpr, stride_xh: tl.constexpr,
        stride_gr: tl.constexpr, stride_gh: tl.constexpr,
        stride_im: tl.constexpr, stride_ii: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_I: tl.constexpr,
    ):
        pid_i = tl.program_id(0)
        offs_m = tl.arange(0, BLOCK_M)
        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        offs_h = tl.arange(0, BLOCK_H)
        mask_m = offs_m < 1
        mask_i = offs_i < INTER

        acc_g = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

        for h0 in range(0, HIDDEN, BLOCK_H):
            h = h0 + offs_h
            mask_h = h < HIDDEN
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + h[None, :] * stride_xh,
                mask=mask_m[:, None] & mask_h[None, :],
                other=0.0,
            )
            wg = tl.load(
                gate_up_ptr + offs_i[:, None] * stride_gr + h[None, :] * stride_gh,
                mask=mask_i[:, None] & mask_h[None, :],
                other=0.0,
            )
            wu = tl.load(
                gate_up_ptr + (INTER + offs_i)[:, None] * stride_gr + h[None, :] * stride_gh,
                mask=mask_i[:, None] & mask_h[None, :],
                other=0.0,
            )
            acc_g += tl.dot(x, wg.trans(), out_dtype=tl.float32)
            acc_u += tl.dot(x, wu.trans(), out_dtype=tl.float32)

        inter = (acc_g * tl.sigmoid(acc_g)) * acc_u
        tl.store(
            inter_ptr + offs_m[:, None] * stride_im + offs_i[None, :] * stride_ii,
            inter.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_i[None, :],
        )

    @triton.jit
    def _shared_down_gate_add_kernel(
        inter_ptr, down_ptr, moe_ptr, x_ptr, gate_ptr, out_ptr,
        HIDDEN: tl.constexpr, INTER: tl.constexpr,
        stride_im: tl.constexpr, stride_ii: tl.constexpr,
        stride_dr: tl.constexpr, stride_di: tl.constexpr,
        stride_mm: tl.constexpr, stride_mh: tl.constexpr,
        stride_xm: tl.constexpr, stride_xh: tl.constexpr,
        stride_gm: tl.constexpr, stride_gh: tl.constexpr,
        HAS_GATE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        pid_d = tl.program_id(0)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_i = tl.arange(0, BLOCK_I)
        mask_m = offs_m < 1
        mask_d = offs_d < HIDDEN

        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for i0 in range(0, INTER, BLOCK_I):
            i = i0 + offs_i
            mask_i = i < INTER
            inter = tl.load(
                inter_ptr + offs_m[:, None] * stride_im + i[None, :] * stride_ii,
                mask=mask_m[:, None] & mask_i[None, :],
                other=0.0,
            )
            w = tl.load(
                down_ptr + offs_d[:, None] * stride_dr + i[None, :] * stride_di,
                mask=mask_d[:, None] & mask_i[None, :],
                other=0.0,
            )
            acc += tl.dot(inter, w.trans(), out_dtype=tl.float32)

        gate = 1.0
        if HAS_GATE:
            offs_h = tl.arange(0, BLOCK_H)
            gate_acc = tl.full((), 0.0, dtype=tl.float32)
            for h0 in range(0, HIDDEN, BLOCK_H):
                h = h0 + offs_h
                mask_h = h < HIDDEN
                x = tl.load(x_ptr + h * stride_xh, mask=mask_h, other=0.0).to(tl.float32)
                gw = tl.load(gate_ptr + h * stride_gh, mask=mask_h, other=0.0).to(tl.float32)
                gate_acc += tl.sum(x * gw, axis=0)
            gate = tl.sigmoid(gate_acc)

        moe = tl.load(
            moe_ptr + offs_m[:, None] * stride_mm + offs_d[None, :] * stride_mh,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            out_ptr + offs_m[:, None] * stride_mm + offs_d[None, :] * stride_mh,
            (moe + acc * gate).to(tl.bfloat16),
            mask=mask_m[:, None] & mask_d[None, :],
        )


def shared_expert_decode_fused_triton(
    h_flat: torch.Tensor,
    moe_out: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_weight: torch.Tensor | None = None,
    *,
    inter: torch.Tensor | None = None,
    block_hidden: int = 128,
    block_inter: int = 32,
    block_out: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    _require_triton()
    if h_flat.ndim != 2 or h_flat.shape[0] != 1:
        raise ValueError(f"h_flat must be [1, D], got {tuple(h_flat.shape)}")
    if h_flat.dtype != torch.bfloat16 or moe_out.dtype != torch.bfloat16:
        raise ValueError("h_flat and moe_out must be BF16")
    hidden = h_flat.shape[1]
    if gate_up_weight.shape[1] != hidden or gate_up_weight.shape[0] % 2 != 0:
        raise ValueError("gate_up_weight must be [2 * I, D]")
    intermediate = gate_up_weight.shape[0] // 2
    if down_weight.shape != (hidden, intermediate):
        raise ValueError(f"down_weight must be [{hidden}, {intermediate}]")
    if gate_weight is not None and gate_weight.shape != (1, hidden):
        raise ValueError(f"gate_weight must be [1, {hidden}]")

    h_flat = h_flat.contiguous()
    moe_out = moe_out.contiguous()
    gate_up_weight = gate_up_weight.contiguous()
    down_weight = down_weight.contiguous()
    if inter is None:
        inter = torch.empty((1, intermediate), device=h_flat.device, dtype=torch.bfloat16)
    else:
        inter = inter.contiguous()
        if inter.shape != (1, intermediate) or inter.dtype != torch.bfloat16:
            raise ValueError(f"inter scratch must be BF16 [1, {intermediate}]")

    _shared_gate_up_silu_kernel[(triton.cdiv(intermediate, block_inter),)](
        h_flat, gate_up_weight, inter,
        HIDDEN=hidden, INTER=intermediate,
        stride_xm=h_flat.stride(0), stride_xh=h_flat.stride(1),
        stride_gr=gate_up_weight.stride(0), stride_gh=gate_up_weight.stride(1),
        stride_im=inter.stride(0), stride_ii=inter.stride(1),
        BLOCK_M=16, BLOCK_H=block_hidden, BLOCK_I=block_inter,
        num_warps=num_warps,
    )

    gate_arg = gate_weight.contiguous() if gate_weight is not None else h_flat
    _shared_down_gate_add_kernel[(triton.cdiv(hidden, block_out),)](
        inter, down_weight, moe_out, h_flat, gate_arg, moe_out,
        HIDDEN=hidden, INTER=intermediate,
        stride_im=inter.stride(0), stride_ii=inter.stride(1),
        stride_dr=down_weight.stride(0), stride_di=down_weight.stride(1),
        stride_mm=moe_out.stride(0), stride_mh=moe_out.stride(1),
        stride_xm=h_flat.stride(0), stride_xh=h_flat.stride(1),
        stride_gm=gate_arg.stride(0), stride_gh=gate_arg.stride(1),
        HAS_GATE=gate_weight is not None,
        BLOCK_M=16, BLOCK_I=block_inter, BLOCK_D=block_out, BLOCK_H=block_hidden,
        num_warps=num_warps,
    )
    return moe_out


__all__ = ["HAS_TRITON", "shared_expert_decode_fused_triton"]
```

**Resident wiring**
In [engine/resident_runner.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/resident_runner.py:318), make `_prepare_shared_expert_gate_up_fused()` run when either the old gate/up fusion or the new full shared fusion is enabled. In [engine/resident_runner.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/resident_runner.py:630), also attach `mlp.shared_expert._inter_scratch` when `LYNN_SHARED_EXPERT_FUSED=1`.

```diff
@@
         self.shared_expert_gate_up_fused_attached = 0
+        self.shared_expert_fused_scratch_attached = 0
@@
-        if self.is_moe and os.environ.get("LYNN_SHARED_EXPERT_GATE_UP_FUSED", "1") != "0":
+        if self.is_moe and (
+            os.environ.get("LYNN_SHARED_EXPERT_GATE_UP_FUSED", "1") != "0"
+            or os.environ.get("LYNN_SHARED_EXPERT_FUSED", "0") == "1"
+        ):
             self._prepare_shared_expert_gate_up_fused()
@@
         attached = 0
+        scratch_attached = 0
+        attach_scratch = os.environ.get("LYNN_SHARED_EXPERT_FUSED", "0") == "1"
         for w in self.layer_weights:
             key = "mlp.shared_expert._gate_up_proj.weight"
-            if key in w:
-                continue
             if "mlp.shared_expert.gate_proj.weight" not in w or "mlp.shared_expert.up_proj.weight" not in w:
                 continue
-            w[key] = torch.cat(
-                [
-                    w["mlp.shared_expert.gate_proj.weight"],
-                    w["mlp.shared_expert.up_proj.weight"],
-                ],
-                dim=0,
-            ).contiguous()
-            attached += 1
+            if key not in w:
+                w[key] = torch.cat(
+                    [
+                        w["mlp.shared_expert.gate_proj.weight"],
+                        w["mlp.shared_expert.up_proj.weight"],
+                    ],
+                    dim=0,
+                ).contiguous()
+                attached += 1
+            if attach_scratch and "mlp.shared_expert.down_proj.weight" in w:
+                inter = int(w["mlp.shared_expert.down_proj.weight"].shape[1])
+                w["mlp.shared_expert._inter_scratch"] = torch.empty(
+                    (1, inter),
+                    device=self.device,
+                    dtype=torch.bfloat16,
+                )
+                scratch_attached += 1
         self.shared_expert_gate_up_fused_attached = attached
+        self.shared_expert_fused_scratch_attached = scratch_attached
         if self.verbose:
             print(f"[resident] shared expert fused gate/up attached={attached}", flush=True)
+            if attach_scratch:
+                print(f"[resident] shared expert fused scratch attached={scratch_attached}", flush=True)
```

**Exactness Risks**
This should be token-coherent, not bit-exact. It preserves row order from `resident_runner.py:644-649` (`[gate_proj, up_proj]`), keeps BF16 input/output, has no bias, and computes the same `F.linear -> SwiGLU -> F.linear -> optional scalar gate -> add` contract.

Main drift risks are Triton reduction order versus cuBLAS, `tl.sigmoid` versus Torch SiLU/sigmoid, and the BF16 store after SwiGLU. To keep cosine near 1, keep the intermediate BF16 store, do not fuse into a recomputing one-kernel down path, and gate this strictly behind `LYNN_SHARED_EXPERT_FUSED=1` with fixture comparison against the current lines 827-834 path before promotion.
tokens used
74,537
I can’t apply edits in this session because the workspace is mounted read-only, and I did not run anything beyond file reads. Here is the exact scoped change I’d make.

**Where**
Target is [engine/moe_packed_nvfp4.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/moe_packed_nvfp4.py:680), specifically the eager shared block at lines 826-834. Existing finalize helpers stay at lines 217-248 and remain the fallback path.

**Design**
Use 2 Triton launches, not 1. A true 1-kernel version either recomputes `gate_up` once per output block or needs cross-program synchronization Triton does not provide. The sane launch-minimal version is:

1. `h @ gate_up.T -> chunk -> silu(gate)*up`, stored BF16 `[1, I]`
2. `inter @ down.T`, plus optional `sigmoid(h @ shared_expert_gate.T)` and `moe_out + shared`, stored BF16 into `moe_out`

This replaces the shared expert’s current 4-5+ eager launches per MoE layer with 2 launches under `LYNN_SHARED_EXPERT_FUSED=1`.

```diff
diff --git a/engine/moe_packed_nvfp4.py b/engine/moe_packed_nvfp4.py
@@
 from triton_kernels.shared_expert_gate import (
     HAS_TRITON as HAS_SHARED_EXPERT_GATE_TRITON,
     add_shared_expert_gate_from_scalar_triton,
     apply_shared_expert_gate_triton,
 )
+from triton_kernels.shared_expert_fused import (
+    HAS_TRITON as HAS_SHARED_EXPERT_FUSED_TRITON,
+    shared_expert_decode_fused_triton,
+)
@@
 def _finalize_shared_expert_output(h_flat: torch.Tensor, moe_out: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
@@
     shared = _apply_shared_expert_gate(h_flat, shared, w)
     return _add_shared_expert_output(moe_out, shared)
+
+
+def _try_fused_shared_expert_output(h_flat: torch.Tensor, moe_out: torch.Tensor, w: dict) -> torch.Tensor | None:
+    if not _env_bool("LYNN_SHARED_EXPERT_FUSED", False):
+        return None
+    if not HAS_SHARED_EXPERT_FUSED_TRITON:
+        raise RuntimeError("LYNN_SHARED_EXPERT_FUSED=1 requires Triton")
+    gate_up = w.get("mlp.shared_expert._gate_up_proj.weight")
+    down = w.get("mlp.shared_expert.down_proj.weight")
+    if gate_up is None or down is None:
+        return None
+    if h_flat.ndim != 2 or h_flat.shape[0] != 1 or moe_out.shape != h_flat.shape:
+        return None
+    if not h_flat.is_cuda or h_flat.dtype != torch.bfloat16 or moe_out.dtype != torch.bfloat16:
+        return None
+    if gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
+        return None
+    hidden = h_flat.shape[1]
+    if gate_up.ndim != 2 or gate_up.shape[1] != hidden or gate_up.shape[0] % 2 != 0:
+        return None
+    inter = gate_up.shape[0] // 2
+    if down.ndim != 2 or down.shape != (hidden, inter):
+        return None
+    gate = w.get("mlp.shared_expert_gate.weight")
+    if gate is not None and (gate.ndim != 2 or gate.shape != (1, hidden) or gate.dtype != torch.bfloat16):
+        return None
+    return shared_expert_decode_fused_triton(
+        h_flat,
+        moe_out,
+        gate_up,
+        down,
+        gate,
+        inter=w.get("mlp.shared_expert._inter_scratch"),
+        block_hidden=_env_int("LYNN_SHARED_EXPERT_FUSED_BLOCK_HIDDEN", 128),
+        block_inter=_env_int("LYNN_SHARED_EXPERT_FUSED_BLOCK_INTER", 32),
+        block_out=_env_int("LYNN_SHARED_EXPERT_FUSED_BLOCK_OUT", 32),
+        num_warps=_env_int("LYNN_SHARED_EXPERT_FUSED_NUM_WARPS", 4),
+    )
@@
     if "mlp.shared_expert.gate_proj.weight" in w:
-        if "mlp.shared_expert._gate_up_proj.weight" in w:
+        fused_shared = _try_fused_shared_expert_output(h_flat, moe_out, w)
+        if fused_shared is not None:
+            moe_out = fused_shared
+        elif "mlp.shared_expert._gate_up_proj.weight" in w:
             gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
             gate_s, up_s = gate_up_s.chunk(2, dim=-1)
         else:
             gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
             up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
-        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
-        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
+        if fused_shared is None:
+            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
+            moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
```

Add `triton_kernels/shared_expert_fused.py`:

```python
"""Decode-time fused BF16 shared-expert kernels."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for fused shared expert kernels")


if HAS_TRITON:

    @triton.jit
    def _shared_gate_up_silu_kernel(
        x_ptr, gate_up_ptr, inter_ptr,
        HIDDEN: tl.constexpr, INTER: tl.constexpr,
        stride_xm: tl.constexpr, stride_xh: tl.constexpr,
        stride_gr: tl.constexpr, stride_gh: tl.constexpr,
        stride_im: tl.constexpr, stride_ii: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_I: tl.constexpr,
    ):
        pid_i = tl.program_id(0)
        offs_m = tl.arange(0, BLOCK_M)
        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        offs_h = tl.arange(0, BLOCK_H)
        mask_m = offs_m < 1
        mask_i = offs_i < INTER

        acc_g = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

        for h0 in range(0, HIDDEN, BLOCK_H):
            h = h0 + offs_h
            mask_h = h < HIDDEN
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + h[None, :] * stride_xh,
                mask=mask_m[:, None] & mask_h[None, :],
                other=0.0,
            )
            wg = tl.load(
                gate_up_ptr + offs_i[:, None] * stride_gr + h[None, :] * stride_gh,
                mask=mask_i[:, None] & mask_h[None, :],
                other=0.0,
            )
            wu = tl.load(
                gate_up_ptr + (INTER + offs_i)[:, None] * stride_gr + h[None, :] * stride_gh,
                mask=mask_i[:, None] & mask_h[None, :],
                other=0.0,
            )
            acc_g += tl.dot(x, wg.trans(), out_dtype=tl.float32)
            acc_u += tl.dot(x, wu.trans(), out_dtype=tl.float32)

        inter = (acc_g * tl.sigmoid(acc_g)) * acc_u
        tl.store(
            inter_ptr + offs_m[:, None] * stride_im + offs_i[None, :] * stride_ii,
            inter.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_i[None, :],
        )

    @triton.jit
    def _shared_down_gate_add_kernel(
        inter_ptr, down_ptr, moe_ptr, x_ptr, gate_ptr, out_ptr,
        HIDDEN: tl.constexpr, INTER: tl.constexpr,
        stride_im: tl.constexpr, stride_ii: tl.constexpr,
        stride_dr: tl.constexpr, stride_di: tl.constexpr,
        stride_mm: tl.constexpr, stride_mh: tl.constexpr,
        stride_xm: tl.constexpr, stride_xh: tl.constexpr,
        stride_gm: tl.constexpr, stride_gh: tl.constexpr,
        HAS_GATE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        pid_d = tl.program_id(0)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_i = tl.arange(0, BLOCK_I)
        mask_m = offs_m < 1
        mask_d = offs_d < HIDDEN

        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for i0 in range(0, INTER, BLOCK_I):
            i = i0 + offs_i
            mask_i = i < INTER
            inter = tl.load(
                inter_ptr + offs_m[:, None] * stride_im + i[None, :] * stride_ii,
                mask=mask_m[:, None] & mask_i[None, :],
                other=0.0,
            )
            w = tl.load(
                down_ptr + offs_d[:, None] * stride_dr + i[None, :] * stride_di,
                mask=mask_d[:, None] & mask_i[None, :],
                other=0.0,
            )
            acc += tl.dot(inter, w.trans(), out_dtype=tl.float32)

        gate = 1.0
        if HAS_GATE:
            offs_h = tl.arange(0, BLOCK_H)
            gate_acc = tl.full((), 0.0, dtype=tl.float32)
            for h0 in range(0, HIDDEN, BLOCK_H):
                h = h0 + offs_h
                mask_h = h < HIDDEN
                x = tl.load(x_ptr + h * stride_xh, mask=mask_h, other=0.0).to(tl.float32)
                gw = tl.load(gate_ptr + h * stride_gh, mask=mask_h, other=0.0).to(tl.float32)
                gate_acc += tl.sum(x * gw, axis=0)
            gate = tl.sigmoid(gate_acc)

        moe = tl.load(
            moe_ptr + offs_m[:, None] * stride_mm + offs_d[None, :] * stride_mh,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            out_ptr + offs_m[:, None] * stride_mm + offs_d[None, :] * stride_mh,
            (moe + acc * gate).to(tl.bfloat16),
            mask=mask_m[:, None] & mask_d[None, :],
        )


def shared_expert_decode_fused_triton(
    h_flat: torch.Tensor,
    moe_out: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_weight: torch.Tensor | None = None,
    *,
    inter: torch.Tensor | None = None,
    block_hidden: int = 128,
    block_inter: int = 32,
    block_out: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    _require_triton()
    if h_flat.ndim != 2 or h_flat.shape[0] != 1:
        raise ValueError(f"h_flat must be [1, D], got {tuple(h_flat.shape)}")
    if h_flat.dtype != torch.bfloat16 or moe_out.dtype != torch.bfloat16:
        raise ValueError("h_flat and moe_out must be BF16")
    hidden = h_flat.shape[1]
    if gate_up_weight.shape[1] != hidden or gate_up_weight.shape[0] % 2 != 0:
        raise ValueError("gate_up_weight must be [2 * I, D]")
    intermediate = gate_up_weight.shape[0] // 2
    if down_weight.shape != (hidden, intermediate):
        raise ValueError(f"down_weight must be [{hidden}, {intermediate}]")
    if gate_weight is not None and gate_weight.shape != (1, hidden):
        raise ValueError(f"gate_weight must be [1, {hidden}]")

    h_flat = h_flat.contiguous()
    moe_out = moe_out.contiguous()
    gate_up_weight = gate_up_weight.contiguous()
    down_weight = down_weight.contiguous()
    if inter is None:
        inter = torch.empty((1, intermediate), device=h_flat.device, dtype=torch.bfloat16)
    else:
        inter = inter.contiguous()
        if inter.shape != (1, intermediate) or inter.dtype != torch.bfloat16:
            raise ValueError(f"inter scratch must be BF16 [1, {intermediate}]")

    _shared_gate_up_silu_kernel[(triton.cdiv(intermediate, block_inter),)](
        h_flat, gate_up_weight, inter,
        HIDDEN=hidden, INTER=intermediate,
        stride_xm=h_flat.stride(0), stride_xh=h_flat.stride(1),
        stride_gr=gate_up_weight.stride(0), stride_gh=gate_up_weight.stride(1),
        stride_im=inter.stride(0), stride_ii=inter.stride(1),
        BLOCK_M=16, BLOCK_H=block_hidden, BLOCK_I=block_inter,
        num_warps=num_warps,
    )

    gate_arg = gate_weight.contiguous() if gate_weight is not None else h_flat
    _shared_down_gate_add_kernel[(triton.cdiv(hidden, block_out),)](
        inter, down_weight, moe_out, h_flat, gate_arg, moe_out,
        HIDDEN=hidden, INTER=intermediate,
        stride_im=inter.stride(0), stride_ii=inter.stride(1),
        stride_dr=down_weight.stride(0), stride_di=down_weight.stride(1),
        stride_mm=moe_out.stride(0), stride_mh=moe_out.stride(1),
        stride_xm=h_flat.stride(0), stride_xh=h_flat.stride(1),
        stride_gm=gate_arg.stride(0), stride_gh=gate_arg.stride(1),
        HAS_GATE=gate_weight is not None,
        BLOCK_M=16, BLOCK_I=block_inter, BLOCK_D=block_out, BLOCK_H=block_hidden,
        num_warps=num_warps,
    )
    return moe_out


__all__ = ["HAS_TRITON", "shared_expert_decode_fused_triton"]
```

**Resident wiring**
In [engine/resident_runner.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/resident_runner.py:318), make `_prepare_shared_expert_gate_up_fused()` run when either the old gate/up fusion or the new full shared fusion is enabled. In [engine/resident_runner.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/resident_runner.py:630), also attach `mlp.shared_expert._inter_scratch` when `LYNN_SHARED_EXPERT_FUSED=1`.

```diff
@@
         self.shared_expert_gate_up_fused_attached = 0
+        self.shared_expert_fused_scratch_attached = 0
@@
-        if self.is_moe and os.environ.get("LYNN_SHARED_EXPERT_GATE_UP_FUSED", "1") != "0":
+        if self.is_moe and (
+            os.environ.get("LYNN_SHARED_EXPERT_GATE_UP_FUSED", "1") != "0"
+            or os.environ.get("LYNN_SHARED_EXPERT_FUSED", "0") == "1"
+        ):
             self._prepare_shared_expert_gate_up_fused()
@@
         attached = 0
+        scratch_attached = 0
+        attach_scratch = os.environ.get("LYNN_SHARED_EXPERT_FUSED", "0") == "1"
         for w in self.layer_weights:
             key = "mlp.shared_expert._gate_up_proj.weight"
-            if key in w:
-                continue
             if "mlp.shared_expert.gate_proj.weight" not in w or "mlp.shared_expert.up_proj.weight" not in w:
                 continue
-            w[key] = torch.cat(
-                [
-                    w["mlp.shared_expert.gate_proj.weight"],
-                    w["mlp.shared_expert.up_proj.weight"],
-                ],
-                dim=0,
-            ).contiguous()
-            attached += 1
+            if key not in w:
+                w[key] = torch.cat(
+                    [
+                        w["mlp.shared_expert.gate_proj.weight"],
+                        w["mlp.shared_expert.up_proj.weight"],
+                    ],
+                    dim=0,
+                ).contiguous()
+                attached += 1
+            if attach_scratch and "mlp.shared_expert.down_proj.weight" in w:
+                inter = int(w["mlp.shared_expert.down_proj.weight"].shape[1])
+                w["mlp.shared_expert._inter_scratch"] = torch.empty(
+                    (1, inter),
+                    device=self.device,
+                    dtype=torch.bfloat16,
+                )
+                scratch_attached += 1
         self.shared_expert_gate_up_fused_attached = attached
+        self.shared_expert_fused_scratch_attached = scratch_attached
         if self.verbose:
             print(f"[resident] shared expert fused gate/up attached={attached}", flush=True)
+            if attach_scratch:
+                print(f"[resident] shared expert fused scratch attached={scratch_attached}", flush=True)
```

**Exactness Risks**
This should be token-coherent, not bit-exact. It preserves row order from `resident_runner.py:644-649` (`[gate_proj, up_proj]`), keeps BF16 input/output, has no bias, and computes the same `F.linear -> SwiGLU -> F.linear -> optional scalar gate -> add` contract.

Main drift risks are Triton reduction order versus cuBLAS, `tl.sigmoid` versus Torch SiLU/sigmoid, and the BF16 store after SwiGLU. To keep cosine near 1, keep the intermediate BF16 store, do not fuse into a recomputing one-kernel down path, and gate this strictly behind `LYNN_SHARED_EXPERT_FUSED=1` with fixture comparison against the current lines 827-834 path before promotion.
