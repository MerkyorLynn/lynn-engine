"""Runtime wrappers for packed NVFP4 weights.

P2 proved Lynn Engine can dequantize compressed-tensors NVFP4 v8-RTN weights
into BF16 and run the model. P3 starts replacing that resident dequant path
with native packed-weight kernels.

The first runtime contract is deliberately narrow: single-token Linear forward
from packed NVFP4 tensors. It is enough for decode-path integration and keeps
the old P2 path untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import torch
from triton_kernels.nvfp4_linear import (
    nvfp4_dual_matvec_packed,
    nvfp4_matvec_packed,
    quantize_fp4_m1_native,
    quantize_fp4_m1_native_out,
)

_SWIZZLE_INDEX_CACHE: dict[tuple[int, int, int, int, str], torch.Tensor] = {}
_SWIZZLE_FP8_ONES_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _scale_shape(dim: int, k: int) -> tuple[int, int]:
    # torch._scaled_mm's FP4 scale_b layout is tiled in 128-row blocks. Single
    # projection weights in Qwen3.6 are usually already 128-aligned, but fused
    # projections such as qkv/z/b/a are not necessarily aligned. Pad the logical
    # scale rows to full tiles so the swizzled index never writes past storage.
    rows = max(((dim + 127) // 128) * 128, 128)
    return rows, max(k // 16, 4)


def _torch_scaled_mm_scale_index(row: int, group: int, groups: int) -> int:
    tile = row // 128
    row_in_tile = row % 128
    return (
        tile * (128 * groups)
        + (group // 4) * 512
        + (row_in_tile % 32) * 16
        + (row_in_tile // 32) * 4
        + (group % 4)
    )


def _compact_scale_to_swizzled_fp8(scale: torch.Tensor, *, outer_dim: int, k: int) -> torch.Tensor:
    rows, groups = _scale_shape(outer_dim, k)
    actual_groups = scale.shape[1]
    ones_key = (rows, groups, str(scale.device))
    ones = _SWIZZLE_FP8_ONES_CACHE.get(ones_key)
    if ones is None:
        ones = torch.ones(rows * groups, device=scale.device, dtype=torch.float8_e4m3fn)
        _SWIZZLE_FP8_ONES_CACHE[ones_key] = ones
    expanded = ones.clone()
    key = (scale.shape[0], actual_groups, rows, groups, str(scale.device))
    idx = _SWIZZLE_INDEX_CACHE.get(key)
    if idx is None:
        row = torch.arange(scale.shape[0], device=scale.device, dtype=torch.long)[:, None]
        group = torch.arange(actual_groups, device=scale.device, dtype=torch.long)[None, :]
        tile = row // 128
        row_in_tile = row % 128
        idx = (
            tile * (128 * groups)
            + (group // 4) * 512
            + (row_in_tile % 32) * 16
            + (row_in_tile // 32) * 4
            + (group % 4)
        ).reshape(-1)
        _SWIZZLE_INDEX_CACHE[key] = idx
    expanded[idx] = scale.reshape(-1).to(torch.float8_e4m3fn)
    return expanded


_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _read_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.exists():
        return json.loads(index_path.read_text())["weight_map"]
    if single_path.exists():
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as st:
            return {k: single_path.name for k in st.keys()}
    raise FileNotFoundError(
        f"Expected model.safetensors.index.json or model.safetensors under {model_dir}"
    )


def _load_tensor_by_key(model_dir: Path, weight_map: dict[str, str], key: str, *, device: str | torch.device):
    from safetensors import safe_open

    file_name = weight_map[key]
    with safe_open(model_dir / file_name, framework="pt", device=device) as st:
        return st.get_tensor(key)


def load_packed_nvfp4_linear(
    model_dir: str | Path,
    base_key: str,
    *,
    name: str | None = None,
    device: str | torch.device = "cuda",
    default_backend: str = "scalar_bridge",
) -> "PackedNVFP4Linear":
    """Load a packed NVFP4 Linear from either v8-RTN or Lynn variable manifest.

    Early P3/P5 probes assumed a single `model.safetensors` file and the
    compressed-tensors suffix layout. The 27B Lynn variable artifact is sharded
    and records the canonical `.weight` tensor in `lynn_quant_manifest.json`.
    This helper preserves the old contract while making native FP4 probes work
    against production 27B artifacts.
    """
    model_dir = Path(model_dir)
    weight_map = _read_weight_map(model_dir)

    manifest_path = model_dir / "lynn_quant_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        rec = manifest.get("quantized_tensors", {}).get(base_key + ".weight")
        if rec is not None:
            weight_packed = _load_tensor_by_key(
                model_dir, weight_map, rec["packed_key"], device=device
            ).contiguous()
            weight_scale = _load_tensor_by_key(
                model_dir, weight_map, rec["scale_key"], device=device
            ).float().contiguous()
            weight_global_scale = _load_tensor_by_key(
                model_dir, weight_map, rec["global_scale_key"], device=device
            ).float().contiguous()
            return PackedNVFP4Linear(
                name=name or base_key,
                weight_packed=weight_packed,
                weight_scale=weight_scale,
                weight_global_scale=weight_global_scale,
                default_backend=default_backend,
            )

    packed_key = base_key + ".weight_packed"
    scale_key = base_key + ".weight_scale"
    global_key = base_key + ".weight_global_scale"
    missing = [k for k in (packed_key, scale_key, global_key) if k not in weight_map]
    if missing:
        raise KeyError(f"{base_key}: missing packed NVFP4 tensors: {missing}")
    return PackedNVFP4Linear(
        name=name or base_key,
        weight_packed=_load_tensor_by_key(model_dir, weight_map, packed_key, device=device).contiguous(),
        weight_scale=_load_tensor_by_key(model_dir, weight_map, scale_key, device=device).float().contiguous(),
        weight_global_scale=_load_tensor_by_key(model_dir, weight_map, global_key, device=device).float().contiguous(),
        default_backend=default_backend,
    )


def load_grouped_nvfp4_weight(
    model_dir: str | Path,
    base_key: str,
    *,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a grouped packed NVFP4 expert tensor.

    Lynn variable 27B stores expert groups as flattened 2D tensors in shards,
    with the original `[experts, out, in]` shape recorded in
    `lynn_quant_manifest.json`. Runtime MoE kernels want the natural 3D view.
    """
    model_dir = Path(model_dir)
    manifest_path = model_dir / "lynn_quant_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} is required for grouped NVFP4 weights")
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("quantized_tensors", {})
    rec = records.get(base_key) or records.get(base_key + ".weight")
    if rec is None:
        raise KeyError(f"{base_key}.weight missing from lynn_quant_manifest.json")
    weight_map = _read_weight_map(model_dir)
    packed = _load_tensor_by_key(model_dir, weight_map, rec["packed_key"], device=device).contiguous()
    scale = _load_tensor_by_key(model_dir, weight_map, rec["scale_key"], device=device).float().contiguous()
    global_scale = _load_tensor_by_key(model_dir, weight_map, rec["global_scale_key"], device=device).float().contiguous()
    original_shape = rec["original_shape"]
    if len(original_shape) != 3:
        raise ValueError(f"{base_key}: expected grouped original_shape, got {original_shape}")
    experts, out_features, in_features = map(int, original_shape)
    if packed.ndim == 2:
        packed = packed.reshape(experts, out_features, in_features // 2).contiguous()
    if scale.ndim == 2:
        scale = scale.reshape(experts, out_features, in_features // 16).contiguous()
    return packed, scale, global_scale


def _quantize_activation_to_fp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `[M, K]` activations to torch.float4_e2m1fn_x2 storage."""
    if x.ndim != 2:
        raise ValueError(f"activation must be [M, K], got {tuple(x.shape)}")
    if x.shape[1] % 16 != 0:
        raise ValueError(f"activation K must be divisible by 16, got {x.shape[1]}")
    table = _E2M1_TABLE.to(device=x.device)
    x32 = x.float()
    m, k = x32.shape
    groups = k // 16
    xg = x32.reshape(m, groups, 16)
    scale = (xg.abs().amax(dim=-1) / float(table[-1])).clamp_min(1e-8)
    normalized = xg.abs() / scale.unsqueeze(-1)
    mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
    sign = (xg < 0).to(torch.uint8) * 8
    codes = (mag.to(torch.uint8) | sign).reshape(m, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, scale


@dataclass(slots=True)
class PackedNVFP4Linear:
    """A single compressed-tensors NVFP4 Linear weight group.

    Attributes mirror the v8-RTN checkpoint schema:

    - `weight_packed`: uint8 `[out_features, in_features / 2]`
    - `weight_scale`: per-group scale `[out_features, in_features / 16]`
    - `weight_global_scale`: scalar global scale
    """

    name: str
    weight_packed: torch.Tensor
    weight_scale: torch.Tensor
    weight_global_scale: torch.Tensor
    native_scale_b: torch.Tensor | None = None
    native_weight_t: torch.Tensor | None = None
    default_backend: str = "scalar_bridge"

    @classmethod
    def from_safetensors(
        cls,
        st,
        base_key: str,
        *,
        name: str | None = None,
        device: str | torch.device = "cuda",
        default_backend: str = "scalar_bridge",
    ) -> "PackedNVFP4Linear":
        """Load one packed NVFP4 Linear group from an open safetensors file."""
        weight_packed = st.get_tensor(base_key + ".weight_packed").to(device).contiguous()
        weight_scale = st.get_tensor(base_key + ".weight_scale").to(device).float().contiguous()
        weight_global_scale = st.get_tensor(base_key + ".weight_global_scale").to(device).float().contiguous()
        return cls(
            name=name or base_key,
            weight_packed=weight_packed,
            weight_scale=weight_scale,
            weight_global_scale=weight_global_scale,
            default_backend=default_backend,
        )

    @property
    def out_features(self) -> int:
        return int(self.weight_packed.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weight_packed.shape[1] * 2)

    def _native_scale_b(self) -> torch.Tensor:
        if self.native_scale_b is None:
            effective = self.weight_scale.float() / self.weight_global_scale.to(self.weight_scale.device).float()
            self.native_scale_b = _compact_scale_to_swizzled_fp8(
                effective,
                outer_dim=self.out_features,
                k=self.in_features,
            )
        return self.native_scale_b

    def _native_weight_t(self) -> torch.Tensor:
        if self.native_weight_t is None:
            self.native_weight_t = self.weight_packed.view(torch.float4_e2m1fn_x2).t()
        return self.native_weight_t

    def forward_native_fast_2d(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Specialized native FP4 path for single-token decode.

        This skips the generic reshape/backend dispatch used by `forward`.
        It is intentionally narrow: callers must pass `[1, in_features]`.
        """
        if x_2d.ndim != 2 or x_2d.shape[0] != 1 or x_2d.shape[1] != self.in_features:
            raise ValueError(
                f"{self.name}: forward_native_fast_2d expects [1, {self.in_features}], "
                f"got {tuple(x_2d.shape)}"
            )
        if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("native_scaled_mm requires torch.float4_e2m1fn_x2 and torch._scaled_mm")
        act_packed, scale_a = quantize_fp4_m1_native(x_2d)
        return torch._scaled_mm(
            act_packed.view(torch.float4_e2m1fn_x2),
            self._native_weight_t(),
            scale_a=scale_a,
            scale_b=self._native_scale_b(),
            out_dtype=torch.float16,
        )[0].float()

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
        backend: str | None = None,
    ) -> torch.Tensor:
        """Run a single-token Linear forward.

        The current P3-A kernel supports one activation vector. This wrapper
        accepts `[D]`, `[1, D]`, or `[1, 1, D]` and reshapes the result back to
        the corresponding leading dimensions. Multi-token prefill is left to a
        later batched GEMM kernel.
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"{self.name}: expected input last dim {self.in_features}, got {x.shape[-1]}"
            )
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] != 1:
            raise NotImplementedError(
                f"{self.name}: PackedNVFP4Linear.forward currently supports one token, "
                f"got {flat.shape[0]}"
            )
        backend = backend or self.default_backend
        if backend == "scalar_bridge":
            out = nvfp4_matvec_packed(
                flat[0],
                self.weight_packed,
                self.weight_scale,
                self.weight_global_scale,
            )
        elif backend == "native_scaled_mm":
            if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
                raise RuntimeError("native_scaled_mm requires torch.float4_e2m1fn_x2 and torch._scaled_mm")
            act_packed, scale_a = quantize_fp4_m1_native(flat)
            out = torch._scaled_mm(
                act_packed.view(torch.float4_e2m1fn_x2),
                self._native_weight_t(),
                scale_a=scale_a,
                scale_b=self._native_scale_b(),
                out_dtype=torch.float16,
            )[0].float()
        else:
            raise ValueError(f"{self.name}: unknown PackedNVFP4Linear backend {backend!r}")
        dtype = output_dtype or x.dtype
        return out.to(dtype).reshape(*x.shape[:-1], self.out_features)

    __call__ = forward


@dataclass(slots=True)
class PackedNVFP4FusedLinear:
    """A fused native-FP4 Linear built by concatenating packed NVFP4 rows.

    This is the first packed-resident runtime shape that can represent natural
    decode fusions such as linear-attention qkv/z/b/a input projection. Each
    source projection may have its own global scale, so callers pass the already
    normalized and swizzled `scale_b` rather than a single scalar global scale.
    """

    name: str
    weight_packed: torch.Tensor
    native_scale_b: torch.Tensor
    native_weight_t: torch.Tensor | None = None
    default_backend: str = "native_fast_2d"
    act_packed_scratch: torch.Tensor | None = None
    scale_a_scratch: torch.Tensor | None = None

    @property
    def out_features(self) -> int:
        return int(self.weight_packed.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weight_packed.shape[1] * 2)

    def _native_weight_t(self) -> torch.Tensor:
        if self.native_weight_t is None:
            self.native_weight_t = self.weight_packed.view(torch.float4_e2m1fn_x2).t()
        return self.native_weight_t

    def _activation_scratch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        expected_packed = (1, self.in_features // 2)
        groups = self.in_features // 16
        expected_scale = (max(1, 128) * max(groups, 4),)
        if (
            self.act_packed_scratch is None
            or tuple(self.act_packed_scratch.shape) != expected_packed
            or self.act_packed_scratch.device != device
        ):
            self.act_packed_scratch = torch.empty(expected_packed, device=device, dtype=torch.uint8)
        if (
            self.scale_a_scratch is None
            or tuple(self.scale_a_scratch.shape) != expected_scale
            or self.scale_a_scratch.device != device
        ):
            self.scale_a_scratch = torch.ones(expected_scale, device=device, dtype=torch.float8_e4m3fn)
        return self.act_packed_scratch, self.scale_a_scratch

    def forward_native_fast_2d(self, x_2d: torch.Tensor) -> torch.Tensor:
        if x_2d.ndim != 2 or x_2d.shape[0] != 1 or x_2d.shape[1] != self.in_features:
            raise ValueError(
                f"{self.name}: forward_native_fast_2d expects [1, {self.in_features}], "
                f"got {tuple(x_2d.shape)}"
            )
        if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("native_scaled_mm requires torch.float4_e2m1fn_x2 and torch._scaled_mm")
        if os.environ.get("LYNN_NATIVE_FP4_ACT_SCRATCH", "0") == "1":
            act_packed_scratch, scale_a_scratch = self._activation_scratch(x_2d.device)
            act_packed, scale_a = quantize_fp4_m1_native_out(x_2d, act_packed_scratch, scale_a_scratch)
        else:
            act_packed, scale_a = quantize_fp4_m1_native(x_2d)
        return torch._scaled_mm(
            act_packed.view(torch.float4_e2m1fn_x2),
            self._native_weight_t(),
            scale_a=scale_a,
            scale_b=self.native_scale_b,
            out_dtype=torch.float16,
        )[0].float()

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
        backend: str | None = None,
    ) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] != 1:
            raise NotImplementedError(
                f"{self.name}: PackedNVFP4FusedLinear.forward currently supports one token, "
                f"got {flat.shape[0]}"
            )
        backend = backend or self.default_backend
        if backend != "native_fast_2d":
            raise ValueError(f"{self.name}: unknown fused backend {backend!r}")
        out = self.forward_native_fast_2d(flat)
        dtype = output_dtype or x.dtype
        return out.to(dtype).reshape(*x.shape[:-1], self.out_features)

    __call__ = forward


def fuse_packed_nvfp4_linears(name: str, linears: list[PackedNVFP4Linear]) -> PackedNVFP4FusedLinear:
    """Concatenate packed NVFP4 rows and build a native FP4 scale_b layout."""
    if not linears:
        raise ValueError("linears must be non-empty")
    in_features = linears[0].in_features
    for linear in linears:
        if linear.in_features != in_features:
            raise ValueError(
                f"all fused linears must share input dim {in_features}; "
                f"{linear.name} has {linear.in_features}"
            )
    weight_packed = torch.cat([linear.weight_packed for linear in linears], dim=0).contiguous()
    effective_scale = torch.cat(
        [
            linear.weight_scale.float()
            / linear.weight_global_scale.to(linear.weight_scale.device).float()
            for linear in linears
        ],
        dim=0,
    ).contiguous()
    scale_b = _compact_scale_to_swizzled_fp8(
        effective_scale,
        outer_dim=int(weight_packed.shape[0]),
        k=in_features,
    )
    return PackedNVFP4FusedLinear(name=name, weight_packed=weight_packed, native_scale_b=scale_b)


def dual_scalar_bridge(
    x: torch.Tensor,
    left: PackedNVFP4Linear,
    right: PackedNVFP4Linear,
    *,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run two same-shaped packed Linear projections in one scalar-bridge launch.

    This keeps BF16 activations and the P3 scalar-bridge numerics, so it avoids
    the activation-FP4 router drift seen in P5-D while still reducing launch
    overhead for natural pairs such as expert gate/up or linear-attn a/b.
    """
    if x.shape[-1] != left.in_features or x.shape[-1] != right.in_features:
        raise ValueError(
            f"dual_scalar_bridge input dim mismatch: x={x.shape[-1]}, "
            f"left={left.in_features}, right={right.in_features}"
        )
    if left.weight_packed.shape != right.weight_packed.shape:
        raise ValueError("dual_scalar_bridge requires matching packed weight shapes")
    if x.reshape(-1, x.shape[-1]).shape[0] != 1:
        raise NotImplementedError("dual_scalar_bridge currently supports one token")

    flat = x.reshape(-1, x.shape[-1])[0]
    out_left, out_right = nvfp4_dual_matvec_packed(
        flat,
        left.weight_packed,
        left.weight_scale,
        left.weight_global_scale,
        right.weight_packed,
        right.weight_scale,
        right.weight_global_scale,
    )
    dtype = output_dtype or x.dtype
    leading = x.shape[:-1]
    return (
        out_left.to(dtype).reshape(*leading, left.out_features),
        out_right.to(dtype).reshape(*leading, right.out_features),
    )
