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

import torch
from triton_kernels.nvfp4_linear import nvfp4_matvec_packed

_SWIZZLE_INDEX_CACHE: dict[tuple[int, int, int, int, str], torch.Tensor] = {}
_SWIZZLE_FP8_ONES_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _scale_shape(dim: int, k: int) -> tuple[int, int]:
    return max(dim, 128), max(k // 16, 4)


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

    @classmethod
    def from_safetensors(
        cls,
        st,
        base_key: str,
        *,
        name: str | None = None,
        device: str | torch.device = "cuda",
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
        backend: str = "scalar_bridge",
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
            act_packed, act_scale = _quantize_activation_to_fp4(flat)
            scale_a = _compact_scale_to_swizzled_fp8(act_scale, outer_dim=flat.shape[0], k=self.in_features)
            out = torch._scaled_mm(
                act_packed.view(torch.float4_e2m1fn_x2),
                self.weight_packed.view(torch.float4_e2m1fn_x2).t(),
                scale_a=scale_a,
                scale_b=self._native_scale_b(),
                out_dtype=torch.float16,
            )[0].float()
        else:
            raise ValueError(f"{self.name}: unknown PackedNVFP4Linear backend {backend!r}")
        dtype = output_dtype or x.dtype
        return out.to(dtype).reshape(*x.shape[:-1], self.out_features)

    __call__ = forward
