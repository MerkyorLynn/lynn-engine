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
        return cls(
            name=name or base_key,
            weight_packed=st.get_tensor(base_key + ".weight_packed").to(device).contiguous(),
            weight_scale=st.get_tensor(base_key + ".weight_scale").to(device).float().contiguous(),
            weight_global_scale=st.get_tensor(base_key + ".weight_global_scale").to(device).float().contiguous(),
        )

    @property
    def out_features(self) -> int:
        return int(self.weight_packed.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weight_packed.shape[1] * 2)

    def forward(self, x: torch.Tensor, *, output_dtype: torch.dtype | None = None) -> torch.Tensor:
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
        out = nvfp4_matvec_packed(
            flat[0],
            self.weight_packed,
            self.weight_scale,
            self.weight_global_scale,
        )
        dtype = output_dtype or x.dtype
        return out.to(dtype).reshape(*x.shape[:-1], self.out_features)

    __call__ = forward
