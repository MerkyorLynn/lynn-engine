# Stage 6 R6000 FP4-MMA Bring-Up Census

| Field | Value |
|---|---|
| Verdict | **PASS** (R6000 FP4-MMA bring-up and public-kernel census passed) |
| Decision | `PASS_R6000_FP4_MMA_BRINGUP` |
| Device | `NVIDIA RTX PRO 6000 Blackwell Server Edition` |
| Capability | `[12, 0]` |
| Memory GiB | `94.973` |
| Disk workspace | `/dev/md0            880G    1G      880G   1% /root/autodl-tmp` |
| Public packages | `vllm:no, cutlass:no, triton:yes, flashinfer:no, tensorrt_llm:no` |
| Explicit NVFP4 imports | `none` |
| Public source candidates | `/root/autodl-tmp/src/vllm:200` |
| Contract passes | `{'p76_cutlass_cute_toolchain': True, 'p79_nvcc_fp4_mma_target_matrix': True, 'p85_blockscaled_fp4_mma_contract': True, 'p87_layout_tile_contract': True, 'p103_fp8_activation_fp4_weight_mma': True}` |
| Promotion boundary | `{'kernel_promoted': False, 'default_runtime_changed': False, 'speed_claim': False}` |

## Pass Gates

| Gate | Value |
|---|---|
| `torch_cuda_recorded` | `True` |
| `blackwell_capability` | `True` |
| `r6000_class_memory` | `True` |
| `disk_headroom` | `True` |
| `public_kernel_census_recorded` | `True` |
| `vllm_nvfp4_or_marlin_seen` | `True` |
| `contract_suite_recorded` | `True` |
| `contract_suite_all_pass` | `True` |
| `all` | `True` |

## Boundary

- This banks machine/toolchain/public-kernel census only.
- It does not promote any Lynn kernel or runtime default.
- If PASS, the next step is a Lynn NVFP4 grouped-MoE FP4-MMA POC using CUTLASS/CuTe plus the public Marlin/Machete census.
