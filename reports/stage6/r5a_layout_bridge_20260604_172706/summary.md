# Stage 6 R5-A Layout Bridge Summary

| Field | Value |
|---|---|
| Result | `/root/autodl-tmp/src/lynn-engine-r5a-github-20260604_172629/reports/stage6/r5a_layout_bridge_20260604_172706/result.json` |
| Decision | `PASS_R5A_LAYOUT_BRIDGE_E8M0_REPACK_REQUIRED` |
| Layout bridge banked | `True` |
| Grouped MoE POC banked | `False` |
| Kernel speed banked | `False` |
| Default promotion banked | `False` |
| Fold-pair group32 supported | `False` |
| Current Lynn E4M3 scales zero-copy supported | `False` |

## Candidate Rows

| Scale case | M | Candidate | rel_l2 | cosine | median ms | packed ratio | scale ratio |
|---|---:|---|---:|---:|---:|---:|---:|
| power2 | 1 | `fold_pair_group32_max` | 1.24727 | 0.726106 | 0.046864 | 1 | 0.125 |
| power2 | 1 | `fold_pair_group32_mean` | 0.781268 | 0.755706 | 0.046192 | 1 | 0.125 |
| power2 | 1 | `fold_pair_group32_geom` | 0.80961 | 0.632761 | 0.045952 | 1 | 0.125 |
| power2 | 1 | `padded_per16_group32` | 6.69752e-08 | 1 | 0.060784 | 2 | 0.25 |
| power2 | 16 | `fold_pair_group32_max` | 1.2308 | 0.736726 | 0.046368 | 1 | 0.125 |
| power2 | 16 | `fold_pair_group32_mean` | 0.795773 | 0.756258 | 0.044832 | 1 | 0.125 |
| power2 | 16 | `fold_pair_group32_geom` | 0.804481 | 0.640212 | 0.044384 | 1 | 0.125 |
| power2 | 16 | `padded_per16_group32` | 9.53564e-08 | 1 | 0.063328 | 2 | 0.25 |
| power2 | 64 | `fold_pair_group32_max` | 1.21876 | 0.73724 | 0.047584 | 1 | 0.125 |
| power2 | 64 | `fold_pair_group32_mean` | 0.789952 | 0.750695 | 0.045584 | 1 | 0.125 |
| power2 | 64 | `fold_pair_group32_geom` | 0.813241 | 0.624958 | 0.04496 | 1 | 0.125 |
| power2 | 64 | `padded_per16_group32` | 8.91437e-08 | 1 | 0.062688 | 2 | 0.25 |
| e4m3_like | 1 | `fold_pair_group32_max` | 1.25845 | 0.760795 | 0.045568 | 1 | 0.125 |
| e4m3_like | 1 | `fold_pair_group32_mean` | 0.571984 | 0.8203 | 0.044432 | 1 | 0.125 |
| e4m3_like | 1 | `fold_pair_group32_geom` | 0.632266 | 0.79231 | 0.04408 | 1 | 0.125 |
| e4m3_like | 1 | `padded_per16_group32` | 0.32689 | 0.963315 | 0.059984 | 2 | 0.25 |
| e4m3_like | 16 | `fold_pair_group32_max` | 1.15094 | 0.790887 | 0.044864 | 1 | 0.125 |
| e4m3_like | 16 | `fold_pair_group32_mean` | 0.568403 | 0.822765 | 0.044544 | 1 | 0.125 |
| e4m3_like | 16 | `fold_pair_group32_geom` | 0.620505 | 0.797604 | 0.044176 | 1 | 0.125 |
| e4m3_like | 16 | `padded_per16_group32` | 0.321478 | 0.965439 | 0.061312 | 2 | 0.25 |
| e4m3_like | 64 | `fold_pair_group32_max` | 1.14332 | 0.791566 | 0.044816 | 1 | 0.125 |
| e4m3_like | 64 | `fold_pair_group32_mean` | 0.567991 | 0.823036 | 0.045072 | 1 | 0.125 |
| e4m3_like | 64 | `fold_pair_group32_geom` | 0.629394 | 0.790951 | 0.045584 | 1 | 0.125 |
| e4m3_like | 64 | `padded_per16_group32` | 0.317324 | 0.966415 | 0.061184 | 2 | 0.25 |

## Boundary

- This summary may bank only `banked_layout_bridge=true`.
- `banked_grouped_moe_fp4_mma_poc`, `banked_kernel_speed`, and `banked_default_promotion` must remain false.
- If current Lynn E4M3 scales are not zero-copy supported, R5-B must use explicit repack or custom scale handling.

