# Stage 6 R5-B E8M0 Repack Summary

| Field | Value |
|---|---|
| Result | `/root/autodl-tmp/src/lynn-engine-r5b-github-20260604_173427/reports/stage6/r5b_e8m0_repack_20260604_173435/result.json` |
| Decision | `FAIL_R5B_E8M0_REPACK_NUMERIC` |
| Repack numeric banked | `False` |
| Grouped MoE POC banked | `False` |
| Kernel speed banked | `False` |
| Default promotion banked | `False` |

## Candidate Rows

| M | Candidate | rel_l2 | cosine | median ms | act value rel_l2 | weight value rel_l2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `e8m0_repack_nearest` | 0.189118 | 0.984502 | 0.062848 | 0.132215 | 0.133102 |
| 1 | `e8m0_repack_floor` | 0.356706 | 0.962568 | 0.06168 | 0.216079 | 0.236368 |
| 1 | `e8m0_repack_ceil` | 0.165278 | 0.986248 | 0.06192 | 0.116771 | 0.118569 |
| 16 | `e8m0_repack_nearest` | 0.194018 | 0.983163 | 0.063168 | 0.133796 | 0.133102 |
| 16 | `e8m0_repack_floor` | 0.370299 | 0.961615 | 0.06232 | 0.241455 | 0.236368 |
| 16 | `e8m0_repack_ceil` | 0.166929 | 0.985979 | 0.060832 | 0.117933 | 0.118569 |
| 64 | `e8m0_repack_nearest` | 0.192458 | 0.983501 | 0.063456 | 0.133666 | 0.133102 |
| 64 | `e8m0_repack_floor` | 0.368003 | 0.962685 | 0.06184 | 0.238487 | 0.236368 |
| 64 | `e8m0_repack_ceil` | 0.166553 | 0.986035 | 0.062592 | 0.119171 | 0.118569 |

## Boundary

- This summary may bank only `banked_repack_numeric=true`.
- `banked_grouped_moe_fp4_mma_poc`, `banked_kernel_speed`, and `banked_default_promotion` must remain false.
- A PASS means R5-C can build the first selected/grouped-MoE FP4-MMA POC using the repacked artifact contract.

