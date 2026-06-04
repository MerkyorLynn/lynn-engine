# Stage 6 P4C Gate/Up Shape Sweep Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (P4C gate/up launch-shape sweep recorded; promotion still closed) |
| Decision | `PASS_P4C_GATEUP_SHAPE_SWEEP_RECORDED` |
| Device | `NVIDIA GB10` |
| Capability | `[12, 1]` |
| Banked shape sweep | `True` |
| Banked gate/up candidate | `False` |
| Banked fused kernel speed | `False` |
| Banked default promotion | `False` |
| Current baseline symbol | `gate_up_silu_tile_inter_scalar` |
| Variant symbol | `gate_up_silu_tile_inter_threads_scalar` |
| Baseline tile_inter | `8` |
| Baseline median | `151.68639421463013` us |
| Baseline rel L2 / max abs | `0.0` / `0.0` |
| Best shape | `tile_inter_2_threads_128` |
| Best median | `91.41119718551636` us |
| Best speedup vs current | `1.6593852710055532` |
| Best actionable >= floor | `True` (floor `1.05`) |
| Caveat | `gate/up symbols allocate output tensors; use sweep for direction only` |

## Shape Sweep

| Shape | Median us | Speedup vs current | Numeric ok | rel L2 / max abs |
|---|---:|---:|---|---|
| `tile_inter_1_threads_64` | `106.47039413452148` | `1.4246814379495945` | `True` | `6.7576849396227244e-06` / `7.450580596923828e-09` |
| `tile_inter_1_threads_128` | `94.72000002861023` | `1.601418857356558` | `True` | `0.0` / `0.0` |
| `tile_inter_1_threads_256` | `100.83520412445068` | `1.5042999667796475` | `True` | `6.7576849396227244e-06` / `7.450580596923828e-09` |
| `tile_inter_2_threads_64` | `93.50079894065857` | `1.6223005143613773` | `True` | `6.7576849396227244e-06` / `7.450580596923828e-09` |
| `tile_inter_2_threads_128` | `91.41119718551636` | `1.6593852710055532` | `True` | `0.0` / `0.0` |
| `tile_inter_2_threads_256` | `94.74239945411682` | `1.601040242685546` | `True` | `6.7576849396227244e-06` / `7.450580596923828e-09` |
| `tile_inter_4_threads_64` | `149.00799989700317` | `1.0179748357100176` | `True` | `6.7576849396227244e-06` / `7.450580596923828e-09` |
| `tile_inter_4_threads_128` | `119.24159526824951` | `1.2720929628070792` | `True` | `0.0` / `0.0` |
| `tile_inter_4_threads_256` | `107.0080041885376` | `1.4175238138950204` | `True` | `6.7576849396227244e-06` / `7.450580596923828e-09` |
| `tile_inter_8_threads_64` | `154.52159643173218` | `0.9816517413580137` | `True` | `6.7576849396227244e-06` / `7.450580596923828e-09` |
| `tile_inter_8_threads_128` | `152.25919485092163` | `0.9962379898511066` | `True` | `0.0` / `0.0` |
| `tile_inter_8_threads_256` | `158.22080373764038` | `0.9587006931538186` | `True` | `6.7576849396227244e-06` / `7.450580596923828e-09` |

## Boundary

- This banks only `banked_p4c_gateup_shape_sweep=true`.
- It does not bank a gate/up speed candidate, fused kernel speed, or default promotion.
- If `best_is_actionable=false`, scalar launch-shape tuning is exhausted and the next cut should be a real CUDA/CUTLASS gate/up kernel.
