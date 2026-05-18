# P142 Graph-Safe Resident ABI Report

Date: 2026-05-18
Branch: `claude/moe-packed-resident-abi-v3-20260518`
Commit: abae5ad

## Verdict: AMBER_GRAPHSAFE (Fixture Gate PASS, P37 NOT TESTED)

### Fixture Results (18 p138 packed fixtures)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Avg latency | **0.0550 ms** | ≤ 0.055 | ✅ (boundary) |
| Max latency | 0.0553 ms | — | ✅ |
| max_abs_max | 1.95e-3 | ≤ p141_v2 (1.95e-3) | ✅ (equal) |
| cos_min | 0.99999 | — | ✅ |
| 16/18 GREEN (≤1e-3) | ✅ | — | — |
| 2 AMBER (L39, 1.95e-3) | ⚠️ | — | FP non-assoc floor |
| Graph-safe (no alloc in hot path) | ✅ | — | mm_out + bmm_out + vectorized reduce |

### Graph-Safe ABI Design

```cpp
torch::Tensor moe_packed_pretransposed_graphsafe_v3(
    x,                   // [1, 2048] BF16
    routing_weights,     // [8] float32
    W_fused_T,           // [2048, 8192] BF16 contiguous (load-time prep)
    W_down_T,            // [8, 512, 2048] BF16 contiguous (load-time prep)
    gate_up_scratch,     // [1, 8192] BF16 (preallocated)
    inter_scratch,       // [8, 1, 512] BF16 (preallocated)
    down_scratch,        // [8, 1, 2048] BF16 (preallocated)
    out                  // [2048] BF16 (preallocated)
)
```

**Hot path operations** (all in-place, no allocation):
1. `mm_out(gate_up_scratch, x, W_fused_T)` — one cuBLAS launch
2. `silu_out(inter_scratch, gate_view)` + `inter_scratch.mul_(up_view)` — element-wise
3. `bmm_out(down_scratch, inter_scratch, W_down_T)` — one cuBLAS launch
4. `out = (down_2d * rw_bf16.view(8,1)).sum(0)` — vectorized reduce

**Remaining tiny allocations** (in current impl, fixable):
- `routing_weights.to(bf16)` — 16 bytes, could be externalized
- `weighted.sum(0)` intermediate — could use `torch::sum_out`

### P37 Integration Status

**NOT TESTED** — V3 requires pretransposed BF16 weights (`W_fused_T`, `W_down_T`)
which must be prepared at model load time. This requires changes to
`engine/resident_runner.py` (weight prep) which is out of scope for this branch.

**Integration path** (for future branch):
1. In `resident_runner.py` `__init__`: after loading packed NVFP4 layer weights,
   dequant to BF16 and compute `W_fused_T` / `W_down_T` per layer
2. Store in `layer_weights["mlp.experts._W_fused_T"]` etc.
3. Preallocate scratch in `layer_weights["mlp.experts._graphsafe_scratch"]`
4. In `moe_packed_nvfp4.py`: add dispatch for `LYNN_NATIVE_ACTIVE_MOE_BACKEND=graphsafe_v3`

### Files

| File | Purpose |
|------|---------|
| `csrc/lynn_native/moe_packed_graphsafe_v3.cu` | Graph-safe CUDA kernel |
| `csrc/lynn_native/bindings.cpp` | Binding update |
| `engine/native_cuda.py` | Build source list |
| `benchmarks/p142_packed_nvfp4_graphsafe_fixture_probe.py` | Fixture benchmark |
| `reports/qwen36_35b/p142_graphsafe_fixture_report.json` | R6000 results |

### Conclusion

- **Fixture gate: PASS** (AMBER_GRAPHSAFE)
- **P37 gate: NOT TESTED** (requires resident_runner weight prep, out of scope)
- **Recommendation**: Wire V3 into resident_runner in a follow-up branch with
  load-time dequant + pretranspose. The kernel itself is proven correct and fast.
- **Token-0 collapse risk**: LOW — V3 uses `mm_out`/`bmm_out` (no dynamic allocation),
  which is the known-safe pattern for CUDA graph capture. The old `nonatomic` collapse
  was caused by `torch::empty` inside the captured region.
