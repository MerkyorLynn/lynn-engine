# Lynn Engine P98: split16 FP4 runtime gate

## Verdict

**Do not promote.** P98 wires the P93 split16 SM120a FP4 MMA gate/up kernel into
the production MoE dispatch as an explicit research backend:

```bash
LYNN_NATIVE_GATEUP_BACKEND=split16_fp4
LYNN_NATIVE_DOWN_BACKEND=cuda_tile
LYNN_NATIVE_DOWN_TILE_HIDDEN=1
LYNN_NATIVE_CUDA_ARCH=sm_120a
```

The native extension builds and exposes the runtime symbols, but the full
generate gate fails. This path remains useful as a kernel-research bridge, not
as a production backend.

## What Passed

Runtime extension smoke on R6000:

```text
has_split16 True
has_down_tile True
```

This confirms that the benchmark-local P93 kernel can be carried into the
Lynn native CUDA extension behind an opt-in ABI.

## What Failed

### Graph-on gate

The default graph-enabled runner fails during CUDA graph capture:

```text
CUDA error: operation not permitted when stream is capturing
```

Root cause: `_gate_up_native_split16_fp4` quantizes the BF16 activation at
runtime with regular PyTorch tensor ops. Those ops allocate / dispatch during
capture and are not graph-safe.

### Graph-off generate parity

Command output:

```text
new_ids_all_match: false
baseline median:  28.28 tok/s
candidate median: 22.63 tok/s
median_speedup:   0.80x
promote_default:  false
```

All three prompts diverged from the baseline greedy token ids. Candidate text
remained coherent, but production promotion requires exact greedy parity.

## Interpretation

P93/P97 compare against a **quantized-activation reference**. That is the right
contract for proving the split16 FP4 MMA kernel, but it is not the same semantic
contract as the current production path, where active MoE consumes BF16
activation values.

So P98 teaches two things:

1. `BF16 activation -> runtime E2M1 quant -> FP4 MMA` is not a drop-in
   replacement for the current BF16-activation active MoE path.
2. The next production-safe path must either preserve BF16 activation semantics
   or explicitly train / calibrate for activation quantization.

## Code Decision

Keep the backend as an explicit opt-in research path, but make it fail loudly if
CUDA graph capture is enabled. It must not become the default and must not be
used in serving without a later parity gate.

## Next

P99 should avoid reusing P98 as-is. Useful follow-ups are:

1. a capture-safe activation quantization CUDA kernel for research-only
   measurement;
2. a BF16-activation + native FP4-weight path that preserves current semantics;
3. an activation-quant-aware calibration / retraining line bundled with the
   MTP or vendor-friendly re-quant cycle.

