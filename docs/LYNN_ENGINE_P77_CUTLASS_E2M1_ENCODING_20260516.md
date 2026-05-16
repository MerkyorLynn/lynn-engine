# Lynn Engine P77: CUTLASS E2M1 encoding compatibility

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P77

P76 proves that R6000 can compile CUTLASS/CuTe sm_120 FP4 code. The next risk
is subtler: can the grouped kernel consume Lynn's current packed E2M1 nibbles
directly as `cutlass::float_e2m1_t`, or do we need a repack/remap layer?

P77 answers that with a GPU bitcast table probe.

## Probe

```bash
python benchmarks/p77_cutlass_e2m1_encoding_probe.py \
  --out reports/p16_155/p77_cutlass_e2m1_encoding_probe.json
```

The probe launches a tiny CUDA kernel that writes:

```cpp
static_cast<float>(cutlass::float_e2m1_t::bitcast(nibble))
```

for all 16 possible nibbles.

## Result

| Nibble | Lynn table | CUTLASS bitcast |
|---:|---:|---:|
| 0 | 0.0 | 0.0 |
| 1 | 0.5 | 0.5 |
| 2 | 1.0 | 1.0 |
| 3 | 1.5 | 1.5 |
| 4 | 2.0 | 2.0 |
| 5 | 3.0 | 3.0 |
| 6 | 4.0 | 4.0 |
| 7 | 6.0 | 6.0 |
| 8 | -0.0 | -0.0 |
| 9 | -0.5 | -0.5 |
| 10 | -1.0 | -1.0 |
| 11 | -1.5 | -1.5 |
| 12 | -2.0 | -2.0 |
| 13 | -3.0 | -3.0 |
| 14 | -4.0 | -4.0 |
| 15 | -6.0 | -6.0 |

Report:

```text
exact_storage_compatible: true
```

## Decision

Lynn packed E2M1 nibbles are storage-compatible with CUTLASS
`float_e2m1_t::bitcast`.

This is a major simplification for P78:

- no nibble remap is required;
- no offline repack is required for weight codes;
- the current Lynn-native packed bytes can be interpreted as CUTLASS E2M1
  storage inside the custom kernel.

The scale issue remains separate. P54 still shows that Lynn's FP32 per-16 scale
contract cannot be blindly converted to vendor e8m0/group32. P77 only clears
the 4-bit value encoding.

## Next path

P78 should build the first tiny selected-row FP4 MMA smoke:

```text
input codes:  Lynn/CUTLASS-compatible E2M1 packed bytes
scale:        start synthetic neutral / explicit FP32 per-16 multiply
shape:        one small tile, not full active MoE yet
gate:         compare against scalar decode on synthetic data
```

If that passes, P79 can connect the same path to one real layer's selected
gate/up rows before re-entering the P69 active boundary gate.
