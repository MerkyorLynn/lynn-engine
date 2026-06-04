# Stage 6 Phase 3-A — grouped active-MoE contract probe

Date: 2026-06-04

Verdict: **PASS** (P3-A contract probe pass; fused kernel not banked).

P3-A tests the grouped active-MoE zero-shadow contract after P2-N/P2-O.
It excludes router and shared expert from the measured candidate, builds a
BF16 active-expert reference, deletes the active BF16 shadows, then runs
`active_moe_grouped_prefill_p3a(...)` from packed NVFP4 tensors only.

**Boundary:** this report cannot bank a fused P3 kernel. Even a PASS only
banks the P3-A contract probe and its artifact schema.

**Batched-down diagnostic:** `LYNN_P3A_BATCHED_DOWN=1` removes the P3-A
per-token down launch loop. It is numerically valid (`argmax 3/3`, min cosine
0.999981), and improves the previous P3-A average slightly (`0.744x -> 0.760x`
vs BF16 active), but it is still slower than BF16 active. Do not promote this
candidate; keep it as an opt-in diagnostic and move the next cut to route
materialization / full gate-up+down batching.

## Artifact

Artifact directory: `reports/stage6/p3a_batched_down_layer0_grouped_moe_contract_probe_20260604_152955`

| File | Present |
|---|---|
| `expected_git_head.txt` | `True` |
| `expected_provenance_manifest.txt` | `True` |
| `git_head.txt` | `True` |
| `git_status.txt` | `True` |
| `head_check.txt` | `True` |
| `nvidia_smi_before.txt` | `True` |
| `nvidia_smi_after.txt` | `True` |
| `provenance_manifest.txt` | `True` |
| `docker_exit_code.txt` | `True` |
| `run.log` | `True` |
| `result.json` | `True` |
| `summary.md` | `True` |

## Provenance

| Field | Value |
|---|---|
| Expected HEAD | `56166d72ffea000fb05d30a3e25919e9b79bac8e` |
| Remote HEAD | `56166d72ffea000fb05d30a3e25919e9b79bac8e` |
| Head check | `remote HEAD ok` |
| Manifest matches | `True` |
| Docker exit code | `0` |
| Git status dirty | `True` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Layer | `0` |
| Batches | `[1, 16, 64]` |
| Candidate | `{'batched_down': True, 'env': {'LYNN_P3A_BATCHED_DOWN': '1', 'previous_LYNN_P3A_BATCHED_DOWN': None}}` |
| Shape | `H=2048 I=512 E=256 top_k=8` |

GPU before:

```text
NVIDIA GB10, 0 %, [N/A], [N/A]
```

GPU after:

```text
NVIDIA GB10, 0 %, [N/A], [N/A]
```

Provenance manifest:

```text
461fa857f6cf056f04b9eea480671dc027ddbb3a36dbaeab3750cd7d7a16a657 scripts/run_spark_stage6_p3a_contract_probe.sh
18ebd21f6fd27e4885db3076786196aa4d1303a41c574b2c88d9667620172723 scripts/spark_stage6_p3a_grouped_moe_contract_probe.py
d226d01f8bf0b6b9bd461fac7f5d0d825e2ae6e8f48212b261902158b55c20d3 scripts/summarize_stage6_p3a_contract_probe.py
58cc16e66db10c5451d1359abe71caf48de571cf47740bcb9651714b322a9dc2 scripts/write_stage6_p3a_report.py
5896563e6d13dab613c1de09c43480fd3162bd707b355bcf75e3a17e711b7738 engine/moe_packed_nvfp4.py
af337e57f15c34892e48d91028ca5a257404fb8b2e209138e061b5fb1e742584 triton_kernels/nvfp4_moe.py
3cbac9cd80b97912259325cf023b46175f4345d84b9ecb64d902985e6c0fcd6e reports/stage6/P3_GROUPED_MOE_ZERO_SHADOW_CONTRACT_20260604.md
```

## Gate Summary

# Stage 6 P3-A Grouped-MoE Contract Probe Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (P3-A contract probe pass; fused kernel not banked) |
| Banked fused kernel | `False` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Layer | `0` |
| Batches | `[1, 16, 64]` |
| Shape | `H=2048 I=512 E=256 top_k=8` |
| Tiles | `gate T=32 I=8 H=128; down H=8 I=512` |
| Numeric gate | `True` |
| Shadow absent at candidate start | `True` |
| Aggregate pass | `True` |
| BF16 active expert bytes | `1.500 GiB` |
| Packed active expert bytes | `0.563 GiB` |
| Inter scratch estimate | `0.000 GiB` |
| Memory after deleting BF16 active | `0.641 GiB` |
| Max candidate peak | `0.642 GiB` |
| Average P3-A vs BF16 speed | `0.760x` |
| Min cosine | `1.000` |
| Argmax matches | `3/3` |

## Per Batch

| Batch | Unique experts | BF16 active us | P3-A us | Speed | Cosine | Argmax |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 607.232 | 739.600 | 0.821x | 1.000 | True |
| 16 | 99 | 6442.240 | 8484.512 | 0.759x | 1.000 | True |
| 64 | 210 | 13835.768 | 19757.048 | 0.700x | 1.000 | True |

## Caveats

- Active MoE only: shared expert and router are excluded from the P3-A contract.
- The candidate consumes packed active weights after BF16 active shadows are deleted.
- Speed is reported, but this probe does not bank a fused P3 kernel.

## Hard Gates

| Gate | Value |
|---|---|
| Banked fused kernel flag is false | `True` |
| Numeric | `true` |
| Active BF16 shadow absent | `true` |
| Aggregate pass | `true` |
| BF16 active expert bytes | `1610612736` |
| Packed active expert bytes | `603979784` |
| Inter scratch estimate | `524288` |
| Memory after deleting BF16 active GiB | `0.640723705291748` |

## Decision

Bank P3-A as a contract-shaped grouped active-MoE probe only. Do not promote P3 or claim a fused kernel.

Do not treat P3-A as real-prompt, residual-stack, server, or RC quality proof.
Those remain P3-B/P3-C/P3-D gates.

## Run Log Tail

```text
        "iters": 2,
        "repeats": 2,
        "median_us": 13835.768222808838,
        "min_us": 13617.504119873047,
        "max_us": 14054.032325744629,
        "all_us": [
          14054.032325744629,
          13617.504119873047
        ]
      }
    },
    "p3a_contract": {
      "1": {
        "warmup": 1,
        "iters": 2,
        "repeats": 2,
        "median_us": 739.5999729633331,
        "min_us": 733.519971370697,
        "max_us": 745.6799745559692,
        "all_us": [
          745.6799745559692,
          733.519971370697
        ]
      },
      "16": {
        "warmup": 1,
        "iters": 2,
        "repeats": 2,
        "median_us": 8484.512329101562,
        "min_us": 8305.10425567627,
        "max_us": 8663.920402526855,
        "all_us": [
          8663.920402526855,
          8305.10425567627
        ]
      },
      "64": {
        "warmup": 1,
        "iters": 2,
        "repeats": 2,
        "median_us": 19757.047653198242,
        "min_us": 19553.407669067383,
        "max_us": 19960.6876373291,
        "all_us": [
          19960.6876373291,
          19553.407669067383
        ]
      }
    }
  },
  "memory": {
    "p3a_candidate_peak": {
      "1": {
        "before_gib": 0.6407275199890137,
        "after_gib": 0.6407313346862793,
        "peak_gib": 0.640744686126709
      },
      "16": {
        "before_gib": 0.640784740447998,
        "after_gib": 0.640845775604248,
        "peak_gib": 0.6409769058227539
      },
      "64": {
        "before_gib": 0.640967845916748,
        "after_gib": 0.641211986541748,
        "peak_gib": 0.6417365074157715
      }
    }
  },
  "passes": {
    "numeric": true,
    "shadow_absent_at_candidate_start": true,
    "all": true
  },
  "notes": [
    "Active MoE only: shared expert and router are excluded from the P3-A contract.",
    "The candidate consumes packed active weights after BF16 active shadows are deleted.",
    "Speed is reported, but this probe does not bank a fused P3 kernel."
  ]
}
```
