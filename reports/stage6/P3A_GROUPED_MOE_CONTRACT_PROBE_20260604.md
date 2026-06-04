# Stage 6 Phase 3-A — grouped active-MoE contract probe

Date: 2026-06-04

Verdict: **PASS** (P3-A contract probe pass; fused kernel not banked).

P3-A tests the grouped active-MoE zero-shadow contract after P2-N/P2-O.
It excludes router and shared expert from the measured candidate, builds a
BF16 active-expert reference, deletes the active BF16 shadows, then runs
`active_moe_grouped_prefill_p3a(...)` from packed NVFP4 tensors only.

**Boundary:** this report cannot bank a fused P3 kernel. Even a PASS only
banks the P3-A contract probe and its artifact schema.

## Artifact

Artifact directory: `reports/stage6/p3a_layer0_grouped_moe_contract_probe_20260604_143854`

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
| Expected HEAD | `825d1dd793ceb751351b106339369dc578386f66` |
| Remote HEAD | `825d1dd793ceb751351b106339369dc578386f66` |
| Head check | `remote HEAD ok` |
| Manifest matches | `True` |
| Docker exit code | `0` |
| Git status dirty | `True` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Layer | `0` |
| Batches | `[1, 16, 64]` |
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
82e174f988b9003723808d5055db31b996fb365255809d0bcf57379dcc91d385 scripts/run_spark_stage6_p3a_contract_probe.sh
ed99e0e54dc05c67bc02d9dd90778f9750210fcaa7130d3bf871b1f65054a45b scripts/spark_stage6_p3a_grouped_moe_contract_probe.py
d226d01f8bf0b6b9bd461fac7f5d0d825e2ae6e8f48212b261902158b55c20d3 scripts/summarize_stage6_p3a_contract_probe.py
58cc16e66db10c5451d1359abe71caf48de571cf47740bcb9651714b322a9dc2 scripts/write_stage6_p3a_report.py
2e868cc1727712fb36e7667c42f9f7d5eb2e99d0fd94267f42281121093867cc engine/moe_packed_nvfp4.py
1ca3909a5f5867510d24f26c889895332cf1e91f56ca79f75fbbab6f14463eed triton_kernels/nvfp4_moe.py
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
| Average P3-A vs BF16 speed | `0.744x` |
| Min cosine | `1.000` |
| Argmax matches | `3/3` |

## Per Batch

| Batch | Unique experts | BF16 active us | P3-A us | Speed | Cosine | Argmax |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 614.240 | 749.456 | 0.820x | 1.000 | True |
| 16 | 99 | 6447.056 | 8823.416 | 0.731x | 1.000 | True |
| 64 | 210 | 13835.160 | 20288.073 | 0.682x | 1.000 | True |

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
        "median_us": 13835.160255432129,
        "min_us": 13623.392105102539,
        "max_us": 14046.928405761719,
        "all_us": [
          14046.928405761719,
          13623.392105102539
        ]
      }
    },
    "p3a_contract": {
      "1": {
        "warmup": 1,
        "iters": 2,
        "repeats": 2,
        "median_us": 749.455988407135,
        "min_us": 744.0639734268188,
        "max_us": 754.8480033874512,
        "all_us": [
          754.8480033874512,
          744.0639734268188
        ]
      },
      "16": {
        "warmup": 1,
        "iters": 2,
        "repeats": 2,
        "median_us": 8823.415756225586,
        "min_us": 8808.719635009766,
        "max_us": 8838.111877441406,
        "all_us": [
          8838.111877441406,
          8808.719635009766
        ]
      },
      "64": {
        "warmup": 1,
        "iters": 2,
        "repeats": 2,
        "median_us": 20288.07258605957,
        "min_us": 20263.61656188965,
        "max_us": 20312.528610229492,
        "all_us": [
          20263.61656188965,
          20312.528610229492
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
