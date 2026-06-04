# Stage 6 Phase 3-B — selected-prefill composition gate

Date: 2026-06-04

Verdict: **PASS** (selected-prefill composition gates passed).

P3-B places the P3-A grouped active-MoE contract inside a selected
transformer prefill stack. It compares BF16 prefill, P2-N reference
(`p2e_hybrid` + block linear-attn), and P3-B candidate
(`p3a_grouped` + block linear-attn).

**Boundary:** a PASS here only banks selected-layer composition. It does
not bank a fused grouped-MoE kernel, server path, RC quality, or default
promotion.

## Artifact

Artifact directory: `reports/stage6/p3b_layers0-3_selected_prefill_gate_20260604_144842`

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
| Expected HEAD | `27cb1251bac28be78e5d95bd0b4a2509f9c07c7c` |
| Remote HEAD | `27cb1251bac28be78e5d95bd0b4a2509f9c07c7c` |
| Head check | `remote HEAD ok` |
| Manifest matches | `True` |
| Docker exit code | `0` |
| Git status dirty | `True` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Layers | `[0, 1, 2, 3]` |
| Layer types | `['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']` |
| Seq lens | `[16, 64]` |

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
88693beccd6f2bf2d1299d5e315de7b2476bb96698c3c0a731ff2ad2672ef593 scripts/run_spark_stage6_p3b_selected_prefill_gate.sh
e6147f66ac5c0b244fd1707585c3c0eb0e1e34a5ad032b9b1fdeb2ab36f11fb5 scripts/spark_stage6_p3b_selected_prefill_gate.py
f17b183c6ad471655e8b64ca42b77f0d653514ab4c53a2ab7909f356025233a2 scripts/summarize_stage6_p3b_selected_prefill_gate.py
3d6385ac5030d181dab3c22c85e9a2e63738cf6175d75b094382bfd06b3e6c30 scripts/write_stage6_p3b_report.py
cd0e4f30b931972ee5e603b9e0c7dad24ace08e353d0feedf38dababcd26b923 engine/full_forward.py
2e868cc1727712fb36e7667c42f9f7d5eb2e99d0fd94267f42281121093867cc engine/moe_packed_nvfp4.py
1ca3909a5f5867510d24f26c889895332cf1e91f56ca79f75fbbab6f14463eed triton_kernels/nvfp4_moe.py
f6b193629e2b4002df177d7ac1b613df40eee927c4a59b9074b7c25664cd4ac6 reports/stage6/P3B_SELECTED_PREFILL_GATE_RUNBOOK_20260604.md
```

## Gate Summary

# Stage 6 P3-B Selected-Prefill Gate Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (selected-prefill composition gates passed) |
| Schema | `lynn-stage6-p3b-selected-prefill-gate-v1` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Layers | `[0, 1, 2, 3]` |
| Layer types | `['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']` |
| Seq lens | `[16, 64]` |
| Banked fused kernel | `False` |
| Banked server path | `False` |
| Predecessors pass | `True` |
| Numeric | `True` |
| Final stack cosine min | `1.000` |
| Final stack argmax | `True` |
| Active BF16 shadow absent | `True` |
| Reload trap installed | `True` |
| Reload not called | `True` |
| Speed vs P2-N reference | `True` |
| BF16 active expert bytes | `6442450944` |
| Packed active expert bytes | `2415919136` |
| Memory after load | `8.517 GiB` |
| Memory after active-shadow delete | `2.525 GiB` |
| Active-shadow memory drop | `5.991 GiB` |
| Avg BF16 prefill | `75136.610 us` |
| Avg P2-N reference | `62453.167 us` |
| Avg P3-B candidate | `61278.111 us` |
| Avg P3-B vs BF16 | `1.297x` |
| Avg P3-B vs P2-N | `1.020x` |

## Per Sequence

| Seq | BF16 us | P2-N us | P3-B us | P3-B/BF16 | P3-B/P2-N | Cosine | Argmax |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 57279.362 | 39298.401 | 38503.967 | 1.488x | 1.021x | 1.000 | `True` |
| 64 | 92993.858 | 85607.933 | 84052.254 | 1.106x | 1.019x | 1.000 | `True` |

## Caveats

- P3-B is selected-layer composition only; it does not bank a fused P3 kernel.
- Router and shared expert remain on the existing BF16 paths.
- Active routed expert BF16 shadows are deleted before P2-N/P3-B candidates run.
- P3-C server readiness and RC quality promotion remain separate gates.

## Hard Gates

| Gate | Value |
|---|---|
| Predecessors pass | `true` |
| Numeric | `true` |
| Final stack cosine min | `0.9999491846441616` |
| Final stack argmax | `true` |
| Active BF16 shadow absent | `true` |
| Reload trap installed | `true` |
| Reload not called | `true` |
| Speed vs P2-N reference | `true` |
| Banked fused kernel flag is false | `True` |
| Banked server path flag is false | `True` |
| BF16 active expert bytes | `6442450944` |
| Packed active expert bytes | `2415919136` |
| Memory after deleting active BF16 GiB | `2.5251412391662598` |
| Reload trap status | `installed` |
| Shadow absence checks | `{'after_delete': True, 'after_p2n_T16': True, 'after_p3b_T16': True, 'after_p2n_T64': True, 'after_p3b_T64': True}` |

## Decision

Bank P3-B selected-prefill composition only. Do not claim fused-kernel, server, or RC promotion.

P3-C server integration and P3-D/RC quality remain separate gates.

## Run Log Tail

```text
        "median_us": 39298.40087890625,
        "min_us": 39298.40087890625,
        "max_us": 39298.40087890625,
        "all_us": [
          39298.40087890625
        ]
      },
      "64": {
        "warmup": 0,
        "iters": 1,
        "repeats": 1,
        "median_us": 85607.9330444336,
        "min_us": 85607.9330444336,
        "max_us": 85607.9330444336,
        "all_us": [
          85607.9330444336
        ]
      }
    },
    "p3b_selected_prefill": {
      "16": {
        "warmup": 0,
        "iters": 1,
        "repeats": 1,
        "median_us": 38503.96728515625,
        "min_us": 38503.96728515625,
        "max_us": 38503.96728515625,
        "all_us": [
          38503.96728515625
        ]
      },
      "64": {
        "warmup": 0,
        "iters": 1,
        "repeats": 1,
        "median_us": 84052.25372314453,
        "min_us": 84052.25372314453,
        "max_us": 84052.25372314453,
        "all_us": [
          84052.25372314453
        ]
      }
    }
  },
  "memory": {
    "p2n_peak": {
      "16": {
        "before_gib": 2.5252022743225098,
        "after_gib": 2.5252633094787598,
        "peak_gib": 2.5908203125
      },
      "64": {
        "before_gib": 2.5254464149475098,
        "after_gib": 2.5256905555725098,
        "peak_gib": 2.59891414642334
      }
    },
    "p3b_peak": {
      "16": {
        "before_gib": 2.5252633094787598,
        "after_gib": 2.5253243446350098,
        "peak_gib": 2.59088134765625
      },
      "64": {
        "before_gib": 2.5256295204162598,
        "after_gib": 2.5258736610412598,
        "peak_gib": 2.59909725189209
      }
    }
  },
  "shadow_absence_checks": {
    "after_delete": true,
    "after_p2n_T16": true,
    "after_p3b_T16": true,
    "after_p2n_T64": true,
    "after_p3b_T64": true
  },
  "reload_trap": {
    "installed": true,
    "status": "installed"
  },
  "passes": {
    "predecessors_pass": true,
    "numeric": true,
    "final_stack_cosine_min": 0.9999491846441616,
    "final_stack_argmax_match": true,
    "no_active_bf16_shadow": true,
    "reload_trap_installed": true,
    "reload_not_called": true,
    "speed_vs_p2n_reference": true,
    "all": true
  },
  "reload_calls": [],
  "notes": [
    "P3-B is selected-layer composition only; it does not bank a fused P3 kernel.",
    "Router and shared expert remain on the existing BF16 paths.",
    "Active routed expert BF16 shadows are deleted before P2-N/P3-B candidates run.",
    "P3-C server readiness and RC quality promotion remain separate gates."
  ]
}
```
