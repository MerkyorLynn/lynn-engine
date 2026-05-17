# Lynn V4-Pro 35B NVFP4 Pivot Probe

Date: 2026-05-17

## Question

If Lynn V4-Pro 35B keeps its quality after NVFP4 and the disk/VRAM delta versus
27B A3B is only a few GiB, pruning should stop being the default quality route.
The 27B model would remain the speed/edge/runtime specialization, while V4-Pro
35B NVFP4 could become the high-quality single-card default.

## Starting Signal

Spark quality matrix:

| Candidate | MMLU 500 5-shot | GPQA Diamond 198 0-shot |
|---|---:|---:|
| Lynn V4-Pro Q4_K_M | 83.60% | 42.42% |
| Lynn V4-Flash Q4_K_M | 80.60% | 43.94% |
| Lynn 27B BF16 | 72.00% | 44.44% |
| Lynn 27B NVFP4-V2 dequant | 69.00% | 41.92% |
| Lynn 27B NVFP4-V0 production gamma | 68.00% | 40.40% |

The important asymmetry: V4-Pro has about `+11.6pp` MMLU over 27B BF16, while
GPQA is clustered in the low-mid 40s. If V4-Pro NVFP4 loses only a few MMLU
points, it may still dominate the 27B line on factual coverage.

## Runner

```bash
scripts/r6000_v4pro_nvfp4_pivot_probe.sh
```

Default source/target:

```text
BF16_MODEL=/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged
OUT_MODEL=/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-lynn-native-nvfp4-v0
COMPARE_27B_MODEL=/root/autodl-tmp/models/lynn-27b-a3b-w4a8-nvfp4-v2
```

If the BF16 oracle was deleted, the runner downloads it first:

```text
DOWNLOAD_BF16_IF_MISSING=1
DOWNLOAD_SOURCE=modelscope
MS_REPO=Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged
```

Set `DOWNLOAD_SOURCE=hf` to use:

```text
HF_REPO=nerkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged
```

Disk handling defaults:

```text
CLEAN_DISK_IF_NEEDED=1
MIN_FREE_GIB_FOR_DOWNLOAD=90
MIN_FREE_GIB_FOR_PACK=35
```

The runner may delete regenerable pivot scratch logs and incomplete pivot output
directories. It does not automatically delete unrelated model directories; when
space is still tight it prints the largest model/report directories into the
log for manual or follow-up cleanup.

## Probe Steps

1. Ensure or download V4-Pro BF16 oracle.
2. Pack Lynn-native per16 NVFP4 from BF16 with
   `scripts/a100_pack_lynn_native_nvfp4.py`.
3. Scan BF16/native-NVFP4/existing-v8 manifests.
4. Run resident BF16-vs-native-NVFP4 top-k/logit smoke.
5. Run a structured generation smoke on native NVFP4.
6. Compare disk size against the current 27B NVFP4-V2 artifact.

## Decision Gate

Promote this line from curiosity to mainline if all hold:

| Gate | Threshold |
|---|---|
| Pack/load | native NVFP4 manifest builds and resident runner loads. |
| Smoke quality | BF16-vs-NVFP4 top-k/logit smoke has no catastrophic first-token drift. |
| MMLU | V4-Pro NVFP4 remains at least `80%` on the 500-question gate, or clearly above 27B BF16. |
| GPQA | no worse than the 27B NVFP4 band unless MMLU gain is decisive. |
| Size | native 35B NVFP4 delta over 27B NVFP4 is small enough for R6000/Spark product memory. |
| TPS | not a 155 replacement yet; measure after quality/size pass. |

Until this gate lands, keep 27B A3B W4A8+MTP as the active R6000 155 TPS line.
