# Lynn V4-Pro 35B NVFP4 Pivot Probe

Date: 2026-05-17

## Question

If Lynn V4-Pro 35B keeps its quality after W4A16 NVFP4 and the disk/VRAM delta
versus 27B A3B is only a few GiB, pruning should stop being the default quality
route. The 27B model remains a speed/edge/runtime specialization, while
V4-Pro 35B W4A16 NVFP4 becomes the primary quality-serving candidate.

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

## 2026-05-17 V8-RTN W4A4 Result

Spark V8-RTN compressed-tensors NVFP4 results were strongly negative:

| Candidate | Q4_K_M / W4A16 | V8-RTN / W4A4 | Delta |
|---|---:|---:|---:|
| V4-Pro | 83.60% / 42.42% | 63.40% / 35.86% | -20.2pp / -6.6pp |
| V4-Flash | 80.60% / 43.94% | 63.00% / 37.88% | -17.6pp / -6.1pp |

Decision: do not treat V8-RTN W4A4 as the quality pivot route. The data says
activation quantization erased most of Lynn's distillation gain and pulled the
35B candidates back toward the Qwen base band.

The next 35B quantization gate is therefore Lynn-native **W4A16 weight-only**
NVFP4: quantize weights to per-16 E2M1 NVFP4 while keeping runtime activations
in BF16. If W4A16 holds quality, test a separate W4A8 recovery bridge. W4A4 is
now an extreme compression candidate, not the main quality route.

## Route Decision

As of the V8-RTN result, V4-Pro W4A16 is no longer a side experiment. It is the
primary candidate for the default high-quality Lynn serving artifact.

The 155 TPS work becomes a race between two branches instead of a commitment to
27B:

| Branch | Purpose | Continue Condition |
|---|---|---|
| V4-Pro 35B W4A16 NVFP4 | quality-first default if it preserves distillation gain | MMLU stays near the Q4_K_M band, size delta over 27B is small, smoke gates pass |
| Lynn 27B W4A8+MTP | smallest/fastest runtime specialization | only if it gives a decisive TPS win without unacceptable structured quality loss |

Do not spend more time making V4-Pro W4A4 look good unless W4A16 is impossible
to serve. Do not start a 35B W4A8 bridge until W4A16 has a measured quality and
speed baseline.

## Runner

R6000 Lynn-native pivot runner:

```bash
scripts/r6000_v4pro_nvfp4_pivot_probe.sh
```

Spark/SGLang-compatible v8-RTN batch quantization runner:

```bash
scripts/spark_v4_35b_v8_rtn_quant_batch.sh
```

Default source/target:

```text
BF16_MODEL=/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged
OUT_MODEL=/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-lynn-native-w4a16-nvfp4-v0
COMPARE_27B_MODEL=/root/autodl-tmp/models/lynn-27b-a3b-w4a8-nvfp4-v2
```

If the BF16 oracle was deleted, the runner downloads it first:

```text
DOWNLOAD_BF16_IF_MISSING=1
DOWNLOAD_SOURCE=modelscope
MS_REPO=Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B
```

Set `DOWNLOAD_SOURCE=hf` to use:

```text
HF_REPO=Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B
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
2. Pack Lynn-native per16 W4A16/weight-only NVFP4 from BF16 with
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

## Parallel Work Policy

Do not pause the active 27B A3B work while V4-Pro/Flash quality numbers are
running.

| Machine | Continue Now | Why |
|---|---|---|
| R6000 | V4-Pro W4A16 pack/smoke/size/TPS plus P118/P119 native verifier prep | V4-Pro W4A16 is now the quality-first serving candidate; 27B remains the smaller speed branch. |
| A100 | 27B W4A8 quality/MTP label construction and sidecar training | These labels remain useful even if 35B becomes the quality default; they also keep the 27B speed route alive. |
| Spark | V4-Pro/Flash BF16 and v8-RTN MMLU/GPQA | This decides whether 35B NVFP4 should become the quality-first route. |

Only start a 35B W4A8 conversion line if W4A16 quality passes and W4A16 speed
is too slow. MTP work should be tested against the V4-Pro W4A16 artifact once it
loads, because quality-preserving 35B may be the better default even if the 27B
branch remains the most aggressive TPS candidate.
