# Lynn Engine Route Decision: V4-Pro W4A16 First

Date: 2026-05-17

## Decision

Promote V4-Pro 35B W4A16 NVFP4 from pivot experiment to primary quality-serving
candidate.

Keep Lynn 27B W4A8+MTP alive as the small/fast runtime branch, but stop treating
it as the only path to the user-facing default. Today's quality data says the
main risk is activation quantization erasing distillation gains, so the next
serious artifact is W4A16 weight-only, not W4A4 or immediate W4A8.

## Evidence

Spark V8-RTN W4A4 erased most of the 35B distillation advantage:

| Candidate | Q4_K_M / W4A16 | V8-RTN / W4A4 | Delta |
|---|---:|---:|---:|
| V4-Pro | 83.60% / 42.42% | 63.40% / 35.86% | -20.2pp / -6.6pp |
| V4-Flash | 80.60% / 43.94% | 63.00% / 37.88% | -17.6pp / -6.1pp |

That points at activation quantization as the quality killer. A larger 35B
W4A16 artifact that costs only a few GiB more is more attractive than spending
another day trying to recover a 27B W4A8 artifact that still trails the
distilled 35B family on MMLU.

## Immediate Plan

1. Finish direct R6000 V4-Pro BF16 download.
2. Pack Lynn-native W4A16 NVFP4 from BF16.
3. Run manifest, logit, generation, size, and load smoke.
4. Run a W4A16/W4A8 generation matrix on the same artifact:
   `off` = W4A16, `gateup/full` = W4A8 fake-FP8 activation probes.
5. Use Spark quality numbers as the main MMLU/GPQA gate when ready.
6. If quality holds, benchmark V4-Pro W4A16 serving and test MTP sidecar wiring.

## Branch Policy

| Branch | Role |
|---|---|
| V4-Pro 35B W4A16 NVFP4 | primary quality/default-serving candidate |
| Lynn 27B W4A8+MTP | smallest high-speed specialization and native verifier development path |
| V4-Pro/Flash W4A4 V8-RTN | deprioritized extreme compression candidate |
| 35B W4A8 | only after W4A16 quality passes and speed needs a middle ground |

Current R6000 automation:

- `watch_and_pack_v4pro_w4a16_20260517_2219.sh` waits for direct BF16 download,
  then packs W4A16 and runs load/logit/generation smoke.
- `watch_and_run_matrix_20260517_2222.sh` waits for the W4A16 artifact, then
  runs `scripts/r6000_v4pro_w4a16_w4a8_matrix.sh`.

Renewal decision tomorrow should be based on whether V4-Pro W4A16 quality holds
and whether A100 is still needed for 35B W4A8/MTP adaptation after that.
