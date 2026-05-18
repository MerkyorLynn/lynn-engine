# V4 W4A16 Disk And Runner Status

Date: 2026-05-17 22:32 CST

## Route

V4-Pro 35B W4A16 is now the primary quality-serving candidate. The old
27B W4A8+MTP track remains useful as the small native speed verifier, but it is
not the only mainline.

## R6000

Freed `/root/autodl-tmp` from 87G available to 153G available.

Removed:

- `/root/autodl-tmp/models/Qwen3.6-35B-A3B-S1-S5v2-S4`
- `/root/autodl-tmp/reports/q4km_build.log`
- `/root/autodl-tmp/reports/q4km_build_vflash_single_20260514T195300.log`

Preserved:

- `/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-bf16-final`
- `/root/autodl-tmp/models/lynn-27b-w4a8-nvfp4-v2`
- `/root/autodl-tmp/models/mtp_sidecars`
- `/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged`

Active runners:

- V4-Pro BF16 direct download to
  `/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged`
- W4A16 pack watcher:
  `/root/autodl-tmp/reports/v4pro_download/watch_and_pack_v4pro_w4a16_20260517_2219.sh`
- W4A16/W4A8 matrix watcher:
  `/root/autodl-tmp/reports/v4pro_w4a16_w4a8/watch_and_run_matrix_20260517_2222.sh`

## A100

Freed `/mnt/data2` from 39G available to 127G available.

Preserved:

- 27B BF16 base
- official Qwen3.6 MTP sidecar
- v27 best MTP sidecar
- v45/v46 hard-miss comparison sidecars
- warm-start-aligned MTP sidecar
- structured_v10 and structured_v16 W4A8 recovery overlays

Removed old failed/low-ROI MTP sidecars and old structured/json-focus overlay
artifacts. This keeps A100 ready for 35B W4A8/MTP adaptation if V4-Pro W4A16
quality holds.
