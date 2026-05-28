# Qwen3.6-35B-A3B APEX-MTP Think-Off Check

Date: 2026-05-28

## Operational Status

- Q4_K_M-MTP temporary server on `18099`: stopped.
- Q4_K_M-MTP quality run: stopped during `mmlu500_5shot_thinking_on_32k` after
  266/500 partial rows.
- Production fallback `lynn-apex-mtp-llamacpp.service` on `18098`: active and
  healthy after cleanup.
- Voice services were left untouched.

## Completed Results

| Model / Runtime | Endpoint | MMLU 500 5-shot | GPQA Diamond 198 | Parse Fail |
|---|---|---:|---:|---:|
| Q4_K_M-MTP self build | `18099`, stopped | 81.40% (`407/500`) | 41.41% (`82/198`) | 12 / 14 |
| APEX-MTP I-Balanced production | `18098` | 82.20% (`411/500`) | 42.42% (`84/198`) | 17 / 19 |

Historical anchors from the 35B quality table:

| Model | MMLU 500 5-shot | GPQA Diamond 198 |
|---|---:|---:|
| BF16 official | 86.40% (`432/500`) | 45.45% (`90/198`) |
| Q4_K_M-imatrix base GGUF | 83.00% (`415/500`) | 50.00% (`99/198`) |
| Lynn-native W4A16 NVFP4 | 84.40% (`422/500`) | 49.49% (`98/198`) |

## Conclusion

The think-off quality drop is real. Q4_K_M-MTP is not just slightly below the
base Q4_K_M anchor; GPQA falls into the low-40% band. The same pattern appears
on the existing APEX-MTP I-Balanced production package, so this is not enough
evidence to blame only the self-quantized Q4_K_M build.

Current recommended split:

- **Think-off quality default:** keep the older Q4_K_M base / W4A16 anchors.
- **Thinking-on workflows:** APEX-MTP I-Balanced remains useful because previous
  32K thinking-on results were much stronger.
- **Q4_K_M-MTP artifact:** keep as a valid research artifact, but do not publish
  it as the default speed+quality answer until AR-only quality isolation and
  MTP accept-rate work are done.

## Copied Artifacts

```text
reports/qwen36_35b/q4km_mtp_quality_20260528_181812/
reports/qwen36_35b/apex_thinkoff_20260528_191853/
reports/mtp/qwen36_q4km_mtp_tps_20260528/
```
