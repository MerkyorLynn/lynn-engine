# P196 W4A8 Structured Content Gate — Report Template

<!-- Fill in after running: bash scripts/r6000_qwen35_9b_w4a8_structured_content_gate.sh -->

**Date:** __________
**Model:** Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0
**Branch:** mimo/qwen35-9b-nvfp4-release-gates-20260519

## Verdict

| Variant | Pass Rate | Rate vs A16 | TPS Mean | TPS vs A16 | Verdict |
|---|---|---|---|---|---|
| W4A16 (reference) | _/10 = __% | 1.00 | ___ | 1.00 | BASELINE |
| W4A8 full | _/10 = __% | ___ | ___ | ___ | ________ |
| W4A8 gateup | _/10 = __% | ___ | ___ | ___ | ________ |

**Overall: __________**

## Per-Test Results

| Test ID | W4A16 | W4A8 full | W4A8 gateup |
|---|---|---|---|
| json_object | | | |
| json_array | | | |
| python_function | | | |
| markdown_table | | | |
| yaml_config | | | |
| csv_data | | | |
| key_value_pairs | | | |
| numbered_list | | | |
| regex_pattern | | | |
| json_nested | | | |

## Failure Details

<!-- List any failures with the actual output that failed validation -->

## Decision

<!-- 
Choose ONE:

### ✅ W4A8_CONTENT_GREEN
W4A8 matches W4A16 structural quality. Proceed to numeric admission (P193).

### ⚠️ W4A8_CONTENT_AMBER
W4A8 has some structural degradation. Investigate before proceeding.

### 🔴 RED_FALLBACK_A16
W4A8 structural quality is unacceptable. **Fall back to W4A16.**
质量漂移就回 A16，不纠结。
-->
