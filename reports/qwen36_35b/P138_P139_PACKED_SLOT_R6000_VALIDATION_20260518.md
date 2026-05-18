# P138/P139 Packed Slot NVFP4 R6000 Validation 2026-05-18

Verdict: GREEN.

Kimi's packed slot repack path was validated on R6000 against the official Qwen3.6-35B-A3B W4A16 NVFP4 model and the p135 slot-order BF16 fixtures.

| Stage | Result |
|---|---:|
| p138 packed fixtures | 18/18 exported |
| packed size per fixture | 15.73 MB |
| BF16 slot equivalent per fixture | 50.33 MB |
| size reduction | ~68.7% |
| p139 exact contract | 18/18 GREEN |
| p139 max_abs_max | 0.0 |
| p139 cosine | 1.0 |

This confirms the packed slot fixture layout is bit-exact with the BF16 p135 slot weights after NVFP4 unpack + scale application. It is suitable as the next native packed MoE kernel input contract.

R6000 artifacts:

- `/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518/manifest.json`
- `/root/autodl-tmp/reports/qwen36_35b/p139_slot_packed_contract_kimi_20260518.json`
