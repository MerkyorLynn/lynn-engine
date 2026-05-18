# P138/P139 Packed Slot NVFP4 Gzip R6000 Validation 2026-05-18

Verdict: GREEN for offline distribution.

Kimi's compressed packed-slot path was validated on R6000 after fixing the gzip load path to move `safetensors.torch.load()` tensors onto the requested device.

| Metric | Result |
|---|---:|
| p138 compressed fixtures | 18/18 exported |
| p139 gzip decode contract | 18/18 GREEN |
| p139 max_abs_max | 0.0 |
| p139 cosine | 1.0 |
| mean gzip fixture size | 14.16 MB |
| BF16 slot equivalent | 50.33 MB |
| size reduction vs BF16 | 71.86% |
| p139 gzip load mean | 115.58 ms |
| p139 unpack mean | 6.58 ms |

Conclusion: gzip is useful for offline fixture distribution/storage, but it is not a decode hot-path format. Production/native kernels should consume uncompressed packed slot sidecars or memory-mapped buffers.

R6000 artifacts:

- `/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_gz_20260518/manifest.json`
- `/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_summary_kimi_gz_20260518.json`
- `/root/autodl-tmp/reports/qwen36_35b/p139_slot_packed_contract_kimi_gz_20260518.json`
