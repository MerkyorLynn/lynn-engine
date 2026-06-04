# Stage 6 Decode GPU-Idle Probe Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (decode GPU-idle ROI probe recorded) |
| Decision | `PASS_DECODE_GPU_IDLE_PROBE_RECORDED` |
| ROI signal | `BORDERLINE_REMEASURE_OR_NSIGHT` |
| Token delta | `48` |
| Wall ms/token | `46.824` |
| CUDA kernel busy ms/token | `35.257` |
| Estimated host gap/idle ms/token | `11.568` |
| Estimated GPU busy ratio | `0.753` |
| Estimated host gap fraction | `0.247` |
| CUDA launches/token | `1969.021` |
| CPU CUDA API ms/token | `3.960` |
| CPU CUDA API calls/token | `758.458` |
| Short runner TPS | `41.363` |
| Long runner TPS | `42.188` |
| Speed promotion | `False` |
| Compiled-loop default | `False` |
| CUDA graph route | `False` |

## Boundary

- This banks only a compiled-loop ROI measurement.
- It does not bank a speed gain, a CUDA graph route, or a default runtime change.
- If the host-gap fraction is high, the next step is a small compiled-loop/MTP-light prototype; if low, do not spend month-scale runtime work here.

## Caveat

GPU busy is estimated from PyTorch profiler self CUDA time using N/2N delta; treat as a go/no-go ROI probe and confirm deep investments with Nsight if borderline.
