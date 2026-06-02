#!/usr/bin/env python3
"""Quantify the qkv-fusion lever on Spark: 3 separate q/k/v GEMVs vs 1 fused.

full_attn_t1.qkv is 2.2 ms/tok and (without LYNN_FULL_ATTN_QKV_FUSED) runs q_proj,
k_proj, v_proj as THREE separate M=1 GEMVs. Fusing them into one [hidden -> q+2kv]
GEMV cuts 3 launches to 1 (the decode bottleneck is launch overhead). This bf16
microbench measures the 3-vs-1 speedup at representative Qwen3.6 full-attn dims, no
model load. (Precision-agnostic: the launch-reduction trend holds for the NVFP4 path.)
"""
import torch

HID = 2048
QD, KVD = 4096, 512          # q: 32x128 ; k,v: 4x128 (GQA), representative
dev = "cuda"
assert torch.cuda.is_available()
torch.manual_seed(0)
cap = torch.cuda.get_device_capability()
print(f"device sm_{cap[0]}{cap[1]}  HID={HID} QD={QD} KVD={KVD}")

x = (torch.randn(1, HID, device=dev) * 0.3).bfloat16()
Wq = (torch.randn(QD, HID, device=dev) * 0.02).bfloat16()
Wk = (torch.randn(KVD, HID, device=dev) * 0.02).bfloat16()
Wv = (torch.randn(KVD, HID, device=dev) * 0.02).bfloat16()
Wqkv = torch.cat([Wq, Wk, Wv], dim=0).contiguous()   # [QD+2KVD, HID]


def separate():
    q = x @ Wq.t(); k = x @ Wk.t(); v = x @ Wv.t()
    return q, k, v


def fused():
    y = x @ Wqkv.t()
    return y.split((QD, KVD, KVD), dim=-1)


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


# correctness
qs, ks, vs = separate(); qf, kf, vf = fused()
ok = torch.allclose(qs, qf) and torch.allclose(ks, kf) and torch.allclose(vs, vf)
ts = bench(separate); tf = bench(fused)
print(f"separate (3 GEMVs): {ts:7.2f} us")
print(f"fused    (1 GEMV) : {tf:7.2f} us   ({ts/tf:.2f}x faster)  exact={ok}")
print("VERDICT:", "WIRE IT (LYNN_FULL_ATTN_QKV_FUSED + prep fused weight)" if ts/tf > 1.15 else "marginal")
