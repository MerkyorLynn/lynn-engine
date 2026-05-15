#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.nvfp4_runtime import load_packed_nvfp4_linear
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native

def bench(fn,warmup,iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return float(s.elapsed_time(e)/iters)

def cmp(a,b):
    af=a.float().flatten(); bf=b.float().flatten(); d=af-bf; denom=torch.linalg.vector_norm(bf).clamp_min(1e-12)
    return {'cosine':float(torch.dot(af,bf)/(torch.linalg.vector_norm(af).clamp_min(1e-12)*torch.linalg.vector_norm(bf).clamp_min(1e-12))),'rel_l2':float(torch.linalg.vector_norm(d)/denom),'max_abs':float(d.abs().max())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v8',required=True); ap.add_argument('--layer',type=int,default=0); ap.add_argument('--weight',default='linear_attn.in_proj_qkv.weight'); ap.add_argument('--out',required=True); ap.add_argument('--iters',type=int,default=500); ap.add_argument('--warmup',type=int,default=50); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    base=f"model.language_model.layers.{args.layer}.{args.weight.removesuffix('.weight')}"
    linear=load_packed_nvfp4_linear(Path(args.v8),base,name=args.weight,device=args.device)
    gen=torch.Generator(device=args.device); gen.manual_seed(20260515)
    x=torch.randn((1,linear.in_features),device=args.device,dtype=torch.bfloat16,generator=gen)
    linear._native_scale_b()
    def native_forward(): return linear(x[0],backend='native_scaled_mm')
    def fast_forward(): return linear.forward_native_fast_2d(x)
    def scalar_forward(): return linear(x[0],backend='scalar_bridge')
    nf=native_forward(); ff=fast_forward(); sf=scalar_forward()
    out={'schema_version':'lynn-engine-p5d-fastpath-probe-v1','layer':args.layer,'weight':args.weight,'shape':{'n':linear.out_features,'k':linear.in_features},'comparisons':{'fast_vs_native':cmp(ff,nf),'fast_vs_scalar':cmp(ff,sf)},'latency_ms':{'native_forward':bench(native_forward,args.warmup,args.iters),'fast_native_forward_2d':bench(fast_forward,args.warmup,args.iters),'scalar_bridge':bench(scalar_forward,args.warmup,args.iters)}}
    out['derived']={'fast_vs_scalar_ratio':out['latency_ms']['scalar_bridge']/out['latency_ms']['fast_native_forward_2d'],'fast_vs_native_ratio':out['latency_ms']['native_forward']/out['latency_ms']['fast_native_forward_2d']}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
