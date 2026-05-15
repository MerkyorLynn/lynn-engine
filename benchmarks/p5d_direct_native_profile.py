#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.nvfp4_runtime import PackedNVFP4Linear
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native

def bench(fn,warmup,iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return float(s.elapsed_time(e)/iters)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v8',required=True); ap.add_argument('--layer',type=int,default=0); ap.add_argument('--weight',default='linear_attn.in_proj_qkv.weight'); ap.add_argument('--out',required=True); ap.add_argument('--iters',type=int,default=500); ap.add_argument('--warmup',type=int,default=50); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    base=f"model.language_model.layers.{args.layer}.{args.weight.removesuffix('.weight')}"
    with safe_open(Path(args.v8)/'model.safetensors',framework='pt',device='cpu') as st:
        linear=PackedNVFP4Linear.from_safetensors(st,base,name=args.weight,device=args.device)
    gen=torch.Generator(device=args.device); gen.manual_seed(20260515)
    x=torch.randn((1,linear.in_features),device=args.device,dtype=torch.bfloat16,generator=gen)
    scale_b=linear._native_scale_b()
    act_packed, scale_a = quantize_fp4_m1_native(x)
    def fused_quant(): return quantize_fp4_m1_native(x)
    def direct_fused_plus_mm():
        ap,sa=quantize_fp4_m1_native(x)
        return torch._scaled_mm(ap.view(torch.float4_e2m1fn_x2), linear.weight_packed.view(torch.float4_e2m1fn_x2).t(), scale_a=sa, scale_b=scale_b, out_dtype=torch.float16)
    def scaled_mm_only():
        return torch._scaled_mm(act_packed.view(torch.float4_e2m1fn_x2), linear.weight_packed.view(torch.float4_e2m1fn_x2).t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
    def native_forward(): return linear(x[0],backend='native_scaled_mm')
    def scalar_forward(): return linear(x[0],backend='scalar_bridge')
    lat={
      'fused_quantize': bench(fused_quant,args.warmup,args.iters),
      'scaled_mm_only': bench(scaled_mm_only,args.warmup,args.iters),
      'direct_fused_quantize_plus_scaled_mm': bench(direct_fused_plus_mm,args.warmup,args.iters),
      'packed_linear_native_forward': bench(native_forward,args.warmup,args.iters),
      'packed_linear_scalar_bridge': bench(scalar_forward,args.warmup,args.iters),
    }
    out={'schema_version':'lynn-engine-p5d-direct-native-profile-v1','layer':args.layer,'weight':args.weight,'shape':{'n':linear.out_features,'k':linear.in_features},'latency_ms':lat,'derived':{'wrapper_over_direct_ms':lat['packed_linear_native_forward']-lat['direct_fused_quantize_plus_scaled_mm'],'direct_over_mm_ms':lat['direct_fused_quantize_plus_scaled_mm']-lat['scaled_mm_only'],'native_vs_scalar_ratio':lat['packed_linear_scalar_bridge']/lat['packed_linear_native_forward'],'direct_vs_scalar_ratio':lat['packed_linear_scalar_bridge']/lat['direct_fused_quantize_plus_scaled_mm']}}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
