#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.incremental_decode import decode_linear_attn
from engine.loader import load_qwen36_layer
from engine.nvfp4_runtime import PackedNVFP4Linear
from engine.qwen36_linear_attn_block import CONV_KERNEL, HEAD_K_DIM, HEAD_V_DIM, NUM_V_HEADS
LINEAR_ATTN_WEIGHT_NAMES=["linear_attn.in_proj_qkv.weight","linear_attn.in_proj_z.weight","linear_attn.in_proj_b.weight","linear_attn.in_proj_a.weight","linear_attn.out_proj.weight"]
def bench(fn,warmup,iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return float(s.elapsed_time(e)/iters)
def cmp(a,b):
    af=a.float().flatten(); bf=b.float().flatten(); d=af-bf; denom=torch.linalg.vector_norm(bf).clamp_min(1e-12); cos=torch.dot(af,bf)/(torch.linalg.vector_norm(af).clamp_min(1e-12)*torch.linalg.vector_norm(bf).clamp_min(1e-12))
    return {"mean_abs":float(d.abs().mean()),"max_abs":float(d.abs().max()),"rel_l2":float(torch.linalg.vector_norm(d)/denom),"cosine":float(cos)}
def load_packed(v8,layer,name,device):
    base=f"model.language_model.layers.{layer}.{name.removesuffix('.weight')}"
    with safe_open(Path(v8)/'model.safetensors',framework='pt',device='cpu') as st:
        return PackedNVFP4Linear.from_safetensors(st,base,name=name,device=device,default_backend='scalar_bridge')
def with_packed(resident,v8,layer,device):
    w=dict(resident)
    for name in LINEAR_ATTN_WEIGHT_NAMES: w[name]=load_packed(v8,layer,name,device)
    return w
def make_inputs(device,dtype,seed):
    gen=torch.Generator(device=device); gen.manual_seed(seed)
    h=torch.randn(1,1,2048,device=device,dtype=dtype,generator=gen)
    state=torch.randn(1,NUM_V_HEADS,HEAD_K_DIM,HEAD_V_DIM,device=device,dtype=torch.float32,generator=gen)*0.01
    conv=torch.randn(1,8192,CONV_KERNEL-1,device=device,dtype=dtype,generator=gen)*0.01
    return h,state,conv
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v8',required=True); ap.add_argument('--layer',type=int,default=0); ap.add_argument('--out',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--dtype',choices=['bf16','fp16'],default='bf16'); ap.add_argument('--iters',type=int,default=80); ap.add_argument('--warmup',type=int,default=8); ap.add_argument('--seed',type=int,default=20260515); args=ap.parse_args()
    dtype=torch.bfloat16 if args.dtype=='bf16' else torch.float16
    resident,_=load_qwen36_layer(args.v8,args.layer,num_experts=256,device=args.device,dequant_dtype=dtype)
    packed=with_packed(resident,args.v8,args.layer,args.device)
    h,state,conv=make_inputs(args.device,dtype,args.seed)
    cases={}
    for name,w in [('resident',resident),('packed_scalar',packed)]:
        ref=decode_linear_attn(h,w,state.clone(),conv.clone(),recurrent_backend='torch')
        fused=decode_linear_attn(h,w,state.clone(),conv.clone(),recurrent_backend='triton_fused_prepare')
        cases[name]={
          'comparisons':{'out':cmp(fused[0],ref[0]),'state':cmp(fused[1],ref[1]),'conv':cmp(fused[2],ref[2])},
          'latency_ms':{
            'torch_recurrent_backend':bench(lambda: decode_linear_attn(h,w,state.clone(),conv.clone(),recurrent_backend='torch')[0],args.warmup,args.iters),
            'triton_fused_prepare_backend':bench(lambda: decode_linear_attn(h,w,state.clone(),conv.clone(),recurrent_backend='triton_fused_prepare')[0],args.warmup,args.iters),
          }
        }
        cases[name]['derived']={'speed_ratio':cases[name]['latency_ms']['torch_recurrent_backend']/cases[name]['latency_ms']['triton_fused_prepare_backend']}
    result={'schema_version':'lynn-engine-p6e-decode-recurrent-backend-gate-v1','layer':args.layer,'model':args.v8,'device':torch.cuda.get_device_name(args.device),'cases':cases}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
