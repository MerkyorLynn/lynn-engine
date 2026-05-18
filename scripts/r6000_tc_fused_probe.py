#!/usr/bin/env python3
"""Quick fused-GEMM latency probe on R6000."""
import torch, torch.nn.functional as F
from safetensors.torch import load_file

d = load_file("/root/autodl-tmp/reports/qwen36_35b/p135_repacked_fixtures_official_w4a16_slot/layer_28_prompt_00_slots.safetensors", device="cuda")
h = d["hidden_in"].to(torch.bfloat16).view(1, 2048)
rw = d["routing_weights"]
sg = d["slot_gate_up_weight"].to(torch.bfloat16)  # [8, 1024, 2048]
sd = d["slot_down_weight"].to(torch.bfloat16)      # [8, 2048, 512]
expected = d["routed_output"].to(torch.bfloat16)

top_k = 8
W_fused_gu = sg.reshape(top_k * 1024, 2048)  # [8192, 2048]
W_down_t = sd.transpose(1, 2).contiguous()     # [8, 512, 2048]
rw_bf16 = rw.to(torch.bfloat16).view(top_k, 1, 1)

def run():
    gate_up_f = torch.mm(h, W_fused_gu.t())  # [1, 8192]
    gur = gate_up_f.view(top_k, 1, 1024)
    inter_f = F.silu(gur[:, :, :512]) * gur[:, :, 512:]
    do = torch.bmm(inter_f, W_down_t)  # [8, 1, 2048]
    return (do * rw_bf16).sum(0)

out = run()
diff = (out.float() - expected.view(1, 2048).float()).abs().max().item()
print(f"Fused gate_up mm + bmm down: max_abs vs stored = {diff:.6e}")

for _ in range(50):
    run()
torch.cuda.synchronize()
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(500):
    run()
e.record()
torch.cuda.synchronize()
ms = s.elapsed_time(e) / 500
print(f"Latency: {ms:.4f} ms")
print(f"vs Triton 0.059: {(ms/0.059 - 1)*100:+.0f}%")
