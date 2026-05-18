# Qwen3.6 35B W4A16 Repack Inventory

This is a read-only inventory for the offline serving-layout repack route.

- model: `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0`
- quantized tensors: `553`
- missing index keys: `0`
- language layers: `40`; linear-attn `30`, full-attn `11`

## Bucket Summary

| Bucket | Records | Packed GiB | Scale Elements | Shards | Top Modules |
|---|---:|---:|---:|---:|---|
| `active_moe` | 82 | 15.3750 | 2063597568 | 7 | mlp.experts.gate_up_proj (41), mlp.experts.down_proj (41) |
| `full_attn` | 44 | 0.1396 | 18743296 | 7 | self_attn.k_proj (11), self_attn.o_proj (11), self_attn.q_proj (11), self_attn.v_proj (11) |
| `linear_attn` | 150 | 0.4706 | 63160320 | 7 | linear_attn.in_proj_qkv (30), linear_attn.in_proj_z (30), linear_attn.out_proj (30), linear_attn.in_proj_a (30) |
| `mtp` | 1 | 0.0039 | 524288 | 1 | mtp.fc (1) |
| `shared_moe` | 164 | 0.0601 | 8066176 | 7 | mlp.shared_expert.down_proj (41), mlp.shared_expert.gate_proj (41), mlp.shared_expert.up_proj (41), mlp.shared_expert_gate (41) |
| `visual` | 112 | 0.2078 | 27885312 | 1 | model.visual.blocks.0.mlp.linear_fc1 (1), model.visual.blocks.0.mlp.linear_fc2 (1), model.visual.blocks.1.mlp.linear_fc1 (1), model.visual.blocks.1.mlp.linear_fc2 (1) |

## Repack Order

1. `active_moe_gateup_down` (`active_moe`): largest repeated decode boundary; Q4_K_M llama.cpp suggests serving layout matters more than another scalar tile sweep.
2. `shared_moe_gateup_down_gate` (`shared_moe`): hot per-token shared expert path; keep BF16 activation semantics.
3. `linear_attn_projection_pack` (`linear_attn`): 30 hybrid SSM layers carry multiple projection launches per token.
4. `full_attn_qkvo_pack` (`full_attn`): 11 full-attention layers still need exact RoPE/cache order.

## Language Layer Summary

| Layer | Packed MiB | Buckets | Shards |
|---:|---:|---|---:|
| 0 | 800.06 | active_moe:4, full_attn:4, linear_attn:5, shared_moe:8 | 2 |
| 1 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 2 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 3 | 398.50 | active_moe:2, full_attn:4, shared_moe:4 | 1 |
| 4 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 2 |
| 5 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 6 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 7 | 398.50 | active_moe:2, full_attn:4, shared_moe:4 | 1 |
| 8 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 2 |
| 9 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 2 |
| 10 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 2 |
| 11 | 398.50 | active_moe:2, full_attn:4, shared_moe:4 | 1 |
| 12 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 13 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 14 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 15 | 398.50 | active_moe:2, full_attn:4, shared_moe:4 | 1 |
| 16 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 2 |
| 17 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 18 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 19 | 398.50 | active_moe:2, full_attn:4, shared_moe:4 | 1 |
| 20 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 21 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 22 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 23 | 398.50 | active_moe:2, full_attn:4, shared_moe:4 | 2 |
| 24 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 25 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 26 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 27 | 398.50 | active_moe:2, full_attn:4, shared_moe:4 | 1 |
| 28 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 29 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 2 |
| 30 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 2 |
| 31 | 398.50 | active_moe:2, full_attn:4, shared_moe:4 | 1 |
| 32 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 33 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 34 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 35 | 398.50 | active_moe:2, full_attn:4, shared_moe:4 | 1 |
| 36 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 2 |
| 37 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 38 | 401.56 | active_moe:2, linear_attn:5, shared_moe:4 | 1 |
| 39 | 398.50 | active_moe:2, full_attn:4, shared_moe:4 | 1 |
