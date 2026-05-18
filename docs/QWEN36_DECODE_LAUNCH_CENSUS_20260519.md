# Qwen3.6 Decode Launch Census

P144 is a read-only profiler for the official Qwen3.6-35B-A3B W4A16 NVFP4
serving path. It measures a short greedy decode with `torch.profiler` and
records CUDA launch count, launches per token, top kernels by self CUDA time,
and coarse kernel groups.

This is the baseline needed before larger boundary work:

- active MoE strict boundary and repack;
- linear/GDN island fusion;
- full-attention workspace/cache boundary.

## Run On R6000

```bash
cd /root/autodl-tmp/lynn-engine
bash scripts/r6000_qwen36_decode_launch_census.sh
```

Optional env:

```bash
MAX_NEW=16 WARMUP_NEW=2 \
PROFILE_ENV='LYNN_SHARED_EXPERT_GATE_BACKEND=triton LYNN_LINEAR_ATTN_CONV_BACKEND=triton_inplace' \
bash scripts/r6000_qwen36_decode_launch_census.sh
```

## Output

The JSON report includes:

- `tokens_profiled`
- `cuda_launch_count_total`
- `cuda_launches_per_token`
- `top_kernels_by_self_cuda_time`
- `grouped_by_name_prefix`
- `wall_ms_per_token`
- `decode_tps_estimate`
- `env_summary`

If CUDA profiling is unavailable, the script writes `status=FAILED` with a
failure reason rather than emitting placeholder data.
