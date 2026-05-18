# KIMI Task - Slot-Repacked Output-Owned Native MoE

## Context

Your previous `native_output_owned_bf16` result is accepted as a research fixture candidate:

- p134 routed-only relaxed: 18/18 GREEN
- `candidate_ms_mean=0.05047ms`
- `max_abs_max=3.90625e-3`
- `cosine_min=0.999980211`
- exact 0/18

It proves the output-owned scheduling idea is good, but it is not production-safe because it uses BF16 dequantized full expert weights rather than packed W4A16/NVFP4 serving weights.

The next useful step is not resident integration. The next useful step is a fixture-level slot-repacked candidate that keeps your output-owned scheduling and removes dynamic 256-expert indexing from the kernel target.

## Branch

```bash
git checkout main
git pull --ff-only
git checkout -b kimi/native-moe-slot-output-owned-20260518
```

If `git pull` is blocked by local dirt, do not clean/reset. Create the branch from the current main-equivalent workspace and list the dirty files in your final note.

## Allowed Write Scope

You may create or edit:

- `benchmarks/p135_repack_moe_fixture_slots.py`
- `benchmarks/p136_moe_slot_repack_contract.py`
- `benchmarks/candidates/native_slot_output_owned_bf16.py`
- new CUDA file under `csrc/lynn_native/` only if needed
- `csrc/lynn_native/bindings.cpp` only if adding a native extension binding
- `engine/native_cuda.py` only if adding that new CUDA file to extension sources
- `scripts/r6000_qwen36_moe_slot_repack.sh`
- reports under `reports/qwen36_35b/`

Do not touch:

- `server/*`
- `engine/resident_runner.py`
- `engine/incremental_decode.py`
- `engine/moe_packed_nvfp4.py`
- `triton_kernels/*`

## Goal

Build a self-contained fixture target for slot-repacked active MoE, then run an output-owned candidate against it.

This is a fixture-kernel task, not a serving integration task.

## Required Artifact 1 - Slot Repack

Implement:

```bash
python benchmarks/p135_repack_moe_fixture_slots.py \
  --fixtures /root/autodl-tmp/reports/qwen36_35b/p133_fixtures_official_w4a16 \
  --model /root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0 \
  --out-dir /root/autodl-tmp/reports/qwen36_35b/p135_slot_repacked_fixtures_20260518
```

Each output safetensors file should include:

- `hidden_in`: `[1, 2048]` BF16
- `expert_ids`: `[8]` int32
- `routing_weights`: `[8]` float32
- `slot_gate_up_weight`: `[8, 1024, 2048]` BF16, F.linear layout
- `slot_down_weight`: `[8, 2048, 512]` BF16, F.linear layout
- `routed_output`: `[1, 2048]` BF16

Also write a manifest with:

- source fixture file
- source fixture sha256
- layer_id
- prompt_id
- tensor shapes/dtypes
- reference compute time

## Required Artifact 2 - Slot Contract

Implement:

```bash
python benchmarks/p136_moe_slot_repack_contract.py \
  --fixtures /root/autodl-tmp/reports/qwen36_35b/p135_slot_repacked_fixtures_20260518 \
  --out /root/autodl-tmp/reports/qwen36_35b/p136_slot_repack_contract_20260518.json
```

The contract should recompute routed-only output using only the slot tensors, not the full model or the 256-expert table.

Acceptance:

- 18/18 fixtures GREEN
- ideal: `max_abs_max == 0`
- acceptable: `max_abs_max <= 1e-3` and `cosine_min >= 0.999999`

## Required Artifact 3 - Output-Owned Candidate

Implement candidate module:

```text
benchmarks/candidates/native_slot_output_owned_bf16.py
```

Candidate interface:

```python
def moe_forward_slot_repacked(hidden_in, slot_gate_up_weight, slot_down_weight, routing_weights, cfg) -> torch.Tensor:
    ...
```

Expected math:

```python
out = 0
for slot in range(8):
    gu = F.linear(hidden, slot_gate_up_weight[slot])
    gate, up = gu[:512], gu[512:]
    inter = F.silu(gate) * up
    down = F.linear(inter, slot_down_weight[slot])
    out += routing_weights[slot] * down
```

CUDA is optional for the first version. A torch version is acceptable if it proves the contract and output format. Native CUDA is preferred if you can keep it scoped.

## R6000 One-Command Wrapper

Provide:

```bash
bash scripts/r6000_qwen36_moe_slot_repack.sh
```

It should:

1. run p135 slot repack
2. run p136 contract
3. run `native_slot_output_owned_bf16` candidate if implemented
4. write a concise summary JSON

## Stop Conditions

Stop and report immediately if:

- p135 produces files much larger than expected and disk risk appears
- p136 cannot reach 18/18 GREEN
- slot ordering is ambiguous
- candidate output differs from p136 baseline by `max_abs > 1e-3`

## Final Report Format

Return:

- branch name
- changed files
- p135 output dir
- p136 JSON path
- candidate JSON path if any
- table:
  - p136 passed/total
  - max_abs_max
  - cosine_min
  - slot_repack_ms_mean
  - candidate_ms_mean
- verdict: `SLOT_REPACK_GREEN`, `CANDIDATE_FAST`, or `CLOSED_NUMERIC`
