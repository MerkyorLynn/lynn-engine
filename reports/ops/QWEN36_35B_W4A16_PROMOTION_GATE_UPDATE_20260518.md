# Qwen3.6-35B-A3B W4A16 Promotion Gate Update - 2026-05-18

## Current R6000 Default

- Default promotion remains official Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4.
- Current safe-default profile is still the only publishable serving path: exact-greedy clean, hard structured clean, and P25 512 decode around 107 TPS.
- AMBER remains opt-in research only until it can pass a stricter 70-prompt structured gate and reach the 118 TPS P25 512 bar.

## New Gate Rules

- DEFAULT requires P37 exact-greedy, 40/40 hard structured, and P25 512 decode TPS >= 108.
- AMBER requires 70/70 hard structured and P25 512 decode TPS >= 118.
- The 70-prompt set adds more JSON/tool-call, Python code-only, OpenAPI YAML, Chinese exact-term, and number-only prompts.

## Latest Candidate Results

| Candidate | Decision | P37 | P25 512 Decode TPS | Structured | Structured Decode TPS | Notes |
|---|---:|---:|---:|---:|---:|---|
| `safe_default_full` | DEFAULT | exact | 107.43 | 40/40 | 107.86 | Publishable baseline |
| `linear_attn_conv_inplace_strict` | RESEARCH_ONLY | drift | 108.40 | 40/40 | 110.43 | Single conv-inplace knob clears speed/format but fails exact-greedy |
| `amber_sharedgate_convinplace` | CLOSED under new gate | drift | 113.08 | 69/70 | 113.60 | Fails 70-prompt gate and below 118 AMBER TPS |
| `safe_default_70` | CLOSED diagnostic | exact | 107.31 | 59/70 | 107.49 | Ran against the later hard-v2 70 set, not the earlier AMBER 70 set |
| `strict_fused_boundary_fullattn` | CLOSED | drift | 96.28 | 40/40 | 96.91 | Full-attn-only strict boundary is slower and not exact |
| `strict_fused_boundary_fullattn_fastfixed` | CLOSED | drift | 96.68 | 40/40 | 96.96 | Fast-fixed variant is still slower and not exact |

## Failure Detail

- `amber_sharedgate_convinplace` failed on `yaml_content_json`.
- It expanded the requested content block into a full OpenAPI document, hit `max_new`, and did not start with an allowed YAML block prefix.
- This confirms the 70-prompt gate is catching exactly the structured-format drift that 40/40 could miss.
- A later safe-default 70 run used a harder remote prompt file version and got 59/70, while keeping P37 exact. Treat that run as a diagnostic of the hard-v2 prompt set rather than as a default downgrade.
- Stream B's single `LYNN_LINEAR_ATTN_CONV_BACKEND=triton_inplace` strict candidate is not strict enough: all three P37 prompts drift even though 40/40 structured serving passes.
- Stream A's `strict_fused_boundary_fullattn_fastfixed` follow-up does not rescue the strict boundary path: it remains P37-drifting and stays around 96.7 TPS.

## Implication

- Do not promote `LYNN_SHARED_EXPERT_GATE_BACKEND=triton + LYNN_LINEAR_ATTN_CONV_BACKEND=triton_inplace` yet.
- Continue the kernel route through numerically strict boundaries: active MoE strict boundary, linear-attn boundary, or repacked W4A16 kernels that preserve P37 exactness.
- The next publishable speed step needs strict parity plus a real P25 512 lift above 108, then 70/70 + 118 for AMBER.
