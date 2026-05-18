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
| `amber_sharedgate_convinplace` | CLOSED under new gate | drift | 113.08 | 69/70 | 113.60 | Fails 70-prompt gate and below 118 AMBER TPS |
| `strict_fused_boundary_fullattn` | CLOSED | drift | 96.28 | 40/40 | 96.91 | Full-attn-only strict boundary is slower and not exact |

## Failure Detail

- `amber_sharedgate_convinplace` failed on `yaml_content_json`.
- It expanded the requested content block into a full OpenAPI document, hit `max_new`, and did not start with an allowed YAML block prefix.
- This confirms the 70-prompt gate is catching exactly the structured-format drift that 40/40 could miss.

## Implication

- Do not promote `LYNN_SHARED_EXPERT_GATE_BACKEND=triton + LYNN_LINEAR_ATTN_CONV_BACKEND=triton_inplace` yet.
- Continue the kernel route through numerically strict boundaries: active MoE strict boundary, linear-attn boundary, or repacked W4A16 kernels that preserve P37 exactness.
- The next publishable speed step needs strict parity plus a real P25 512 lift above 108, then 70/70 + 118 for AMBER.

