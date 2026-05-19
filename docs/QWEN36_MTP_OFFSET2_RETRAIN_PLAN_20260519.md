# Qwen3.5/3.6 MTP Contract Plan - 2026-05-19

## Decision

MTP remains a required acceleration branch for both current Lynn release tracks:

- Qwen3.5-9B dense, which is now the release mainline;
- Qwen3.6-35B-A3B MoE, which remains the high-end/NVFP4 research line.

Both local configs inspected on R6000 expose `text_config.mtp_num_hidden_layers=1`.
The public model cards also document MTP/NextN serving for Qwen3.5-9B and
Qwen3.6-35B-A3B through SGLang `NEXTN` and vLLM `qwen3_next_mtp`.

The current Qwen3.6 35B sidecar must not be counted as TPS credit because Spark
measured about 0.30% accept rate. The root cause is not a serving wire failure;
it is likely a target-contract mismatch:

- existing calibration scripts train the head to reproduce the base model's
  immediate next-token prediction from the current hidden state (`offset=1`);
- speculative verification needs the draft head to predict the token after the
  accepted draft token (`offset=2`).

The next useful MTP task is therefore a public-head contract audit first, then
an explicit offset-2 trainer and gate only if the public head cannot be made to
match the SGLang/vLLM contract.

## Public-Head First

Before retraining, Lynn should first verify the public Qwen MTP head contract.
The official Qwen3.5-9B and Qwen3.6-35B-A3B model pages document SGLang
`--speculative-algo NEXTN` and vLLM `qwen3_next_mtp` serving modes. SGLang's
Qwen3.6 documentation also states that both MoE and dense variants ship
`mtp.safetensors`.

This changes the near-term plan:

1. Extract the official public `mtp.*` tensors directly from the HF safetensors
   shards for both Qwen3.5-9B and Qwen3.6-35B-A3B, then record their SHA256/key
   inventory.
2. Compare them against the current Lynn sidecars and any calibrated sidecars.
   A mismatch means Lynn may have been testing a warm-start/calibrated head
   rather than the official production NextN head.
3. Run a Spark/SGLang reference probe with the exact official model and record
   accept rate, accept length, command-line flags, and output quality.
4. Run the same prompt set through Lynn's MTP wire using the official public
   head unchanged.
5. Only start offset-2 retraining if the public head cannot reach useful accept
   in Lynn after the SGLang/vLLM contract is reproduced.

In short: public NextN first, retrain second. Retraining is still the fallback
if the public head is incompatible with Lynn's serving state layout.

## 9B and 35B Unified Work Items

| Track | Public MTP expected? | First action | Promotion target |
| --- | --- | --- | --- |
| Qwen3.5-9B dense | Yes | Extract official `mtp.*`, run SGLang/vLLM contract reference, then Lynn accept probe | 10-20% TPS over Q4_K_M/NVFP4 baseline without quality drift |
| Qwen3.6-35B-A3B MoE | Yes | Re-extract official `mtp.*`, compare with current sidecar, reproduce SGLang NEXTN accept contract | 10-20% TPS over 107-108 TPS safe W4A16 baseline |

The first pass should be GPU-light and evidence-first. Do not train until the
public-head inventory and reference accept numbers are known.

## R6000 Feasibility

R6000 is sufficient for a practical first pass if the base model is frozen and
used only in no-grad teacher-forcing mode.

Supported on R6000:

- load official Qwen3.6-35B-A3B Lynn-native W4A16/NVFP4 for no-grad teacher data;
- run teacher-forced one-step rollout to collect `(hidden_N, token_N+1, label_N+2)`;
- train only the MTP sidecar bridge/head parameters, starting with `mtp.fc.weight`
  and optionally the MTP norms;
- run iterative accept-rate probes and small end-to-end speculative smoke tests.

Not suitable on R6000:

- full 35B base-model backpropagation;
- large-scale multi-epoch MTP retraining through the frozen base layers with
  retained activations;
- expensive K=2 batched verify work before the offset-2 head clears accept-rate
  gates.

This means R6000 can answer the key question quickly: can an offset-2 sidecar
reach a useful accept rate? A100 is only needed later if the small R6000 pass is
promising and we want a broader corpus or more trainable MTP layers.

## Offset-2 Training Contract

For each prompt:

1. Prefill the base model to the last prompt token.
2. Read `hidden_N` and base greedy `token_N+1`.
3. Teacher-force `token_N+1` through the base model to obtain base logits for
   `token_N+2`.
4. Run MTP with `(hidden_N, token_N+1, position_N)` as input.
5. Train MTP logits against `token_N+2`.

The report must explicitly include:

- `target_offset: 2`;
- the draft input token text/id (`token_N+1`);
- the label token text/id (`token_N+2`);
- accept-rate measured against the same offset-2 contract.

## Gates

Do not run full service TPS until the offset-2 accept gate clears.

| Gate | Requirement | Decision |
| --- | --- | --- |
| Contract smoke | Labels are `N+2`, not `N+1` | Required |
| Tiny train | 32-64 prompts, no NaN, loss falls | Required |
| Heldout accept | `<30%` | CLOSED |
| Heldout accept | `30-50%` | AMBER, tune data/head |
| Heldout accept | `>50%` | Run end-to-end speculative TPS |
| TPS gate | `>1.10x` over baseline | Keep branch |
| TPS gate | `>1.20x` over baseline | Promotion candidate |

## Implementation Steps

1. Add `--target-offset {1,2}` to the calibration trainer, defaulting to `1` for
   backwards compatibility but printing the contract in every report.
2. Add an offset-2 case collector that runs a no-grad teacher-forced base decode
   for the first draft token.
3. Save an offset-2 sidecar under a distinct metadata key:
   `lynn_mtp_target_offset=2`.
4. Add a strict accept probe that refuses to evaluate a sidecar without matching
   `target_offset=2` metadata.
5. Only after accept-rate clears 50%, reuse the existing `engine/mtp_serving.py`
   speculative wire for end-to-end TPS.

## Expected ROI

If the offset-2 head reaches 50-70% accept, the expected useful gain is roughly
10-20% on the current 35B line. On a 107-108 TPS safe R6000 baseline, that is
the difference between staying near 108 and having a realistic 118-130 TPS
candidate.

The branch is therefore worth keeping even while Qwen3.5-9B becomes the release
mainline.
