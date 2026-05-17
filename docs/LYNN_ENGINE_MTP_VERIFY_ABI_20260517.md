# Lynn Engine MTP K=2/K=3 Verify ABI

Date: 2026-05-17

## Purpose

This is the clean-room ABI for turning Lynn's current MTP shadow signal into a
real serving speed path.

Current state:

- `engine.resident_runner` can load a sidecar and run one-token MTP shadow.
- v34 on R6000 is the best native sidecar at `65/121 = 53.72%`.
- P113 cut the one-token draft path from about `7.6 ms` to `2.24 ms`.
- P115 shows serial one-token MTP still misses 155 TPS.

The next speed path therefore needs a verifier that can process candidate spans
as K=2/K=3 batches, then commit or roll back state exactly.

This file is not a Rust/CUDA implementation spec copied from Atlas. Atlas is
AGPL and must stay clean-room. The state contract below is derived from Lynn's
own `LynnInferenceState`, resident runner, and P9/P112/P115 reports.

## State Model

Lynn decode state is:

```text
LynnInferenceState
  seq_len: int
  kv_cache[full_attn_layer] = (K, V)
    K,V shape [1, 2, max_seq_len, 256] bf16
  recurrent_state[linear_attn_layer]
    shape [1, 32, 128, 128] fp32
  conv_state[linear_attn_layer]
    shape [1, 8192, 3] bf16
```

There are 10 full-attention layers and 30 linear-attention layers. The layer
pattern is `linear, linear, linear, full` repeated 10 times.

Important serving convention:

```text
S_n = cache/state after positions [0, n)
logits(S_n) predict token x_n
current Lynn generate may emit x_n before x_n has been folded into S_n
next decode step folds x_n into state at position n
```

The verifier must respect that convention. It verifies a pending token plus
drafts against a checkpointed state before those tokens are folded in.

## Verify Inputs

Minimum K=2 call:

```text
state_before: S_n
verify_tokens: [x_n, d_{n+1}]
position_start: n
mode: greedy argmax
```

Minimum K=3 call:

```text
state_before: S_n
verify_tokens: [x_n, d_{n+1}, d_{n+2}]
position_start: n
mode: greedy argmax
```

Where:

- `x_n` is the pending base token already chosen by logits from `S_n`.
- `d_*` are MTP draft tokens.
- `position_start` must equal `state_before.seq_len`.
- `verify_tokens` is a device `int64` or `int32` buffer in the native ABI.

The verifier returns base argmax tokens after each verify input:

```text
K=2 output argmax_after = [v_{n+1}, v_{n+2}]
K=3 output argmax_after = [v_{n+1}, v_{n+2}, v_{n+3}]
```

`v_{n+1}` is the base model's next token after processing `x_n`, so `d_{n+1}`
is accepted when `d_{n+1} == v_{n+1}`.

## Accept/Reject Rules

K=2:

```text
if d_{n+1} == v_{n+1}:
  emit d_{n+1}
  keep pending_next = v_{n+2}
  commit state after [x_n, d_{n+1}]  # S_{n+2}
else:
  emit v_{n+1}
  keep pending_next = v_{n+1}
  commit state after [x_n]           # S_{n+1}
  discard state after d_{n+1}
```

K=3:

```text
accepted = 0 if d_{n+1} != v_{n+1}
accepted = 1 if d_{n+1} == v_{n+1} and d_{n+2} != v_{n+2}
accepted = 2 if d_{n+1} == v_{n+1} and d_{n+2} == v_{n+2}

accepted 0:
  emit v_{n+1}
  commit state after [x_n]

accepted 1:
  emit d_{n+1}
  emit v_{n+2}
  commit state after [x_n, d_{n+1}]

accepted 2:
  emit d_{n+1}
  emit d_{n+2}
  keep pending_next = v_{n+3}
  commit state after [x_n, d_{n+1}, d_{n+2}]
```

The serving loop may choose whether to immediately emit `pending_next` or keep
it as the next pending base token. The state invariant is stricter: canonical
state must only include tokens that have been folded through the base model.

## Commit/Rollback Contract

The verifier must not rely on Python `clone()` snapshots for production.

Required native scratch state:

```text
VerifyScratch(K)
  full_attn_kv_writes:
    per full-attn layer, positions [n, n+K)
  linear_recurrent_intermediates:
    per linear layer, state after each verify token
  linear_conv_intermediates:
    per linear layer, state after each verify token
  hidden_after_token:
    [K, 1, 1, 2048] bf16 optional, used for MTP proposer chaining
  logits_or_argmax:
    argmax ids required, logits optional for diagnostics
```

Commit operation:

```text
commit_count = 1 + num_accepted_drafts
state.seq_len = position_start + commit_count
full-attn KV: keep writes for positions < state.seq_len
linear recurrent/conv: copy intermediate[commit_count - 1] into canonical state
pending_next: not folded into state yet
```

Reject operation is not a separate primitive. It is `commit_count = 1` with
scratch positions beyond `state.seq_len` ignored. Full-attention KV beyond
`seq_len` may contain stale data as long as future writes overwrite it and all
attention reads are clipped by `seq_len`.

## Native ABI Sketch

First C++ extension surface should stay narrow:

```text
lynn_verify_k(
  state_handle,
  verify_tokens_dev,
  num_tokens,            # 2 or 3
  position_start,
  out_argmax_dev,        # [num_tokens]
  scratch_handle,
  flags
) -> status

lynn_commit_verify(
  state_handle,
  scratch_handle,
  commit_count
) -> status
```

For the first prototype, `state_handle` can wrap Python-owned tensors from
`LynnInferenceState`. A later `liblynn_decode_core` can own the buffer arena
directly, but the prototype should avoid a loader rewrite.

Required flags:

```text
LYNN_VERIFY_K=2|3
LYNN_VERIFY_ARGMAX_ONLY=1
LYNN_VERIFY_RECORD_HIDDEN=0|1
LYNN_VERIFY_STRICT_PARITY=1
```

Optional future flags:

```text
LYNN_VERIFY_GRAMMAR_MASK=1
LYNN_VERIFY_TOPK_LOGITS=N
LYNN_VERIFY_OVERLAP_MTP=1
```

## Proposer Contract

The verifier is independent from the proposer.

Current `engine/mtp_sidecar.py` proposer:

```text
mtp.fc -> one full-attention MTP layer -> mtp.norm -> shared lm_head
```

It currently provides a one-token shadow draft from a base hidden state plus a
current token id. It does not yet provide a cheap K=2/K=3 draft chain. That is
why the verifier ABI must expose `hidden_after_token`: later proposer variants
can use the verified hidden/state surface to draft the next span without
falling back to serial Python recursion.

Serving policy:

- format-guarded structured requests keep MTP disabled until the sidecar is
  guard-aware;
- raw route-allowlisted requests can run shadow or verify experiments;
- no global MTP enable until P117-style policy says the route is eligible.

## Quality Gates

The first native verifier is promotable only if all pass:

| Gate | Requirement |
|---|---|
| P107 raw trace | K=2/K=3 verifier reproduces Python base argmax and state for every event. |
| P116 route policy | route classification does not change versus shadow traces except where expected by recursive verify. |
| Structured guard | MTP remains disabled with `LYNN_MTP_DISABLE_FOR_FORMAT_GUARD=1`. |
| P105 generation | W4A8 serving text does not regress versus Config D fallback. |
| State parity | after accept/reject, next one-token base decode matches Python runner logits/argmax. |
| Cost | K=2 iteration cost <= 1.65 ms on R6000, or measured overlap makes effective cost <= 1.65 ms. |

If state parity passes but cost fails, keep the verifier as a diagnostic and
continue optimizing active-MoE/MTP layer kernels.

## Implementation Order

1. Add a Python-only `p118_mtp_verify_state_parity.py` that simulates the ABI
   with cloned states. This freezes accept/commit semantics before C++ work.
2. Add a tiny native extension boundary that only copies/commits linear
   recurrent and conv intermediates for K=2; leave layer math in Python.
3. Move one full-attention layer verify into native/static boundary.
4. Move active-MoE verify into the transposed/native layout.
5. Only then replace the full verify loop with `lynn_verify_k`.

This sequence keeps each failure diagnosable. It also avoids the false shortcut
of moving the token loop to C++ before the costly math and state ABI are ready.
