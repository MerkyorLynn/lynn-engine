#!/usr/bin/env python3
"""MTP block-verifier correctness probe (Stage 5 / task A step 2, 2026-06-03).

CONTEXT: spark_mtp_offset_align_probe.py PROVED the MTP draft head is excellent --
pair_next (h_p, embed(x_{p+1})) predicts x_{p+2} at 91.5% top-1. So the 2.4% A/B
accept is NOT the draft; it must be the T>=2 BLOCK VERIFIER used by
speculative_step_kn_batched (LYNN_MTP_SPECULATIVE_BATCHED=1, K=2), which calls
decode_block_to_logits_and_hidden over [pending, draft...]. Known hazard
(project_lynn_engine_t1_only_kernel_contract): NVFP4 decode kernels hard-code
h.shape[1]==1; T>=2 silently mis-handles the time dim -> wrong block logits ->
spurious rejects. Reject path re-decodes with the correct decode_one -> output
stays TOKEN_EXACT, hiding the bug as "low accept".

This probe DIRECTLY confirms + localizes. At each greedy step it compares, from
identical state, the T=2 block forward over [pending, draft] vs the canonical T=1
decode of pending:

  truth   = decode_one(pending)            -> true_next (= x_{p+2}), h_true
  block   = decode_block([pending, draft]) -> argmax_ids[0]=b0, h_block[:,0]

Reports:
  intrinsic draft accept  : draft_0 == true_next        (sanity, expect ~0.9)
  BLOCK CORRECT (pos0)    : b0 == true_next             (LOW => block verifier broken)
  kn_batched would-accept : draft_0 == b0               (== the 2.4% number)
  hidden cos pos0         : cos(h_block[:,0], h_true)    (<1 => LAYER kernels wrong at T=2)
  lm_head all_pos vs 1pos : b0 vs argmax(lm_head(norm(h_block[:,0:1])))  (differ => lm_head all_positions bug)

Run in docker lynn-eval-base:cu13, PYTHONNOUSERSITE=1, APEX stopped (see offset probe header).
"""
import os, sys, pathlib, json

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_ENV = {
    "LYNN_MOE_IMPL": "packed_nvfp4", "LYNN_MOE_FAST_FIXED": "1",
    "LYNN_NATIVE_ACTIVE_MOE_BACKEND": "triton", "LYNN_NATIVE_GATEUP_BACKEND": "triton_fast_decode",
    "LYNN_NATIVE_DOWN_BACKEND": "triton", "LYNN_ROUTER_TOPK_SORTED": "0",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare", "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1", "LYNN_LINEAR_STATE_UPDATE": "inplace",
    "LYNN_PACKED_DECODE": "0", "LYNN_PACKED_SHARED_EXPERT": "0", "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair", "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_FULL_ATTN_ROPE_CACHE": "1", "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
for k, v in BASE_ENV.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("LYNN_MOE_DOWN_BLOCK_HIDDEN", "4")
os.environ.setdefault("LYNN_LINEAR_ATTN_GQA_RECURRENT", "1")
os.environ.setdefault("LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV", "1")
os.environ.setdefault("LYNN_RMSNORM_FUSED", "1")
os.environ.setdefault("LYNN_FULL_ATTN_FUSED", "1")
os.environ.setdefault("LYNN_SHARED_EXPERT_FUSED", "1")
os.environ.setdefault("LYNN_LINEAR_ATTN_FUSE_GBETA", "1")
os.environ.setdefault("LYNN_NVFP4_BF16_OUT", "1")
os.environ.setdefault("LYNN_DECODE_OPROJ_NOCOPY", "1")

SIDECAR = os.environ.get(
    "LYNN_MTP_SIDECAR",
    "/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp/mtp.safetensors",
)
os.environ["LYNN_MTP_SIDECAR"] = SIDECAR
os.environ["LYNN_MTP_SPECULATIVE"] = "1"

import torch
import torch.nn.functional as F

from engine.resident_runner import LynnIncrementalRunner, _encode_prompt
from engine.full_forward import _prefill_layer, _rms_norm
from engine.inference_state import LynnInferenceState
from engine.mtp_serving import decode_one_to_logits_and_hidden, decode_block_to_logits_and_hidden

EMBED = "model.language_model.embed_tokens.weight"
NORM = "model.language_model.norm.weight"
MODEL = os.environ.get(
    "MODEL",
    "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526",
)
STEPS = int(os.environ.get("PROBE_STEPS", "36"))
P = "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step."


def main() -> None:
    runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=True)
    dev = runner.device
    print("MTP sidecar loaded?:", getattr(runner, "mtp_sidecar_loaded", None), flush=True)
    if not getattr(runner, "mtp_sidecar_loaded", False):
        print("FATAL: sidecar not loaded", flush=True)
        return

    tok = runner.tokenizer
    ids = _encode_prompt(tok, P, dev, use_chat_template=False)
    T = ids.shape[1]
    state = LynnInferenceState.from_config(
        runner.cfg, batch=1, max_seq_len=runner.max_seq_len, device=dev, dtype=runner.dtype
    )
    h = F.embedding(ids, runner.outside[EMBED])
    pos = torch.arange(T, device=dev, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, runner.layer_types[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = T
    logits = runner._lm_head_logits(_rms_norm(h, runner.outside[NORM]))
    pending = int(logits[0].argmax().item())
    h_pending = h[:, -1:, :].contiguous()  # h_p that produced `pending`

    n = 0
    draft_correct = 0     # draft_0 == true_next  (intrinsic head, expect ~0.9)
    block_correct = 0     # b0 == true_next       (block verifier; LOW => broken)
    kn_accept = 0         # draft_0 == b0         (== the A/B 2.4% number)
    cos_sum = 0.0
    cos_min = 2.0
    lmhead_mismatch = 0   # b0 != argmax(single-position lm_head on h_block[:,0])
    rows = []
    norm_w = runner.outside[NORM]

    for step in range(STEPS):
        # draft from (h_pending, pending) -- the serving pair_next contract
        _mtp_hidden, draft_logits = runner._mtp_draft_hidden_logits(
            base_hidden=h_pending, current_token_id=pending, current_pos=int(state.seq_len - 1)
        )
        draft_0 = int(draft_logits[0].argmax().item())

        snap = runner._snapshot_state(state)
        # T=2 block forward over [pending, draft_0]
        h_block, _lg_block, argmax_ids = decode_block_to_logits_and_hidden(runner, state, [pending, draft_0])
        b0 = int(argmax_ids[0])
        # independent single-position lm_head on the block's pos-0 hidden
        b0_single = int(runner._lm_head_logits(_rms_norm(h_block[:, 0:1, :], norm_w))[0].argmax().item())
        h_block_pos0 = h_block[:, 0:1, :].float()
        runner._restore_state(state, snap)

        # canonical T=1 truth
        h_true, _lg_true, true_next = decode_one_to_logits_and_hidden(runner, state, pending)
        cos = float(F.cosine_similarity(h_block_pos0.reshape(1, -1), h_true.float().reshape(1, -1)).item())

        n += 1
        draft_correct += int(draft_0 == true_next)
        block_correct += int(b0 == true_next)
        kn_accept += int(draft_0 == b0)
        cos_sum += cos
        cos_min = min(cos_min, cos)
        lmhead_mismatch += int(b0 != b0_single)
        if len(rows) < 16:
            rows.append({
                "p": int(state.seq_len - 1),
                "pending": tok.decode([pending]),
                "draft_0": tok.decode([draft_0]),
                "true_next": tok.decode([true_next]),
                "block_b0": tok.decode([b0]),
                "draft==true": draft_0 == true_next,
                "block==true": b0 == true_next,
                "draft==block": draft_0 == b0,
                "b0_all==b0_1pos": b0 == b0_single,
                "cos_pos0": round(cos, 4),
            })
        pending = int(true_next)
        h_pending = h_true[:, -1:, :].contiguous()

    def pct(x):
        return f"{100.0*x/max(n,1):5.1f}% ({x}/{n})"

    print("\n=============== BLOCK-VERIFIER CORRECTNESS (n=%d) ===============" % n, flush=True)
    print(f"intrinsic draft accept (draft_0==true_next) : {pct(draft_correct)}   [sanity ~0.9]", flush=True)
    print(f"BLOCK CORRECT          (b0==true_next)       : {pct(block_correct)}   [LOW => block verifier broken]", flush=True)
    print(f"kn_batched would-accept(draft_0==b0)         : {pct(kn_accept)}        [== the A/B accept number]", flush=True)
    print(f"hidden cos pos0 (block vs truth)             : mean={cos_sum/max(n,1):.4f}  min={cos_min:.4f}", flush=True)
    print(f"lm_head all_pos != 1pos (on block hidden)    : {pct(lmhead_mismatch)}   [>0 => all_positions lm_head bug]", flush=True)
    print("\n--- per-step rows ---", flush=True)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False), flush=True)

    print("\n=============== VERDICT ===============", flush=True)
    if block_correct / max(n, 1) >= 0.80:
        print("Block verifier is CORRECT -> bug is NOT here; re-examine kn accept accounting / chaining.", flush=True)
    else:
        cos_mean = cos_sum / max(n, 1)
        if lmhead_mismatch / max(n, 1) >= 0.3 and cos_mean >= 0.99:
            print("=> LOCALIZED: layer hiddens are fine (cos~1) but lm_head ALL_POSITIONS path is wrong for T>=2.", flush=True)
            print("   FIX = _lm_head_logits(all_positions=True) per-row dispatch (native FP4 lm_head T>=2).", flush=True)
        elif cos_mean < 0.99:
            print(f"=> LOCALIZED: block LAYER kernels are wrong at T=2 (pos0 hidden cos={cos_mean:.4f} < 1).", flush=True)
            print("   FIX = thread T-dim through _decode_layer_block_fast kernels (full-attn / linear-attn / MoE", flush=True)
            print("   that hard-code h.shape[1]==1). Matches the M12 decode_full_attn_k2 'pending' item.", flush=True)
        else:
            print("=> Block wrong but neither lm_head nor pos0-hidden fully explains -> inspect per-layer-type.", flush=True)


if __name__ == "__main__":
    main()
