#!/usr/bin/env python3
"""MTP draft<->serving ALIGNMENT probe (Stage 5 / task A step 1, 2026-06-03).

WHY: spark_mtp_ab.py shows accept ~= 2.4% with TOKEN_EXACT=True on both sidecars.
The handoff hypothesis was "serving applies offset-1 vs trained offset=2". But a
code trace shows the speculative serving path (k1 / k1_batched / kn_batched first
draft) ALREADY pairs base_hidden=h_p with current_token_id=x_{p+1} (the token h_p
produced) and compares the draft to base's real x_{p+2} -- i.e. it ALREADY does
offset=2 correctly. So either (a) the draft logits are garbage in this engine, or
(b) the trained contract is actually offset-1 (then record_mtp_shadow's pairing is
right and speculative is wrong). This probe decides empirically.

For a clean greedy decode it captures, per step, the pre-final-norm hidden h_p and
the real token sequence, then computes the MTP draft under TWO embed pairings and
scores draft-argmax against THREE targets -- a 2x3 accept grid:

  pair_next = MTP(h_p, embed(x_{p+1}))   # what speculative serving feeds (offset=2 intent)
  pair_same = MTP(h_p, embed(x_p))       # what record_mtp_shadow feeds  (offset=1 intent)

  targets: x_{p+1} (next) | x_{p+2} (next-next) | x_{p+3}

Also reports the rank of the contract-expected token inside the draft logits and
the draft top-1 margin, so "wrong pairing" (some cell ~60%, low rank) is
distinguishable from "garbage draft" (every cell ~random, high rank).

Run in docker lynn-eval-base:cu13 with PYTHONNOUSERSITE=1, APEX stopped:
  docker run -d --gpus all --ipc=host -e HOME=/home/merkyor -e PYTHONNOUSERSITE=1 \
    -e PYTHONUNBUFFERED=1 -v /home/merkyor:/home/merkyor -w /home/merkyor/lynn-engine \
    lynn-eval-base:cu13 bash -lc \
    "python3 -u scripts/spark_mtp_offset_align_probe.py > reports/mtp_offset_probe.log 2>&1"

Optional env:
  PROBE_STEPS=48                 # greedy decode steps to sample
  LYNN_MTP_LAYER_MOE=...         # MTP-layer MoE mode (default decode_slot_sorted)
  LYNN_MTP_SIDECAR=<path>        # which sidecar (default base qwen36-35b-a3b-mtp)
"""
import os, sys, pathlib, json

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Full RC-validated stack, IDENTICAL to spark_mtp_ab.py (apples-to-apples). ---
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
os.environ["LYNN_MTP_SPECULATIVE"] = "1"  # init-time: triggers sidecar load

import torch
import torch.nn.functional as F

from engine.resident_runner import LynnIncrementalRunner, _encode_prompt
from engine.full_forward import _prefill_layer, _rms_norm
from engine.inference_state import LynnInferenceState
from engine.mtp_serving import decode_one_to_logits_and_hidden

EMBED = "model.language_model.embed_tokens.weight"
NORM = "model.language_model.norm.weight"
MODEL = os.environ.get(
    "MODEL",
    "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526",
)
STEPS = int(os.environ.get("PROBE_STEPS", "48"))
P = "If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step."


def _rank_of(logits_1d: torch.Tensor, target_id: int) -> int:
    """0-based rank of target_id in a [V] logit row (0 == argmax/top-1)."""
    tv = logits_1d[int(target_id)]
    return int((logits_1d > tv).sum().item())


def main() -> None:
    runner = LynnIncrementalRunner(MODEL, device="cuda", dtype=torch.bfloat16, verbose=True)
    dev = runner.device
    print("MTP sidecar loaded?:", getattr(runner, "mtp_sidecar_loaded", None), "from", SIDECAR, flush=True)
    print("MTP layer MoE mode:", os.environ.get("LYNN_MTP_LAYER_MOE", "decode_slot_sorted (default)"), flush=True)
    if not getattr(runner, "mtp_sidecar_loaded", False):
        print("FATAL: sidecar not loaded", flush=True)
        return

    tok = runner.tokenizer
    ids = _encode_prompt(tok, P, dev, use_chat_template=False)
    T = ids.shape[1]
    state = LynnInferenceState.from_config(
        runner.cfg, batch=1, max_seq_len=runner.max_seq_len, device=dev, dtype=runner.dtype
    )

    # --- exact prefill (mirrors generate()) ---
    h = F.embedding(ids, runner.outside[EMBED])
    pos = torch.arange(T, device=dev, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, runner.layer_types[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = T
    logits = runner._lm_head_logits(_rms_norm(h, runner.outside[NORM]))
    x_next = int(logits[0].argmax().item())

    # steps[k] = (position p, h_p pre-final-norm, produced token x_{p+1})
    steps = [(T - 1, h[:, -1:, :].contiguous(), x_next)]
    seq = [int(t) for t in ids[0].tolist()] + [x_next]  # seq[i] = token at position i
    cur = x_next
    for _ in range(STEPS):
        p = state.seq_len  # cur will be consumed at this position
        h_p, lg, am = decode_one_to_logits_and_hidden(runner, state, cur)
        steps.append((p, h_p[:, -1:, :].contiguous(), am))
        seq.append(int(am))
        cur = am
    print(f"decoded {len(steps)} steps; sample text: {tok.decode(seq[T:T+40])!r}", flush=True)

    # --- 2x3 accept grid + ranks ---
    pairings = ("pair_next", "pair_same")
    targets = ("x_p+1", "x_p+2", "x_p+3")
    hit = {pp: {tg: 0 for tg in targets} for pp in pairings}
    total = 0
    rank_next_xp2 = []   # rank of x_{p+2} in pair_next logits (offset=2 expectation)
    rank_same_xp1 = []   # rank of x_{p+1} in pair_same logits (offset=1 expectation)
    margin_next = []
    rows = []
    EMB = runner.outside[EMBED]

    for (p, h_p, produced) in steps:
        # need x_p, x_{p+1}, x_{p+2}, x_{p+3}
        if p + 3 >= len(seq):
            continue
        x_p = seq[p]
        x_p1 = seq[p + 1]   # == produced
        x_p2 = seq[p + 2]
        x_p3 = seq[p + 3]
        # pair_next: feed embed(x_{p+1}) -- what speculative serving does
        dl_next = runner._mtp_draft_logits(base_hidden=h_p, current_token_id=x_p1, current_pos=p)[0].float()
        d_next = int(dl_next.argmax().item())
        # pair_same: feed embed(x_p) -- what record_mtp_shadow does
        dl_same = runner._mtp_draft_logits(base_hidden=h_p, current_token_id=x_p, current_pos=p)[0].float()
        d_same = int(dl_same.argmax().item())

        total += 1
        for d, pp in ((d_next, "pair_next"), (d_same, "pair_same")):
            hit[pp]["x_p+1"] += int(d == x_p1)
            hit[pp]["x_p+2"] += int(d == x_p2)
            hit[pp]["x_p+3"] += int(d == x_p3)
        rank_next_xp2.append(_rank_of(dl_next, x_p2))
        rank_same_xp1.append(_rank_of(dl_same, x_p1))
        top2 = torch.topk(dl_next, 2).values
        margin_next.append(float((top2[0] - top2[1]).item()))
        if len(rows) < 14:
            rows.append({
                "p": p,
                "x_p": tok.decode([x_p]), "x_p+1": tok.decode([x_p1]),
                "x_p+2": tok.decode([x_p2]),
                "draft_next": tok.decode([d_next]), "next==x_p+2": d_next == x_p2,
                "draft_same": tok.decode([d_same]), "same==x_p+1": d_same == x_p1,
                "rank_xp2_in_next": _rank_of(dl_next, x_p2),
            })

    def pct(n):
        return f"{100.0 * n / max(total, 1):5.1f}% ({n}/{total})"

    print("\n================ MTP ALIGNMENT GRID (n=%d) ================" % total, flush=True)
    print(f"{'pairing':>10} | {'==x_p+1':>15} | {'==x_p+2':>15} | {'==x_p+3':>15}", flush=True)
    for pp in pairings:
        print(f"{pp:>10} | {pct(hit[pp]['x_p+1']):>15} | {pct(hit[pp]['x_p+2']):>15} | {pct(hit[pp]['x_p+3']):>15}", flush=True)

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2] if s else -1

    print("\n--- draft sanity (rank: 0 = top-1; ~random ~ V/2 ~ 75000) ---", flush=True)
    print(f"pair_next  rank of x_p+2 : mean={mean(rank_next_xp2):8.1f}  median={med(rank_next_xp2)}", flush=True)
    print(f"pair_same  rank of x_p+1 : mean={mean(rank_same_xp1):8.1f}  median={med(rank_same_xp1)}", flush=True)
    print(f"pair_next  top1 margin   : mean={mean(margin_next):.3f}", flush=True)

    print("\n--- per-step rows ---", flush=True)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False), flush=True)

    print("\n================ VERDICT ================", flush=True)
    best_pp, best_tg, best_n = None, None, -1
    for pp in pairings:
        for tg in targets:
            if hit[pp][tg] > best_n:
                best_pp, best_tg, best_n = pp, tg, hit[pp][tg]
    print(f"highest cell: {best_pp} -> {best_tg} = {pct(best_n)}", flush=True)
    if best_n / max(total, 1) >= 0.40:
        print(f"=> ALIGNMENT: trained contract matches [{best_pp} predicting {best_tg}].", flush=True)
        if best_pp == "pair_next" and best_tg == "x_p+2":
            print("   => speculative serving pairing is ALREADY correct; 2.4% must come from elsewhere"
                  " (accounting / chaining / state). Re-examine spec accept stats, not the offset.", flush=True)
        else:
            print(f"   => FIX: change serving to use [{best_pp}] and compare draft vs [{best_tg}].", flush=True)
    else:
        print("=> NO cell >= 40%: draft logits look RANDOM -> not an offset bug; the MTP-layer"
              " forward is producing garbage under this engine config. Next: bisect"
              " LYNN_FULL_ATTN_FUSED / LYNN_RMSNORM_FUSED / LYNN_MTP_LAYER_MOE for the MTP layer.", flush=True)


if __name__ == "__main__":
    main()
