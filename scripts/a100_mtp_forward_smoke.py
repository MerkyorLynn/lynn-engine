#!/usr/bin/env python3
"""Forward-smoke a Lynn-shaped Qwen3-Next style MTP sidecar.

This is a wiring check, not an accept-rate or TPS claim. It loads the aligned
`mtp.*` sidecar, reuses the frozen base model embedding/lm_head, and runs the
minimal Qwen3-Next MTP body:

    fc([pre_fc_norm_hidden(base_hidden), pre_fc_norm_embedding(input_embed)])
    -> one full-attention MTP decoder layer
    -> mtp.norm
    -> shared lm_head

The goal is to turn the MTP asset from "shape-compatible file exists" into a
concrete forward path with finite logits and inspectable draft top-k tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _layer_forward, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _topk(tokenizer: Any, logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.float(), k=k, dim=-1)
    rows: list[dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(values[0].tolist(), indices[0].tolist()), start=1):
        token_id = int(idx)
        rows.append(
            {
                "rank": rank,
                "token_id": token_id,
                "score": float(score),
                "text": tokenizer.decode([token_id]),
            }
        )
    return rows


def _load_sidecar(path: Path, device: str, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    inventory: dict[str, Any] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            inventory[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "numel": int(tensor.numel()),
            }
            tensors[key] = tensor.to(device=device, dtype=dtype)
    return tensors, {"metadata": metadata, "tensors": inventory}


def _mtp_layer_weights(sidecar: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "mtp.layers.0."
    out: dict[str, torch.Tensor] = {}
    for key, tensor in sidecar.items():
        if key.startswith(prefix):
            out[key.removeprefix(prefix)] = tensor
    required = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    }
    missing = sorted(required - set(out))
    if missing:
        raise KeyError(f"MTP sidecar is missing layer tensors after prefix strip: {missing}")
    return out


def _base_prefill_last_hidden(
    runner: LynnIncrementalRunner,
    prompt: str,
    *,
    use_chat_template: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=use_chat_template)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    position_ids = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)

    from engine.inference_state import LynnInferenceState
    from engine.full_forward import _prefill_layer

    state = LynnInferenceState(
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )
    for layer_idx in range(runner.n_layers):
        h = _prefill_layer(
            h,
            position_ids,
            LAYER_TYPES[layer_idx],
            runner.layer_weights[layer_idx],
            runner.layer_cfgs[layer_idx],
            state,
            layer_idx,
        )
    last_token_id = int(ids[0, -1].item())
    last_pos = int(ids.shape[1] - 1)
    input_embed = runner.outside["model.language_model.embed_tokens.weight"][last_token_id].view(1, 1, -1)
    return h[:, -1:, :].contiguous(), input_embed.contiguous(), ids, last_token_id, last_pos


def run_smoke(
    *,
    base_model: Path,
    sidecar_file: Path,
    prompt: str,
    use_chat_template: bool,
    device: str,
    dtype: torch.dtype,
    top_k: int,
    train_fc_steps: int,
    train_lr: float,
) -> dict[str, Any]:
    runner = LynnIncrementalRunner(str(base_model), device=device, dtype=dtype, max_seq_len=4096, verbose=True)
    sidecar, sidecar_inventory = _load_sidecar(sidecar_file, device, dtype)
    mtp_w = _mtp_layer_weights(sidecar)

    base_hidden, input_embed, ids, last_token_id, last_pos = _base_prefill_last_hidden(
        runner,
        prompt,
        use_chat_template=use_chat_template,
    )
    base_normed = _rms_norm(base_hidden, runner.outside["model.language_model.norm.weight"])
    base_logits = runner._lm_head_logits(base_normed)

    text_cfg = (runner.cfg.get("text_config") or runner.cfg) if isinstance(runner.cfg, dict) else runner.cfg
    mtp_cfg = dict(text_cfg)
    mtp_cfg["num_experts"] = int(mtp_w["mlp.experts.gate_up_proj"].shape[0])
    mtp_cfg["num_experts_per_tok"] = int(text_cfg.get("num_experts_per_tok", 8))
    mtp_cfg["expert_intermediate"] = int(mtp_w["mlp.experts.down_proj"].shape[-1])
    mtp_cfg["layer_idx"] = 0
    pos = torch.tensor([[last_pos]], device=runner.device, dtype=torch.long)

    def mtp_forward() -> tuple[torch.Tensor, torch.Tensor]:
        hidden_part = _rms_norm(base_hidden.detach(), sidecar["mtp.pre_fc_norm_hidden.weight"])
        embed_part = _rms_norm(input_embed.detach(), sidecar["mtp.pre_fc_norm_embedding.weight"])
        mtp_hidden_local = F.linear(torch.cat([hidden_part, embed_part], dim=-1), sidecar["mtp.fc.weight"])
        mtp_out = _layer_forward(mtp_hidden_local, pos, "full_attention", mtp_w, mtp_cfg)
        mtp_normed = _rms_norm(mtp_out, sidecar["mtp.norm.weight"])
        return mtp_hidden_local, runner._lm_head_logits(mtp_normed)

    mtp_hidden, mtp_logits = mtp_forward()
    base_argmax = int(base_logits[0].argmax().item())
    train_report = None
    if train_fc_steps > 0:
        for tensor in sidecar.values():
            tensor.requires_grad_(False)
        sidecar["mtp.fc.weight"].requires_grad_(True)
        label = torch.tensor([base_argmax], device=runner.device, dtype=torch.long)
        optimizer = torch.optim.AdamW([sidecar["mtp.fc.weight"]], lr=train_lr)
        history: list[dict[str, float]] = []
        for step in range(train_fc_steps + 1):
            _, step_logits = mtp_forward()
            loss = F.cross_entropy(step_logits.float(), label)
            history.append({"step": step, "loss": float(loss.detach().item())})
            if step == train_fc_steps:
                mtp_logits = step_logits.detach()
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        sidecar["mtp.fc.weight"].requires_grad_(False)
        train_report = {
            "mode": "fc_only_single_prompt_ce_to_base_argmax",
            "steps": train_fc_steps,
            "lr": train_lr,
            "label_token_id": base_argmax,
            "label_text": runner.tokenizer.decode([base_argmax]),
            "loss_before": history[0]["loss"],
            "loss_after": history[-1]["loss"],
            "loss_delta": history[-1]["loss"] - history[0]["loss"],
            "history": history,
            "weights_saved": False,
        }

    finite = bool(torch.isfinite(mtp_logits).all().item())
    mtp_argmax = int(mtp_logits[0].argmax().item())
    result = {
        "schema_version": "lynn-a100-mtp-forward-smoke-v1",
        "decision": "GREEN: MTP sidecar forward path produced finite draft logits." if finite else "RED: MTP logits contain non-finite values.",
        "base_model": str(base_model),
        "sidecar_file": str(sidecar_file),
        "prompt": prompt,
        "use_chat_template": use_chat_template,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "prompt_tokens": int(ids.numel()),
        "last_token_id": last_token_id,
        "last_token_text": runner.tokenizer.decode([last_token_id]),
        "last_position": last_pos,
        "mtp_hidden_shape": list(mtp_hidden.shape),
        "mtp_logits_shape": list(mtp_logits.shape),
        "mtp_logits_finite": finite,
        "base_next_argmax": {
            "token_id": base_argmax,
            "text": runner.tokenizer.decode([base_argmax]),
        },
        "mtp_draft_argmax": {
            "token_id": mtp_argmax,
            "text": runner.tokenizer.decode([mtp_argmax]),
        },
        "argmax_match": base_argmax == mtp_argmax,
        "train_smoke": train_report,
        "base_next_topk": _topk(runner.tokenizer, base_logits, top_k),
        "mtp_draft_topk": _topk(runner.tokenizer, mtp_logits, top_k),
        "sidecar": sidecar_inventory,
    }
    del runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--sidecar-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Return one JSON object with keys city and unit for Tokyo in celsius.")
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--train-fc-steps", type=int, default=0)
    ap.add_argument("--train-lr", type=float, default=1e-3)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    result = run_smoke(
        base_model=Path(args.base_model),
        sidecar_file=Path(args.sidecar_file),
        prompt=args.prompt,
        use_chat_template=args.use_chat_template,
        device=args.device,
        dtype=dtype,
        top_k=args.top_k,
        train_fc_steps=args.train_fc_steps,
        train_lr=args.train_lr,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["mtp_logits_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
