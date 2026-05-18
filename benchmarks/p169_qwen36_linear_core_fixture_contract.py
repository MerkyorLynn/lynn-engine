#!/usr/bin/env python3
"""P169: Qwen3.6 linear/GDN core fixture export and self-check contract.

P168 showed the repeated hot region is the linear-attention core, especially
`in_proj -> recurrent/GDN -> conv`. P169 creates a fixture target for exact
boundary work without changing serving defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LynnInferenceState  # noqa: E402
from engine.incremental_decode import (  # noqa: E402
    _linear,
    _linear_conv_update_decode,
    _rms_norm_gated_decode,
)
from engine.qwen36_linear_attn_block import (  # noqa: E402
    HEAD_K_DIM,
    HEAD_V_DIM,
    KEY_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    VALUE_DIM,
    V_PER_K,
)
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.gated_delta import (  # noqa: E402
    recurrent_gated_delta_fused_prepare,
    recurrent_gated_delta_fused_prepare_gqa,
)


DEFAULT_PROMPTS = [
    "用一句话解释 MoE active parameters",
    "Return a JSON object with keys name and score.",
]
CHECK_KEYS = [
    "proj_all",
    "out_conv",
    "conv_state_out",
    "core_attn_out",
    "recurrent_state_out",
    "gated_norm_out",
    "linear_core_out",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_snapshot() -> dict[str, str | None]:
    names = [
        "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4",
        "LYNN_LINEAR_ATTN_CONV_BACKEND",
        "LYNN_LINEAR_ATTN_RECURRENT_BACKEND",
        "LYNN_LINEAR_ATTN_RECURRENT_INPLACE",
        "LYNN_LINEAR_ATTN_GQA_RECURRENT",
        "LYNN_RMSNORM_GATED_BACKEND",
    ]
    return {name: os.environ.get(name) for name in names}


def _parse_layers(raw: str, runner: LynnIncrementalRunner) -> list[int]:
    linear = [i for i, kind in enumerate(runner.layer_types) if kind == "linear_attention"]
    if raw == "all":
        return linear
    if raw == "first-of-block":
        return [i for i in linear if i % 4 == 0]
    layers: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            layers.extend(range(a, b + 1))
        else:
            layers.append(int(part))
    bad = [i for i in layers if i not in linear]
    if bad:
        raise ValueError(f"requested non-linear-attention layers: {bad}")
    return sorted(dict.fromkeys(layers))


def _prefill_prompt(runner: LynnIncrementalRunner, prompt: str, max_seq_len: int):
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState.from_config(
        runner.cfg,
        batch=1,
        max_seq_len=max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for layer_idx in range(runner.n_layers):
        h = _prefill_layer(
            h,
            pos,
            runner.layer_types[layer_idx],
            runner.layer_weights[layer_idx],
            runner.layer_cfgs[layer_idx],
            state,
            layer_idx,
        )
    state.seq_len = int(ids.shape[1])
    logits = runner._lm_head_logits(_rms_norm(h, runner.outside["model.language_model.norm.weight"]))
    return int(logits[0].argmax().item()), int(ids.shape[1]), state


def _linear_core_reference(
    h_norm: torch.Tensor,
    w: dict[str, Any],
    recurrent_state_in: torch.Tensor,
    conv_state_in: torch.Tensor,
) -> dict[str, torch.Tensor]:
    fused_key = "linear_attn._in_proj_qkv_z_b_a.weight"
    if fused_key not in w:
        raise RuntimeError(f"{fused_key} missing; set LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1")
    B = h_norm.shape[0]
    use_gqa = V_PER_K > 1 and os.environ.get("LYNN_LINEAR_ATTN_GQA_RECURRENT", "0") == "1"

    proj_all = _linear(h_norm, w[fused_key])
    mixed_new, z_raw, b_raw, a_raw = torch.split(
        proj_all,
        [KEY_DIM + KEY_DIM + VALUE_DIM, VALUE_DIM, NUM_V_HEADS, NUM_V_HEADS],
        dim=-1,
    )
    out_conv, conv_state_out = _linear_conv_update_decode(
        mixed_new.transpose(1, 2),
        conv_state_in.clone(),
        w["linear_attn.conv1d.weight"],
    )
    q, k, v = torch.split(out_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
    k = k.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
    v = v.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
    if V_PER_K > 1 and not use_gqa:
        q = q.repeat_interleave(V_PER_K, dim=2)
        k = k.repeat_interleave(V_PER_K, dim=2)
    z = z_raw.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
    beta = b_raw.sigmoid()
    g = -w["linear_attn.A_log"].float().exp() * F.softplus(a_raw.float() + w["linear_attn.dt_bias"].float())
    state_work = recurrent_state_in.clone()
    if use_gqa:
        core_attn_out, recurrent_state_out = recurrent_gated_delta_fused_prepare_gqa(q, k, v, g, beta, state_work)
    else:
        core_attn_out, recurrent_state_out = recurrent_gated_delta_fused_prepare(q, k, v, g, beta, state_work)
    flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
    flat_z = z.reshape(-1, HEAD_V_DIM)
    gated_norm_out = _rms_norm_gated_decode(flat_x, w["linear_attn.norm.weight"], flat_z)
    gated_norm_out = gated_norm_out.reshape(B, 1, NUM_V_HEADS * HEAD_V_DIM)
    linear_core_out = _linear(gated_norm_out, w["linear_attn.out_proj.weight"])
    return {
        "proj_all": proj_all.detach().clone(),
        "mixed_new": mixed_new.detach().clone(),
        "z": z.detach().clone(),
        "b": b_raw.detach().clone(),
        "a": a_raw.detach().clone(),
        "out_conv": out_conv.detach().clone(),
        "conv_state_out": conv_state_out.detach().clone(),
        "q_for_recurrent": q.detach().clone(),
        "k_for_recurrent": k.detach().clone(),
        "v_for_recurrent": v.detach().clone(),
        "beta": beta.detach().clone(),
        "g": g.detach().clone(),
        "core_attn_out": core_attn_out.detach().clone(),
        "recurrent_state_out": recurrent_state_out.detach().clone(),
        "gated_norm_out": gated_norm_out.detach().clone(),
        "linear_core_out": linear_core_out.detach().clone(),
    }


def _metrics(ref: torch.Tensor, got: torch.Tensor) -> dict[str, float | int]:
    rf = ref.float().reshape(-1)
    gf = got.float().reshape(-1)
    diff = rf - gf
    ref_norm = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    got_norm = torch.linalg.vector_norm(gf).clamp_min(1e-12)
    max_abs = float(diff.abs().max().item())
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(diff).item() / float(ref_norm.item())),
        "cosine": float((torch.dot(rf, gf) / (ref_norm * got_norm)).item()),
        "exact": 1 if max_abs == 0.0 else 0,
    }


def export_fixtures(args: argparse.Namespace) -> dict[str, Any]:
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=torch.bfloat16, max_seq_len=args.max_seq_len, verbose=False)
    layers = _parse_layers(args.layers, runner)
    out_dir = Path(args.fixtures)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.prompt or DEFAULT_PROMPTS
    entries: list[dict[str, Any]] = []
    for prompt_id, prompt in enumerate(prompts):
        next_id, prompt_tokens, state = _prefill_prompt(runner, prompt, args.max_seq_len)
        token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
        pos = torch.tensor([[state.seq_len]], device=runner.device, dtype=torch.long)
        h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        selected = set(layers)
        for layer_idx in range(runner.n_layers):
            w = runner.layer_weights[layer_idx]
            if runner.layer_types[layer_idx] == "linear_attention":
                h_norm = _rms_norm(h, w["input_layernorm.weight"])
                if layer_idx in selected:
                    tensors = {
                        "h_norm": h_norm.detach().clone(),
                        "recurrent_state_in": state.recurrent_state[layer_idx].detach().clone(),
                        "conv_state_in": state.conv_state[layer_idx].detach().clone(),
                    }
                    tensors.update(
                        _linear_core_reference(
                            h_norm,
                            w,
                            state.recurrent_state[layer_idx],
                            state.conv_state[layer_idx],
                        )
                    )
                    fixture_name = f"layer_{layer_idx:02d}_prompt_{prompt_id:02d}.safetensors"
                    fixture_path = out_dir / fixture_name
                    save_file(tensors, str(fixture_path))
                    entries.append(
                        {
                            "file": fixture_name,
                            "sha256": _sha256(fixture_path),
                            "layer_id": layer_idx,
                            "prompt_id": prompt_id,
                            "prompt": prompt,
                            "prompt_tokens": prompt_tokens,
                            "decode_token_id": next_id,
                            "seq_len": int(state.seq_len),
                            "tensor_shapes": {k: list(v.shape) for k, v in tensors.items()},
                            "tensor_dtypes": {k: str(v.dtype) for k, v in tensors.items()},
                        }
                    )
            h = runner._decode_layer_fast(h, pos, state, layer_idx)
        state.seq_len += 1
    manifest = {
        "schema_version": "lynn-qwen36-linear-core-fixture-v1",
        "model": args.model,
        "layers": layers,
        "prompts": prompts,
        "env": _env_snapshot(),
        "created_unix": time.time(),
        "fixtures": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def check_fixtures(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_dir = Path(args.fixtures)
    manifest = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    candidate_dir = Path(args.candidate_output_dir) if args.candidate_output_dir else None
    runner = None
    if candidate_dir is None:
        runner = LynnIncrementalRunner(
            args.model or manifest["model"],
            device=args.device,
            dtype=torch.bfloat16,
            max_seq_len=args.max_seq_len,
            verbose=False,
        )
    rows = []
    for item in manifest["fixtures"]:
        tensors = load_file(str(fixtures_dir / item["file"]), device=args.device)
        layer = int(item["layer_id"])
        if candidate_dir is None:
            assert runner is not None
            candidate = _linear_core_reference(
                tensors["h_norm"],
                runner.layer_weights[layer],
                tensors["recurrent_state_in"],
                tensors["conv_state_in"],
            )
        else:
            candidates = [
                candidate_dir / item["file"],
                candidate_dir / Path(item["file"]).name,
                candidate_dir / f"{Path(item['file']).stem}.safetensors",
            ]
            found = next((path for path in candidates if path.exists()), None)
            if found is None:
                rows.append(
                    {
                        "fixture_file": item["file"],
                        "layer_id": layer,
                        "prompt_id": int(item["prompt_id"]),
                        "passed": False,
                        "per_tensor": {},
                        "fail_reasons": ["candidate file missing"],
                    }
                )
                continue
            candidate = load_file(str(found), device=args.device)
        per_tensor = {}
        passed = True
        fail_reasons: list[str] = []
        for key in CHECK_KEYS:
            if key not in candidate:
                if args.require_all_keys:
                    passed = False
                    fail_reasons.append(f"missing candidate tensor {key}")
                continue
            m = _metrics(tensors[key], candidate[key])
            per_tensor[key] = m
            # Exact tensor equality is the primary contract. Cosine is computed
            # in FP32 and can print as 0.9999998 even when max_abs is exactly 0.
            if m["exact"] != 1 and (
                m["max_abs"] > args.max_abs_threshold or m["cosine"] < args.cosine_threshold
            ):
                passed = False
                fail_reasons.append(f"{key} exceeds thresholds")
        rows.append(
            {
                "fixture_file": item["file"],
                "layer_id": layer,
                "prompt_id": int(item["prompt_id"]),
                "passed": passed,
                "per_tensor": per_tensor,
                "fail_reasons": fail_reasons,
            }
        )
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    max_abs_max = max(
        (float(metric["max_abs"]) for row in rows for metric in row["per_tensor"].values()),
        default=0.0,
    )
    cosine_min = min(
        (float(metric["cosine"]) for row in rows for metric in row["per_tensor"].values()),
        default=1.0,
    )
    return {
        "schema_version": "lynn-qwen36-linear-core-fixture-contract-v1",
        "mode": "candidate-output-dir" if candidate_dir is not None else "selfcheck",
        "fixtures": str(fixtures_dir),
        "candidate_output_dir": str(candidate_dir) if candidate_dir is not None else None,
        "model": args.model or manifest["model"],
        "thresholds": {
            "max_abs_threshold": args.max_abs_threshold,
            "cosine_threshold": args.cosine_threshold,
        },
        "summary": {
            "passed": passed,
            "total": total,
            "max_abs_max": max_abs_max,
            "cosine_min": cosine_min,
            "all_passed": passed == total,
        },
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--layers", default="first-of-block")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-abs-threshold", type=float, default=0.0)
    parser.add_argument("--cosine-threshold", type=float, default=0.999999)
    parser.add_argument("--candidate-output-dir", default="")
    parser.add_argument("--require-all-keys", action="store_true")
    args = parser.parse_args()
    if not args.export and not args.check:
        args.export = True
        args.check = True
    report: dict[str, Any] = {}
    if args.export:
        if not args.model:
            raise ValueError("--model is required when --export is set")
        report["export"] = export_fixtures(args)
    if args.check:
        report["contract"] = check_fixtures(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.get("contract", report.get("export", {})).get("summary", report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
