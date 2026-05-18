#!/usr/bin/env python3
"""P147 · Triton-contract active-MoE stage gate.

The fixture line now has three useful but separate contracts:

* p135/p136: slot-order PyTorch fixtures are self-consistent.
* p138/p139: slot-packed NVFP4 tensors round-trip to BF16 weights.
* p146: resident candidates must match greedy generation before P25/structured.

P147 fills the missing admission gate between fixture work and resident P37:
it runs the *current production Triton active-MoE kernels* on p138 slot-packed
NVFP4 fixtures, then optionally compares candidate stage outputs against those
Triton stage references.

The important contract details are intentionally strict:

* slot order is preserved;
* routing weights stay FP32 for the down kernel;
* gate/up writes BF16 intermediate values before down;
* down reloads that BF16 intermediate, accumulates each slot in FP32, applies
  the FP32 route weight after the per-slot down projection, then stores BF16.

Any Native MoE kernel that cannot match this stage output should remain a
fixture research artifact and should not be escalated to resident P37.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512


@dataclass
class StageResult:
    fixture_file: str
    layer_id: int
    prompt_id: int
    gateup_ms: float
    down_ms: float
    candidate_path: str | None
    inter_compared: bool
    inter_max_abs: float | None
    inter_mean_abs: float | None
    inter_rel_l2: float | None
    inter_cosine: float | None
    inter_exact: int | None
    out_compared: bool
    out_max_abs: float | None
    out_mean_abs: float | None
    out_rel_l2: float | None
    out_cosine: float | None
    out_exact: int | None
    passed: bool
    fail_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_fixture(path: Path, device: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load as load_buffer
    from safetensors.torch import load_file

    if len(path.suffixes) >= 2 and path.suffixes[-2:] == [".safetensors", ".gz"]:
        with gzip.open(str(path), "rb") as f:
            raw = f.read()
        return {k: v.to(device) for k, v in load_buffer(raw).items()}
    return load_file(str(path), device=device)


def _load_candidate(
    candidate_dir: str,
    fixture_file: str,
    device: str,
) -> tuple[dict[str, torch.Tensor], str]:
    base = Path(candidate_dir)
    stem = Path(fixture_file).stem
    if stem.endswith(".safetensors"):
        stem = Path(stem).stem
    candidates = [
        base / fixture_file,
        base / Path(fixture_file).name,
        base / f"{stem}.safetensors",
        base / f"{stem}_candidate.safetensors",
        base / f"{stem}_triton_stage.safetensors",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"no candidate output found for {fixture_file!r} in {candidate_dir}")
    return _load_fixture(path, device), str(path)


def _metric(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float | int]:
    rf = ref.float().flatten()
    cf = cand.float().flatten()
    diff = rf - cf
    ref_norm = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    cand_norm = torch.linalg.vector_norm(cf).clamp_min(1e-12)
    diff_norm = torch.linalg.vector_norm(diff)
    max_abs = float(diff.abs().max())
    cosine = float(torch.dot(rf, cf) / (ref_norm * cand_norm))
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.abs().mean()),
        "rel_l2": float(diff_norm / ref_norm),
        "cosine": cosine,
        "exact": 1 if max_abs == 0.0 else 0,
    }


def _bench_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _find_tensor(data: dict[str, torch.Tensor], keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _run_triton_reference(
    data: dict[str, torch.Tensor],
    *,
    gate_block_inter: int,
    gate_block_hidden: int,
    gate_num_warps: int,
    down_block_hidden: int,
    down_block_inter: int,
    down_num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from triton_kernels.nvfp4_moe import (
        nvfp4_grouped_down_weighted_sum,
        nvfp4_grouped_gate_up_silu_fast_decode,
    )

    hidden = data["hidden_in"].to(torch.bfloat16).view(-1).contiguous()
    top_k = int(data["slot_gate_up_packed"].shape[0])
    slot_ids = torch.arange(top_k, device=hidden.device, dtype=torch.int32)
    routing_weights = data["routing_weights"].to(torch.float32).contiguous()

    inter = nvfp4_grouped_gate_up_silu_fast_decode(
        hidden,
        slot_ids,
        data["slot_gate_up_packed"].contiguous(),
        data["slot_gate_up_scale"].contiguous(),
        data["slot_gate_up_global_scale"].to(hidden.device).contiguous(),
        block_inter=gate_block_inter,
        block_hidden=gate_block_hidden,
        num_warps=gate_num_warps,
    )
    out = nvfp4_grouped_down_weighted_sum(
        inter,
        slot_ids,
        routing_weights,
        data["slot_down_packed"].contiguous(),
        data["slot_down_scale"].contiguous(),
        data["slot_down_global_scale"].to(hidden.device).contiguous(),
        block_hidden=down_block_hidden,
        block_inter=down_block_inter,
        num_warps=down_num_warps,
    )
    return inter.contiguous(), out.view(1, HIDDEN_SIZE).contiguous()


def run_gate(
    *,
    packed_fixtures: str,
    candidate_output_dir: str | None,
    write_reference_dir: str | None,
    device: str,
    max_abs_threshold: float,
    cosine_threshold: float,
    warmup: int,
    iters: int,
    gate_block_inter: int,
    gate_block_hidden: int,
    gate_num_warps: int,
    down_block_hidden: int,
    down_block_inter: int,
    down_num_warps: int,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    packed_path = Path(packed_fixtures)
    manifest_path = packed_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"p138 manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    ref_dir = Path(write_reference_dir) if write_reference_dir else None
    if ref_dir is not None:
        ref_dir.mkdir(parents=True, exist_ok=True)

    results: list[StageResult] = []
    print("[p147] Triton-contract active-MoE stage gate")
    print(f"[p147] packed_fixtures={packed_path}")
    print(f"[p147] candidate_output_dir={candidate_output_dir or '<none>'}")
    print(f"[p147] write_reference_dir={ref_dir or '<none>'}")
    print()

    for entry in manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = int(entry["layer_id"])
        prompt_id = int(entry["prompt_id"])
        data = _load_fixture(packed_path / fixture_file, device)

        def gate_fn() -> torch.Tensor:
            return _run_triton_reference(
                data,
                gate_block_inter=gate_block_inter,
                gate_block_hidden=gate_block_hidden,
                gate_num_warps=gate_num_warps,
                down_block_hidden=down_block_hidden,
                down_block_inter=down_block_inter,
                down_num_warps=down_num_warps,
            )[0]

        # Run once and then benchmark stages separately. The down benchmark
        # reuses the exact BF16 inter tensor from the reference run.
        inter_ref, out_ref = _run_triton_reference(
            data,
            gate_block_inter=gate_block_inter,
            gate_block_hidden=gate_block_hidden,
            gate_num_warps=gate_num_warps,
            down_block_hidden=down_block_hidden,
            down_block_inter=down_block_inter,
            down_num_warps=down_num_warps,
        )
        from triton_kernels.nvfp4_moe import nvfp4_grouped_down_weighted_sum

        hidden = data["hidden_in"].to(torch.bfloat16).view(-1).contiguous()
        top_k = int(data["slot_gate_up_packed"].shape[0])
        slot_ids = torch.arange(top_k, device=hidden.device, dtype=torch.int32)
        routing_weights = data["routing_weights"].to(torch.float32).contiguous()

        gate_ms = _bench_ms(gate_fn, warmup=warmup, iters=iters)
        down_ms = _bench_ms(
            lambda: nvfp4_grouped_down_weighted_sum(
                inter_ref,
                slot_ids,
                routing_weights,
                data["slot_down_packed"].contiguous(),
                data["slot_down_scale"].contiguous(),
                data["slot_down_global_scale"].to(device).contiguous(),
                block_hidden=down_block_hidden,
                block_inter=down_block_inter,
                num_warps=down_num_warps,
            ),
            warmup=warmup,
            iters=iters,
        )

        if ref_dir is not None:
            out_name = f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_triton_stage.safetensors"
            save_file(
                {
                    "hidden_in": data["hidden_in"].detach().cpu().contiguous(),
                    "expert_ids": data["expert_ids"].detach().cpu().contiguous(),
                    "slot_expert_ids": torch.arange(top_k, dtype=torch.int32),
                    "routing_weights": data["routing_weights"].detach().cpu().contiguous(),
                    "triton_inter": inter_ref.detach().cpu().contiguous(),
                    "routed_output": out_ref.detach().cpu().contiguous(),
                },
                str(ref_dir / out_name),
            )

        fail_reasons: list[str] = []
        candidate_path: str | None = None
        inter_metrics: dict[str, float | int] | None = None
        out_metrics: dict[str, float | int] | None = None

        if candidate_output_dir:
            cand, candidate_path = _load_candidate(candidate_output_dir, fixture_file, device)
            cand_inter = _find_tensor(cand, ("triton_inter", "candidate_inter", "slot_intermediate", "inter"))
            cand_out = _find_tensor(cand, ("routed_output", "candidate_output", "output", "moe_output"))
            if cand_inter is None and cand_out is None:
                fail_reasons.append("candidate contains neither inter nor routed output tensor")
            if cand_inter is not None:
                inter_metrics = _metric(inter_ref, cand_inter.to(torch.bfloat16))
                if float(inter_metrics["max_abs"]) > max_abs_threshold:
                    fail_reasons.append(
                        f"inter max_abs {inter_metrics['max_abs']:.6e} > {max_abs_threshold}"
                    )
                if int(inter_metrics["exact"]) == 0 and float(inter_metrics["cosine"]) < cosine_threshold:
                    fail_reasons.append(
                        f"inter cosine {inter_metrics['cosine']:.9f} < {cosine_threshold}"
                    )
            if cand_out is not None:
                out_metrics = _metric(out_ref, cand_out.to(torch.bfloat16).view(1, HIDDEN_SIZE))
                if float(out_metrics["max_abs"]) > max_abs_threshold:
                    fail_reasons.append(
                        f"out max_abs {out_metrics['max_abs']:.6e} > {max_abs_threshold}"
                    )
                if int(out_metrics["exact"]) == 0 and float(out_metrics["cosine"]) < cosine_threshold:
                    fail_reasons.append(
                        f"out cosine {out_metrics['cosine']:.9f} < {cosine_threshold}"
                    )

        passed = not fail_reasons
        result = StageResult(
            fixture_file=fixture_file,
            layer_id=layer_id,
            prompt_id=prompt_id,
            gateup_ms=gate_ms,
            down_ms=down_ms,
            candidate_path=candidate_path,
            inter_compared=inter_metrics is not None,
            inter_max_abs=None if inter_metrics is None else float(inter_metrics["max_abs"]),
            inter_mean_abs=None if inter_metrics is None else float(inter_metrics["mean_abs"]),
            inter_rel_l2=None if inter_metrics is None else float(inter_metrics["rel_l2"]),
            inter_cosine=None if inter_metrics is None else float(inter_metrics["cosine"]),
            inter_exact=None if inter_metrics is None else int(inter_metrics["exact"]),
            out_compared=out_metrics is not None,
            out_max_abs=None if out_metrics is None else float(out_metrics["max_abs"]),
            out_mean_abs=None if out_metrics is None else float(out_metrics["mean_abs"]),
            out_rel_l2=None if out_metrics is None else float(out_metrics["rel_l2"]),
            out_cosine=None if out_metrics is None else float(out_metrics["cosine"]),
            out_exact=None if out_metrics is None else int(out_metrics["exact"]),
            passed=passed,
            fail_reasons=fail_reasons,
        )
        results.append(result)
        status = "GREEN" if passed else "RED"
        out_abs = "-" if out_metrics is None else f"{out_metrics['max_abs']:.2e}"
        inter_abs = "-" if inter_metrics is None else f"{inter_metrics['max_abs']:.2e}"
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} {status} "
            f"gate={gate_ms:.4f}ms down={down_ms:.4f}ms "
            f"inter_abs={inter_abs} out_abs={out_abs}",
            flush=True,
        )

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    gate_mean = sum(r.gateup_ms for r in results) / total if total else 0.0
    down_mean = sum(r.down_ms for r in results) / total if total else 0.0
    inter_abs = [r.inter_max_abs for r in results if r.inter_max_abs is not None]
    out_abs = [r.out_max_abs for r in results if r.out_max_abs is not None]
    verdict = (
        "REFERENCE_READY"
        if candidate_output_dir is None
        else ("TRITON_STAGE_EXACT" if passed_count == total else "CLOSED_STAGE_DRIFT")
    )
    report = {
        "schema": "lynn-moe-triton-stage-contract-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_fixtures": str(packed_path),
        "candidate_output_dir": candidate_output_dir,
        "reference_output_dir": str(ref_dir) if ref_dir is not None else None,
        "max_abs_threshold": max_abs_threshold,
        "cosine_threshold": cosine_threshold,
        "kernel_config": {
            "gate_block_inter": gate_block_inter,
            "gate_block_hidden": gate_block_hidden,
            "gate_num_warps": gate_num_warps,
            "down_block_hidden": down_block_hidden,
            "down_block_inter": down_block_inter,
            "down_num_warps": down_num_warps,
        },
        "total": total,
        "passed": passed_count,
        "verdict": verdict,
        "gateup_ms_mean": gate_mean,
        "down_ms_mean": down_mean,
        "active_moe_ms_mean": gate_mean + down_mean,
        "inter_max_abs_max": max(inter_abs) if inter_abs else None,
        "out_max_abs_max": max(out_abs) if out_abs else None,
        "results": [r.to_dict() for r in results],
    }
    print()
    print(f"[p147] verdict={verdict} passed={passed_count}/{total}")
    print(f"[p147] gateup_ms_mean={gate_mean:.4f} down_ms_mean={down_mean:.4f}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Triton-contract active-MoE stage gate.")
    ap.add_argument("--packed-fixtures", required=True, help="p138 packed slot fixture directory.")
    ap.add_argument("--candidate-output-dir", default=None, help="Optional candidate stage output directory.")
    ap.add_argument("--write-reference-dir", default=None, help="Optional directory for Triton stage reference outputs.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-abs-threshold", type=float, default=0.0)
    ap.add_argument("--cosine-threshold", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--gate-block-inter", type=int, default=8)
    ap.add_argument("--gate-block-hidden", type=int, default=256)
    ap.add_argument("--gate-num-warps", type=int, default=4)
    ap.add_argument("--down-block-hidden", type=int, default=8)
    ap.add_argument("--down-block-inter", type=int, default=512)
    ap.add_argument("--down-num-warps", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = run_gate(
        packed_fixtures=args.packed_fixtures,
        candidate_output_dir=args.candidate_output_dir,
        write_reference_dir=args.write_reference_dir,
        device=args.device,
        max_abs_threshold=args.max_abs_threshold,
        cosine_threshold=args.cosine_threshold,
        warmup=args.warmup,
        iters=args.iters,
        gate_block_inter=args.gate_block_inter,
        gate_block_hidden=args.gate_block_hidden,
        gate_num_warps=args.gate_num_warps,
        down_block_hidden=args.down_block_hidden,
        down_block_inter=args.down_block_inter,
        down_num_warps=args.down_num_warps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[p147] report={out_path}")
    return 0 if report["verdict"] in {"REFERENCE_READY", "TRITON_STAGE_EXACT"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
