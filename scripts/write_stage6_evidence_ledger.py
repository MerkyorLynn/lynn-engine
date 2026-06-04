#!/usr/bin/env python3
"""Write the Stage 6 evidence ledger.

This is intentionally conservative: runbooks and contracts can make a gate
READY, but only reports/result.json artifacts with explicit PASS evidence can
make it BANKED.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
STAGE6 = ROOT / "reports" / "stage6"


@dataclass(frozen=True)
class Gate:
    gate: str
    title: str
    status: str
    evidence: str
    artifact: str
    next_step: str
    decision: str = ""


def _read(rel: str) -> str:
    path = ROOT / rel
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _contains(rel: str, needles: Iterable[str]) -> bool:
    text = _read(rel)
    return bool(text) and all(needle in text for needle in needles)


def _latest_result(prefixes: Iterable[str]) -> tuple[Path | None, dict[str, Any] | None]:
    dirs = [
        path
        for path in STAGE6.iterdir()
        if path.is_dir() and any(path.name.startswith(prefix) for prefix in prefixes)
    ]
    for run_dir in sorted(dirs, reverse=True):
        result = run_dir / "result.json"
        if not result.exists():
            continue
        try:
            return run_dir, json.loads(result.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return run_dir, {"_json_error": True}
    return None, None


def _latest_run(prefixes: Iterable[str]) -> Path | None:
    dirs = [
        path
        for path in STAGE6.iterdir()
        if path.is_dir() and any(path.name.startswith(prefix) for prefix in prefixes)
    ]
    return sorted(dirs, reverse=True)[0] if dirs else None


def _passes_all(data: dict[str, Any] | None) -> bool:
    if not data:
        return False
    passes = data.get("passes")
    if isinstance(passes, dict) and passes.get("all") is True:
        return True
    return data.get("all_pass") is True or data.get("ALL_PASS") is True


def _decision(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    if data.get("_json_error"):
        return "JSON_DECODE_ERROR"
    for key in ("decision", "verdict", "status"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    passes = data.get("passes")
    if isinstance(passes, dict):
        return "passes.all=true" if passes.get("all") is True else "passes.all=false"
    return "all_pass=true" if data.get("all_pass") is True else ""


def _artifact(path: Path | None, fallback: str = "") -> str:
    if path is None:
        return fallback
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _report_gate(gate: str, title: str, report: str, pass_needles: list[str], next_step: str) -> Gate:
    status = "BANKED" if _contains(report, pass_needles) else "MISSING_OR_WEAK"
    evidence = "report contains explicit pass evidence" if status == "BANKED" else "report missing or pass evidence too weak"
    return Gate(gate, title, status, evidence, report, next_step)


def _result_gate(
    gate: str,
    title: str,
    prefixes: list[str],
    next_step: str,
    *,
    negative_if_fail: bool = False,
) -> Gate:
    run_dir, data = _latest_result(prefixes)
    if data is None:
        return Gate(gate, title, "NOT_RUN", "no result.json artifact found", "", next_step)
    if data.get("_json_error"):
        return Gate(gate, title, "FAILED_ARTIFACT", "latest result.json is not valid JSON", _artifact(run_dir), next_step)
    if _passes_all(data):
        return Gate(gate, title, "BANKED", "latest result.json has passes.all=true", _artifact(run_dir), next_step, _decision(data))
    status = "CLOSED_NEGATIVE" if negative_if_fail else "FAILED_ARTIFACT"
    evidence = "latest result.json failed as an intentional negative gate" if negative_if_fail else "latest result.json did not pass"
    return Gate(gate, title, status, evidence, _artifact(run_dir), next_step, _decision(data))


def _ready_gate(gate: str, title: str, required: list[str], next_step: str) -> Gate:
    missing = [rel for rel in required if not (ROOT / rel).exists()]
    if missing:
        return Gate(gate, title, "MISSING_TOOLING", f"missing: {', '.join(missing)}", "", next_step)
    return Gate(gate, title, "READY_WAITING_SPARK", "tooling/runbook exists, but no Spark PASS artifact is banked", ", ".join(required), next_step)


def _result_or_ready_gate(
    gate: str,
    title: str,
    prefixes: list[str],
    required: list[str],
    next_step: str,
) -> Gate:
    run_dir, data = _latest_result(prefixes)
    if data is not None:
        if data.get("_json_error"):
            return Gate(gate, title, "FAILED_ARTIFACT", "latest result.json is not valid JSON", _artifact(run_dir), next_step)
        if _passes_all(data):
            return Gate(gate, title, "BANKED", "latest Spark result.json has passes.all=true", _artifact(run_dir), next_step, _decision(data))
        return Gate(gate, title, "FAILED_ARTIFACT", "latest Spark result.json did not pass", _artifact(run_dir), next_step, _decision(data))
    return _ready_gate(gate, title, required, next_step)


def _result_timeout_or_ready_gate(
    gate: str,
    title: str,
    prefixes: list[str],
    required: list[str],
    next_step: str,
) -> Gate:
    run_dir = _latest_run(prefixes)
    if run_dir is None:
        return _ready_gate(gate, title, required, next_step)
    result = run_dir / "result.json"
    if result.exists():
        try:
            data = json.loads(result.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return Gate(gate, title, "FAILED_ARTIFACT", "latest result.json is not valid JSON", _artifact(run_dir), next_step)
        if _passes_all(data):
            return Gate(gate, title, "BANKED", "latest Spark result.json has passes.all=true", _artifact(run_dir), next_step, _decision(data))
        return Gate(gate, title, "FAILED_ARTIFACT", "latest Spark result.json did not pass", _artifact(run_dir), next_step, _decision(data))
    if (run_dir / "timeout.txt").exists():
        return Gate(
            gate,
            title,
            "TIMEOUT_NOT_CLEAN",
            "latest Spark run was terminated after slow-mode opt-in produced no result.json",
            _artifact(run_dir),
            next_step,
            (run_dir / "timeout.txt").read_text(encoding="utf-8").strip(),
        )
    if (run_dir / "docker_exit_code.txt").exists():
        return Gate(
            gate,
            title,
            "FAILED_ARTIFACT",
            "latest Spark run exited without result.json",
            _artifact(run_dir),
            next_step,
            (run_dir / "docker_exit_code.txt").read_text(encoding="utf-8").strip(),
        )
    return Gate(gate, title, "INCOMPLETE_ARTIFACT", "latest Spark run has no result.json yet", _artifact(run_dir), next_step)


def _contract_gate(gate: str, title: str, required: list[str], next_step: str) -> Gate:
    missing = [rel for rel in required if not (ROOT / rel).exists()]
    if missing:
        return Gate(gate, title, "MISSING_TOOLING", f"missing: {', '.join(missing)}", "", next_step)
    return Gate(
        gate,
        title,
        "CONTRACT_READY_UNIMPLEMENTED",
        "fail-loud ABI/static gate exists, but no fused implementation or GPU result is banked",
        ", ".join(required),
        next_step,
    )


def _p3a_batched_down_candidate_gate() -> Gate:
    required = [
        "reports/stage6/P3A_BATCHED_DOWN_CONTRACT_PROBE_20260604.md",
        "scripts/run_spark_stage6_p3a_contract_probe.sh",
        "scripts/spark_stage6_p3a_grouped_moe_contract_probe.py",
        "engine/moe_packed_nvfp4.py",
        "triton_kernels/nvfp4_moe.py",
    ]
    run_dir, data = _latest_result(["p3a_batched_down_"])
    if data is None:
        return _ready_gate(
            "p3a_batched_down_candidate",
            "P3-A batched-down speed candidate",
            required,
            "Run after the opt-in batched-down path exists; do not promote without numeric and speed gates.",
        )
    if data.get("_json_error"):
        return Gate(
            "p3a_batched_down_candidate",
            "P3-A batched-down speed candidate",
            "FAILED_ARTIFACT",
            "latest result.json is not valid JSON",
            _artifact(run_dir),
            "Fix artifact JSON before interpreting the candidate.",
        )
    rows = ((data.get("bench") or {}).get("rows") or [])
    speeds = [row.get("p3a_vs_bf16") for row in rows if isinstance(row.get("p3a_vs_bf16"), (int, float))]
    avg_speed = sum(speeds) / len(speeds) if speeds else None
    if not _passes_all(data):
        return Gate(
            "p3a_batched_down_candidate",
            "P3-A batched-down speed candidate",
            "FAILED_ARTIFACT",
            "latest Spark result.json did not pass numeric/shadow gates",
            _artifact(run_dir),
            "Fix correctness before interpreting speed.",
            _decision(data),
        )
    if avg_speed is not None and avg_speed < 1.0:
        return Gate(
            "p3a_batched_down_candidate",
            "P3-A batched-down speed candidate",
            "DIAGNOSTIC_BANKED",
            f"numeric/shadow gates pass, but average speed is {avg_speed:.3f}x vs BF16 active",
            _artifact(run_dir),
            "Do not promote; continue with route/materialization or true batched gate-up/down fusion.",
            "PASS_BUT_SPEED_GATE_CLOSED",
        )
    return Gate(
        "p3a_batched_down_candidate",
        "P3-A batched-down speed candidate",
        "BANKED",
        "numeric/shadow gates pass and candidate is not slower than BF16 active",
        _artifact(run_dir),
        "Escalate to P3-B/P3-C selected/resident gates before promotion.",
        _decision(data),
    )


def _decode_gpu_idle_probe_gate() -> Gate:
    required = [
        "scripts/run_spark_stage6_decode_gpu_idle_probe.sh",
        "scripts/spark_stage6_decode_gpu_idle_probe.py",
        "scripts/summarize_stage6_decode_gpu_idle_probe.py",
    ]
    run_dir, data = _latest_result(["decode_gpu_idle_probe_"])
    if data is None:
        return _ready_gate(
            "decode_gpu_idle_probe",
            "Decode GPU-idle / compiled-loop ROI probe",
            required,
            "Run before month-scale C++/compiled runtime work; this decides whether host dispatch is recoverable.",
        )
    if data.get("_json_error"):
        return Gate(
            "decode_gpu_idle_probe",
            "Decode GPU-idle / compiled-loop ROI probe",
            "FAILED_ARTIFACT",
            "latest result.json is not valid JSON",
            _artifact(run_dir),
            "Fix artifact JSON before interpreting compiled-loop ROI.",
        )
    if not _passes_all(data):
        return Gate(
            "decode_gpu_idle_probe",
            "Decode GPU-idle / compiled-loop ROI probe",
            "FAILED_ARTIFACT",
            "latest Spark result.json did not pass timing/launch gates",
            _artifact(run_dir),
            "Rerun before investing in runtime work.",
            _decision(data),
        )
    delta = data.get("delta") or {}
    signal = str(delta.get("compiled_loop_roi_signal") or _decision(data))
    gap = delta.get("host_gap_fraction_est")
    launches = delta.get("cuda_launches_per_token")
    evidence = "latest Spark result.json records decode GPU busy/host-gap estimate"
    if isinstance(gap, (int, float)) and isinstance(launches, (int, float)):
        evidence = f"host-gap fraction {gap:.3f}, CUDA launches/token {launches:.1f}"
    return Gate(
        "decode_gpu_idle_probe",
        "Decode GPU-idle / compiled-loop ROI probe",
        "DIAGNOSTIC_BANKED",
        evidence,
        _artifact(run_dir),
        "Use this ROI signal to choose MTP-light/compiled-loop prototype scope; do not treat it as speed promotion.",
        signal,
    )


def _r6000_fp4_mma_census_gate() -> Gate:
    required = [
        "reports/stage6/R6000_FP4_MMA_BRINGUP_RUNBOOK_20260604.md",
        "scripts/r6000_stage6_fp4_mma_census.py",
        "scripts/summarize_stage6_r6000_fp4_mma_census.py",
        "scripts/test_stage6_r6000_fp4_mma_census_tools.py",
    ]
    run_dir, data = _latest_result(["r6000_fp4_mma_census_"])
    if data is None:
        missing = [rel for rel in required if not (ROOT / rel).exists()]
        if missing:
            return Gate(
                "r6000_fp4_mma_census",
                "R6000 FP4-MMA bring-up and public-kernel census",
                "MISSING_TOOLING",
                f"missing: {', '.join(missing)}",
                "",
                "Add the R6000 bring-up tooling before renting/using the FP4-MMA host.",
            )
        return Gate(
            "r6000_fp4_mma_census",
            "R6000 FP4-MMA bring-up and public-kernel census",
            "READY_WAITING_R6000",
            "tooling/runbook exists, but no R6000 PASS artifact is banked",
            ", ".join(required),
            "Run on the RTX PRO 6000 96GB host with --run-contracts before starting a new FP4-MMA kernel.",
        )
    if data.get("_json_error"):
        return Gate(
            "r6000_fp4_mma_census",
            "R6000 FP4-MMA bring-up and public-kernel census",
            "FAILED_ARTIFACT",
            "latest result.json is not valid JSON",
            _artifact(run_dir),
            "Fix artifact JSON before interpreting R6000 readiness.",
        )
    if _passes_all(data) and data.get("decision") == "PASS_R6000_FP4_MMA_BRINGUP":
        torch_info = ((data.get("torch") or {}).get("json") or {})
        device = torch_info.get("device_name", "unknown")
        cap = torch_info.get("capability", "unknown")
        return Gate(
            "r6000_fp4_mma_census",
            "R6000 FP4-MMA bring-up and public-kernel census",
            "BANKED",
            f"R6000 FP4-MMA census passed on {device}, capability {cap}",
            _artifact(run_dir),
            "Start Lynn NVFP4 grouped-MoE FP4-MMA POC from CUTLASS/CuTe plus public Marlin/Machete census.",
            _decision(data),
        )
    return Gate(
        "r6000_fp4_mma_census",
        "R6000 FP4-MMA bring-up and public-kernel census",
        "FAILED_ARTIFACT",
        "latest R6000 census did not pass all machine/toolchain/public-kernel gates",
        _artifact(run_dir),
        "Fix machine/toolchain/public-kernel readiness before writing kernels.",
        _decision(data),
    )


def _p4b_contract_gate() -> Gate:
    required = [
        "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md",
        "scripts/test_stage6_p4b_single_kernel_static.py",
        "scripts/run_spark_stage6_p4b_single_cta_numeric_preflight.sh",
    ]
    missing = [rel for rel in required if not (ROOT / rel).exists()]
    if missing:
        return Gate(
            "p4b_single_kernel_contract",
            "P4B true fused single-kernel ABI",
            "MISSING_TOOLING",
            f"missing: {', '.join(missing)}",
            "",
            "Restore contract/static/numeric tooling before implementation work continues.",
        )
    if _contains(
        "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md",
        [
            "LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE=1",
            "PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE",
            "banked_fused_kernel=false",
        ],
    ):
        return Gate(
            "p4b_single_kernel_contract",
            "P4B true fused single-kernel ABI",
            "REFERENCE_IMPL_BANKED_SPEED_CLOSED",
            "opt-in single-CTA output-returning numeric reference is banked; speed/default promotion remain closed",
            ", ".join(required),
            "Scale beyond T=1/top_k=8, add byte-count profiler and speed gate before any promotion.",
        )
    return Gate(
        "p4b_single_kernel_contract",
        "P4B true fused single-kernel ABI",
        "CONTRACT_READY_UNIMPLEMENTED",
        "fail-loud ABI/static gate exists, but no output-returning implementation evidence is recorded",
        ", ".join(required),
        "Replace fail-loud-only contract with an opt-in numeric reference, then run Spark.",
    )


def _p4b_multi_cta_microbench_gate() -> Gate:
    required = [
        "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md",
        "scripts/run_spark_stage6_p4b_single_cta_microbench.sh",
        "scripts/spark_stage6_p4b_single_cta_microbench.py",
        "scripts/summarize_stage6_p4b_single_cta_microbench.py",
    ]
    run_dir, data = _latest_result(["p4b_multi_cta_microbench_"])
    if data is None:
        return _ready_gate(
            "p4b_multi_cta_recompute_microbench",
            "P4B multi-CTA recompute microbench",
            required,
            "Run with --candidate-mode multi_cta after the multi-CTA candidate is present.",
        )
    if data.get("_json_error"):
        return Gate(
            "p4b_multi_cta_recompute_microbench",
            "P4B multi-CTA recompute microbench",
            "FAILED_ARTIFACT",
            "latest result.json is not valid JSON",
            _artifact(run_dir),
            "Fix artifact JSON before using this gate.",
        )
    if not _passes_all(data):
        return Gate(
            "p4b_multi_cta_recompute_microbench",
            "P4B multi-CTA recompute microbench",
            "FAILED_ARTIFACT",
            "latest Spark result.json did not pass numeric/timing checks",
            _artifact(run_dir),
            "Fix numeric correctness before interpreting speed.",
            _decision(data),
        )
    speedup = ((data.get("bench") or {}).get("candidate_vs_reference_speedup"))
    if isinstance(speedup, (float, int)) and speedup < 1.0:
        return Gate(
            "p4b_multi_cta_recompute_microbench",
            "P4B multi-CTA recompute microbench",
            "CLOSED_NEGATIVE",
            f"numeric exact but slower than P4A two-stage; speedup={speedup:.6f}x",
            _artifact(run_dir),
            "Reject per-output-tile active recompute; next candidate must preserve active reuse.",
            _decision(data),
        )
    return Gate(
        "p4b_multi_cta_recompute_microbench",
        "P4B multi-CTA recompute microbench",
        "BANKED",
        "numeric/timing passed and candidate is not slower than P4A two-stage",
        _artifact(run_dir),
        "Escalate to real layer/e2e speed and RC quality gates before promotion.",
        _decision(data),
    )


def _p4c_active_reuse_decision_gate() -> Gate:
    required = [
        "reports/stage6/P4C_ACTIVE_REUSE_KERNEL_DECISION_20260604.md",
        "scripts/test_stage6_p4c_active_reuse_decision_static.py",
    ]
    missing = [rel for rel in required if not (ROOT / rel).exists()]
    if missing:
        return Gate(
            "p4c_active_reuse_decision",
            "P4C active-reuse kernel decision",
            "MISSING_TOOLING",
            f"missing: {', '.join(missing)}",
            "",
            "Restore the active-reuse decision doc/static gate before writing another P4B/P4C CUDA candidate.",
        )
    if _contains(
        "reports/stage6/P4C_ACTIVE_REUSE_KERNEL_DECISION_20260604.md",
        [
            "The next kernel must preserve",
            "active reuse.",
            "P4C, not P4B",
            "39.54 ms vs P4A 0.283 ms = 0.007x",
            "0.0058x",
            "fused_zero_shadow_active_reuse_contract",
        ],
    ):
        return Gate(
            "p4c_active_reuse_decision",
            "P4C active-reuse kernel decision",
            "DECISION_BANKED",
            "single-CTA and multi-CTA recompute anti-proofs are captured; next candidate must preserve active reuse",
            ", ".join(required),
            "Implement P4C active-reuse two-phase/CUTLASS-style candidate; do not report it as P4B out-only fused speed.",
        )
    return Gate(
        "p4c_active_reuse_decision",
        "P4C active-reuse kernel decision",
        "MISSING_OR_WEAK",
        "decision doc exists but does not preserve measured anti-proofs and naming boundary",
        ", ".join(required),
        "Strengthen the decision doc before another CUDA candidate is accepted.",
    )


def _p4c_shadow_cycle_first_divergence_gate() -> Gate:
    run_dir, data = _latest_result(["p4c_tile2_shadow_cycle_first_divergence_"])
    if data is None:
        return Gate(
            "p4c_tile2_shadow_cycle_first_divergence",
            "P4C tile2 shadow-cycle first-divergence diagnostic",
            "NOT_RUN",
            "no result.json artifact found",
            "",
            "Run after rc-mini rejection to explain whether P4C fails immediately or drifts under server-like shadow release.",
        )
    if data.get("_json_error"):
        return Gate(
            "p4c_tile2_shadow_cycle_first_divergence",
            "P4C tile2 shadow-cycle first-divergence diagnostic",
            "FAILED_ARTIFACT",
            "latest result.json is not valid JSON",
            _artifact(run_dir),
            "Fix artifact JSON before using this diagnostic.",
        )
    if (
        data.get("candidate_release_shadows_before_decode") is True
        and data.get("pass") is True
        and data.get("first_top1_divergence") is None
        and data.get("first_layer_divergence")
    ):
        first_layer = data.get("first_layer_divergence") or {}
        return Gate(
            "p4c_tile2_shadow_cycle_first_divergence",
            "P4C tile2 shadow-cycle first-divergence diagnostic",
            "DIAGNOSTIC_BANKED",
            (
                "server-like shadow cycle kept top-1 stable on the arithmetic prompt, "
                f"but first hidden drift appears at step {first_layer.get('step')}/layer {first_layer.get('layer')}"
            ),
            _artifact(run_dir),
            "Use as a go/no-go diagnostic only; do not promote P4C tile2 without wider RC exactness and e2e speed.",
            "pass=true; first_top1_divergence=null",
        )
    return Gate(
        "p4c_tile2_shadow_cycle_first_divergence",
        "P4C tile2 shadow-cycle first-divergence diagnostic",
        "FAILED_ARTIFACT",
        "latest diagnostic result did not satisfy the shadow-cycle evidence contract",
        _artifact(run_dir),
        "Inspect the diagnostic before making P4C promotion decisions.",
        _decision(data),
    )


def collect_gates() -> list[Gate]:
    gates: list[Gate] = [
        _report_gate(
            "stage6_60g_decode_release",
            "60 GiB decode-only shadow release",
            "reports/stage6/DECODE_ONLY_SHADOW_FREE_SERVING_RECIPE.md",
            ["88", "28", "60", "token-exact", "0.998"],
            "Use as serving capability; high-throughput multi-request still needs packed prefill.",
        ),
        _report_gate(
            "p0_1_no_reload_prefill",
            "P0.1 packed-prefill no-reload smoke",
            "reports/stage6/P01_NO_RELOAD_SMOKE_20260603.md",
            ["P0.1 PASSED", "| token-exact | true |", "| ALL_PASS | true |"],
            "Replace slow stream_bf16 proof path with real grouped kernels.",
        ),
        _report_gate(
            "p0_2_resident_inventory",
            "P0.2 resident BF16 inventory",
            "reports/stage6/P02_RESIDENT_INVENTORY_20260603.md",
            ["P0.2 PASSED", "4.72 GiB", "60.00 GiB"],
            "Prioritize projection/embed/lm-head residents after MoE shadow removal.",
        ),
        _report_gate(
            "p1_single_dense_projection",
            "P1 single dense projection packed-NVFP4 PoC",
            "reports/stage6/P1_DENSE_PROJECTION_POC_20260604.md",
            ["Phase 1 single dense projection PoC PASSED", "all | PASS"],
            "Do not infer M>1 serving win from this M=1 gate.",
        ),
        Gate(
            "p1a_batched_dense_rejected",
            "P1-A batched dense scalar bridge rejection",
            "CLOSED_NEGATIVE"
            if _contains("reports/stage6/P1A_TILED_PROJECTION_SWEEP_20260604.md", ["NOT promoted", "loses to Spark's BF16"])
            else "MISSING_OR_WEAK",
            "report closes the scalar M>1 dense bridge as slower than BF16"
            if _contains("reports/stage6/P1A_TILED_PROJECTION_SWEEP_20260604.md", ["NOT promoted", "loses to Spark's BF16"])
            else "negative report missing or evidence too weak",
            "reports/stage6/P1A_TILED_PROJECTION_SWEEP_20260604.md",
            "Keep as correctness probe; real M>1 dense win needs FP4-MMA/CUTLASS.",
        ),
        _report_gate(
            "p2_grouped_moe_census",
            "P2 grouped MoE packed-prefill census",
            "reports/stage6/P2_GROUPED_MOE_PREFILL_CENSUS_20260604.md",
            ["P2 census is banked", "Numeric gate for census: **PASS**"],
            "Use routed grouped MoE as the real packed-prefill target.",
        ),
        _result_gate(
            "p2ka_gated_delta_loop_rejected",
            "P2-KA gated-delta token loop rejection",
            ["p2ka_gated_delta_native_loop_long_", "p2ka_gated_delta_native_loop_"],
            "Use block linear-attn instead of token-loop decode recurrence.",
            negative_if_fail=True,
        ),
        _result_gate(
            "p2kb_block_linear_kernel",
            "P2-KB gated-delta block kernel PoC",
            ["p2kb_gated_delta_block_kernel_long_", "p2kb_gated_delta_block_kernel_"],
            "Keep composing into selected-layer prefill gates.",
        ),
        _result_gate(
            "p2l_linear_attn_block_integration",
            "P2-L block linear-attn integration smoke",
            ["p2l_linear_attn_block_integration_long_", "p2l_linear_attn_block_integration_"],
            "Combine with packed MoE in selected-layer gates.",
        ),
        _result_gate(
            "p2m_selected_layer_block_linear",
            "P2-M selected-layer block-linear smoke",
            ["p2m_selected_layer_block_linear_"],
            "Expand layer coverage before server RC.",
        ),
        _result_gate(
            "p2n_wider_layer_block_linear",
            "P2-N wider selected-layer block-linear smoke",
            ["p2n_wider_layer_block_linear_"],
            "Next real proof is P2-O/P3 server-shaped RC.",
        ),
        _result_or_ready_gate(
            "p2o_basic_packed_prefill_rc",
            "P2-O basic packed-prefill RC smoke",
            ["p2o_basic_packed_prefill_rc_smoke_"],
            [
                "reports/stage6/P2O_PACKED_PREFILL_RC_GATE_RUNBOOK_20260604.md",
                "scripts/run_spark_stage6_p2o_rc_smoke.sh",
                "scripts/spark_stage6_p2o_packed_prefill_rc_smoke.py",
            ],
            "Basic smoke is banked only as active-MoE no-reload correctness/memory evidence; do not treat the slow-mode prefill as a speed win.",
        ),
        _result_or_ready_gate(
            "p2o_rcmini_nonlong_packed_prefill_rc",
            "P2-O rc-mini non-long shard packed-prefill smoke",
            ["p2o_rc-mini_idx0-4_packed_prefill_rc_smoke_"],
            [
                "reports/stage6/P2O_PACKED_PREFILL_RC_GATE_RUNBOOK_20260604.md",
                "scripts/run_spark_stage6_p2o_rc_smoke.sh",
                "scripts/spark_stage6_p2o_packed_prefill_rc_smoke.py",
            ],
            "Banks only the rc-mini 0-4 shard; long-context remains a separate slow-mode scale gate.",
        ),
        _result_timeout_or_ready_gate(
            "p2o_rcmini_packed_prefill_rc",
            "P2-O rc-mini packed-prefill RC smoke",
            ["p2o_rc-mini_packed_prefill_rc_smoke_"],
            [
                "reports/stage6/P2O_PACKED_PREFILL_RC_GATE_RUNBOOK_20260604.md",
                "scripts/run_spark_stage6_p2o_rc_smoke.sh",
                "scripts/spark_stage6_p2o_packed_prefill_rc_smoke.py",
            ],
            "Rerun only after slow-mode prefill is accelerated or the rc-mini prompt set is split/chunked; the 2048 run was max_seq_len-invalid and the 8192 run timed out.",
        ),
        _result_or_ready_gate(
            "p3a_grouped_moe_contract_probe",
            "P3-A grouped active-MoE contract probe",
            ["p3a_layer"],
            [
                "reports/stage6/P3_GROUPED_MOE_ZERO_SHADOW_CONTRACT_20260604.md",
                "scripts/run_spark_stage6_p3a_contract_probe.sh",
                "scripts/spark_stage6_p3a_grouped_moe_contract_probe.py",
            ],
            "Banks only the P3-A active-MoE contract probe; speed/default/fused-kernel promotion remain closed.",
        ),
        _p3a_batched_down_candidate_gate(),
        _result_or_ready_gate(
            "p3b_selected_prefill",
            "P3-B selected-prefill composition gate",
            ["p3b_layers"],
            [
                "reports/stage6/P3B_SELECTED_PREFILL_GATE_RUNBOOK_20260604.md",
                "scripts/run_spark_stage6_p3b_selected_prefill_gate.sh",
                "scripts/spark_stage6_p3b_selected_prefill_gate.py",
                "scripts/summarize_stage6_p3b_selected_prefill_gate.py",
                "scripts/write_stage6_p3b_report.py",
            ],
            "Banks selected-layer composition only; P3-C server behavior and full rc-mini long-context remain separate gates.",
        ),
        _result_or_ready_gate(
            "p3c_resident_prompt",
            "P3-C resident-prompt gate",
            ["p3c_basic_resident_prompt_gate_", "p3c_rc-mini_resident_prompt_gate_"],
            [
                "reports/stage6/P3C_RESIDENT_PROMPT_GATE_RUNBOOK_20260604.md",
                "scripts/run_spark_stage6_p3c_resident_prompt_gate.sh",
                "scripts/spark_stage6_p3c_resident_prompt_gate.py",
                "scripts/summarize_stage6_p3c_resident_prompt_gate.py",
                "scripts/write_stage6_p3c_report.py",
            ],
            "Banks resident-prompt basic smoke only; P3-D server behavior and RC quality remain separate gates.",
        ),
        _ready_gate(
            "p3d_server_rc",
            "P3-D OpenAI server RC smoke",
            ["reports/stage6/P3D_SERVER_RC_PROMOTION_GATE_RUNBOOK_20260604.md", "scripts/run_spark_stage6_p3d_server_rc_gate.sh"],
            "Run after P3-C PASS; banks opt-in server smoke only.",
        ),
        _ready_gate(
            "p3e_rc_quality_battery",
            "P3-E RC quality-battery smoke",
            ["reports/stage6/P3E_RC_QUALITY_BATTERY_RUNBOOK_20260604.md", "scripts/run_spark_stage6_p3e_rc_quality_battery.sh"],
            "Run after P3-D PASS; full leaderboard/default promotion remains closed.",
        ),
        _ready_gate(
            "p4_native_abi_preflight",
            "P4 native ABI two-stage reference preflight",
            ["reports/stage6/P4_NATIVE_FUSED_MOE_ABI_CONTRACT_20260604.md", "scripts/run_spark_stage6_p4_native_abi_preflight.sh"],
            "Run on Spark; may bank native ABI preflight only, not fused kernel.",
        ),
        _ready_gate(
            "p4_runtime_bridge_preflight",
            "P4 real runtime bridge preflight",
            ["reports/stage6/P4_NATIVE_FUSED_MOE_ABI_CONTRACT_20260604.md", "scripts/run_spark_stage6_p4_runtime_bridge_preflight.sh"],
            "Run on Spark; must prove native layer selection and no fallback.",
        ),
        _result_or_ready_gate(
            "p4b_single_kernel_preflight",
            "P4B single-kernel fail-loud preflight",
            ["p4b_single_kernel_preflight_"],
            [
                "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md",
                "scripts/run_spark_stage6_p4b_single_kernel_preflight.sh",
                "scripts/spark_stage6_p4b_single_kernel_preflight.py",
                "scripts/summarize_stage6_p4b_single_kernel_preflight.py",
            ],
            "Run on Spark; may bank fail-loud single-kernel contract preflight only, not fused-kernel speed.",
        ),
        _result_or_ready_gate(
            "p4b_runtime_bridge_preflight",
            "P4B real runtime bridge fail-loud preflight",
            ["p4b_runtime_bridge_preflight_"],
            [
                "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md",
                "scripts/run_spark_stage6_p4b_runtime_bridge_preflight.sh",
                "scripts/spark_stage6_p4b_runtime_bridge_preflight.py",
                "scripts/summarize_stage6_p4b_runtime_bridge_preflight.py",
            ],
            "Run on Spark; proves resident-runner routing reaches P4B fail-loud symbol after active BF16 shadows are removed.",
        ),
        _result_or_ready_gate(
            "p4b_single_cta_numeric_preflight",
            "P4B single-CTA output-returning numeric preflight",
            ["p4b_single_cta_numeric_preflight_"],
            [
                "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md",
                "scripts/run_spark_stage6_p4b_single_cta_numeric_preflight.sh",
                "scripts/spark_stage6_p4b_single_cta_numeric_preflight.py",
                "scripts/summarize_stage6_p4b_single_cta_numeric_preflight.py",
            ],
            "Use as correctness reference only; next gate needs byte-count profiler plus speed/RC evidence.",
        ),
        _result_or_ready_gate(
            "p4b_single_cta_microbench",
            "P4B single-CTA reference microbench",
            ["p4b_single_cta_microbench_"],
            [
                "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md",
                "scripts/run_spark_stage6_p4b_single_cta_microbench.sh",
                "scripts/spark_stage6_p4b_single_cta_microbench.py",
                "scripts/summarize_stage6_p4b_single_cta_microbench.py",
            ],
            "Treat as an intentional speed anti-proof for single-CTA; next implementation must be multi-CTA/CUTLASS-style.",
        ),
        _result_or_ready_gate(
            "p4b_multi_cta_numeric_preflight",
            "P4B multi-CTA recompute numeric preflight",
            ["p4b_multi_cta_numeric_preflight_"],
            [
                "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md",
                "scripts/run_spark_stage6_p4b_single_cta_numeric_preflight.sh",
                "scripts/spark_stage6_p4b_single_cta_numeric_preflight.py",
                "scripts/summarize_stage6_p4b_single_cta_numeric_preflight.py",
            ],
            "If numeric passes, use microbench to accept or reject this recompute strategy.",
        ),
        _p4b_multi_cta_microbench_gate(),
        _p4b_contract_gate(),
        _result_or_ready_gate(
            "p4c_active_reuse_runtime_bridge",
            "P4C active-reuse runtime bridge preflight",
            ["p4c_runtime_bridge_preflight_"],
            [
                "reports/stage6/P4C_ACTIVE_REUSE_KERNEL_DECISION_20260604.md",
                "scripts/run_spark_stage6_p4c_runtime_bridge_preflight.sh",
                "scripts/spark_stage6_p4c_runtime_bridge_preflight.py",
                "scripts/summarize_stage6_p4c_runtime_bridge_preflight.py",
                "scripts/test_stage6_p4c_runtime_bridge_tools.py",
            ],
            "Run on Spark; banks P4C active-reuse route/numeric evidence only, not fused speed or default promotion.",
        ),
        _result_or_ready_gate(
            "p4c_active_reuse_microbench",
            "P4C active-reuse speed-baseline microbench",
            ["p4c_active_reuse_microbench_"],
            [
                "reports/stage6/P4C_ACTIVE_REUSE_KERNEL_DECISION_20260604.md",
                "scripts/run_spark_stage6_p4c_active_reuse_microbench.sh",
                "scripts/spark_stage6_p4c_active_reuse_microbench.py",
                "scripts/summarize_stage6_p4c_active_reuse_microbench.py",
                "scripts/test_stage6_p4c_active_reuse_microbench_tools.py",
            ],
            "Run on Spark; banks the current P4C symbol speed baseline before replacing it with a real active-reuse speed candidate.",
        ),
        _result_or_ready_gate(
            "p4c_component_profile",
            "P4C gate/up vs down component profile",
            ["p4c_component_profile_"],
            [
                "reports/stage6/P4C_ACTIVE_REUSE_KERNEL_DECISION_20260604.md",
                "scripts/run_spark_stage6_p4c_component_profile.sh",
                "scripts/spark_stage6_p4c_component_profile.py",
                "scripts/summarize_stage6_p4c_component_profile.py",
                "scripts/test_stage6_p4c_component_profile_tools.py",
            ],
            "Run on Spark; use the larger component to choose the first active-reuse speed candidate.",
        ),
        _result_or_ready_gate(
            "p4c_gateup_shape_sweep",
            "P4C gate/up launch-shape sweep",
            ["p4c_gateup_shape_sweep_"],
            [
                "reports/stage6/P4C_ACTIVE_REUSE_KERNEL_DECISION_20260604.md",
                "scripts/run_spark_stage6_p4c_gateup_shape_sweep.sh",
                "scripts/spark_stage6_p4c_gateup_shape_sweep.py",
                "scripts/summarize_stage6_p4c_gateup_shape_sweep.py",
                "scripts/test_stage6_p4c_gateup_shape_sweep_tools.py",
            ],
            "Run on Spark; bank only launch-shape diagnostic evidence before writing a real gate/up CUDA/CUTLASS candidate.",
        ),
        _result_or_ready_gate(
            "p4c_gateup_shape_candidate",
            "P4C gate/up launch-shape speed candidate",
            ["p4c_gateup_shape_candidate_"],
            [
                "reports/stage6/P4C_ACTIVE_REUSE_KERNEL_DECISION_20260604.md",
                "scripts/run_spark_stage6_p4c_gateup_shape_candidate_microbench.sh",
                "scripts/spark_stage6_p4c_gateup_shape_candidate_microbench.py",
                "scripts/summarize_stage6_p4c_gateup_shape_candidate_microbench.py",
                "scripts/test_stage6_p4c_gateup_shape_candidate_tools.py",
            ],
            "Run on Spark; compare current tile_inter=8 vs candidate tile_inter=2 on the full P4C active-reuse symbol.",
        ),
        _result_or_ready_gate(
            "p4c_tile2_server_smoke",
            "P4C tile2 OpenAI server smoke",
            ["p4c_tile2_server_smoke_"],
            [
                "scripts/run_spark_stage6_p4c_tile2_server_smoke.sh",
                "scripts/spark_stage6_p4c_tile2_server_smoke.py",
                "scripts/summarize_stage6_p4c_tile2_server_smoke.py",
                "scripts/test_stage6_p4c_tile2_server_smoke_tools.py",
                "server/openai_http.py",
            ],
            "Run on Spark; bank opt-in server evidence for tile_inter=2 P4C decode after prefill releases shadows.",
        ),
        _result_gate(
            "p4c_tile2_rcmini_agreement",
            "P4C tile2 RC-mini server agreement rejection",
            ["p4c_tile2_rcmini_agreement_"],
            "Run first-divergence diagnostics before widening P4C tile2 server prompts or considering promotion.",
            negative_if_fail=True,
        ),
        _p4c_shadow_cycle_first_divergence_gate(),
        _p4c_active_reuse_decision_gate(),
        _decode_gpu_idle_probe_gate(),
        _r6000_fp4_mma_census_gate(),
    ]
    return gates


def _derive_current_status(gates: list[Gate]) -> tuple[str, str]:
    by_gate = {gate.gate: gate for gate in gates}
    r6000 = by_gate.get("r6000_fp4_mma_census")
    decode_idle = by_gate.get("decode_gpu_idle_probe")
    if r6000 and r6000.status == "BANKED":
        note = (
            "Stage 6 lanes are split and active: Spark owns 35B serving/memory/MTP/compiled-loop ROI "
            "and RC regressions; R6000 owns FP4-MMA/CUTLASS/CuTe/vLLM Marlin-Machete census plus the "
            "next native grouped-MoE FP4-MMA POC. No kernel/default/speed promotion is implied."
        )
        if decode_idle and decode_idle.status == "DIAGNOSTIC_BANKED":
            note += f" Spark decode ROI probe remains {decode_idle.decision or decode_idle.status}."
        return "R6000_FP4_MMA_CENSUS_BANKED", note
    if r6000 and r6000.status == "READY_WAITING_R6000":
        return (
            "R6000_FP4_MMA_CENSUS_TOOLING_READY",
            "Stage 6 lanes are split; R6000 tooling exists but still needs a PASS artifact before FP4-MMA kernel work.",
        )
    return (
        "STAGE6_EVIDENCE_LEDGER_UPDATED",
        "Stage 6 gates are recorded; inspect individual gate rows before promoting any runtime/default.",
    )


def to_json(gates: list[Gate], spark_status: str, spark_note: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for gate in gates:
        counts[gate.status] = counts.get(gate.status, 0) + 1
    return {
        "schema": "lynn-stage6-evidence-ledger-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage6_status": spark_status,
        "stage6_note": spark_note,
        # Compatibility keys kept for older scripts that still read spark_*.
        "spark_status": spark_status,
        "spark_note": spark_note,
        "counts": counts,
        "promotion_boundaries": {
            "p3_default_promotion": False,
            "p4_banked_fused_kernel": False,
            "p4_default_promotion": False,
        },
        "gates": [gate.__dict__ for gate in gates],
    }


def to_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Stage 6 Evidence Ledger",
        "",
        "Date: 2026-06-04",
        "",
        "Verdict: **evidence ledger only; this document does not bank new GPU results.**",
        "",
        "This ledger separates banked positive evidence, intentional negative closures,",
        "and gates that are wired but still waiting for lane-specific PASS/FAIL artifacts.",
        "",
        "## Current Stage 6 Status",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Status | `{data['spark_status']}` |",
        f"| Note | {data['spark_note'] or '`not recorded`'} |",
        "",
        "## Promotion Boundaries",
        "",
        "| Boundary | Value |",
        "|---|---|",
        "| P3 default promotion | `false` |",
        "| P4 fused kernel banked | `false` |",
        "| P4 default promotion | `false` |",
        "",
        "## Gate Ledger",
        "",
        "| Gate | Status | Evidence | Artifact | Next step |",
        "|---|---|---|---|---|",
    ]
    for gate in data["gates"]:
        decision = f" `{gate['decision']}`" if gate.get("decision") else ""
        artifact = gate["artifact"] or "`none`"
        lines.append(
            f"| `{gate['gate']}` | **{gate['status']}**{decision} | "
            f"{gate['evidence']} | `{artifact}` | {gate['next_step']} |"
        )
    lines.extend([
        "",
        "## Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ])
    for status, count in sorted(data["counts"].items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend([
        "",
        "## Local Check",
        "",
        "```bash",
        "python3 scripts/write_stage6_evidence_ledger.py \\",
        "  --markdown-out reports/stage6/STAGE6_EVIDENCE_LEDGER_20260604.md \\",
        "  --json-out reports/stage6/stage6_evidence_ledger_20260604.json",
        "```",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Write Stage 6 evidence ledger.")
    ap.add_argument("--markdown-out", default="reports/stage6/STAGE6_EVIDENCE_LEDGER_20260604.md")
    ap.add_argument("--json-out", default="reports/stage6/stage6_evidence_ledger_20260604.json")
    ap.add_argument("--spark-status", default="UNVERIFIED_THIS_RUN")
    ap.add_argument("--spark-note", default="")
    args = ap.parse_args()

    gates = collect_gates()
    status = args.spark_status
    note = args.spark_note
    if status == "UNVERIFIED_THIS_RUN" and not note:
        status, note = _derive_current_status(gates)
    data = to_json(gates, status, note)
    md = to_markdown(data)

    md_path = ROOT / args.markdown_out
    json_path = ROOT / args.json_out
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[stage6-ledger] markdown={md_path}")
    print(f"[stage6-ledger] json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
