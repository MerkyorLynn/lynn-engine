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
        _ready_gate(
            "p2o_packed_prefill_rc",
            "P2-O packed-prefill RC smoke",
            ["reports/stage6/P2O_PACKED_PREFILL_RC_GATE_RUNBOOK_20260604.md", "scripts/run_spark_stage6_p2o_rc_smoke.sh"],
            "Run on Spark when SSH is reachable.",
        ),
        _ready_gate(
            "p3b_selected_prefill",
            "P3-B selected-prefill composition gate",
            ["reports/stage6/P3B_SELECTED_PREFILL_GATE_RUNBOOK_20260604.md", "scripts/run_spark_stage6_p3b_selected_prefill_gate.sh"],
            "Run after P2-O basic/rc-mini plus P3-A predecessor evidence.",
        ),
        _ready_gate(
            "p3c_resident_prompt",
            "P3-C resident-prompt gate",
            ["reports/stage6/P3C_RESIDENT_PROMPT_GATE_RUNBOOK_20260604.md", "scripts/run_spark_stage6_p3c_resident_prompt_gate.sh"],
            "Run after P3-B PASS.",
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
        _ready_gate(
            "p4b_single_kernel_preflight",
            "P4B single-kernel fail-loud preflight",
            [
                "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md",
                "scripts/run_spark_stage6_p4b_single_kernel_preflight.sh",
                "scripts/spark_stage6_p4b_single_kernel_preflight.py",
                "scripts/summarize_stage6_p4b_single_kernel_preflight.py",
            ],
            "Run on Spark; may bank fail-loud single-kernel contract preflight only, not fused-kernel speed.",
        ),
        _contract_gate(
            "p4b_single_kernel_contract",
            "P4B true fused single-kernel ABI",
            ["reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md", "scripts/test_stage6_p4b_single_kernel_static.py"],
            "Replace fail-loud contract with a real fused CUDA/CUTLASS kernel, then run byte-count/numeric/speed/RC gates.",
        ),
    ]
    return gates


def to_json(gates: list[Gate], spark_status: str, spark_note: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for gate in gates:
        counts[gate.status] = counts.get(gate.status, 0) + 1
    return {
        "schema": "lynn-stage6-evidence-ledger-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
        "and gates that are wired but still waiting for Spark PASS/FAIL artifacts.",
        "",
        "## Current Spark Status",
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
    data = to_json(gates, args.spark_status, args.spark_note)
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
