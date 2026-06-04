#!/usr/bin/env python3
"""GPU-free static checks for the R5-C4 full active-MoE speed A/B contract."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "reports" / "stage6" / "R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB_CONTRACT_20260604.md"


def main() -> int:
    failures: list[str] = []
    if not CONTRACT.exists():
        failures.append(f"missing {CONTRACT.relative_to(ROOT)}")
    else:
        text = CONTRACT.read_text(encoding="utf-8")
        for needle in [
            "PASS_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB",
            "input_r5c3c_decision=PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE",
            "same_scope_ab=true",
            "real_model_weights=true",
            "real_router_outputs=true",
            "candidate_no_active_bf16_shadow=true",
            "candidate_no_reload=true",
            "candidate_no_bf16_weight_materialization=true",
            "candidate_full_active_moe_boundary_timed=true",
            "timing_includes_gateup_swiglu_down_weighted_scatter=true",
            "numeric_vs_w4a16_or_p3_reference=true",
            "candidate_median_speedup_vs_best_reference_ge_1p05=true",
            "banked_full_active_moe_prefill_speed=true",
            "banked_grouped_moe_fp4_mma_poc=true",
            "banked_kernel_speed=true",
            "banked_decode_tps=false",
            "banked_server_rc=false",
            "banked_default_promotion=false",
            "banked_full_transformer_prefill=false",
            "route/order -> grouped gate/up FP4-MMA -> SwiGLU -> down projection -> top-k weighted sum",
            "hidden[T,H] + expert_ids[T,top_k] + routing_weights[T,top_k]",
            "H=2048",
            "I=512",
            "top_k=8",
            "candidate_no_bf16_shadow_in_timed_region",
            "candidate_covers_gateup_swiglu_down_weighted",
            "timing_includes_gateup_swiglu_down_weighted_scatter",
            "speedup_vs_w4a16",
            "speedup_vs_packed_p3",
            "median_speedup_vs_best_reference",
            "DIAGNOSTIC_BANKED_SPEED_CLOSED",
            "R5-C4 does not bank Spark decode TPS",
            "does not bank default promotion",
            "does not bank full transformer prefill",
        ]:
            if needle not in text:
                failures.append(f"contract missing {needle!r}")
    if failures:
        print("Stage 6 R5-C4 full active-MoE speed contract static check FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C4 full active-MoE speed contract static check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
