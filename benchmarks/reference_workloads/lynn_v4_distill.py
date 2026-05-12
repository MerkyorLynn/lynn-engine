"""
Lynn-V4-Distill-Qwen-35B-A3B reference workload config (Phase 4 placeholder).

This file declares the canonical reference workload for Lynn engine P1-P3
verification: 4 published HF checkpoints of the same model, of which BF16
merged is ground truth, plus SGLang dev-cu13 nightly running same model as
the production oracle.

Status: PLACEHOLDER as of 2026-05-12.
Will populate concrete loaders / paths after 2026-05-13 V Pro Distill ship.

See docs/PHASE4_REFERENCE_WORKLOAD.md for the full Phase 4 plan.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class CheckpointSpec:
    """One published quantization of the reference workload."""
    name: str
    hf_repo: str
    modelscope_repo: str
    quant_format: Literal["bf16", "compressed_tensors_nvfp4", "modelopt_fp4", "compressed_tensors_fp8"]
    role: Literal["ground_truth", "dequant_target", "candidate", "fallback"]
    expected_serving_stack: str = ""
    expected_size_gb: float = 0.0


@dataclass
class ReferenceWorkload:
    """The reference workload spec — 4 checkpoints + production oracle."""
    name: str = "Lynn-V4-Distill-Qwen-35B-A3B"
    base_model: str = "Qwen/Qwen3.6-35B-A3B"
    base_after_stages: str = "S1-S5v2-S4"   # Lynn pipeline internal stage SFT
    lora_method: str = "LoRA r=64, lora_target=all, 1 epoch SFT"
    train_data: str = "50K DS V4 Pro/Flash + 5400 HAS = ~55K samples"
    hardware: str = "2x A100-80GB"
    ckpts: list = field(default_factory=lambda: [
        CheckpointSpec(
            name="bf16_merged",
            hf_repo="nerkyor/Lynn-V4-Distill-Qwen-35B-A3B-bf16-merged",  # if published; else use R6000 merged dir
            modelscope_repo="Merkyor/Lynn-V4-Distill-Qwen-35B-A3B-bf16-merged",
            quant_format="bf16",
            role="ground_truth",
            expected_size_gb=67.0,
        ),
        CheckpointSpec(
            name="nvfp4_v8rtn",
            hf_repo="nerkyor/Lynn-V4-Distill-Qwen-35B-A3B-NVFP4-v8-RTN",
            modelscope_repo="Merkyor/Lynn-V4-Distill-Qwen-35B-A3B-NVFP4-v8-RTN",
            quant_format="compressed_tensors_nvfp4",
            role="dequant_target",
            expected_serving_stack="SGLang dev-cu13 nightly",
            expected_size_gb=16.0,
        ),
        CheckpointSpec(
            name="nvfp4_modelopt",
            hf_repo="nerkyor/Lynn-V4-Distill-Qwen-35B-A3B-NVFP4-modelopt",
            modelscope_repo="Merkyor/Lynn-V4-Distill-Qwen-35B-A3B-NVFP4-modelopt",
            quant_format="modelopt_fp4",
            role="candidate",
            expected_serving_stack="not-ready (4-engine blocked 2026-05-12)",
            expected_size_gb=24.0,
        ),
        CheckpointSpec(
            name="fp8",
            hf_repo="nerkyor/Lynn-V4-Distill-Qwen-35B-A3B-FP8",
            modelscope_repo="Merkyor/Lynn-V4-Distill-Qwen-35B-A3B-FP8",
            quant_format="compressed_tensors_fp8",
            role="fallback",
            expected_serving_stack="vLLM / SGLang FP8 path",
            expected_size_gb=35.0,
        ),
    ])
    # Production oracle for parity testing
    production_oracle_url: str = "http://127.0.0.1:18000/v1/chat/completions"
    production_oracle_stack: str = "SGLang dev-cu13 nightly (Spark sm_121)"
    production_oracle_model_name: str = "Qwen3.6-35B-A3B-NVFP4-v8-RTN"


WORKLOAD = ReferenceWorkload()


# ============================================================================
# Phase 4 progress tracking
# ============================================================================
PHASE_STATUS = {
    "P1_canonical_spec": "PENDING",       # design in docs/PHASE4_REFERENCE_WORKLOAD.md
    "P1_loader_normalize": "PENDING",     # extend L4 guard from raise to parse+normalize
    "P2_dequant_to_bf16": "PENDING",
    "P2_1_single_tensor_unpack": "PENDING",
    "P2_2_one_linear_parity": "PENDING",
    "P2_3_one_transformer_layer_parity": "PENDING",
    "P2_4_five_token_decode": "PENDING",
    "P3_n20_parity_bf16": "PENDING",
    "P3_n20_parity_dequant_v8rtn": "PENDING",
    "P3_daily_regression_runner": "PENDING",
    "P4_native_nvfp4_gemm": "DEFERRED_UNTIL_P3_DONE",
}


def cli_main():
    """Print the workload spec + phase status — humans can `python3 -m benchmarks.reference_workloads.lynn_v4_distill`."""
    print(f"Reference workload: {WORKLOAD.name}")
    print(f"  base: {WORKLOAD.base_model} (post {WORKLOAD.base_after_stages})")
    print(f"  data: {WORKLOAD.train_data}")
    print(f"  hardware: {WORKLOAD.hardware}")
    print()
    print("Checkpoints:")
    for c in WORKLOAD.ckpts:
        print(f"  [{c.role:<14}] {c.name:<18} {c.quant_format:<28} ~{c.expected_size_gb:>5.1f} GB  HF:{c.hf_repo}")
    print()
    print(f"Production oracle: {WORKLOAD.production_oracle_stack}")
    print(f"  endpoint: {WORKLOAD.production_oracle_url}")
    print(f"  model_name: {WORKLOAD.production_oracle_model_name}")
    print()
    print("Phase 4 status:")
    for k, v in PHASE_STATUS.items():
        print(f"  {k:<38} {v}")


if __name__ == "__main__":
    cli_main()
