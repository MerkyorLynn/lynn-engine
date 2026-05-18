#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3.5-9B-BF16}"
REPORT="${REPORT:-$ROOT/reports/qwen35_9b/r6000_qwen35_9b_dense_runtime_smoke.json}"
RUN_GENERATION="${RUN_GENERATION:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
PROMPT="${PROMPT:-Say OK.}"
MAX_NEW="${MAX_NEW:-1}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/r6000_qwen35_9b_dense_runtime_smoke.sh [--model-dir PATH] [--report PATH]

Environment:
  PYTHON=/path/to/python
  RUN_GENERATION=0|1     default 1
  MAX_SEQ_LEN=4096       default 4096
  PROMPT='Say OK.'       default Say OK.
  MAX_NEW=1              default 1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --report)
      REPORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "$REPORT")"
export ROOT MODEL_DIR REPORT RUN_GENERATION MAX_SEQ_LEN PROMPT MAX_NEW

cd "$ROOT"
echo "[qwen35-9b-smoke] root=$ROOT"
echo "[qwen35-9b-smoke] model=$MODEL_DIR"
echo "[qwen35-9b-smoke] report=$REPORT"

"$PYTHON" -m py_compile \
  engine/inference_state.py \
  engine/full_forward.py \
  engine/loader.py \
  engine/resident_runner.py

"$PYTHON" - <<'PY'
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch

root = Path(os.environ["ROOT"])
model_dir = Path(os.environ["MODEL_DIR"])
report_path = Path(os.environ["REPORT"])
run_generation = os.environ.get("RUN_GENERATION", "1") == "1"
max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "4096"))
prompt = os.environ.get("PROMPT", "Say OK.")
max_new = int(os.environ.get("MAX_NEW", "1"))

report: dict = {
    "benchmark": "r6000_qwen35_9b_dense_runtime_smoke",
    "model_dir": str(model_dir),
    "max_seq_len": max_seq_len,
    "prompt": prompt,
    "max_new": max_new,
    "run_generation": run_generation,
    "checks": {},
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

def finish(status: str, **extra) -> None:
    report["status"] = status
    report.update(extra)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))

try:
    if not (model_dir / "config.json").exists():
        finish("PENDING", reason="config.json not found")
        raise SystemExit(0)

    import sys
    sys.path.insert(0, str(root))

    config = json.loads((model_dir / "config.json").read_text())
    tc = config.get("text_config", config)

    from engine.inference_state import LynnInferenceState, infer_layer_types
    from engine.loader import load_qwen36_layer
    from engine.resident_runner import _runtime_config

    layer_types = infer_layer_types(tc)
    linear_count = sum(1 for x in layer_types if x == "linear_attention")
    full_count = sum(1 for x in layer_types if x == "full_attention")
    report["model_config"] = {
        "model_type": tc.get("model_type") or config.get("model_type"),
        "hidden_size": tc.get("hidden_size"),
        "num_hidden_layers": tc.get("num_hidden_layers"),
        "num_attention_heads": tc.get("num_attention_heads"),
        "num_key_value_heads": tc.get("num_key_value_heads"),
        "head_dim": tc.get("head_dim"),
        "num_experts": tc.get("num_experts"),
        "layer_type_counts": {
            "linear_attention": linear_count,
            "full_attention": full_count,
        },
    }
    if int(tc.get("num_hidden_layers", 0)) != 32 or linear_count != 24 or full_count != 8:
        raise RuntimeError(f"unexpected Qwen3.5-9B layer layout: linear={linear_count} full={full_count}")

    cfg, n_layers = _runtime_config(str(model_dir))
    report["checks"]["runtime_config"] = {
        "status": "PASS",
        "n_layers": n_layers,
        "is_moe": cfg.get("is_moe"),
        "num_experts": cfg.get("num_experts"),
        "linear_conv_dim": cfg.get("linear_conv_dim"),
    }

    state = LynnInferenceState.from_config(tc, batch=1, max_seq_len=64, device="cpu", dtype=torch.bfloat16)
    report["checks"]["state_from_config"] = {
        "status": "PASS",
        "kv_layers": len(state.kv_cache),
        "recurrent_layers": len(state.recurrent_state),
        "conv_layers": len(state.conv_state),
        "kv_shape_layer3": list(state.kv_cache[3][0].shape) if 3 in state.kv_cache else None,
        "recurrent_shape_layer0": list(state.recurrent_state[0].shape) if 0 in state.recurrent_state else None,
        "conv_shape_layer0": list(state.conv_state[0].shape) if 0 in state.conv_state else None,
    }

    weights, inferred = load_qwen36_layer(
        str(model_dir),
        0,
        num_experts=0,
        device="cpu",
        dequant_dtype=torch.bfloat16,
    )
    dense_keys = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    missing = [k for k in dense_keys if k not in weights]
    if missing:
        raise RuntimeError(f"dense FFN keys missing from layer 0 load: {missing}")
    report["checks"]["dense_layer0_load"] = {
        "status": "PASS",
        "inferred": inferred,
        "gate_shape": list(weights["mlp.gate_proj.weight"].shape),
        "up_shape": list(weights["mlp.up_proj.weight"].shape),
        "down_shape": list(weights["mlp.down_proj.weight"].shape),
    }

    if not run_generation:
        finish("STATIC_PASS", reason="RUN_GENERATION=0")
        raise SystemExit(0)

    from engine.resident_runner import LynnIncrementalRunner

    t0 = time.time()
    runner = LynnIncrementalRunner(
        str(model_dir),
        device="cuda",
        dtype=torch.bfloat16,
        max_seq_len=max_seq_len,
        verbose=True,
    )
    result = runner.generate(prompt, max_new=max_new, use_chat_template=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    finish(
        "GENERATION_PASS",
        generated_text=result.get("text"),
        new_token_ids=result.get("new_ids"),
        completion_text=result.get("completion_text"),
        timings=result.get("timings"),
        load_seconds=getattr(runner, "load_seconds", None),
        elapsed_seconds=elapsed,
    )
except SystemExit:
    raise
except Exception as exc:
    tb = traceback.format_exc().splitlines()
    finish(
        "RUNTIME_BLOCKED_NEXT",
        error_type=type(exc).__name__,
        error=str(exc),
        traceback_tail=tb[-24:],
    )
PY
