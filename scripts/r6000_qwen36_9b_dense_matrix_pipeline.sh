#!/usr/bin/env bash
#
# r6000_qwen36_9b_dense_matrix_pipeline.sh
# Orchestration skeleton for Qwen3.6-9B-Dense R6000 benchmark pipeline.
# DRY_RUN=1 by default — prints commands, does not execute.
#
# Covers three tracks:
#   1. BF16 baseline       — Lynn Engine full-precision
#   2. Q4_K_M llama.cpp    — GGUF cross-platform reference
#   3. W4A16 NVFP4 Lynn    — Native packed NVFP4 on Blackwell
#
# Env vars:
#   MODEL_DIR          — base model directory (default: /data/models/Qwen3.6-9B-Dense)
#   OUTPUT_DIR         — report output dir (default: ./r6000_9b_reports)
#   DRY_RUN            — 1 = print only (default), 0 = execute
#   LLAMACPP_BIN       — path to llama-bench / llama-perplexity (default: llama-bench)
#   LYNN_ENGINE_BIN    — path to lynn-engine-bench (default: lynn-engine-bench)
#   MMLU_DATASET       — MMLU eval dataset path (default: data/mmlu.jsonl)
#
# Usage:
#   DRY_RUN=1 bash scripts/r6000_qwen36_9b_dense_matrix_pipeline.sh
#   DRY_RUN=0 MODEL_DIR=/mnt/models ./scripts/r6000_qwen36_9b_dense_matrix_pipeline.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-/data/models/Qwen3.6-9B-Dense}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/r6000_9b_reports}"
DRY_RUN="${DRY_RUN:-1}"
LLAMACPP_BIN="${LLAMACPP_BIN:-llama-bench}"
LYNN_ENGINE_BIN="${LYNN_ENGINE_BIN:-lynn-engine-bench}"
MMLU_DATASET="${MMLU_DATASET:-data/mmlu.jsonl}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
log_warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
log_dry()   { printf "\033[1;36m[DRY]\033[0m   %s\n" "$*"; }
log_cmd()   { printf "\033[1;32m[CMD]\033[0m   %s\n" "$*"; }
log_err()   { printf "\033[1;31m[ERR]\033[0m   %s\n" "$*" >&2; }

run_or_dry() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        log_dry "$1"
    else
        log_cmd "$1"
        eval "$1"
    fi
}

detect_model_path() {
    local quant="$1"
    local path=""
    case "${quant}" in
        bf16)
            path="${MODEL_DIR}/bf16"
            [[ -d "${path}" ]] || path="${MODEL_DIR}/model.safetensors.d"
            ;;
        q4_km)
            path="${MODEL_DIR}/gguf/qwen3.6-9b-dense-q4_k_m.gguf"
            [[ -f "${path}" ]] || path="${MODEL_DIR}/qwen3.6-9b-dense-q4_k_m.gguf"
            ;;
        w4a16_nvfp4)
            path="${MODEL_DIR}/w4a16_nvfp4"
            [[ -d "${path}" ]] || path="${MODEL_DIR}/nvfp4"
            ;;
    esac
    echo "${path}"
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "═══════════════════════════════════════════════════════════════════════"
echo " Qwen3.6-9B-Dense R6000 Benchmark Pipeline"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  MODEL_DIR:    ${MODEL_DIR}"
echo "  OUTPUT_DIR:   ${OUTPUT_DIR}"
echo "  DRY_RUN:      ${DRY_RUN}"
echo "  LLAMACPP_BIN: ${LLAMACPP_BIN}"
echo "  LYNN_ENGINE:  ${LYNN_ENGINE_BIN}"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
log_info "Pre-flight checks..."

[[ "${DRY_RUN}" == "1" ]] && log_info "DRY_RUN=1 — printing commands only."

# Output dir
if [[ "${DRY_RUN}" != "1" ]]; then
    run_or_dry "mkdir -p '${OUTPUT_DIR}'"
else
    log_dry "mkdir -p '${OUTPUT_DIR}'"
fi

# Model path detection
BF16_PATH="$(detect_model_path bf16)"
Q4KM_PATH="$(detect_model_path q4_km)"
NVFP4_PATH="$(detect_model_path w4a16_nvfp4)"

echo ""
echo "  Detected paths:"
printf "    BF16:      %s %s\n" "${BF16_PATH}"   "$([ -e "${BF16_PATH}" ]   && echo '(found)' || echo '(MISSING)')"
printf "    Q4_K_M:    %s %s\n" "${Q4KM_PATH}"   "$([ -e "${Q4KM_PATH}" ]   && echo '(found)' || echo '(MISSING)')"
printf "    W4A16:     %s %s\n" "${NVFP4_PATH}"  "$([ -e "${NVFP4_PATH}" ]  && echo '(found)' || echo '(MISSING)')"
echo ""

# Warnings for missing paths
if [[ ! -e "${BF16_PATH}" ]]; then
    log_warn "BF16 model path not found: ${BF16_PATH}"
fi
if [[ ! -e "${Q4KM_PATH}" ]]; then
    log_warn "Q4_K_M model path not found: ${Q4KM_PATH}"
fi
if [[ ! -e "${NVFP4_PATH}" ]]; then
    log_warn "W4A16/NVFP4 model path not found: ${NVFP4_PATH}"
fi

# ---------------------------------------------------------------------------
# Stage 1: BF16 Baseline
# ---------------------------------------------------------------------------
echo "────────────────────────────────────────────────────────────────────────"
echo " Stage 1/3 — BF16 Baseline (Lynn Engine)"
echo "────────────────────────────────────────────────────────────────────────"

BF16_REPORT="${OUTPUT_DIR}/bf16_report.json"
log_info "Output: ${BF16_REPORT}"

if [[ -e "${BF16_PATH}" ]]; then
    run_or_dry "${LYNN_ENGINE_BIN} --model '${BF16_PATH}' --benchmark mmlu,gpqa,single_tps,concurrent_tps --output '${BF16_REPORT}'"
else
    log_warn "Skipping BF16 — model path missing."
fi

# ---------------------------------------------------------------------------
# Stage 2: Q4_K_M llama.cpp
# ---------------------------------------------------------------------------
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo " Stage 2/3 — Q4_K_M llama.cpp"
echo "────────────────────────────────────────────────────────────────────────"

Q4KM_REPORT="${OUTPUT_DIR}/q4km_llamacpp_report.json"
Q4KM_PPL="${OUTPUT_DIR}/q4km_mmlu_ppl.txt"
log_info "Output: ${Q4KM_REPORT}"

if [[ -e "${Q4KM_PATH}" ]]; then
    run_or_dry "${LLAMACPP_BIN} -m '${Q4KM_PATH}' --output-format json > '${Q4KM_REPORT}'"
    run_or_dry "llama-perplexity -m '${Q4KM_PATH}' -f '${MMLU_DATASET}' > '${Q4KM_PPL}'"
else
    log_warn "Skipping Q4_K_M — model path missing."
fi

# ---------------------------------------------------------------------------
# Stage 3: W4A16 NVFP4 Lynn-native
# ---------------------------------------------------------------------------
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo " Stage 3/3 — W4A16 NVFP4 Lynn-native"
echo "────────────────────────────────────────────────────────────────────────"

NVFP4_REPORT="${OUTPUT_DIR}/w4a16_nvfp4_report.json"
NVFP4_TRACE="${OUTPUT_DIR}/w4a16_nvfp4_trace.json"
log_info "Output: ${NVFP4_REPORT}"

if [[ -e "${NVFP4_PATH}" ]]; then
    run_or_dry "${LYNN_ENGINE_BIN} --model '${NVFP4_PATH}' --benchmark mmlu,gpqa,single_tps --output '${NVFP4_REPORT}'"
    run_or_dry "lynn-engine-profile --model '${NVFP4_PATH}' --kernel packed_nvfp4 --trace '${NVFP4_TRACE}'"
else
    log_warn "Skipping W4A16/NVFP4 — model path missing."
fi

# ---------------------------------------------------------------------------
# Post-process: summarize
# ---------------------------------------------------------------------------
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo " Post-process — Summarize reports"
echo "────────────────────────────────────────────────────────────────────────"

SUMMARIZER="${SCRIPT_DIR}/summarize_qwen36_9b_r6000_reports.py"
SCHEMA="${REPO_ROOT}/reports/qwen36_9b/qwen36_9b_dense_matrix_schema_v1.json"
SUMMARY_MD="${REPO_ROOT}/docs/QWEN36_9B_R6000_PIPELINE_20260518.md"

if [[ -f "${SUMMARIZER}" ]]; then
    run_or_dry "python3 '${SUMMARIZER}' --schema '${SCHEMA}' --reports '${OUTPUT_DIR}' --out '${SUMMARY_MD}'"
else
    log_warn "Summarizer not found: ${SUMMARIZER}"
fi

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " Pipeline complete"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Reports dir:  ${OUTPUT_DIR}"
echo "  DRY_RUN:      ${DRY_RUN}"
echo ""
echo "  To execute for real, run with DRY_RUN=0 and verify model paths exist."
echo ""
