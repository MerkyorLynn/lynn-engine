#!/bin/bash
# Start Lynn engine OpenAI HTTP server for 27B NVFP4 on Spark sm_121.
# Run inside lmsysorg/sglang:dev-cu13 docker (verified sm_121 triton JIT path).
#
# Env vars below mirror the R6000 P10 production config that hits 88-89 TPS
# stable / 103.49 TPS strict full graph / 107.15 TPS replay-only. On Spark
# sm_121 the same config is what we use to validate sm_121 portability.
# DO NOT drop any of these env vars without testing — each is load-bearing.
set -e
MODEL=/models/lynn-27b-variable-recovery-step5000-nvfp4-final
PORT=18099
LOG=/home/merkyor/reports/lynn_27b_server_$(date +%H%M).log

docker rm -f lynn-27b-nvfp4-server 2>/dev/null || true

docker run -d --name lynn-27b-nvfp4-server \
  --gpus all --restart=no --ipc=host \
  -p ${PORT}:${PORT} \
  -v /home/merkyor/models:/models \
  -v /home/merkyor/lynn-engine:/lynn-engine \
  -w /lynn-engine \
  -e PYTHONPATH=/lynn-engine \
  -e LYNN_PREFILL_WARMUP=1 \
  -e LYNN_MOE_IMPL=packed_nvfp4 \
  -e LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare \
  -e LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1 \
  -e LYNN_LINEAR_STATE_UPDATE=inplace \
  -e LYNN_QK_NORM_ROPE_BACKEND=triton_pair \
  -e LYNN_RMSNORM_GATED_BACKEND=triton \
  -e LYNN_LINEAR_ATTN_INPROJ_FUSED=1 \
  -e LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1 \
  -e LYNN_NATIVE_FP4_LM_HEAD=1 \
  -e LYNN_LINEAR_BLOCK_GRAPH=${LYNN_LINEAR_BLOCK_GRAPH:-1} \
  -e LYNN_LINEAR_BLOCK_GRAPH_REUSE=${LYNN_LINEAR_BLOCK_GRAPH_REUSE:-1} \
  -e LYNN_LINEAR_BLOCK_GRAPH_PREWARM=${LYNN_LINEAR_BLOCK_GRAPH_PREWARM:-1} \
  -e LYNN_PACKED_DECODE=1 \
  -e LYNN_PACKED_DECODE_BACKEND=native_fast_2d \
  -e LYNN_PACKED_DECODE_FULL_ATTN=1 \
  -e LYNN_PACKED_DECODE_LINEAR_ATTN=1 \
  -e LYNN_PACKED_DECODE_PREPARE_NATIVE=1 \
  -e LYNN_PACKED_SHARED_EXPERT=1 \
  ${LYNN_SP_TRITON_AUTOTUNE:+-e LYNN_SP_TRITON_AUTOTUNE=$LYNN_SP_TRITON_AUTOTUNE} \
  ${LYNN_SP_NGRAM_SPEC:+-e LYNN_SP_NGRAM_SPEC=$LYNN_SP_NGRAM_SPEC} \
  ${LYNN_NATIVE_ACTIVE_MOE_BACKEND:+-e LYNN_NATIVE_ACTIVE_MOE_BACKEND=$LYNN_NATIVE_ACTIVE_MOE_BACKEND} \
  ${LYNN_NATIVE_CUDA_ARCH:+-e LYNN_NATIVE_CUDA_ARCH=$LYNN_NATIVE_CUDA_ARCH} \
  ${LYNN_SPARK_FP8_BUILD_DIR:+-e LYNN_SPARK_FP8_BUILD_DIR=$LYNN_SPARK_FP8_BUILD_DIR} \
  lmsysorg/sglang:dev-cu13 \
  python3 -m server.openai_http \
    --model ${MODEL} \
    --host 0.0.0.0 --port ${PORT} \
    --served-name Lynn-V4-Distill-Qwen-27B-A3B-NVFP4

echo "[start] container launched. tailing logs (Ctrl+C to detach, container keeps running)…"
docker logs -f lynn-27b-nvfp4-server 2>&1 | tee "${LOG}" &
LOG_PID=$!

# wait for READY
echo "[wait] waiting up to 600s for /health to return OK"
for i in $(seq 1 120); do
  if curl -sf -m 3 http://127.0.0.1:${PORT}/health 2>/dev/null | grep -q -iE "ok|ready|healthy"; then
    echo "[ready] server up after ${i}*5 = $((i*5))s"
    kill $LOG_PID 2>/dev/null
    exit 0
  fi
  sleep 5
done

echo "[FAIL] server did not become healthy in 600s. check ${LOG} + docker logs lynn-27b-nvfp4-server"
exit 1
