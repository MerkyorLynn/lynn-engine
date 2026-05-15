#!/bin/bash
# Auto-pipeline: wait NVFP4 transfer → sanity → server → master eval.
# Designed to run nohup in background; status written to /home/merkyor/reports/27b_pipeline_status.json
# Each phase appends to log. Any failure halts and surfaces in status.
set -u

MODEL_TMP=/home/merkyor/models/lynn-27b-variable-recovery-step5000-nvfp4-final.tmp
MODEL_FINAL=/home/merkyor/models/lynn-27b-variable-recovery-step5000-nvfp4-final
EXPECTED_TENSORS=1026
PORT=18099
TS=$(date +%H%M)
LOG=/home/merkyor/reports/27b_pipeline_${TS}.log
STATUS=/home/merkyor/reports/27b_pipeline_status.json
OUT=/home/merkyor/reports/27b_nvfp4_eval_${TS}

log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a "$LOG"; }
status() {
  phase=$1; state=$2; detail=${3:-}
  python3 -c "
import json,os
p='$STATUS'
d={}
if os.path.exists(p):
    try: d=json.load(open(p))
    except: d={}
d['$phase']={'state':'$state','detail':'''$detail''','t':'$(date +%H:%M:%S)'}
json.dump(d, open(p,'w'), indent=2, ensure_ascii=False)
" 2>/dev/null
}

mkdir -p "$OUT" "$(dirname "$LOG")"
log "=== pipeline start ==="
status pipeline running ""

# -------- Phase 1: wait transfer --------
log "[phase 1] wait NVFP4 transfer (expect $EXPECTED_TENSORS tensors)"
status phase1_wait_transfer running ""
last_count=0; stalled=0
while true; do
  count=$(ls "$MODEL_TMP/tensors/" 2>/dev/null | wc -l)
  sz=$(du -sh "$MODEL_TMP" 2>/dev/null | awk '{print $1}')
  log "  tensors=$count/$EXPECTED_TENSORS size=$sz"
  if [ "$count" -ge "$EXPECTED_TENSORS" ]; then
    log "[phase 1] DONE — all $EXPECTED_TENSORS tensors present"
    status phase1_wait_transfer done "tensors=$count size=$sz"
    break
  fi
  if [ "$count" -eq "$last_count" ]; then stalled=$((stalled+1)); else stalled=0; fi
  if [ "$stalled" -ge 12 ]; then  # 12 * 60s = 12 min no progress
    log "[phase 1] STALLED 12min, surface and exit"
    status phase1_wait_transfer stalled "no progress 12min, count=$count"
    exit 1
  fi
  last_count=$count
  sleep 60
done

# -------- Phase 2: rename to final --------
log "[phase 2] rename .tmp → final"
status phase2_rename running ""
if [ -d "$MODEL_FINAL" ]; then
  log "  final dir already exists, removing first"
  rm -rf "$MODEL_FINAL"
fi
mv "$MODEL_TMP" "$MODEL_FINAL"
log "[phase 2] DONE"
status phase2_rename done ""

# -------- Phase 3: sanity in docker --------
log "[phase 3] sanity check in sglang docker"
status phase3_sanity running ""
docker run --rm --gpus all \
  -v /home/merkyor/models:/models \
  -v /home/merkyor/lynn-engine:/lynn-engine \
  -v /tmp/nvfp4_landing_sanity.py:/tmp/sanity.py \
  -e PYTHONPATH=/lynn-engine \
  lmsysorg/sglang:dev-cu13 \
  python3 /tmp/sanity.py 2>&1 | tee -a "$LOG"

if grep -q "ALL GATES PASS" "$LOG"; then
  log "[phase 3] DONE"
  status phase3_sanity done ""
else
  log "[phase 3] FAIL — sanity did not report ALL GATES PASS"
  status phase3_sanity fail "see $LOG"
  exit 2
fi

# -------- Phase 4: start server --------
log "[phase 4] start Lynn engine server"
status phase4_server running ""
docker rm -f lynn-27b-nvfp4-server 2>/dev/null || true

docker run -d --name lynn-27b-nvfp4-server \
  --gpus all --restart=no --ipc=host \
  -p ${PORT}:${PORT} \
  -v /home/merkyor/models:/models \
  -v /home/merkyor/lynn-engine:/lynn-engine \
  -w /lynn-engine \
  -e PYTHONPATH=/lynn-engine \
  -e LYNN_PREFILL_WARMUP=1 \
  -e LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare \
  -e LYNN_MOE_IMPL=triton \
  -e LYNN_QK_NORM_ROPE_BACKEND=triton_pair \
  -e LYNN_RMSNORM_GATED_BACKEND=triton \
  -e LYNN_LINEAR_ATTN_INPROJ_FUSED=1 \
  -e LYNN_LINEAR_BLOCK_GRAPH=1 \
  -e LYNN_LINEAR_BLOCK_GRAPH_REUSE=1 \
  -e LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1 \
  -e LYNN_LINEAR_STATE_UPDATE=inplace \
  lmsysorg/sglang:dev-cu13 \
  python3 -m server.openai_http \
    --model /models/lynn-27b-variable-recovery-step5000-nvfp4-final \
    --host 0.0.0.0 --port ${PORT} \
    --served-name Lynn-V4-Distill-Qwen-27B-A3B-NVFP4

log "  container started, waiting for /health"
for i in $(seq 1 180); do
  if curl -sf -m 3 http://127.0.0.1:${PORT}/health 2>/dev/null | grep -qiE "ok|ready|healthy"; then
    log "[phase 4] DONE — server healthy after $((i*5))s"
    status phase4_server done "ready_s=$((i*5))"
    break
  fi
  sleep 5
done

if ! curl -sf -m 3 http://127.0.0.1:${PORT}/health >/dev/null 2>&1; then
  log "[phase 4] FAIL — server did not become healthy in 900s"
  docker logs --tail 80 lynn-27b-nvfp4-server 2>&1 | tee -a "$LOG"
  status phase4_server fail "see docker logs lynn-27b-nvfp4-server"
  exit 3
fi

# -------- Phase 5: master eval --------
log "[phase 5] run master_27b_eval.py"
status phase5_eval running ""

# install requests if missing (host python3)
python3 -c "import requests" 2>/dev/null || pip3 install --user requests 2>&1 | tail -3

python3 /home/merkyor/scripts/master_27b_eval.py \
  --base http://127.0.0.1:${PORT} \
  --out "$OUT" 2>&1 | tee -a "$LOG"

if [ -f "$OUT/summary.json" ]; then
  log "[phase 5] DONE — see $OUT/summary.json"
  status phase5_eval done "$OUT/summary.json"
else
  log "[phase 5] FAIL — no summary.json"
  status phase5_eval fail "see $LOG"
  exit 4
fi

log "=== pipeline DONE ==="
status pipeline done "$OUT/summary.json"
