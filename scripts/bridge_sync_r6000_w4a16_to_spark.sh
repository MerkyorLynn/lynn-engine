#!/usr/bin/env bash
# Bridge-copy the official Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4 artifact
# from an R6000 box to Spark once packing has completed.
#
# This script is intended to run from an operator machine that can SSH to both
# remotes. It avoids requiring Spark to know the R6000 SSH alias.

set -euo pipefail

R6000_HOST="${R6000_HOST:-r6000}"
SPARK_HOST="${SPARK_HOST:-dgx-via-ssh}"
R6000_MODEL="${R6000_MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
SPARK_MODEL="${SPARK_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000}"
POLL_SECONDS="${POLL_SECONDS:-180}"
LOG_PREFIX="${LOG_PREFIX:-[sync-r6000-w4a16]}"

log() {
    printf '%s %s %s\n' "$LOG_PREFIX" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

remote_ready() {
    ssh -n "$R6000_HOST" "MODEL_DIR='$R6000_MODEL' python3 -c 'import json, os, pathlib, sys; d=pathlib.Path(os.environ[\"MODEL_DIR\"]); index_path=d/\"model.safetensors.index.json\"; manifest_path=d/\"lynn_quant_manifest.json\"; index=json.loads(index_path.read_text()); manifest=json.loads(manifest_path.read_text()); files=set(index.get(\"weight_map\", {}).values()); missing=[name for name in files if not (d/name).exists() or (d/name).stat().st_size <= 0]; bad=(not files or missing or int(manifest.get(\"quantized_count\", 0)) <= 0); bad and sys.exit(3); print(json.dumps({\"dir\": str(d), \"shards\": len(files), \"quantized_count\": manifest.get(\"quantized_count\"), \"kept_count\": manifest.get(\"kept_count\")}, ensure_ascii=False))'"
}

log "watching R6000 artifact: $R6000_HOST:$R6000_MODEL"
while true; do
    if READY_JSON="$(remote_ready 2>/dev/null)"; then
        if ssh -n "$R6000_HOST" "pgrep -af '[a]100_pack_lynn_native_nvfp4.py' | grep -F '$R6000_MODEL' >/dev/null"; then
            log "artifact index exists but packer is still active; waiting"
        else
            log "artifact ready: $READY_JSON"
            break
        fi
    else
        log "artifact not ready yet"
    fi
    sleep "$POLL_SECONDS"
done

TMP_DST="${SPARK_MODEL}.tmp"
log "preparing Spark destination: $SPARK_HOST:$SPARK_MODEL"
ssh -n "$SPARK_HOST" "rm -rf '$TMP_DST' && mkdir -p '$TMP_DST'"

log "streaming artifact through local bridge; this can take a while"
ssh -n "$R6000_HOST" "tar -C '$R6000_MODEL' -cf - ." | ssh "$SPARK_HOST" "tar -C '$TMP_DST' -xf -"

log "validating Spark copy"
ssh -n "$SPARK_HOST" "MODEL_DIR='$TMP_DST' python3 -c 'import json, os, pathlib, sys; d=pathlib.Path(os.environ[\"MODEL_DIR\"]); index=json.loads((d/\"model.safetensors.index.json\").read_text()); manifest=json.loads((d/\"lynn_quant_manifest.json\").read_text()); files=set(index.get(\"weight_map\", {}).values()); missing=[name for name in files if not (d/name).exists() or (d/name).stat().st_size <= 0]; missing and sys.exit(4); print(json.dumps({\"dir\": str(d), \"shards\": len(files), \"quantized_count\": manifest.get(\"quantized_count\"), \"kept_count\": manifest.get(\"kept_count\")}, ensure_ascii=False))'"

ssh -n "$SPARK_HOST" "rm -rf '$SPARK_MODEL' && mv '$TMP_DST' '$SPARK_MODEL' && du -sh '$SPARK_MODEL'"
log "Spark copy complete: $SPARK_HOST:$SPARK_MODEL"
