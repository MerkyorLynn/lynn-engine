#!/usr/bin/env bash
set -euo pipefail
# Compatibility wrapper. The official 9B route is Qwen3.5-9B; the earlier
# qwen36 filename is kept only so remote/background scripts do not break.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/r6000_qwen35_9b_official_w4a16_pack.sh" "$@"
