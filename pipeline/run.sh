#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# run.sh — Batch runner for the Store Intelligence detection pipeline
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./pipeline/run.sh [DATA_DIR] [STORE_ID] [LAYOUT_PATH]
#
# Arguments:
#   DATA_DIR     Directory containing .mp4 / .avi video clips  (default: data/)
#   STORE_ID     Store identifier                              (default: STORE_01)
#   LAYOUT_PATH  Path to store_layout.json                     (default: store_layout.json)
#
# The script will:
#   1. Check that the API is reachable (GET /health).
#   2. Discover all video clips under DATA_DIR.
#   3. Run pipeline/detect.py for each clip.
#   4. Print a summary of processed clips.
#
# Camera ID is derived from the filename: clip named "cam_entrance_2026.mp4"
# yields camera-id "cam_entrance_2026".  Override by naming clips as
# <camera-id>__<anything>.mp4  (double underscore separator).
#
# Start time is taken from file modification time unless the filename
# contains an ISO timestamp segment (YYYY-MM-DDTHH-MM-SS).
#
# Prerequisites:
#   pip install ultralytics opencv-python-headless scipy requests numpy
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
DATA_DIR="${1:-data}"
STORE_ID="${2:-STORE_01}"
LAYOUT_PATH="${3:-store_layout.json}"
API_URL="${SIS_API_URL:-http://localhost:8000}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Colours ───────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # no colour

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 1. Health check ──────────────────────────────────────────────────────
info "Checking API health at ${API_URL}/health ..."
if curl -sf --max-time 5 "${API_URL}/health" > /dev/null 2>&1; then
    info "API is reachable."
else
    warn "API at ${API_URL} is NOT reachable."
    warn "Events will be emitted but may fail to POST."
    warn "Continuing anyway — events are logged locally."
fi

# ── 3. Process clips using the Python Batch Runner ─────────────
info "Delegating batch execution to Python so Re-ID memory is shared..."
python -m pipeline.batch_run \
    --data-dir "$DATA_DIR" \
    --store-id "$STORE_ID" \
    --layout "$LAYOUT_PATH"

exit $?
