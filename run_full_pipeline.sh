#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

source venv/bin/activate
set -a
source ~/.env
[ -f "$PROJECT_DIR/.env" ] && source "$PROJECT_DIR/.env"
set +a

mkdir -p "$LOG_DIR"
FULL_LOG_FILE="${LOG_DIR}/full_pipeline.log"

{
    echo "============================================================"
    echo "START FULL PIPELINE $(date -Iseconds)"
    echo "PROJECT_DIR=$PROJECT_DIR"
    echo "============================================================"

    bash "$PROJECT_DIR/run_parse.sh"
    bash "$PROJECT_DIR/run_build_sources.sh"

    if [[ $# -ge 1 && -n "${1:-}" ]]; then
        bash "$PROJECT_DIR/run_rag.sh" "$1"
    else
        bash "$PROJECT_DIR/run_rag.sh"
    fi

    echo "============================================================"
    echo "END FULL PIPELINE $(date -Iseconds)"
    echo "============================================================"
} 2>&1 | tee -a "$FULL_LOG_FILE"
