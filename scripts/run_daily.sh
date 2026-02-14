#!/bin/bash
# Daily prediction pipeline runner
# Loads .env and executes the daily pipeline

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Run daily pipeline
cd "$PROJECT_DIR"
/opt/anaconda3/bin/python -m src.pipelines.daily_pipeline 2>&1

# Log result
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exit code: $?" >> "$PROJECT_DIR/data/logs/daily_pipeline.log"
