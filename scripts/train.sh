#!/usr/bin/env bash
# Convenience wrapper: bash scripts/train.sh <dior|dota|fair1m> [extra python args...]
set -euo pipefail

DATASET="${1:-dior}"
shift || true

CONFIG="configs/${DATASET}.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "Unknown dataset '${DATASET}'. Expected one of: dior, dota, fair1m"
    exit 1
fi

echo "Training on ${DATASET} using ${CONFIG} ..."
python src/train.py --config "${CONFIG}" "$@"
