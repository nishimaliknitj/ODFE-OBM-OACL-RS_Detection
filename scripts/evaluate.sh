#!/usr/bin/env bash
# Convenience wrapper: bash scripts/evaluate.sh <dior|dota|fair1m> <weights_path>
set -euo pipefail

DATASET="${1:-dior}"
WEIGHTS="${2:-checkpoints/${DATASET}_full_best.pt}"
shift 2 || true

CONFIG="configs/${DATASET}.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "Unknown dataset '${DATASET}'. Expected one of: dior, dota, fair1m"
    exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "Weights file not found: ${WEIGHTS}"
    echo "Usage: bash scripts/evaluate.sh <dior|dota|fair1m> <path_to_weights.pt>"
    exit 1
fi

echo "Evaluating ${DATASET} with weights ${WEIGHTS} ..."
python src/test.py --config "${CONFIG}" --weights "${WEIGHTS}" "$@"
