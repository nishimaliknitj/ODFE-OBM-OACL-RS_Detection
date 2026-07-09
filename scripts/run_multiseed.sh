#!/usr/bin/env bash
# Multi-seed reproducibility run (paper Table 13): seeds 42, 123, 2024
set -euo pipefail

DATASET="${1:-dior}"
CONFIG="configs/${DATASET}.yaml"

for SEED in 42 123 2024; do
    echo ">>> Training ${DATASET} with seed ${SEED}"
    python src/train.py --config "${CONFIG}" --seed "${SEED}" \
        2>&1 | tee "logs/${DATASET}_seed${SEED}.log"
done

echo "Multi-seed run complete. Aggregate results with:"
echo "  python scripts/aggregate_seeds.py --dataset ${DATASET}"
