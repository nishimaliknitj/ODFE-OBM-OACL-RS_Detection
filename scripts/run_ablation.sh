#!/usr/bin/env bash
# Runs the full component ablation (Table 15 of the paper):
#   Baseline -> +ODFE -> +ODFE+OBM -> Full Model (+ODFE+OBM+OACL)
set -euo pipefail

DATASET="${1:-dior}"
CONFIG="configs/${DATASET}.yaml"

echo ">>> [1/4] Baseline (no ODFE/OBM/OACL)"
python src/train.py --config "${CONFIG}" --no-odfe --no-obm --no-oacl \
    --epochs 100 2>&1 | tee "logs/${DATASET}_ablation_baseline.log"

echo ">>> [2/4] +ODFE only"
python src/train.py --config "${CONFIG}" --no-obm --no-oacl \
    --epochs 100 2>&1 | tee "logs/${DATASET}_ablation_odfe.log"

echo ">>> [3/4] +ODFE +OBM"
python src/train.py --config "${CONFIG}" --no-oacl \
    --epochs 100 2>&1 | tee "logs/${DATASET}_ablation_odfe_obm.log"

echo ">>> [4/4] Full Model (+ODFE +OBM +OACL)"
python src/train.py --config "${CONFIG}" \
    --epochs 100 2>&1 | tee "logs/${DATASET}_ablation_full.log"

echo "Ablation study complete. See logs/${DATASET}_ablation_*.log"
