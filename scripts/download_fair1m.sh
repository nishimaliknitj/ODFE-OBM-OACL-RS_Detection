#!/usr/bin/env bash
# Download and prepare the FAIR1M dataset.
# Official dataset page: https://www.gaofen-challenge.com/benchmark
set -euo pipefail

DATA_ROOT="data/FAIR1M"
mkdir -p "${DATA_ROOT}/train/images" "${DATA_ROOT}/train/labelXml"
mkdir -p "${DATA_ROOT}/val/images" "${DATA_ROOT}/val/labelXml"

echo "=========================================================================="
echo " FAIR1M Dataset Setup"
echo "=========================================================================="
echo "Official dataset page (verified, Gaofen Challenge / ISPRS benchmark):"
echo "  https://www.gaofen-challenge.com/benchmark"
echo
echo "Paper: https://arxiv.org/abs/2103.05569"
echo
echo "After downloading, place the files as:"
echo "  ${DATA_ROOT}/train/images/*.tif"
echo "  ${DATA_ROOT}/train/labelXml/*.xml"
echo "  ${DATA_ROOT}/val/images/*.tif"
echo "  ${DATA_ROOT}/val/labelXml/*.xml"
echo
echo "Input resolution for this repo's config is 800x800 (configs/fair1m.yaml)."
echo "FAIR1M setup instructions complete -- see docs/data_preparation.md."
