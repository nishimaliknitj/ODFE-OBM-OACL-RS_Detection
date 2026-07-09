#!/usr/bin/env bash
# Download and prepare the DOTA dataset (v1.0/v1.5/v2.0).
# Official dataset page: https://captain-whu.github.io/DOTA/dataset.html
set -euo pipefail

DATA_ROOT="data/DOTA"
mkdir -p "${DATA_ROOT}/train/images" "${DATA_ROOT}/train/labelTxt"
mkdir -p "${DATA_ROOT}/val/images" "${DATA_ROOT}/val/labelTxt"

echo "=========================================================================="
echo " DOTA Dataset Setup"
echo "=========================================================================="
echo "Official dataset page (verified, maintained by Wuhan University / CAPTAIN-WHU):"
echo "  https://captain-whu.github.io/DOTA/dataset.html"
echo
echo "Paper: https://arxiv.org/abs/1711.10398"
echo
echo "NOTE: DOTA images/annotations are for academic use only; commercial use"
echo "is prohibited by the original license. Registration may be required."
echo
echo "After downloading and registering, place the raw files as:"
echo "  ${DATA_ROOT}/raw/train/images/*.png"
echo "  ${DATA_ROOT}/raw/train/labelTxt/*.txt"
echo "  ${DATA_ROOT}/raw/val/images/*.png"
echo "  ${DATA_ROOT}/raw/val/labelTxt/*.txt"
echo
echo "This repository trains on 640x640 patches (paper Table 4). Once the raw"
echo "full-resolution images are in place, run the patch-extraction utility:"
echo
echo "  python -m src.datasets.patch_extraction \\"
echo "      --input ${DATA_ROOT}/raw/train \\"
echo "      --output ${DATA_ROOT}/train \\"
echo "      --patch-size 640 --overlap 256"
echo
echo "(See docs/data_preparation.md for the full patch-extraction protocol.)"
