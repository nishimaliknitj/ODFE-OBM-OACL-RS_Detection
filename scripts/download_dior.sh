#!/usr/bin/env bash
# Download and prepare the DIOR dataset.
# Official dataset page: https://gcheng-nwpu.github.io/#Datasets
set -euo pipefail

DATA_ROOT="data/DIOR"
mkdir -p "${DATA_ROOT}"

echo "=========================================================================="
echo " DIOR Dataset Setup"
echo "=========================================================================="
echo "Official dataset page (verified, maintained by the original authors):"
echo "  https://gcheng-nwpu.github.io/#Datasets"
echo
echo "Paper: https://arxiv.org/abs/1909.00133"
echo
echo "After downloading from the official page, place the files as follows:"
echo "  ${DATA_ROOT}/JPEGImages-trainval.zip"
echo "  ${DATA_ROOT}/JPEGImages-test.zip"
echo "  ${DATA_ROOT}/Annotations.zip"
echo
read -p "Have you placed the zip files in ${DATA_ROOT}? [y/N] " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborting. Please download the dataset from https://gcheng-nwpu.github.io/#Datasets first."
    exit 1
fi

cd "${DATA_ROOT}"

if [ -f "JPEGImages-trainval.zip" ]; then
    unzip -q -o JPEGImages-trainval.zip -d JPEGImages-train
    echo "Extracted training images."
fi

if [ -f "JPEGImages-test.zip" ]; then
    unzip -q -o JPEGImages-test.zip -d JPEGImages-test
    echo "Extracted test images."
fi

if [ -f "Annotations.zip" ]; then
    unzip -q -o Annotations.zip -d Annotations
    mkdir -p Annotations/train Annotations/test
    echo "Extracted annotations. Split train/test annotation files manually"
    echo "according to the official train/test image list if not already split."
fi

echo "DIOR setup complete. Verify with: ls ${DATA_ROOT}/JPEGImages-train | wc -l"
