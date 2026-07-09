# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2024-01-01
### Added
- Initial public release of ODFE, OBM, and OACL modules.
- Unified optimization-difficulty score `tau` (gradient sensitivity + convergence instability + curvature proxy), EMA-smoothed (`beta=0.9`).
- YOLOv8-style single-stage baseline detector wrapper with ODFE-reweighted feature pyramid (P3/P4/P5).
- OBM difficulty-proportional mosaic sampling.
- OACL bounded per-sample classification loss reweighting.
- CIoU localization loss implementation.
- Training / evaluation / inference entry points (`src/train.py`, `src/test.py`, `src/predict.py`).
- Dataset loaders for DIOR, DOTA, FAIR1M.
- Configs for all three benchmark datasets.
- Unit tests for tau computation, ODFE, OBM, OACL, and CIoU.
- Docker and Conda environment definitions.
- GitHub Actions CI workflow (lint + unit tests).

### Notes
- YOLOrs baseline comparison values in the original manuscript tables were corrected after a table-formatting duplication issue identified during peer review (see paper Section "Data Verification and Correction Statement"). This repository ships the corrected reproduction protocol only.

## [0.2.0] - 2023-11-15
### Added
- Component ablation scripts for tau (`g` only, `delta` only, `kappa` only, pairwise combinations).
- Multi-seed reproducibility harness (seeds 42, 123, 2024).

### Changed
- Switched EMA decay default from 0.8 to 0.9 after sensitivity sweep.

## [0.1.0] - 2023-09-01
### Added
- Initial internal prototype: YOLOv8 baseline training pipeline on DIOR.
- Draft gradient-sensitivity-only feature reweighting (precursor to ODFE).
