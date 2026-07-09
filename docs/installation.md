# Installation Guide

## Requirements

- Python 3.8+
- CUDA 11.6+ (for GPU training; CPU works for testing/small-scale debugging)
- ~10 GB disk space for dependencies (excluding datasets)

## Option A: pip + venv

```bash
git clone https://github.com/nishimaliknitj/ODFE-OBM-OACL-RSDetection.git
cd ODFE-OBM-OACL-RSDetection

python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                    # editable install of the `src` package
```

## Option B: Conda

```bash
conda env create -f environment.yml
conda activate odfe-obm-oacl
```

## Option C: Docker

```bash
docker build -t odfe-obm-oacl .
docker run --gpus all -it \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    odfe-obm-oacl
```

Or via Docker Compose (also starts a TensorBoard sidecar on port 6006):

```bash
docker-compose up train tensorboard
```

## Verifying your installation

```bash
python -m pytest tests/ -v
```

You should see 48 passing tests, covering the tau tracker, ODFE, OBM, OACL, CIoU loss, evaluation metrics, and a full detector forward/backward smoke test.

```bash
python -c "
from src.utils.config import load_config
from src.models import build_detector
cfg = load_config('configs/dior.yaml')
model = build_detector(cfg)
print(f'Model built successfully with {model.num_parameters():,} parameters.')
"
```

## GPU Setup Notes

This codebase was developed and validated on a single **NVIDIA RTX 3090 (24GB VRAM)**. For multi-GPU training, wrap the model with `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel` — note that the per-sample `OptimizationDifficultyTracker` state (`src/models/tau.py`) is currently maintained on a single process; for distributed training you will need to shard or synchronize this state across ranks (see `docs/methodology.md`, Limitations discussion, and the paper's Limitations section, point 4).

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run scripts from the repo root, or `pip install -e .` |
| `RuntimeError: CUDA out of memory` | Reduce `training.batch_size` in the config YAML |
| `cv2.error` on headless servers | `pip install opencv-python-headless` instead of `opencv-python` |
| Slow training with `deterministic: true` | Deterministic cuDNN mode is much slower; only use for exact reproducibility checks |
