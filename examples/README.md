# Examples

Minimal end-to-end usage snippets. See `docs/` for full documentation.

## Build the model and run a forward pass

```python
from src.utils.config import load_config
from src.models import build_detector
import torch

cfg = load_config("configs/dior.yaml")
model = build_detector(cfg)

x = torch.randn(2, 3, 640, 640)
outputs = model(x)
print(outputs["cls_outputs"][0].shape)  # (2, 20, 80, 80) for DIOR (20 classes)
```

## Compute the optimization difficulty score (tau) manually

```python
import torch
from src.models.tau import OptimizationDifficultyTracker, TauConfig

tracker = OptimizationDifficultyTracker(TauConfig(ema_decay=0.9))

sample_ids = ["img_001", "img_002"]
grad_norms = torch.tensor([1.2, 0.3])       # ||grad_f L_det||_2 per sample
per_sample_losses = torch.tensor([0.8, 0.1])

tau = tracker.update(sample_ids, grad_norms, per_sample_losses)
print(tau)  # tensor([1.2000, 0.3000])  -- first update, no history yet
```

## Compute OACL loss directly

```python
import torch
from src.losses.oacl_loss import OptimizationAwareClassificationLoss

loss_fn = OptimizationAwareClassificationLoss(max_sample_weight=3.0)
logits = torch.randn(8, 20)     # (batch, num_classes)
targets = torch.randint(0, 2, (8, 20)).float()
tau = torch.rand(8) * 2         # per-sample difficulty scores

loss = loss_fn(logits, targets, tau)
print(loss.item())
```

## Full training / evaluation

```bash
python src/train.py --config configs/dior.yaml
python src/test.py --config configs/dior.yaml --weights checkpoints/odfe_obm_oacl_dior_final.pt
python src/predict.py --config configs/dior.yaml --weights checkpoints/odfe_obm_oacl_dior_final.pt --source path/to/image.jpg
```
