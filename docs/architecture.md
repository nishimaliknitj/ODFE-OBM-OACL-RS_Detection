# Architecture Overview

```
                     Optical RS Image Input
                              |
                              v
              +--------------------------------+
              |   OBM: difficulty-proportional  |
              |   mosaic sample selection       |   P_i ∝ τ_i^(t)
              +--------------------------------+
                              |
                              v
              +--------------------------------+
              |  Backbone (CSP) + FPN Neck      |
              |  -> {P3 (stride 8), P4 (16),    |
              |      P5 (32)}                   |
              +--------------------------------+
                              |
                              v
              +--------------------------------+
              |  ODFE: feature-level reweight   |   α_l ∝ g_l^(t)·(1+s_l^(t))
              +--------------------------------+
                              |
                 -------------+-------------
                 |            |            |
                 v            v            v
             Detect(P3)   Detect(P4)   Detect(P5)
                 |            |            |
                 -------------+-------------
                              |
                              v
              +--------------------------------+
              |  OACL: per-sample loss reweight |   w_i ∝ τ_i^(t) / E[τ^(t)]
              +--------------------------------+
                              |
                              v
                 L_final = L_loc (CIoU) + λ·L_cls
```

## Module Boundaries (source files)

- `src/models/tau.py` — the shared `τ` signal (OptimizationDifficultyTracker), independent of any specific detector.
- `src/models/odfe.py` — feature-level reweighting; consumes gradient-hook output, produces reweighted feature maps.
- `src/models/obm.py` — sample-selection probability + mosaic stitching; operates at the *dataset/dataloader* level, not inside the model's forward pass.
- `src/losses/oacl_loss.py` — per-sample classification loss reweighting; a drop-in `nn.Module` loss function.
- `src/losses/ciou.py` — standard localization loss, independent of the optimization-difficulty framework.
- `src/losses/combined_loss.py` — wires CIoU + OACL (+ optional invariance regularization) into the final training objective.
- `src/models/detector.py` — reference backbone/neck/head wiring everything together into a trainable `nn.Module`.

## Data Flow During Training (see `src/train.py`)

1. `OptimizationBalancedMosaicSampler` (OBM) selects which samples enter each mini-batch, weighted by their currently-tracked `τ`.
2. The batch is forward-passed through `ODFEDetector`, which internally applies `OptimizationDrivenFeatureEmphasis` (ODFE) to its FPN outputs before the detection head.
3. `CombinedDetectionLoss` computes `L_loc` (CIoU) and `L_cls` (OACL, using each sample's current `τ`) and combines them into `L_final`.
4. After `backward()`, gradient-magnitude and loss signals for this iteration are fed into `OptimizationDifficultyTracker.update()`, updating the EMA-smoothed `τ` for the *next* iteration each sample is seen.

This creates the mutually-reinforcing loop described in the paper: better features (ODFE) improve difficulty estimation quality → more hard-sample exposure (OBM) stabilizes gradients → loss reweighting (OACL) amplifies tail-sample gradients → which in turn sharpens the `τ` estimate.
