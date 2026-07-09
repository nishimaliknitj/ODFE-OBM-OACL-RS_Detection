# Methodology: Paper-to-Code Mapping

This document maps every equation and module in the paper to its corresponding implementation, for readers who want to verify correctness or extend the method.

## 1. Optimization Difficulty Score (tau)

| Paper | Equation | Code |
|---|---|---|
| Gradient sensitivity `g_i^(t)` | Eq. 1 (feature-level), Eq. 5 (sample-level) | `src/models/tau.py::OptimizationDifficultyTracker.update` |
| Feature-level stability `s_l^(t)` | Eq. 2 | `src/models/odfe.py::OptimizationDrivenFeatureEmphasis.update_gradient_stats` |
| Convergence instability `delta_i^(t)` | Eq. 6 | `src/models/tau.py` (computed as `abs(loss_t - loss_prev)`) |
| Curvature proxy `kappa_i^(t)` | Eq. 7 | `src/models/tau.py` (computed as `abs(g_t - g_prev)`) |
| Unified difficulty score `tau_i^(t)` | Eq. 8 | `src/models/tau.py::OptimizationDifficultyTracker.update` (final line) |
| EMA smoothing (beta=0.9) | Eq. 9 | `src/models/tau.py` (`g_ema`, `delta_ema`, `kappa_ema`) |
| Head/tail partitioning by `E[tau]` | Section "Calculation of Optimization Difficulty" | `src/models/tau.py::partition_head_tail` |

## 2. ODFE (Optimization-Driven Feature Emphasis)

| Paper | Equation | Code |
|---|---|---|
| Feature-level importance `alpha_l^(t)` | Eq. 3 | `src/models/tau.py::feature_level_importance` |
| Normalized reweighting `f_tilde_l` | Eq. 4 | `src/models/tau.py::normalize_importance`, applied in `src/models/odfe.py::forward` |

`ODFEDetector` (`src/models/detector.py`) applies ODFE to the three FPN outputs (P3/P4/P5) immediately before the detection head, matching Figure 1 of the paper.

**Important implementation note**: Eq. 1 defines `g_l^(t)` as the gradient of the *detection loss* with respect to feature level `l`. This repository provides `register_feature_grad_hooks()` (in `src/models/odfe.py`) as the mechanism to capture this signal via PyTorch backward hooks on the feature tensors. Wire this into your training loop as:

```python
grad_norms, handles = register_feature_grad_hooks(features)
loss.backward()
odfe.update_gradient_stats(grad_norms)
for h in handles:
    h.remove()
```

## 3. OBM (Optimization-Balanced Mosaic)

| Paper | Equation | Code |
|---|---|---|
| Difficulty-proportional sampling `P_i` | Eq. 10 | `src/models/obm.py::OptimizationBalancedMosaicSampler.compute_sampling_distribution` |
| Mosaic construction `I'(x,y)` | Figure 5, Step 3 | `src/models/obm.py::build_mosaic_image` |

## 4. OACL (Optimization-Aware Classification Loss)

| Paper | Equation | Code |
|---|---|---|
| Weighted BCE `L_cls^(i)` | Eq. 11 | `src/losses/oacl_loss.py::OptimizationAwareClassificationLoss.forward` |
| Adaptive weight `w_i^(t)` | Eq. 12 | same |
| Expected gradient contribution | Eq. 13 | (emerges automatically from autograd; not separately implemented) |
| Bounded weight `w_hat_i^(t)` | Eq. 14 | `torch.clamp(w, max=self.w_max)` |
| Aggregated loss `L_cls` | Eq. 15 | `reduction="mean"` in the same module |

## 5. Localization Loss and Combined Objective

| Paper | Equation | Code |
|---|---|---|
| CIoU loss `L_loc` | Eq. 17 | `src/losses/ciou.py::ciou_loss` |
| Combined objective `L_final` | Eq. 16 | `src/losses/combined_loss.py::CombinedDetectionLoss.forward` |
| Invariance regularization `L_inv` | Eq. 18 | `src/losses/oacl_loss.py::InvarianceRegularizationLoss` |
| Full objective with `L_inv` | Eq. 19 | `src/losses/combined_loss.py` (when `invariance_enabled=True`) |

## 6. Evaluation Metrics

| Paper | Equation | Code |
|---|---|---|
| Precision `P` | Eq. 20 | `src/utils/metrics.py::precision_recall` |
| Recall `R` | Eq. 21 | same |
| Average Precision `AP` | Eq. 22 | `src/utils/metrics.py::compute_ap` (all-point interpolation) |
| Mean AP `mAP` | Eq. 23 | `src/utils/metrics.py::compute_map` |
| IoU | Eq. 24 | `src/utils/metrics.py::box_iou_matrix` |

## 7. Training Loop (Algorithm 1)

`src/train.py::train_one_epoch` implements the pseudocode of Algorithm 1: per-iteration tau computation, ODFE feature reweighting, OBM-guided sampling, OACL loss weighting, and combined gradient update.

## 8. Inference (Algorithm 2)

`src/predict.py::predict_image` implements Algorithm 2: feature extraction, fixed (frozen) ODFE emphasis, detection head prediction, confidence thresholding, and NMS. As noted in the paper, ODFE/OBM/OACL are **training-time only** — at inference, ODFE simply applies its already-learned fixed importance weights with zero additional FLOPs.

## Extension Points

1. **Detection head**: The reference implementation in `src/models/detector.py` uses a simplified global-average-pooled classification/regression head for clarity and testability. To reproduce the paper's exact YOLOv8-CSP dense per-anchor head with SimOTA/TaskAlignedAssigner label assignment, swap `DetectionHead` for the Ultralytics `Detect` module and update `train.py`'s target-assignment logic (`dummy_targets_from_batch`).
2. **Oriented bounding boxes (DOTA/FAIR1M)**: This repo converts OBB annotations to axis-aligned envelopes. For rotated-box training, replace `CIoULoss` with a rotated-IoU loss (e.g., KLD loss, GWD loss) and adjust the regression head to predict `(cx, cy, w, h, theta)`.
3. **Per-sample gradient computation**: The reference `train.py` uses a coarse proxy (mean parameter-gradient-norm broadcast across the batch) for `g_i^(t)` to keep the training loop simple and fast. For the paper's exact per-sample, per-feature-level gradient sensitivity, use micro-batching (`batch_size=1` inner loop with `retain_graph=True`) or a library such as `functorch`/`torch.func.vmap` with `grad` for efficient per-sample gradients.
