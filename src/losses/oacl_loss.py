"""
OACL: Optimization-Aware Classification Loss
===============================================

Reweights per-sample binary cross-entropy classification loss according to
the shared optimization difficulty score tau_i^(t), rather than a static
class-frequency prior (as in Equalization Loss) or a single-scalar
confidence signal (as in Focal Loss).

    L_cls^(i) = -w_i^(t) * [ y_i log(p_i) + (1 - y_i) log(1 - p_i) ]     (Eq. 11)
    w_i^(t)   = tau_i^(t) / (E[tau^(t)] + eps)                            (Eq. 12)
    w_hat_i^(t) = min(w_i^(t), w_max)                                     (Eq. 14)
    L_cls = (1/|B|) * sum_i w_hat_i^(t) * L_cls^(i)                        (Eq. 15)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizationAwareClassificationLoss(nn.Module):
    """
    Drop-in replacement for standard BCE / Focal classification loss in a
    detection head. Requires the per-sample tau_i^(t) score (typically
    supplied by `OptimizationDifficultyTracker.get_tau(sample_ids)`).
    """

    def __init__(self, max_sample_weight: float = 3.0, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.w_max = max_sample_weight
        self.eps = eps
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        tau: torch.Tensor,
        tau_batch_expectation: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_logits : (B, C) or (B,) raw classification logits (pre-sigmoid)
        targets     : same shape as pred_logits, binary {0, 1} targets
        tau         : (B,) per-sample optimization difficulty scores
        tau_batch_expectation : float, E[tau^(t)] over the current mini-batch.
                                 If None, computed as tau.mean().

        Returns
        -------
        loss : scalar (if reduction != 'none') or (B,) per-sample loss
        """
        assert pred_logits.shape[0] == tau.shape[0], "Batch size mismatch between logits and tau"

        if tau_batch_expectation is None:
            tau_batch_expectation = float(tau.mean().clamp_min(self.eps).item())

        # Eq. 12: adaptive weight
        w = tau / (tau_batch_expectation + self.eps)

        # Eq. 14: bounded weighting constraint
        w = torch.clamp(w, max=self.w_max)

        # Broadcast weight across class dimension if pred_logits is (B, C)
        if pred_logits.dim() > 1:
            w = w.view(-1, *([1] * (pred_logits.dim() - 1)))

        # Eq. 11: weighted BCE (numerically stable version via BCE-with-logits)
        per_sample_bce = F.binary_cross_entropy_with_logits(
            pred_logits, targets.float(), reduction="none"
        )
        weighted = w * per_sample_bce

        if self.reduction == "mean":
            return weighted.mean()
        elif self.reduction == "sum":
            return weighted.sum()
        return weighted


class FocalLossBaseline(nn.Module):
    """Standard Focal Loss (Lin et al. 2017) -- used as an ablation / comparison baseline."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets.float(), reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class EqualizationLossBaseline(nn.Module):
    """Equalization Loss (Tan et al. 2020) -- class-frequency-based baseline for comparison."""

    def __init__(self, class_freq: torch.Tensor, threshold: float = 0.05):
        super().__init__()
        self.register_buffer("class_freq", class_freq)  # (C,) normalized frequency per class
        self.threshold = threshold

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        rare_mask = (self.class_freq < self.threshold).float()  # (C,)
        # Suppress gradient for rare-class negatives (simplified EQL formulation)
        weight = 1.0 - rare_mask.unsqueeze(0) * (1 - targets)
        loss = F.binary_cross_entropy_with_logits(pred_logits, targets.float(), reduction="none")
        return (weight * loss).mean()


class InvarianceRegularizationLoss(nn.Module):
    """
    Eq. 18: L_inv = (1/L) * sum_l || f_tilde_l^(a) - f_tilde_l^(b) ||_2^2

    Penalizes inconsistency between ODFE feature maps of two augmented
    views of the same image, promoting stable tail-object representations.
    Disabled by default (see paper Eq. 19 discussion).
    """

    def __init__(self):
        super().__init__()

    def forward(self, features_a, features_b) -> torch.Tensor:
        assert len(features_a) == len(features_b)
        losses = []
        for fa, fb in zip(features_a, features_b):
            losses.append(F.mse_loss(fa, fb))
        return torch.stack(losses).mean()
