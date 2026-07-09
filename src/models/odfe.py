"""
ODFE: Optimization-Driven Feature Emphasis
============================================

Reweights multi-scale feature pyramid levels {f_l} based on optimization
dynamics (gradient sensitivity + learning instability) rather than
appearance statistics (unlike SE-Net, CBAM, ABNet).

    g_l^(t) = (1/|B|) * sum_i || grad_{f_l} L_det(x_i, y_i; theta_t) ||_2   (Eq. 1)
    s_l^(t) = | g_l^(t) - g_l^(t-1) |                                       (Eq. 2)
    alpha_l^(t) = g_l^(t) * (1 + s_l^(t))                                    (Eq. 3)
    f_tilde_l = ( alpha_l^(t) / (sum_k alpha_k^(t) + eps) ) * f_l            (Eq. 4)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .tau import feature_level_importance, normalize_importance


class OptimizationDrivenFeatureEmphasis(nn.Module):
    """
    Wraps a list of multi-scale feature maps {f_l}_{l=1}^{L} (e.g. P3, P4, P5
    from a feature pyramid neck) and re-emphasizes them according to
    optimization-driven importance scores.

    This module maintains a running estimate of gradient magnitude per
    feature level (g_l) and its iteration-to-iteration change (s_l), then
    uses these to compute a softmax-normalized reweighting applied to the
    feature maps before they are fed to the detection head.
    """

    def __init__(self, num_levels: int = 3, ema_decay: float = 0.9, eps: float = 1e-6):
        super().__init__()
        self.num_levels = num_levels
        self.ema_decay = ema_decay
        self.eps = eps

        # Persistent (non-learnable) buffers tracking gradient magnitude
        # per level across iterations.
        self.register_buffer("g_ema", torch.zeros(num_levels))
        self.register_buffer("g_prev", torch.zeros(num_levels))
        self.register_buffer("initialized", torch.tensor(False))

        # Cache of the last computed importance weights (for logging / Grad-CAM-style analysis)
        self.last_alpha: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update_gradient_stats(self, grad_norms_per_level: torch.Tensor):
        """
        Called after backward() using hooks that capture
        ||grad_{f_l} L_det||_2 per feature level (see
        `register_feature_grad_hooks` below). Updates the EMA state.
        """
        grad_norms_per_level = grad_norms_per_level.detach().to(self.g_ema.device)

        if not bool(self.initialized):
            self.g_prev.copy_(grad_norms_per_level)
            self.g_ema.copy_(grad_norms_per_level)
            self.initialized.fill_(True)
            return

        # s_l^(t) = |g_l^(t) - g_l^(t-1)|
        stability = torch.abs(grad_norms_per_level - self.g_prev)

        # EMA update of g
        self.g_ema.mul_(self.ema_decay).add_(grad_norms_per_level, alpha=1 - self.ema_decay)
        self.g_prev.copy_(grad_norms_per_level)

        # Cache instability for importance computation on next forward
        self._last_stability = stability

    def compute_importance(self) -> torch.Tensor:
        """Eq. 3 + Eq. 4: returns normalized alpha_l for each feature level."""
        stability = getattr(self, "_last_stability", torch.zeros_like(self.g_ema))
        alpha = feature_level_importance(self.g_ema, stability)
        alpha_norm = normalize_importance(alpha, self.eps)
        self.last_alpha = alpha_norm.detach().clone()
        return alpha_norm

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        features : list of L tensors, each (B, C_l, H_l, W_l)

        Returns
        -------
        emphasized_features : list of L reweighted tensors, same shapes.
        """
        assert (
            len(features) == self.num_levels
        ), f"Expected {self.num_levels} feature levels, got {len(features)}"

        if not bool(self.initialized):
            # Before any gradient statistics are available (first forward
            # pass of training), pass features through unchanged.
            return features

        alpha = self.compute_importance()  # (L,)
        emphasized = [alpha[l] * features[l] for l in range(self.num_levels)]
        return emphasized

    def get_importance_scores(self) -> Optional[torch.Tensor]:
        """For logging / visualization (Figure 3-style importance bars)."""
        return self.last_alpha


def register_feature_grad_hooks(
    features: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Utility: given a list of feature tensors that require grad, registers
    hooks that will populate `grad_norms` with ||grad||_2 per level after
    `loss.backward()` is called. Call this BEFORE the forward pass that
    produces `features`, then read `grad_norms` after backward.

    Example
    -------
    >>> grad_norms = torch.zeros(len(features))
    >>> handles = register_feature_grad_hooks(features, grad_norms)
    >>> loss.backward()
    >>> odfe.update_gradient_stats(grad_norms)
    >>> for h in handles: h.remove()
    """
    grad_norms = torch.zeros(len(features))
    handles = []

    def _make_hook(idx):
        def _hook(grad):
            grad_norms[idx] = grad.detach().norm(p=2)

        return _hook

    for i, f in enumerate(features):
        if f.requires_grad:
            handles.append(f.register_hook(_make_hook(i)))

    return grad_norms, handles
