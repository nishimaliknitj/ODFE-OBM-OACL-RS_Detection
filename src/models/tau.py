"""
Optimization Difficulty Score (tau)
====================================

Implements the unified, EMA-smoothed, per-sample optimization difficulty
score described in the paper (Eqs. 1-9). This single signal is shared by
all three proposed modules:

    - ODFE  (Optimization-Driven Feature Emphasis)   -> src/models/odfe.py
    - OBM   (Optimization-Balanced Mosaic)            -> src/models/obm.py
    - OACL  (Optimization-Aware Classification Loss)  -> src/losses/oacl_loss.py

tau_i^(t) = g_i^(t) * (1 + delta_i^(t) + kappa_i^(t))

where:
    g_i^(t)     : gradient sensitivity      (Eq. 5)  - grad magnitude of L_det w.r.t. theta
    delta_i^(t) : convergence instability   (Eq. 6)  - |L^(t)(x_i) - L^(t-1)(x_i)|
    kappa_i^(t) : curvature proxy           (Eq. 7)  - |g_i^(t) - g_i^(t-1)|

All three components are tracked as exponential moving averages (Eq. 9)
with decay beta (default 0.9, effective window T_eff = 1/(1-beta) = 10 iters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class TauConfig:
    ema_decay: float = 0.9
    epsilon: float = 1e-6
    warmup_iters: int = 100


class OptimizationDifficultyTracker:
    """
    Maintains per-sample EMA statistics (g, delta, kappa) across training
    iterations and computes the unified optimization difficulty score tau.

    Sample identity is tracked via a stable integer/string sample id
    (typically the dataset index or image filename), NOT the batch position,
    since the same physical sample must accumulate history across epochs.

    Usage
    -----
    >>> tracker = OptimizationDifficultyTracker(TauConfig(ema_decay=0.9))
    >>> tau = tracker.update(sample_ids, grad_norms, per_sample_losses)
    >>> weights = tracker.get_tau(sample_ids)
    """

    def __init__(self, config: Optional[TauConfig] = None):
        self.config = config or TauConfig()
        self.beta = self.config.ema_decay
        self.eps = self.config.epsilon

        # Per-sample state, keyed by sample id (python object -> dict is fine
        # for dataset sizes considered in the paper; O(N) memory - see
        # Limitations section of the paper for scaling discussion).
        self._g_ema: Dict[str, float] = {}
        self._loss_prev: Dict[str, float] = {}
        self._g_prev: Dict[str, float] = {}
        self._delta_ema: Dict[str, float] = {}
        self._kappa_ema: Dict[str, float] = {}
        self._iter_count: Dict[str, int] = {}

        self.global_iter: int = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        sample_ids,
        grad_norms: torch.Tensor,
        per_sample_losses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update EMA statistics for a batch and return the current tau scores.

        Parameters
        ----------
        sample_ids : sequence of hashable ids, length B
        grad_norms : (B,) tensor -- ||grad_{f_l} L_det(x_i, y_i; theta_t)||_2
                     (Eq. 1/5). Typically computed by backward-hooking the
                     detection loss w.r.t. per-sample feature activations,
                     or approximated via per-sample loss.backward() with
                     `retain_graph=True` in a microbatch loop.
        per_sample_losses : (B,) tensor -- L_det(x_i) for this iteration.

        Returns
        -------
        tau : (B,) tensor of optimization difficulty scores for this batch.
        """
        self.global_iter += 1
        grad_norms = grad_norms.detach().float().cpu()
        per_sample_losses = per_sample_losses.detach().float().cpu()

        tau_values = torch.zeros(len(sample_ids))

        for i, sid in enumerate(sample_ids):
            sid = str(sid)
            g_t = float(grad_norms[i])
            loss_t = float(per_sample_losses[i])

            g_prev = self._g_prev.get(sid, g_t)
            loss_prev = self._loss_prev.get(sid, loss_t)

            # Eq. 6: convergence instability
            delta_t = abs(loss_t - loss_prev)
            # Eq. 7: curvature proxy
            kappa_t = abs(g_t - g_prev)

            # Eq. 9: EMA smoothing of g, delta, kappa
            g_ema_prev = self._g_ema.get(sid, g_t)
            delta_ema_prev = self._delta_ema.get(sid, delta_t)
            kappa_ema_prev = self._kappa_ema.get(sid, kappa_t)

            g_ema = self.beta * g_ema_prev + (1 - self.beta) * g_t
            delta_ema = self.beta * delta_ema_prev + (1 - self.beta) * delta_t
            kappa_ema = self.beta * kappa_ema_prev + (1 - self.beta) * kappa_t

            # Eq. 8: unified difficulty score
            tau = g_ema * (1.0 + delta_ema + kappa_ema)

            # Persist state
            self._g_ema[sid] = g_ema
            self._delta_ema[sid] = delta_ema
            self._kappa_ema[sid] = kappa_ema
            self._g_prev[sid] = g_t
            self._loss_prev[sid] = loss_t
            self._iter_count[sid] = self._iter_count.get(sid, 0) + 1

            tau_values[i] = tau

        return tau_values

    def get_tau(self, sample_ids) -> torch.Tensor:
        """Look up the current (EMA) tau for a list of sample ids without updating."""
        out = torch.zeros(len(sample_ids))
        for i, sid in enumerate(sample_ids):
            sid = str(sid)
            g = self._g_ema.get(sid, 0.0)
            d = self._delta_ema.get(sid, 0.0)
            k = self._kappa_ema.get(sid, 0.0)
            out[i] = g * (1.0 + d + k)
        return out

    def get_all_tau(self) -> Dict[str, float]:
        """Return {sample_id: tau} for every sample seen so far. Used by OBM."""
        return {
            sid: self._g_ema[sid]
            * (1.0 + self._delta_ema.get(sid, 0.0) + self._kappa_ema.get(sid, 0.0))
            for sid in self._g_ema
        }

    def batch_expectation(self, tau_batch: torch.Tensor) -> float:
        """E[tau^(t)] over the current mini-batch -- used by OACL (Eq. 12)."""
        return float(tau_batch.mean().item()) if tau_batch.numel() > 0 else self.eps

    def is_warmed_up(self) -> bool:
        """
        Before ~warmup_iters iterations, tau has not accumulated enough
        history to reliably differentiate hard vs. easy samples (see paper
        'Limitations', point 2: Cold-start bias of tau in early training).
        """
        return self.global_iter >= self.config.warmup_iters

    def reset(self):
        self._g_ema.clear()
        self._loss_prev.clear()
        self._g_prev.clear()
        self._delta_ema.clear()
        self._kappa_ema.clear()
        self._iter_count.clear()
        self.global_iter = 0

    def state_dict(self) -> dict:
        return {
            "g_ema": self._g_ema,
            "loss_prev": self._loss_prev,
            "g_prev": self._g_prev,
            "delta_ema": self._delta_ema,
            "kappa_ema": self._kappa_ema,
            "iter_count": self._iter_count,
            "global_iter": self.global_iter,
            "config": self.config,
        }

    def load_state_dict(self, state: dict):
        self._g_ema = state["g_ema"]
        self._loss_prev = state["loss_prev"]
        self._g_prev = state["g_prev"]
        self._delta_ema = state["delta_ema"]
        self._kappa_ema = state["kappa_ema"]
        self._iter_count = state["iter_count"]
        self.global_iter = state["global_iter"]


def partition_head_tail(tau_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Partitions a batch into optimization-head / optimization-tail subsets
    using the moving threshold E[tau^(t)] (paper: "samples satisfying
    tau_i^(t) > E[tau^(t)] are treated as optimization-tail samples").
    """
    threshold = tau_batch.mean()
    tail_mask = tau_batch > threshold
    head_mask = ~tail_mask
    return {"threshold": threshold, "tail_mask": tail_mask, "head_mask": head_mask}


def feature_level_importance(
    grad_norms_per_level: torch.Tensor, stability_per_level: torch.Tensor
) -> torch.Tensor:
    """
    Eq. 3: alpha_l^(t) = g_l^(t) * (1 + s_l^(t))

    Parameters
    ----------
    grad_norms_per_level : (L,) tensor, one entry per feature pyramid level
    stability_per_level  : (L,) tensor, s_l^(t) = |g_l^(t) - g_l^(t-1)|  (Eq. 2)

    Returns
    -------
    alpha : (L,) unnormalized importance scores per feature level.
    """
    return grad_norms_per_level * (1.0 + stability_per_level)


def normalize_importance(alpha: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. 4 normalization: alpha_l / (sum_k alpha_k + eps)."""
    return alpha / (alpha.sum() + eps)
