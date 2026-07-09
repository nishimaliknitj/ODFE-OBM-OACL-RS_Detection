"""
Combined Training Objective
==============================

    L_final = L_loc + lambda * L_cls                                    (Eq. 16)
    L_final = L_loc + lambda * L_cls + mu * L_inv                        (Eq. 19, optional)

Where L_loc is CIoU (src/losses/ciou.py) and L_cls is OACL
(src/losses/oacl_loss.py). This module wires them together with the
tau tracker into a single callable used by the training loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..models.tau import OptimizationDifficultyTracker
from .ciou import CIoULoss
from .oacl_loss import InvarianceRegularizationLoss, OptimizationAwareClassificationLoss


class CombinedDetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        loc_weight: float = 1.0,
        cls_weight: float = 0.5,  # lambda
        max_sample_weight: float = 3.0,  # w_max
        invariance_enabled: bool = False,
        invariance_coefficient: float = 0.01,  # mu
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loc_weight = loc_weight
        self.cls_weight = cls_weight

        self.loc_loss_fn = CIoULoss(reduction="mean")
        self.cls_loss_fn = OptimizationAwareClassificationLoss(max_sample_weight=max_sample_weight)

        self.invariance_enabled = invariance_enabled
        self.invariance_coefficient = invariance_coefficient
        self.inv_loss_fn = InvarianceRegularizationLoss() if invariance_enabled else None

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        pred_cls_logits: torch.Tensor,
        target_cls: torch.Tensor,
        tau: torch.Tensor,
        tau_batch_expectation: Optional[float] = None,
        features_view_a: Optional[List[torch.Tensor]] = None,
        features_view_b: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with 'loss_loc', 'loss_cls', 'loss_inv' (if enabled),
        and 'loss_total' = L_final.
        """
        loss_loc = self.loc_loss_fn(pred_boxes, target_boxes)
        loss_cls = self.cls_loss_fn(pred_cls_logits, target_cls, tau, tau_batch_expectation)

        loss_total = self.loc_weight * loss_loc + self.cls_weight * loss_cls

        out = {
            "loss_loc": loss_loc,
            "loss_cls": loss_cls,
            "loss_total": loss_total,
        }

        if self.invariance_enabled and features_view_a is not None and features_view_b is not None:
            loss_inv = self.inv_loss_fn(features_view_a, features_view_b)
            loss_total = loss_total + self.invariance_coefficient * loss_inv
            out["loss_inv"] = loss_inv
            out["loss_total"] = loss_total

        return out


def build_loss(config: dict, tau_tracker: OptimizationDifficultyTracker) -> CombinedDetectionLoss:
    oacl_cfg = config.get("oacl", {})
    inv_cfg = config.get("invariance_regularization", {})
    loss_cfg = config.get("loss", {})

    return CombinedDetectionLoss(
        num_classes=config["dataset"]["num_classes"],
        loc_weight=loss_cfg.get("loc_weight", 1.0),
        cls_weight=loss_cfg.get("cls_weight", 0.5),
        max_sample_weight=oacl_cfg.get("max_sample_weight", 3.0),
        invariance_enabled=inv_cfg.get("enabled", False),
        invariance_coefficient=inv_cfg.get("coefficient", 0.01),
    )
