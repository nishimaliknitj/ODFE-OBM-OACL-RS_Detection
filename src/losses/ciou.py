"""
CIoU Localization Loss
========================

Complete Intersection over Union bounding-box regression loss
(Zheng et al., "Distance-IoU Loss", AAAI 2020), used as L_loc in

    L_final = L_loc + lambda * L_cls                                    (Eq. 16)

    L_loc = 1 - IoU + rho^2(b, b_gt) / c^2 + alpha_v * v                 (Eq. 17)

where:
    IoU        : standard Intersection-over-Union
    rho(b,b_gt): Euclidean distance between box centers
    c          : diagonal length of the smallest enclosing box
    v          : aspect-ratio consistency term
    alpha_v    : trade-off weight, alpha_v = v / ((1 - IoU) + v)
"""

from __future__ import annotations

import math

import torch


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise IoU between two sets of boxes in [x1, y1, x2, y2] format, same shape (N, 4)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = area1 + area2 - inter_area + eps
    return inter_area / union_area


def ciou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Parameters
    ----------
    pred_boxes, target_boxes : (N, 4) tensors in [x1, y1, x2, y2] format.
    reduction : "mean" | "sum" | "none"

    Returns
    -------
    loss : CIoU loss per Eq. 17.
    """
    iou = box_iou_xyxy(pred_boxes, target_boxes, eps)

    # Center distance rho^2(b, b_gt)
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    tgt_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    tgt_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    rho2 = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

    # Smallest enclosing box diagonal c^2
    enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Aspect ratio consistency term v
    pred_w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=eps)
    pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=eps)
    tgt_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=eps)
    tgt_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=eps)

    v = (4 / (math.pi**2)) * torch.pow(torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h), 2)

    with torch.no_grad():
        alpha_v = v / ((1 - iou) + v + eps)

    ciou = iou - (rho2 / c2) - alpha_v * v
    loss = 1 - ciou

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class CIoULoss(torch.nn.Module):
    """nn.Module wrapper around `ciou_loss` for use inside a training loop / loss registry."""

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        return ciou_loss(pred_boxes, target_boxes, eps=self.eps, reduction=self.reduction)
