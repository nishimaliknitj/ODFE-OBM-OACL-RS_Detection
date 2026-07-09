"""Unit tests for src/losses/ciou.py -- CIoU localization loss (Eq. 17)."""

import torch

from losses.ciou import box_iou_xyxy, ciou_loss, CIoULoss


def test_iou_identical_boxes_is_one():
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    iou = box_iou_xyxy(boxes, boxes)
    assert torch.allclose(iou, torch.tensor([1.0]), atol=1e-5)


def test_iou_disjoint_boxes_is_zero():
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    iou = box_iou_xyxy(b1, b2)
    assert torch.allclose(iou, torch.tensor([0.0]), atol=1e-5)


def test_ciou_loss_zero_for_perfect_match():
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    loss = ciou_loss(boxes, boxes)
    assert loss.item() < 1e-4


def test_ciou_loss_positive_for_mismatched_boxes():
    pred = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    target = torch.tensor([[30.0, 30.0, 90.0, 90.0]])
    loss = ciou_loss(pred, target)
    assert loss.item() > 0


def test_ciou_loss_decreases_as_boxes_converge():
    target = torch.tensor([[20.0, 20.0, 60.0, 60.0]])

    far_pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    close_pred = torch.tensor([[18.0, 18.0, 58.0, 58.0]])

    loss_far = ciou_loss(far_pred, target)
    loss_close = ciou_loss(close_pred, target)

    assert loss_close.item() < loss_far.item()


def test_ciou_module_wrapper_matches_functional():
    pred = torch.tensor([[5.0, 5.0, 25.0, 25.0]])
    target = torch.tensor([[10.0, 10.0, 30.0, 30.0]])

    module_loss = CIoULoss()(pred, target)
    functional_loss = ciou_loss(pred, target)

    assert torch.allclose(module_loss, functional_loss)


def test_ciou_loss_batch_reduction_modes():
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 20.0, 20.0]])
    target = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 25.0, 25.0]])

    loss_none = ciou_loss(pred, target, reduction="none")
    loss_mean = ciou_loss(pred, target, reduction="mean")
    loss_sum = ciou_loss(pred, target, reduction="sum")

    assert loss_none.shape == (2,)
    assert torch.allclose(loss_mean, loss_none.mean())
    assert torch.allclose(loss_sum, loss_none.sum())
