"""Unit tests for src/utils/metrics.py -- Precision/Recall/AP/mAP (Eqs. 20-24)."""

import numpy as np

from utils.metrics import box_iou_matrix, precision_recall, compute_ap


def test_box_iou_matrix_shape():
    b1 = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    b2 = np.array([[0, 0, 10, 10]])
    iou = box_iou_matrix(b1, b2)
    assert iou.shape == (2, 1)


def test_box_iou_matrix_empty_inputs():
    iou = box_iou_matrix(np.zeros((0, 4)), np.zeros((0, 4)))
    assert iou.shape == (0, 0)


def test_perfect_predictions_give_ap_one():
    gt_boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    pred_boxes = gt_boxes.copy()
    pred_scores = np.array([0.9, 0.95])

    precision, recall = precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
    ap = compute_ap(precision, recall)
    assert ap > 0.99


def test_no_predictions_give_ap_zero():
    gt_boxes = np.array([[0, 0, 10, 10]])
    pred_boxes = np.zeros((0, 4))
    pred_scores = np.zeros((0,))

    precision, recall = precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
    ap = compute_ap(precision, recall)
    assert ap == 0.0


def test_all_false_positives_give_low_ap():
    gt_boxes = np.array([[0, 0, 10, 10]])
    pred_boxes = np.array([[100, 100, 110, 110]])  # far away, no overlap
    pred_scores = np.array([0.9])

    precision, recall = precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
    ap = compute_ap(precision, recall)
    assert ap == 0.0


def test_compute_ap_monotonic_envelope():
    """AP should be well-defined (in [0,1]) even with a non-monotonic raw P-R curve."""
    precision = np.array([1.0, 0.5, 0.66, 0.75])
    recall = np.array([0.25, 0.25, 0.5, 0.75])
    ap = compute_ap(precision, recall)
    assert 0.0 <= ap <= 1.0
