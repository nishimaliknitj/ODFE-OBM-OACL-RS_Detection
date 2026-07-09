"""
Evaluation Metrics
====================

Implements Precision, Recall, Average Precision (AP), and mean Average
Precision (mAP), following Eqs. 20-24 of the paper:

    P = TP / (TP + FP)                                                  (Eq. 20)
    R = TP / (TP + FN)                                                  (Eq. 21)
    AP = integral_0^1 P(R) dR                                            (Eq. 22)
    mAP = (1/N) * sum_i AP_i                                             (Eq. 23)
    IoU = area(X ∩ Y) / area(X ∪ Y)                                       (Eq. 24)

Supports both mAP@0.5 (single IoU threshold) and mAP@0.5:0.95 (averaged
over IoU thresholds 0.5 to 0.95 in steps of 0.05), as used throughout the
paper's result tables.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU (Eq. 24) between two sets of axis-aligned boxes.

    Parameters
    ----------
    boxes1 : (N, 4) array, [x1, y1, x2, y2]
    boxes2 : (M, 4) array, [x1, y1, x2, y2]

    Returns
    -------
    iou : (N, M) array
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-10, None)


def precision_recall(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes cumulative Precision/Recall arrays for one class on one image
    (or pre-concatenated across a dataset), following Eq. 20/21.

    Parameters
    ----------
    pred_boxes  : (N, 4) predicted boxes, sorted by descending confidence
    pred_scores : (N,) confidence scores (same order as pred_boxes)
    gt_boxes    : (M, 4) ground-truth boxes for this class
    iou_threshold : float

    Returns
    -------
    precision, recall : arrays of length N (cumulative at each prediction rank)
    """
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    num_gt = len(gt_boxes)
    matched = np.zeros(num_gt, dtype=bool)

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    if num_gt == 0:
        fp[:] = 1
    else:
        ious = box_iou_matrix(pred_boxes, gt_boxes)
        for i in range(len(pred_boxes)):
            if ious.shape[1] == 0:
                fp[i] = 1
                continue
            best_gt = np.argmax(ious[i])
            best_iou = ious[i, best_gt]
            if best_iou >= iou_threshold and not matched[best_gt]:
                tp[i] = 1
                matched[best_gt] = True
            else:
                fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-10, None)  # Eq. 20
    recall = tp_cum / np.clip(num_gt, 1e-10, None)  # Eq. 21
    return precision, recall


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    11-point / all-point interpolated Average Precision (Eq. 22), using the
    standard PASCAL VOC 2010+ all-point interpolation (monotonically
    decreasing precision envelope).
    """
    # Prepend/append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing (envelope)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate area under the P-R curve where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_map(
    all_pred_boxes: Dict[int, List[np.ndarray]],
    all_pred_scores: Dict[int, List[np.ndarray]],
    all_gt_boxes: Dict[int, List[np.ndarray]],
    num_classes: int,
    iou_thresholds: List[float] = None,
) -> Dict[str, float]:
    """
    Computes mAP (Eq. 23) across all classes at one or more IoU thresholds.

    Parameters
    ----------
    all_pred_boxes  : {class_id: [per-image (N_i, 4) pred boxes]}
    all_pred_scores : {class_id: [per-image (N_i,) pred scores]}
    all_gt_boxes    : {class_id: [per-image (M_i, 4) gt boxes]}
    num_classes     : total number of classes N
    iou_thresholds  : list of IoU thresholds. Default [0.5] for mAP@0.5.
                       Pass [0.5, 0.55, ..., 0.95] for mAP@0.5:0.95.

    Returns
    -------
    dict with keys: "mAP", "per_class_AP" (dict), and "per_threshold_mAP".
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    per_threshold_maps = []
    per_class_ap_at_50 = {}

    for thr in iou_thresholds:
        aps = []
        for c in range(num_classes):
            pred_boxes_c = all_pred_boxes.get(c, [])
            pred_scores_c = all_pred_scores.get(c, [])
            gt_boxes_c = all_gt_boxes.get(c, [])

            # Concatenate across images with a running image-id offset so
            # boxes from different images are never matched to each other.
            pb_list, ps_list, gb_list = [], [], []
            for img_idx, (pb, ps, gb) in enumerate(zip(pred_boxes_c, pred_scores_c, gt_boxes_c)):
                pb_list.append(pb)
                ps_list.append(ps)
                gb_list.append(gb)

            if len(pb_list) == 0:
                aps.append(0.0)
                continue

            # Evaluate per-image then merge P/R rankings by score across images
            all_pb = (
                np.concatenate(pb_list, axis=0)
                if any(len(x) for x in pb_list)
                else np.zeros((0, 4))
            )
            all_ps = (
                np.concatenate(ps_list, axis=0) if any(len(x) for x in ps_list) else np.zeros((0,))
            )

            # Flatten GT with image offsets to avoid cross-image matching
            gt_flat = []
            img_ids_for_gt = []
            for img_idx, gb in enumerate(gb_list):
                for box in gb:
                    gt_flat.append(box)
                    img_ids_for_gt.append(img_idx)
            gt_flat = np.array(gt_flat).reshape(-1, 4) if gt_flat else np.zeros((0, 4))

            pred_img_ids = []
            for img_idx, pb in enumerate(pb_list):
                pred_img_ids.extend([img_idx] * len(pb))
            pred_img_ids = np.array(pred_img_ids)

            precision, recall = _per_image_matched_pr(
                all_pb, all_ps, pred_img_ids, gt_flat, np.array(img_ids_for_gt), thr
            )
            ap = compute_ap(precision, recall)
            aps.append(ap)

            if abs(thr - 0.5) < 1e-6:
                per_class_ap_at_50[c] = ap

        per_threshold_maps.append(float(np.mean(aps)) if aps else 0.0)

    return {
        "mAP": float(np.mean(per_threshold_maps)),
        "mAP@0.5": per_threshold_maps[0] if abs(iou_thresholds[0] - 0.5) < 1e-6 else None,
        "per_class_AP@0.5": per_class_ap_at_50,
        "per_threshold_mAP": dict(zip(iou_thresholds, per_threshold_maps)),
    }


def _per_image_matched_pr(
    pred_boxes, pred_scores, pred_img_ids, gt_boxes, gt_img_ids, iou_threshold
):
    """Helper: computes P/R ensuring predictions only match GT within the same image."""
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_img_ids = pred_img_ids[order]

    num_gt = len(gt_boxes)
    matched = np.zeros(num_gt, dtype=bool)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i in range(len(pred_boxes)):
        same_img_mask = gt_img_ids == pred_img_ids[i]
        candidate_idx = np.where(same_img_mask & ~matched)[0]
        if len(candidate_idx) == 0:
            fp[i] = 1
            continue
        ious = box_iou_matrix(pred_boxes[i : i + 1], gt_boxes[candidate_idx])[0]
        best = np.argmax(ious)
        if ious[best] >= iou_threshold:
            tp[i] = 1
            matched[candidate_idx[best]] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-10, None)
    recall = tp_cum / np.clip(num_gt, 1e-10, None)
    return precision, recall
