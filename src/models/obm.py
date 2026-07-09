"""
OBM: Optimization-Balanced Mosaic Augmentation
=================================================

Reframes mosaic construction as a probabilistic sampling process guided by
per-sample optimization difficulty tau_i^(t), rather than class-frequency
statistics used by conventional instance-balanced mosaic strategies.

    P_i = tau_i^(t) / sum_j tau_j^(t)                                   (Eq. 10)

Samples with high optimization difficulty are more frequently selected
for mosaic construction, increasing their effective training exposure --
regardless of their semantic class frequency.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


class OptimizationBalancedMosaicSampler:
    """
    Given a pool of per-sample difficulty scores (tau), samples 4 indices
    at a time (with replacement, weighted by tau) to construct a mosaic,
    following Figure 5 of the paper: (1) construct mini-batch, (2) compute
    difficulty-aligned sampling distribution, (3) generate mosaic.
    """

    def __init__(self, temperature: float = 2.0, eps: float = 1e-6):
        """
        Parameters
        ----------
        temperature : float
            Softens (T > 1) or sharpens (T < 1) the difficulty-proportional
            distribution: P_i ∝ tau_i^(1/T). T=1.0 reproduces Eq. 10 exactly.
        eps : float
            Numerical stability constant.
        """
        self.temperature = temperature
        self.eps = eps

    def compute_sampling_distribution(self, tau_by_id: Dict[str, float]) -> Dict[str, float]:
        """Eq. 10: P_i = tau_i / sum_j tau_j, with optional temperature."""
        ids = list(tau_by_id.keys())
        tau_vals = np.array([max(tau_by_id[i], self.eps) for i in ids], dtype=np.float64)

        if self.temperature != 1.0:
            tau_vals = np.power(tau_vals, 1.0 / self.temperature)

        probs = tau_vals / (tau_vals.sum() + self.eps)
        return dict(zip(ids, probs))

    def sample_mosaic_group(self, tau_by_id: Dict[str, float], group_size: int = 4) -> List[str]:
        """
        Draws `group_size` sample ids (with replacement) according to the
        difficulty-proportional distribution, for constructing one mosaic
        image (standard 2x2 mosaic uses group_size=4).
        """
        dist = self.compute_sampling_distribution(tau_by_id)
        ids = list(dist.keys())
        probs = np.array(list(dist.values()))
        probs = probs / probs.sum()  # renormalize for float safety

        chosen = np.random.choice(ids, size=group_size, replace=True, p=probs)
        return chosen.tolist()

    def sample_batch_indices(self, tau_by_id: Dict[str, float], batch_size: int) -> List[str]:
        """Standard (non-mosaic) difficulty-proportional batch sampling."""
        dist = self.compute_sampling_distribution(tau_by_id)
        ids = list(dist.keys())
        probs = np.array(list(dist.values()))
        probs = probs / probs.sum()
        chosen = np.random.choice(ids, size=batch_size, replace=True, p=probs)
        return chosen.tolist()


def build_mosaic_image(
    images: Sequence[np.ndarray],
    boxes: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    output_size: int = 640,
):
    """
    Standard 2x2 mosaic stitching (Figure 5, Step 3):
        I'(x, y) = sum_{k=1}^{4} 1[(x,y) in R_k] * I_k(x, y)

    Combines 4 (image, boxes, labels) triples selected by
    `OptimizationBalancedMosaicSampler.sample_mosaic_group` into a single
    mosaic training sample, adjusting bounding box coordinates for each
    quadrant.

    Parameters
    ----------
    images : sequence of 4 HxWx3 uint8 arrays
    boxes  : sequence of 4 (N_k, 4) arrays in [x1, y1, x2, y2] format
    labels : sequence of 4 (N_k,) arrays of class indices
    output_size : int, side length of the square output mosaic

    Returns
    -------
    mosaic_img : (output_size, output_size, 3) uint8 array
    mosaic_boxes : (N_total, 4) array
    mosaic_labels : (N_total,) array
    """
    assert len(images) == 4, "OBM mosaic requires exactly 4 images (2x2 grid)"

    s = output_size
    mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)  # 114 = YOLO gray padding
    all_boxes, all_labels = [], []

    # Quadrant placement: top-left, top-right, bottom-left, bottom-right
    placements = [(0, 0), (s, 0), (0, s), (s, s)]

    for k, (img, box, lab) in enumerate(zip(images, boxes, labels)):
        ox, oy = placements[k]
        h0, w0 = img.shape[:2]
        scale = s / max(h0, w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)

        # Resize (assumes cv2 available at call site; kept import-light here)
        import cv2  # local import to keep module import light when cv2 is unused elsewhere

        resized = cv2.resize(img, (new_w, new_h))
        mosaic_img[oy : oy + new_h, ox : ox + new_w] = resized

        if box is not None and len(box) > 0:
            b = box.copy().astype(np.float64)
            b[:, [0, 2]] = b[:, [0, 2]] * scale + ox
            b[:, [1, 3]] = b[:, [1, 3]] * scale + oy
            all_boxes.append(b)
            all_labels.append(lab)

    # Crop back to a single output_size x output_size canvas (center crop of the 2x canvas)
    crop_x0, crop_y0 = s // 2, s // 2
    mosaic_img = mosaic_img[crop_y0 : crop_y0 + s, crop_x0 : crop_x0 + s]

    if all_boxes:
        mosaic_boxes = np.concatenate(all_boxes, axis=0)
        mosaic_boxes[:, [0, 2]] -= crop_x0
        mosaic_boxes[:, [1, 3]] -= crop_y0
        mosaic_boxes = np.clip(mosaic_boxes, 0, s)
        mosaic_labels = np.concatenate(all_labels, axis=0)

        # Discard degenerate boxes produced by the crop
        valid = (mosaic_boxes[:, 2] > mosaic_boxes[:, 0]) & (
            mosaic_boxes[:, 3] > mosaic_boxes[:, 1]
        )
        mosaic_boxes = mosaic_boxes[valid]
        mosaic_labels = mosaic_labels[valid]
    else:
        mosaic_boxes = np.zeros((0, 4), dtype=np.float64)
        mosaic_labels = np.zeros((0,), dtype=np.int64)

    return mosaic_img, mosaic_boxes, mosaic_labels
