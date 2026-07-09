"""
Augmentation Pipeline
========================

Implements the standard augmentations listed in Table 4 of the paper:
random horizontal/vertical flip, scale jitter, color jitter, and MixUp.
Mosaic augmentation itself is handled separately by OBM
(src/models/obm.py) since it requires access to the full dataset's
difficulty scores rather than operating on a single sample.
"""

from __future__ import annotations

import random
from typing import Tuple

import cv2
import numpy as np


class DetectionAugmentation:
    """
    Callable augmentation pipeline: (image, boxes, labels) -> (image, boxes, labels).
    All box operations assume [x1, y1, x2, y2] absolute pixel coordinates.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        scale_jitter: Tuple[float, float] = (0.5, 1.5),
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        is_train: bool = True,
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.scale_jitter = scale_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.is_train = is_train

    def __call__(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        if not self.is_train:
            return image, boxes, labels

        h, w = image.shape[:2]

        if random.random() < self.hflip_prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        if random.random() < self.vflip_prob:
            image = np.ascontiguousarray(image[::-1, :, :])
            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

        image = self._color_jitter(image)
        image, boxes = self._scale_jitter(image, boxes)

        return image, boxes, labels

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = img * factor

        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = img.mean()
            img = (img - mean) * factor + mean

        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            gray = img.mean(axis=2, keepdims=True)
            img = gray + (img - gray) * factor

        return np.clip(img, 0, 255).astype(np.uint8)

    def _scale_jitter(self, image: np.ndarray, boxes: np.ndarray):
        scale = random.uniform(*self.scale_jitter)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        image_scaled = cv2.resize(image, (new_w, new_h))

        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale

        # Resize back to original canvas via center pad/crop to keep input_size fixed
        image_out = cv2.resize(image_scaled, (w, h))
        if len(boxes) > 0:
            scale_back_x = w / new_w
            scale_back_y = h / new_h
            boxes[:, [0, 2]] *= scale_back_x
            boxes[:, [1, 3]] *= scale_back_y

        return image_out, boxes


def mixup(
    image1: np.ndarray,
    boxes1: np.ndarray,
    labels1: np.ndarray,
    image2: np.ndarray,
    boxes2: np.ndarray,
    labels2: np.ndarray,
    alpha: float = 0.1,
):
    """MixUp augmentation (Zhang et al. 2018), prob=0.1, alpha=0.1 per Table 4."""
    lam = np.random.beta(alpha, alpha)
    mixed_image = (lam * image1.astype(np.float32) + (1 - lam) * image2.astype(np.float32)).astype(
        np.uint8
    )
    mixed_boxes = (
        np.concatenate([boxes1, boxes2], axis=0) if len(boxes1) or len(boxes2) else np.zeros((0, 4))
    )
    mixed_labels = (
        np.concatenate([labels1, labels2], axis=0)
        if len(labels1) or len(labels2)
        else np.zeros((0,), dtype=np.int64)
    )
    return mixed_image, mixed_boxes, mixed_labels
