"""
Base Dataset for Remote Sensing Object Detection
====================================================

Common functionality shared by DIOR / DOTA / FAIR1M loaders:
    - stable sample-id assignment (required by the tau tracker, which
      accumulates per-sample EMA state across epochs)
    - standard augmentation pipeline (flip, scale jitter, color jitter)
    - OBM-driven mosaic hook (optional)
    - collate function producing batches with sample ids attached
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RemoteSensingDetectionDataset(Dataset):
    """
    Abstract base class. Subclasses (DIORDataset, DOTADataset, FAIR1MDataset)
    must implement `_load_annotations()` to populate `self.samples`, a list
    of dicts: {"id": str, "image_path": str, "boxes": (N,4) np.ndarray,
    "labels": (N,) np.ndarray}.
    """

    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        class_names: List[str],
        input_size: Tuple[int, int] = (640, 640),
        is_train: bool = True,
        transform: Optional[Callable] = None,
        obm_sampler=None,  # OptimizationBalancedMosaicSampler, injected by train.py
        tau_tracker=None,  # OptimizationDifficultyTracker, injected by train.py
    ):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.input_size = input_size
        self.is_train = is_train
        self.transform = transform
        self.obm_sampler = obm_sampler
        self.tau_tracker = tau_tracker

        self.samples: List[Dict] = []
        self._load_annotations()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found under images_dir={images_dir}, "
                f"annotations_dir={annotations_dir}. "
                f"Did you run the corresponding scripts/download_*.sh script? "
                f"See docs/data_preparation.md."
            )

    # ------------------------------------------------------------------
    def _load_annotations(self):
        raise NotImplementedError("Subclasses must implement _load_annotations()")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _resize(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h0, w0 = image.shape[:2]
        th, tw = self.input_size
        image_r = cv2.resize(image, (tw, th))

        if len(boxes) > 0:
            scale_x, scale_y = tw / w0, th / h0
            boxes = boxes.copy().astype(np.float32)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        return image_r, boxes

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image_path"])
        boxes = sample["boxes"].copy()
        labels = sample["labels"].copy()

        image, boxes = self._resize(image, boxes)

        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return {
            "id": sample["id"],
            "image": image_tensor,
            "boxes": torch.from_numpy(boxes).float(),
            "labels": torch.from_numpy(labels).long(),
            "image_path": sample["image_path"],
        }

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def get_sample_ids(self) -> List[str]:
        return [s["id"] for s in self.samples]


def detection_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function: images stack normally, but boxes/labels are
    variable-length per image, so they stay as a list. Sample ids are kept
    for tau-tracker lookups during training.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    ids = [b["id"] for b in batch]
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    image_paths = [b["image_path"] for b in batch]

    return {
        "ids": ids,
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "image_paths": image_paths,
    }
