"""
DOTA Dataset Loader
=====================

DOTA (Xia et al., "DOTA: A large-scale dataset for object detection in
aerial images", CVPR 2018) uses oriented bounding boxes stored in
`labelTxt/*.txt` files, one line per instance:

    x1 y1 x2 y2 x3 y3 x4 y4 category difficult

This loader parses the 8-point polygon and converts it to its axis-aligned
bounding rectangle (min/max envelope) for use with the axis-aligned CIoU
detector head implemented in this repository. For full oriented-bounding-box
(OBB) training, swap in an OBB-aware head/loss (see docs/methodology.md).

Expected directory layout:

    data/DOTA/
    ├── train/
    │   ├── images/*.png
    │   └── labelTxt/*.txt
    └── val/
        ├── images/*.png
        └── labelTxt/*.txt
"""

from __future__ import annotations

import os
from glob import glob

import numpy as np

from .base_dataset import RemoteSensingDetectionDataset

DOTA_CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]


class DOTADataset(RemoteSensingDetectionDataset):
    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        class_names=None,
        input_size=(640, 640),
        is_train: bool = True,
        transform=None,
        obm_sampler=None,
        tau_tracker=None,
    ):
        super().__init__(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            class_names=class_names or DOTA_CLASSES,
            input_size=input_size,
            is_train=is_train,
            transform=transform,
            obm_sampler=obm_sampler,
            tau_tracker=tau_tracker,
        )

    def _load_annotations(self):
        txt_files = sorted(glob(os.path.join(self.annotations_dir, "*.txt")))

        for txt_path in txt_files:
            image_id = os.path.splitext(os.path.basename(txt_path))[0]

            image_path = None
            for ext in (".png", ".jpg", ".jpeg", ".tif"):
                candidate = os.path.join(self.images_dir, f"{image_id}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path is None:
                continue

            boxes, labels = self._parse_labeltxt(txt_path)

            self.samples.append(
                {
                    "id": f"DOTA_{image_id}",
                    "image_path": image_path,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

    def _parse_labeltxt(self, txt_path: str):
        boxes, labels = [], []
        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            # Skip DOTA header lines ("imagesource:", "gsd:")
            if len(parts) < 9 or parts[0].startswith(("imagesource", "gsd")):
                continue

            coords = list(map(float, parts[:8]))
            category = parts[8].lower()

            if category not in self.class_to_idx:
                continue

            xs = coords[0::2]
            ys = coords[1::2]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_idx[category])

        boxes_arr = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        labels_arr = np.array(labels, dtype=np.int64)
        return boxes_arr, labels_arr
