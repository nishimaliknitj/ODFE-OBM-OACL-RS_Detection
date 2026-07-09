"""
FAIR1M Dataset Loader
========================

FAIR1M (Sun et al., "FAIR1M: A benchmark dataset for fine-grained object
recognition in high-resolution remote sensing imagery", ISPRS 2022) uses
XML annotations with `<points>` polygon coordinates and fine-grained
`<possibleresult><name>` category labels.

Expected directory layout:

    data/FAIR1M/
    ├── train/
    │   ├── images/*.tif
    │   └── labelXml/*.xml
    └── val/
        ├── images/*.tif
        └── labelXml/*.xml
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from glob import glob

import numpy as np

from .base_dataset import RemoteSensingDetectionDataset

FAIR1M_CLASSES = [
    "Boeing737",
    "Boeing747",
    "Boeing777",
    "Boeing787",
    "A220",
    "A321",
    "A330",
    "A350",
    "ARJ21",
    "other-airplane",
    "Passenger Ship",
    "Motorboat",
    "Fishing Boat",
    "Tugboat",
    "other-ship",
    "Engineering Ship",
    "Liquid Cargo Ship",
    "Dry Cargo Ship",
    "Warship",
    "Small Car",
    "Bus",
    "Cargo Truck",
    "Dump Truck",
    "other-vehicle",
    "Van",
    "Trailer",
    "Tractor",
    "Excavator",
    "Truck Tractor",
    "Baseball Field",
    "Basketball Court",
    "Football Field",
    "Tennis Court",
    "Roundabout",
    "Intersection",
    "Bridge",
    "other",
]


class FAIR1MDataset(RemoteSensingDetectionDataset):
    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        class_names=None,
        input_size=(800, 800),
        is_train: bool = True,
        transform=None,
        obm_sampler=None,
        tau_tracker=None,
    ):
        super().__init__(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            class_names=class_names or FAIR1M_CLASSES,
            input_size=input_size,
            is_train=is_train,
            transform=transform,
            obm_sampler=obm_sampler,
            tau_tracker=tau_tracker,
        )
        # FAIR1M category names are case-sensitive and contain spaces;
        # build a lowercase-normalized lookup for robust matching.
        self._normalized_class_to_idx = {k.lower(): v for k, v in self.class_to_idx.items()}

    def _load_annotations(self):
        xml_files = sorted(glob(os.path.join(self.annotations_dir, "*.xml")))

        for xml_path in xml_files:
            image_id = os.path.splitext(os.path.basename(xml_path))[0]

            image_path = None
            for ext in (".tif", ".png", ".jpg"):
                candidate = os.path.join(self.images_dir, f"{image_id}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path is None:
                continue

            boxes, labels = self._parse_xml(xml_path)

            self.samples.append(
                {
                    "id": f"FAIR1M_{image_id}",
                    "image_path": image_path,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

    def _parse_xml(self, xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes, labels = [], []
        for obj in root.findall(".//object"):
            name_node = obj.find(".//possibleresult/name")
            if name_node is None or name_node.text is None:
                continue
            category = name_node.text.strip()
            category_key = category.lower()

            if category_key not in self._normalized_class_to_idx:
                category_key = "other"
                if category_key not in self._normalized_class_to_idx:
                    continue

            points_node = obj.find(".//points")
            if points_node is None:
                continue

            xs, ys = [], []
            for pt in points_node.findall("point"):
                x_str, y_str = pt.text.strip().split(",")
                xs.append(float(x_str))
                ys.append(float(y_str))

            if not xs:
                continue

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self._normalized_class_to_idx[category_key])

        boxes_arr = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        labels_arr = np.array(labels, dtype=np.int64)
        return boxes_arr, labels_arr
