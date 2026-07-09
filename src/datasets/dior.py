"""
DIOR Dataset Loader
=====================

DIOR (Li et al., "Object detection in optical remote sensing images: A
survey and a new benchmark", ISPRS 2020) uses Pascal-VOC-style XML
annotations, one file per image, with axis-aligned bounding boxes.

Expected directory layout (see docs/data_preparation.md):

    data/DIOR/
    ├── JPEGImages-train/
    │   ├── 00001.jpg
    │   └── ...
    ├── JPEGImages-test/
    ├── Annotations/
    │   ├── train/
    │   │   ├── 00001.xml
    │   │   └── ...
    │   └── test/
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from glob import glob

import numpy as np

from .base_dataset import RemoteSensingDetectionDataset

DIOR_CLASSES = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "expressway-service-area",
    "expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]


class DIORDataset(RemoteSensingDetectionDataset):
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
            class_names=class_names or DIOR_CLASSES,
            input_size=input_size,
            is_train=is_train,
            transform=transform,
            obm_sampler=obm_sampler,
            tau_tracker=tau_tracker,
        )

    def _load_annotations(self):
        xml_files = sorted(glob(os.path.join(self.annotations_dir, "*.xml")))

        for xml_path in xml_files:
            image_id = os.path.splitext(os.path.basename(xml_path))[0]
            image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                continue

            boxes, labels = self._parse_xml(xml_path)

            self.samples.append(
                {
                    "id": f"DIOR_{image_id}",
                    "image_path": image_path,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

    def _parse_xml(self, xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in self.class_to_idx:
                continue
            bnd = obj.find("bndbox")
            x1 = float(bnd.find("xmin").text)
            y1 = float(bnd.find("ymin").text)
            x2 = float(bnd.find("xmax").text)
            y2 = float(bnd.find("ymax").text)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_idx[name])

        boxes_arr = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        labels_arr = np.array(labels, dtype=np.int64)
        return boxes_arr, labels_arr
