"""
Evaluation Entry Point
=========================

Computes mAP@0.5 and mAP@0.5:0.95 on the validation/test split, following
Eqs. 20-24 of the paper. Also reports per-category AP for the tail
categories declared in the dataset config's `reporting.tail_categories`.

Usage
-----
    python src/test.py --config configs/dior.yaml --weights checkpoints/dior_full_best.pt
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset, detection_collate_fn
from datasets.augmentations import DetectionAugmentation
from models import build_detector
from utils import compute_map, get_logger, load_config
from utils.checkpoint import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ODFE+OBM+OACL detector")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--conf-threshold", type=float, default=None)
    p.add_argument("--iou-threshold", type=float, default=None)
    p.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device, conf_threshold, num_classes):
    """
    Runs the model over the validation loader and collects per-class
    prediction boxes/scores and ground-truth boxes for mAP computation.

    NOTE: The simplified detection head in src/models/detector.py performs
    global average pooling over each feature level for demonstration
    purposes (see train.py docstring). For full per-anchor/per-pixel
    dense predictions matching the paper's YOLOv8-style head, replace the
    pooled cls/reg outputs below with a proper decode + NMS pipeline
    (see docs/methodology.md for the extension point).
    """
    model.eval()

    all_pred_boxes = defaultdict(list)
    all_pred_scores = defaultdict(list)
    all_gt_boxes = defaultdict(list)

    for c in range(num_classes):
        all_pred_boxes[c] = []
        all_pred_scores[c] = []
        all_gt_boxes[c] = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["images"].to(device)
        outputs = model(images)

        cls_logits = outputs["cls_outputs"][0].mean(dim=[2, 3])  # (B, C)
        reg_pred = outputs["reg_outputs"][0].mean(dim=[2, 3])  # (B, 4)
        scores = torch.sigmoid(cls_logits)

        batch_size = images.shape[0]
        for b in range(batch_size):
            gt_boxes = batch["boxes"][b].numpy()
            gt_labels = batch["labels"][b].numpy()

            pred_box = reg_pred[b].cpu().numpy().reshape(1, 4)

            for c in range(num_classes):
                score_c = float(scores[b, c].item())
                if score_c >= conf_threshold:
                    all_pred_boxes[c].append(pred_box)
                    all_pred_scores[c].append(np.array([score_c]))
                else:
                    all_pred_boxes[c].append(np.zeros((0, 4)))
                    all_pred_scores[c].append(np.zeros((0,)))

                cls_mask = gt_labels == c
                all_gt_boxes[c].append(gt_boxes[cls_mask])

    return all_pred_boxes, all_pred_scores, all_gt_boxes


def main():
    args = parse_args()
    config = load_config(args.config)

    device = args.device or (config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    conf_threshold = args.conf_threshold or config["inference"]["conf_threshold"]

    logger = get_logger("eval")
    logger.info(f"Loading config from {args.config}")
    logger.info(f"Loading weights from {args.weights}")

    dataset_cfg = config["dataset"]
    val_transform = DetectionAugmentation(is_train=False)
    val_dataset = build_dataset(
        dataset_cfg["name"],
        images_dir=dataset_cfg["val_images"],
        annotations_dir=dataset_cfg["val_annotations"],
        class_names=dataset_cfg["class_names"],
        input_size=tuple(dataset_cfg["input_size"]),
        is_train=False,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        collate_fn=detection_collate_fn,
    )
    logger.info(f"Validation samples: {len(val_dataset)}")

    model = build_detector(config).to(device)
    load_checkpoint(args.weights, model, map_location=device)
    logger.info("Weights loaded successfully.")

    all_pred_boxes, all_pred_scores, all_gt_boxes = run_inference(
        model, val_loader, device, conf_threshold, dataset_cfg["num_classes"]
    )

    results_50 = compute_map(
        all_pred_boxes,
        all_pred_scores,
        all_gt_boxes,
        num_classes=dataset_cfg["num_classes"],
        iou_thresholds=[0.5],
    )
    results_50_95 = compute_map(
        all_pred_boxes,
        all_pred_scores,
        all_gt_boxes,
        num_classes=dataset_cfg["num_classes"],
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )

    logger.info(f"mAP@0.5      = {results_50['mAP'] * 100:.2f}%")
    logger.info(f"mAP@0.5:0.95 = {results_50_95['mAP'] * 100:.2f}%")

    class_names = dataset_cfg["class_names"]
    tail_categories = config["dataset"].get("reporting", {}).get("tail_categories", [])
    logger.info("Per-category AP@0.5:")
    for c, ap in results_50["per_class_AP@0.5"].items():
        name = class_names[c] if c < len(class_names) else str(c)
        marker = " [TAIL]" if name in tail_categories else ""
        logger.info(f"  {name:30s}: {ap * 100:5.2f}%{marker}")

    output = {
        "mAP@0.5": results_50["mAP"],
        "mAP@0.5:0.95": results_50_95["mAP"],
        "per_class_AP@0.5": {
            class_names[c] if c < len(class_names) else str(c): ap
            for c, ap in results_50["per_class_AP@0.5"].items()
        },
        "config": args.config,
        "weights": args.weights,
    }

    output_path = args.output or os.path.join(
        config["experiment"]["results_dir"], f"{config['experiment']['name']}_eval_results.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
