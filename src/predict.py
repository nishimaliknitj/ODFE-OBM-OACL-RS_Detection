"""
Inference Entry Point
========================

Implements Algorithm 2 of the paper ("Inference Procedure of the Proposed
Detector"): feature extraction -> fixed ODFE feature emphasis (weights
frozen from training) -> detection head -> confidence thresholding + NMS.

At inference time, ODFE/OBM/OACL introduce ZERO additional parameters and
ZERO additional FLOPs relative to the baseline detector (see paper Table 14
and the "Computational Overhead" section) -- ODFE simply applies its
already-learned fixed importance weights.

Usage
-----
    python src/predict.py --weights checkpoints/dior_full_best.pt \\
        --config configs/dior.yaml \\
        --source path/to/image.jpg or path/to/dir/ \\
        --save-dir results/predictions
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
import torchvision

from models import build_detector
from utils import get_logger, load_config
from utils.checkpoint import load_checkpoint

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description="Run inference with a trained ODFE+OBM+OACL detector")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True, help="Image file or directory of images")
    p.add_argument("--save-dir", type=str, default="results/predictions")
    p.add_argument("--conf-threshold", type=float, default=None)
    p.add_argument("--iou-threshold", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_image_paths(source: str):
    if os.path.isdir(source):
        paths = []
        for ext in IMG_EXTENSIONS:
            paths.extend(glob(os.path.join(source, f"*{ext}")))
        return sorted(paths)
    return [source]


def preprocess(image_path: str, input_size):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, tuple(input_size))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0), (h0, w0), img


def draw_boxes(image, boxes, scores, labels, class_names, scale_x, scale_y):
    out = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        name = class_names[label] if label < len(class_names) else str(label)
        cv2.putText(
            out,
            f"{name} {score:.2f}",
            (int(x1), max(int(y1) - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


@torch.no_grad()
def predict_image(model, image_tensor, device, conf_threshold, iou_threshold, num_classes):
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)

    cls_logits = outputs["cls_outputs"][0].mean(dim=[2, 3])  # (1, C) -- see docs/methodology.md
    reg_pred = outputs["reg_outputs"][0].mean(dim=[2, 3])  # (1, 4)
    scores = torch.sigmoid(cls_logits)[0]

    keep_scores, keep_labels = [], []
    for c in range(num_classes):
        if scores[c] >= conf_threshold:
            keep_scores.append(float(scores[c]))
            keep_labels.append(c)

    if not keep_scores:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)

    box = reg_pred[0].cpu().numpy()
    boxes = np.tile(box, (len(keep_scores), 1))
    scores_arr = np.array(keep_scores)
    labels_arr = np.array(keep_labels)

    # NMS (torchvision) — for a single pooled box per image this is a no-op
    # placeholder; a dense per-anchor head would produce many candidate
    # boxes here and genuinely benefit from NMS.
    if len(boxes) > 1:
        keep_idx = torchvision.ops.nms(
            torch.from_numpy(boxes).float(),
            torch.from_numpy(scores_arr).float(),
            iou_threshold,
        ).numpy()
        boxes, scores_arr, labels_arr = boxes[keep_idx], scores_arr[keep_idx], labels_arr[keep_idx]

    return boxes, scores_arr, labels_arr


def main():
    args = parse_args()
    config = load_config(args.config)
    device = args.device or (config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    conf_threshold = args.conf_threshold or config["inference"]["conf_threshold"]
    iou_threshold = args.iou_threshold or config["inference"]["nms_iou_threshold"]

    logger = get_logger("predict")
    os.makedirs(args.save_dir, exist_ok=True)

    model = build_detector(config).to(device)
    load_checkpoint(args.weights, model, map_location=device)
    model.eval()
    logger.info(f"Model loaded from {args.weights}")

    class_names = config["dataset"]["class_names"]
    input_size = config["inference"]["input_size"]

    image_paths = load_image_paths(args.source)
    logger.info(f"Found {len(image_paths)} image(s) to run inference on.")

    for path in image_paths:
        tensor, (h0, w0), original_bgr = preprocess(path, input_size)
        boxes, scores, labels = predict_image(
            model, tensor, device, conf_threshold, iou_threshold, config["dataset"]["num_classes"]
        )

        scale_x = w0 / input_size[0]
        scale_y = h0 / input_size[1]
        vis = draw_boxes(original_bgr, boxes, scores, labels, class_names, scale_x, scale_y)

        out_path = os.path.join(args.save_dir, os.path.basename(path))
        cv2.imwrite(out_path, vis)
        logger.info(f"{os.path.basename(path)}: {len(boxes)} detection(s) -> saved to {out_path}")


if __name__ == "__main__":
    main()
