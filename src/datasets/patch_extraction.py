"""
Patch Extraction Utility
===========================

Splits high-resolution remote sensing images (e.g. raw DOTA images up to
4000x4000) into overlapping fixed-size patches, adjusting bounding box
annotations accordingly. Implements the "patch extraction" preprocessing
step referenced in `configs/dota.yaml -> dataset.patch_extraction` and the
paper's "Dataset-specific notes" (DOTA: 640x640 patches, 256px overlap).

Usage
-----
    python -m src.datasets.patch_extraction \\
        --input data/DOTA/raw/train \\
        --output data/DOTA/train \\
        --patch-size 640 --overlap 256 --format dota
"""
from __future__ import annotations

import argparse
import os
from glob import glob

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Extract overlapping patches from high-res RS images")
    p.add_argument("--input", type=str, required=True, help="Directory with images/ and labelTxt/ subfolders")
    p.add_argument("--output", type=str, required=True, help="Output directory (will contain images/ and labelTxt/)")
    p.add_argument("--patch-size", type=int, default=640)
    p.add_argument("--overlap", type=int, default=256)
    p.add_argument("--format", type=str, default="dota", choices=["dota"])
    return p.parse_args()


def extract_patches_for_image(image: np.ndarray, patch_size: int, overlap: int):
    """Yields (patch_image, (x_offset, y_offset)) tuples covering the full image with overlap."""
    h, w = image.shape[:2]
    stride = patch_size - overlap

    x_starts = list(range(0, max(w - patch_size, 0) + 1, stride)) or [0]
    y_starts = list(range(0, max(h - patch_size, 0) + 1, stride)) or [0]

    # Ensure the right/bottom edges are covered
    if x_starts[-1] + patch_size < w:
        x_starts.append(max(w - patch_size, 0))
    if y_starts[-1] + patch_size < h:
        y_starts.append(max(h - patch_size, 0))

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + patch_size, w)
            y1 = min(y0 + patch_size, h)
            patch = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
            patch[: y1 - y0, : x1 - x0] = image[y0:y1, x0:x1]
            yield patch, (x0, y0)


def adjust_dota_annotations(label_path: str, x0: int, y0: int, patch_size: int, min_visibility: float = 0.3):
    """
    Reads a DOTA labelTxt file, shifts polygon coordinates into patch-local
    space, and discards/clips instances that fall (mostly) outside the patch.
    """
    if not os.path.exists(label_path):
        return []

    kept_lines = []
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9 or parts[0].startswith(("imagesource", "gsd")):
            continue

        coords = list(map(float, parts[:8]))
        category = parts[8]
        difficult = parts[9] if len(parts) > 9 else "0"

        xs = coords[0::2]
        ys = coords[1::2]
        poly_x1, poly_x2 = min(xs), max(xs)
        poly_y1, poly_y2 = min(ys), max(ys)

        # Compute overlap between the instance's bbox and the patch
        inter_x1 = max(poly_x1, x0)
        inter_y1 = max(poly_y1, y0)
        inter_x2 = min(poly_x2, x0 + patch_size)
        inter_y2 = min(poly_y2, y0 + patch_size)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        poly_area = max((poly_x2 - poly_x1) * (poly_y2 - poly_y1), 1e-6)

        if inter_area / poly_area < min_visibility:
            continue

        shifted_coords = [
            (coords[i] - x0) if i % 2 == 0 else (coords[i] - y0) for i in range(8)
        ]
        clipped = [max(0, min(patch_size, c)) for c in shifted_coords]

        new_line = " ".join(f"{c:.2f}" for c in clipped) + f" {category} {difficult}\n"
        kept_lines.append(new_line)

    return kept_lines


def main():
    args = parse_args()

    images_dir = os.path.join(args.input, "images")
    labels_dir = os.path.join(args.input, "labelTxt")
    out_images_dir = os.path.join(args.output, "images")
    out_labels_dir = os.path.join(args.output, "labelTxt")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(images_dir, "*.png")) + glob(os.path.join(images_dir, "*.jpg")))
    print(f"Found {len(image_paths)} source images in {images_dir}")

    total_patches = 0
    for img_path in image_paths:
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{image_id}.txt")

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Could not read {img_path}, skipping.")
            continue

        for patch, (x0, y0) in extract_patches_for_image(image, args.patch_size, args.overlap):
            patch_id = f"{image_id}_{x0}_{y0}"
            patch_lines = adjust_dota_annotations(label_path, x0, y0, args.patch_size)

            if not patch_lines:
                continue  # skip empty patches (no annotated instances)

            cv2.imwrite(os.path.join(out_images_dir, f"{patch_id}.png"), patch)
            with open(os.path.join(out_labels_dir, f"{patch_id}.txt"), "w") as f:
                f.writelines(patch_lines)
            total_patches += 1

    print(f"Patch extraction complete. Generated {total_patches} non-empty patches -> {args.output}")


if __name__ == "__main__":
    main()
