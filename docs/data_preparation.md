# Data Preparation

This document describes how to obtain and organize the three benchmark datasets used in the paper: **DIOR**, **DOTA**, and **FAIR1M**. All links below were verified against the official dataset homepages maintained by the original dataset authors.

## 1. DIOR

- **Source**: Li et al., *"Object detection in optical remote sensing images: A survey and a new benchmark"*, ISPRS J. Photogrammetry and Remote Sensing, 2020.
- **Size**: 23,463 images, 20 categories, ~190k instances. License: CC BY-NC 4.0.
- **Format**: Pascal-VOC-style XML annotations, one file per image.
- **Official dataset page**: **https://gcheng-nwpu.github.io/#Datasets**
- **Direct download**: [Google Drive](https://drive.google.com/open?id=1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) · [BaiduNetDisk](https://pan.baidu.com/s/1iLKT0JQoKXEJTGNxt5lSMg)
- **Paper**: https://arxiv.org/abs/1909.00133

```bash
bash scripts/download_dior.sh
```

Expected layout after setup:
```
data/DIOR/
├── JPEGImages-train/
│   ├── 00001.jpg
│   └── ...
├── JPEGImages-test/
│   └── ...
└── Annotations/
    ├── train/
    │   ├── 00001.xml
    │   └── ...
    └── test/
```

Images range from 800×800 to 10,000×10,000 pixels. This repo applies a **sliding-window strategy (stride 512)** to large source images before resizing to the 640×640 training resolution (see `configs/dior.yaml -> dataset.sliding_window`), to avoid losing small objects.

## 2. DOTA

- **Source**: Xia et al., *"DOTA: A large-scale dataset for object detection in aerial images"*, CVPR 2018.
- **Size**: 2,806 images, 15 categories, ~188k instances, oriented bounding boxes. License: academic use only, commercial use prohibited.
- **Format**: `labelTxt/*.txt`, 8-point polygon per instance + category + difficulty flag.
- **Official dataset page**: **https://captain-whu.github.io/DOTA/dataset.html**
- **Direct download (DOTA-v1.0)**: [Google Drive — Training set](https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK) · [Validation set](https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224) · [Testing images](https://drive.google.com/drive/folders/1mYOf5USMGNcJRPcvRVJVV1uHEalG5RPl) · [BaiduNetDisk](https://pan.baidu.com/s/1kWyRGaz)
- **Paper**: https://arxiv.org/abs/1711.10398

```bash
bash scripts/download_dota.sh
```

Expected layout:
```
data/DOTA/
├── train/
│   ├── images/*.png
│   └── labelTxt/*.txt
└── val/
    ├── images/*.png
    └── labelTxt/*.txt
```

DOTA images are very large (up to 4,000×4,000). This repo's `DOTADataset` loader converts the 8-point oriented polygon to its **axis-aligned bounding rectangle** for compatibility with the CIoU-based axis-aligned detection head (`src/losses/ciou.py`). For full oriented bounding box (OBB) support, see `docs/methodology.md`.

Following the paper's protocol, high-resolution images should be cropped into **640×640 patches with 256-pixel overlap** prior to training (`src/datasets/patch_extraction.py`).

**Note**: DOTA images/annotations are for **academic purposes only**; commercial use is prohibited under the original authors' license (Wuhan University / CAPTAIN-WHU).

## 3. FAIR1M

- **Source**: Sun et al., *"FAIR1M: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery"*, ISPRS J. Photogrammetry and Remote Sensing, 2022.
- **Size**: 15,000+ images, 37+ fine-grained categories, 1M+ instances.
- **Format**: XML with `<points>` polygon coordinates and `<possibleresult><name>` fine-grained category labels.
- **Official dataset page**: **https://www.gaofen-challenge.com/benchmark** (ISPRS / Gaofen Challenge; this site requires JavaScript to render — enable it in your browser if the page appears blank)
- **Alternative accessible mirror**: [Hugging Face — blanchon/FAIR1M](https://huggingface.co/datasets/blanchon/FAIR1M)
- **Paper**: https://arxiv.org/abs/2103.05569

```bash
bash scripts/download_fair1m.sh
```

Expected layout:
```
data/FAIR1M/
├── train/
│   ├── images/*.tif
│   └── labelXml/*.xml
└── val/
    ├── images/*.tif
    └── labelXml/*.xml
```

FAIR1M is trained at **800×800** input resolution with **batch size 8** (reduced from 16 due to memory) — see `configs/fair1m.yaml`.

## Quick Reference: All Dataset Links

| Dataset | Official Page | Direct Download | Paper |
|---|---|---|---|
| DIOR | https://gcheng-nwpu.github.io/#Datasets | [Google Drive](https://drive.google.com/open?id=1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) | https://arxiv.org/abs/1909.00133 |
| DOTA | https://captain-whu.github.io/DOTA/dataset.html | [Google Drive](https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK) | https://arxiv.org/abs/1711.10398 |
| FAIR1M | https://www.gaofen-challenge.com/benchmark | [Hugging Face mirror](https://huggingface.co/datasets/blanchon/FAIR1M) | https://arxiv.org/abs/2103.05569 |

## Verifying your setup

After placing each dataset, verify the loader can parse it:

```bash
python -c "
from src.datasets import build_dataset
ds = build_dataset('DIOR', images_dir='data/DIOR/JPEGImages-train',
                    annotations_dir='data/DIOR/Annotations/train',
                    class_names=None, input_size=(640,640), is_train=True)
print(f'Loaded {len(ds)} samples')
print(ds[0]['boxes'].shape, ds[0]['labels'])
"
```

## Notes on Licensing

Each dataset has its own license and usage terms set by the original authors. This repository does **not** redistribute any dataset images or annotations — only loader code and links to the official sources above. Please review and comply with each dataset's license before use (DIOR and DOTA are academic-use only; commercial use is prohibited by the original authors).
