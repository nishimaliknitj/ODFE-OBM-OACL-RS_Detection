# Data Availability Statement

This statement accompanies the manuscript **"Optimization-Driven Feature Learning for Long-Tailed Object Detection in Optical Remote Sensing Imagery"** and addresses the editorial requirement for reproducibility (raw code, not pseudocode, plus training/validation datasets, deposited in a public repository).

## Code Availability

The full source code implementing the proposed framework (ODFE, OBM, OACL), including training, evaluation, and inference scripts, configuration files, unit tests, and documentation, is publicly and permanently available at:

**https://github.com/nishimaliknitj/ODFE-OBM-OACL-RSDetection**

This is the actual, runnable implementation (not pseudocode) used to produce the results reported in the paper. The repository includes:

- Full implementation of the optimization difficulty score (τ), ODFE, OBM, and OACL modules, with direct correspondence to every equation in the paper (see [`docs/methodology.md`](docs/methodology.md) for the equation-to-code mapping).
- Training (`src/train.py`), evaluation (`src/test.py`), and inference (`src/predict.py`) entry points.
- Configuration files for all three benchmark datasets (`configs/dior.yaml`, `configs/dota.yaml`, `configs/fair1m.yaml`).
- Dataset loaders for DIOR, DOTA, and FAIR1M (`src/datasets/`).
- A unit test suite (48 tests, all passing) covering every core module (`tests/`).
- Setup instructions, Docker/Conda environments, and step-by-step reproduction scripts (`scripts/`).

## Dataset Availability

This study uses three publicly available benchmark datasets for optical remote sensing object detection. None of the datasets were created by the authors; all are third-party benchmarks released by their original authors under their own academic-use licenses. This repository does not redistribute dataset images or annotations, only loader code — the datasets must be obtained directly from their official sources below, each verified to be live and accessible.

| Dataset | Official Source | Direct Download | Reference |
|---|---|---|---|
| **DIOR** | https://gcheng-nwpu.github.io/#Datasets | [Google Drive](https://drive.google.com/open?id=1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) / [BaiduNetDisk](https://pan.baidu.com/s/1iLKT0JQoKXEJTGNxt5lSMg) | Li, K., Wan, G., Cheng, G., Meng, L., & Han, J. (2020). Object detection in optical remote sensing images: A survey and a new benchmark. *ISPRS Journal of Photogrammetry and Remote Sensing*, 159, 296–307. https://arxiv.org/abs/1909.00133 |
| **DOTA** | https://captain-whu.github.io/DOTA/dataset.html | [Google Drive](https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK) / [BaiduNetDisk](https://pan.baidu.com/s/1kWyRGaz) | Xia, G.-S., Bai, X., Ding, J., Zhu, Z., Belongie, S., Luo, J., Datcu, M., Pelillo, M., & Zhang, L. (2018). DOTA: A large-scale dataset for object detection in aerial images. *CVPR 2018*. https://arxiv.org/abs/1711.10398 |
| **FAIR1M** | https://www.gaofen-challenge.com/benchmark | [Hugging Face mirror](https://huggingface.co/datasets/blanchon/FAIR1M) | Sun, X., Wang, P., Yan, Z., Xu, F., Wang, R., Diao, W., Chen, J., Li, J., Feng, Y., Xu, T., et al. (2022). FAIR1M: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery. *ISPRS Journal of Photogrammetry and Remote Sensing*, 184, 116–130. https://arxiv.org/abs/2103.05569 |

Step-by-step download and preprocessing instructions for each dataset — including the exact directory layout expected by this repository's data loaders — are provided in [`docs/data_preparation.md`](docs/data_preparation.md), and can be triggered directly via:

```bash
bash scripts/download_dior.sh
bash scripts/download_dota.sh
bash scripts/download_fair1m.sh
```

**Note on the FAIR1M official page**: `gaofen-challenge.com/benchmark` is a JavaScript-rendered site; some automated tools/browsers with scripting disabled may see a blank page. Enabling JavaScript resolves this. A directly browsable mirror of the released portion of the dataset is also provided above (Hugging Face) for immediate access.

## Licensing Note

- DIOR and DOTA are released for **academic/non-commercial use only** by their original authors; commercial use is prohibited under their respective licenses (both CC BY-NC 4.0 / academic-only terms).
- FAIR1M is released under the Gaofen Challenge / ISPRS benchmark terms.
- Users of this repository must independently review and comply with each dataset's license before downloading or using the data.

## Suggested Text for Manuscript "Data Availability" Section

> The code used to generate the results in this study is publicly and permanently available at https://github.com/nishimaliknitj/ODFE-OBM-OACL-RSDetection. This repository contains the full training, evaluation, and inference implementation, configuration files, and documentation required to reproduce all reported experiments. The DIOR, DOTA, and FAIR1M datasets used for training and validation are publicly available third-party benchmarks and can be obtained from their official sources: DIOR at https://gcheng-nwpu.github.io/#Datasets, DOTA at https://captain-whu.github.io/DOTA/dataset.html, and FAIR1M at https://www.gaofen-challenge.com/benchmark. Dataset preparation instructions matching the directory structure expected by our code are provided in the repository at docs/data_preparation.md.
