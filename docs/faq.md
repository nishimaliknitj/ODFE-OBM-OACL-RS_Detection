# FAQ

**Q: Do I need all three datasets (DIOR, DOTA, FAIR1M) to run this repo?**
No. Each is independent. Download only what you need via `scripts/download_*.sh` and train with the matching config.

**Q: Why did my reproduced mAP differ slightly from the paper?**
Small differences (< 1 AP) are expected due to hardware, random seed, and library-version differences (see paper Table 13 for the reported seed variance: DIOR ±0.4%, DOTA ±0.3%, FAIR1M ±0.5%). If the gap is larger, please open a [reproducibility report](../.github/ISSUE_TEMPLATE/reproducibility_report.md).

**Q: Is OBB (oriented bounding box) training supported out of the box?**
Not in the reference implementation — DOTA/FAIR1M oriented annotations are converted to axis-aligned envelope boxes for simplicity. See `docs/methodology.md` Extension Points for how to add OBB support.

**Q: Do ODFE/OBM/OACL add inference latency?**
No. All three modules are training-time only. See paper Table 14 ("Computational Overhead") — zero additional parameters, zero additional FLOPs at inference.

**Q: How much training overhead do the modules add?**
Approximately 8–12% wall-clock training time overhead relative to the YOLOv8 baseline (ODFE ~5%, OBM ~7% combined, OACL negligible). See `docs/methodology.md` and paper Table 14.

**Q: Can I use this on a dataset other than DIOR/DOTA/FAIR1M?**
Yes — implement a new subclass of `RemoteSensingDetectionDataset` (see `src/datasets/dior.py` as a template) and add a corresponding `configs/your_dataset.yaml`.

**Q: The YOLOrs numbers in the paper's tables were corrected after review — where can I read about that?**
See the paper's "Data Verification and Correction Statement" section, and `CHANGELOG.md` in this repo for the corresponding code-side note.
