# Paper Source

This folder contains the manuscript for **"Optimization-Driven Feature Learning for Long-Tailed Object Detection in Optical Remote Sensing Imagery"**.

- `paper.pdf` — compiled manuscript (44 pages, including appendices).
- `latex/` — LaTeX source skeleton (see below). The full bibliography and figure sources used for the final compiled PDF are maintained separately due to size; this skeleton reproduces the structure and compilation pipeline for reference and future revisions.

## Compiling

```bash
cd paper/latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

(Standard 4-pass LaTeX + BibTeX compilation sequence: `pdflatex -> bibtex -> pdflatex x2`.)

## Structure

- Abstract, Introduction, Related Work
- Methodology (ODFE, OBM, OACL, combined objective)
- Experiments (DIOR, DOTA, FAIR1M)
- Discussion, Statistical Validation, Data Verification and Correction Statement
- Ablation Study, Hyperparameter Sensitivity, Robustness Analysis
- Conclusions, Limitations, Declarations
- Appendices A–E (per-category results, tau distribution analysis, failure case analysis)

See the root [`README.md`](../README.md) for the code implementation corresponding to each section.
