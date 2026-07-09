# Contributing to ODFE-OBM-OACL-RSDetection

Thanks for your interest in contributing! This project implements the method described in our paper on optimization-driven long-tailed object detection for remote sensing imagery. Contributions of all kinds are welcome: bug fixes, new dataset loaders, additional baselines, documentation improvements, and reproducibility reports.

## How to Contribute

1. **Fork** the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. **Set up your environment**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt   # linting / testing tools
   pre-commit install
   ```
3. **Make your changes.** Keep changes focused; one logical change per PR.
4. **Add or update tests** under `tests/` for any code you change (see `tests/README` style — pytest-based).
5. **Run the test suite** before submitting:
   ```bash
   pytest tests/ -v
   flake8 src/
   ```
6. **Commit** using clear, descriptive messages (imperative mood, e.g. `Fix tau EMA reset on epoch boundary`).
7. **Open a Pull Request** against `main`, filling in the PR template. Link any related issues.

## Code Style

- Python code follows **PEP8**, enforced via `flake8` and `black` (line length 100).
- Type hints are encouraged for new public functions.
- Docstrings follow the Google style.
- Config files are YAML; keep keys consistent with existing `configs/*.yaml`.

## Reporting Bugs

Please use the **Bug Report** issue template and include:
- Python / PyTorch / CUDA versions
- Full stack trace
- Minimal reproduction steps
- Dataset/config used

## Proposing Features

Open an issue first describing the motivation before submitting a large PR, so we can discuss design before implementation effort is spent.

## Reproducibility Reports

If you re-run our experiments and get different numbers, please open an issue with your hardware, seed, and full config — we take reproducibility seriously and will investigate discrepancies (see our data-verification precedent in the paper).

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
