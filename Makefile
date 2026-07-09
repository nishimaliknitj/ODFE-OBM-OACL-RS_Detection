.PHONY: install install-dev lint format test test-cov clean train-dior train-dota train-fair1m \
        eval-dior eval-dota eval-fair1m docker-build docker-run tensorboard download-data

PYTHON := python
PIP := pip

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install

lint:
	flake8 src/ tests/
	isort --check-only --diff src/ tests/
	black --check src/ tests/

format:
	isort src/ tests/
	black src/ tests/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov .coverage

download-data:
	bash scripts/download_dior.sh
	bash scripts/download_dota.sh
	bash scripts/download_fair1m.sh

train-dior:
	$(PYTHON) src/train.py --config configs/dior.yaml

train-dota:
	$(PYTHON) src/train.py --config configs/dota.yaml

train-fair1m:
	$(PYTHON) src/train.py --config configs/fair1m.yaml

eval-dior:
	$(PYTHON) src/test.py --config configs/dior.yaml --weights checkpoints/dior_full_best.pt

eval-dota:
	$(PYTHON) src/test.py --config configs/dota.yaml --weights checkpoints/dota_full_best.pt

eval-fair1m:
	$(PYTHON) src/test.py --config configs/fair1m.yaml --weights checkpoints/fair1m_full_best.pt

docker-build:
	docker build -t odfe-obm-oacl:latest .

docker-run:
	docker run --gpus all -it -v $(PWD)/data:/workspace/data -v $(PWD)/checkpoints:/workspace/checkpoints odfe-obm-oacl:latest

tensorboard:
	tensorboard --logdir logs --host 0.0.0.0 --port 6006
