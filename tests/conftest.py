"""Shared pytest fixtures for the test suite."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


@pytest.fixture
def device():
    return "cpu"  # CI runs CPU-only for speed; see .github/workflows/ci.yml


@pytest.fixture
def sample_batch_size():
    return 4
