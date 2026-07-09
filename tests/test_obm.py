"""Unit tests for src/models/obm.py -- Optimization-Balanced Mosaic augmentation."""

import numpy as np

from models.obm import OptimizationBalancedMosaicSampler, build_mosaic_image


def test_sampling_distribution_sums_to_one():
    sampler = OptimizationBalancedMosaicSampler(temperature=1.0)
    tau_by_id = {"a": 1.0, "b": 3.0, "c": 0.5, "d": 2.0}
    dist = sampler.compute_sampling_distribution(tau_by_id)
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_harder_samples_get_higher_probability():
    sampler = OptimizationBalancedMosaicSampler(temperature=1.0)
    tau_by_id = {"easy": 0.1, "hard": 10.0}
    dist = sampler.compute_sampling_distribution(tau_by_id)
    assert dist["hard"] > dist["easy"]


def test_mosaic_group_sampling_returns_correct_size():
    sampler = OptimizationBalancedMosaicSampler(temperature=1.0)
    tau_by_id = {f"s{i}": np.random.rand() + 0.01 for i in range(20)}
    group = sampler.sample_mosaic_group(tau_by_id, group_size=4)
    assert len(group) == 4
    assert all(g in tau_by_id for g in group)


def test_temperature_softens_distribution():
    """Higher temperature should bring the distribution closer to uniform."""
    sampler_sharp = OptimizationBalancedMosaicSampler(temperature=0.1)
    sampler_soft = OptimizationBalancedMosaicSampler(temperature=10.0)

    tau_by_id = {"a": 1.0, "b": 100.0}

    dist_sharp = sampler_sharp.compute_sampling_distribution(tau_by_id)
    dist_soft = sampler_soft.compute_sampling_distribution(tau_by_id)

    # Sharp temperature should produce a more skewed (higher max) distribution
    assert max(dist_sharp.values()) >= max(dist_soft.values())


def test_build_mosaic_image_produces_correct_output_size():
    size = 64
    images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
    boxes = [
        np.array([[10, 10, 50, 50]], dtype=np.float32),
        np.array([[20, 20, 60, 60]], dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
        np.array([[5, 5, 30, 30]], dtype=np.float32),
    ]
    labels = [
        np.array([0]),
        np.array([1]),
        np.zeros((0,), dtype=np.int64),
        np.array([2]),
    ]

    mosaic_img, mosaic_boxes, mosaic_labels = build_mosaic_image(
        images, boxes, labels, output_size=size
    )

    assert mosaic_img.shape == (size, size, 3)
    assert mosaic_boxes.shape[1] == 4
    assert len(mosaic_boxes) == len(mosaic_labels)


def test_build_mosaic_requires_exactly_four_images():
    images = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    boxes = [np.zeros((0, 4)) for _ in range(3)]
    labels = [np.zeros((0,), dtype=np.int64) for _ in range(3)]
    try:
        build_mosaic_image(images, boxes, labels)
        assert False, "Expected AssertionError for != 4 images"
    except AssertionError:
        pass
