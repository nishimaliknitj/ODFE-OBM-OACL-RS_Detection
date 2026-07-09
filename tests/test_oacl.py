"""Unit tests for src/losses/oacl_loss.py -- Optimization-Aware Classification Loss."""

import torch

from losses.oacl_loss import (
    OptimizationAwareClassificationLoss,
    FocalLossBaseline,
    InvarianceRegularizationLoss,
)


def test_oacl_returns_scalar_by_default():
    loss_fn = OptimizationAwareClassificationLoss(max_sample_weight=3.0)
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 2, (4, 5)).float()
    tau = torch.tensor([1.0, 2.0, 0.5, 3.0])

    loss = loss_fn(logits, targets, tau)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_oacl_higher_tau_increases_sample_contribution():
    """A sample with much higher tau (before clamping) should be weighted more heavily."""
    loss_fn = OptimizationAwareClassificationLoss(max_sample_weight=100.0, reduction="none")

    logits = torch.tensor([[2.0], [2.0]])  # same prediction
    targets = torch.tensor([[0.0], [0.0]])  # same wrong target -> same raw BCE
    tau = torch.tensor([1.0, 10.0])

    per_sample = loss_fn(logits, targets, tau, tau_batch_expectation=1.0)
    assert per_sample[1] > per_sample[0]


def test_oacl_weight_is_bounded_by_w_max():
    loss_fn = OptimizationAwareClassificationLoss(max_sample_weight=2.0, reduction="none")
    logits = torch.tensor([[0.0], [0.0]])
    targets = torch.tensor([[1.0], [1.0]])
    tau = torch.tensor([1.0, 1000.0])  # second sample would get a huge weight if unbounded

    per_sample = loss_fn(logits, targets, tau, tau_batch_expectation=1.0)
    ratio = (per_sample[1] / per_sample[0]).item()
    # Ratio should not exceed w_max (2.0), confirming the bound (Eq. 14) is applied
    assert ratio <= 2.0 + 1e-3


def test_oacl_batch_expectation_computed_automatically_if_none():
    loss_fn = OptimizationAwareClassificationLoss(max_sample_weight=10.0)
    logits = torch.randn(3, 2)
    targets = torch.randint(0, 2, (3, 2)).float()
    tau = torch.tensor([1.0, 1.0, 1.0])

    # Should not raise, should equal the case with explicit mean expectation
    loss_auto = loss_fn(logits, targets, tau)
    loss_manual = loss_fn(logits, targets, tau, tau_batch_expectation=1.0)
    assert torch.allclose(loss_auto, loss_manual, atol=1e-5)


def test_focal_loss_baseline_runs():
    loss_fn = FocalLossBaseline(alpha=0.25, gamma=2.0)
    logits = torch.randn(4, 3)
    targets = torch.randint(0, 2, (4, 3)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0


def test_invariance_regularization_zero_for_identical_features():
    loss_fn = InvarianceRegularizationLoss()
    feats = [torch.randn(2, 8, 4, 4) for _ in range(3)]
    loss = loss_fn(feats, feats)  # same features twice -> L2 distance should be 0
    assert loss.item() < 1e-6


def test_invariance_regularization_positive_for_different_features():
    loss_fn = InvarianceRegularizationLoss()
    feats_a = [torch.zeros(2, 8, 4, 4) for _ in range(3)]
    feats_b = [torch.ones(2, 8, 4, 4) for _ in range(3)]
    loss = loss_fn(feats_a, feats_b)
    assert loss.item() > 0
