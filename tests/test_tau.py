"""Unit tests for src/models/tau.py -- the core optimization difficulty score."""

import torch

from models.tau import (
    OptimizationDifficultyTracker,
    TauConfig,
    partition_head_tail,
    feature_level_importance,
    normalize_importance,
)


def test_tau_tracker_initializes_empty():
    tracker = OptimizationDifficultyTracker(TauConfig(ema_decay=0.9))
    assert tracker.global_iter == 0
    assert tracker.get_all_tau() == {}


def test_tau_first_update_returns_nonzero_scores():
    tracker = OptimizationDifficultyTracker(TauConfig(ema_decay=0.9))
    ids = ["a", "b", "c"]
    grad_norms = torch.tensor([1.0, 2.0, 0.5])
    losses = torch.tensor([0.8, 1.2, 0.3])

    tau = tracker.update(ids, grad_norms, losses)

    assert tau.shape == (3,)
    assert torch.all(tau >= 0)
    # On first update, delta=0 and kappa=0, so tau == g (Eq. 8 with no history)
    assert torch.allclose(tau, grad_norms, atol=1e-5)


def test_tau_increases_with_gradient_instability():
    """A sample with persistently fluctuating gradients should accumulate
    higher tau over iterations than one with stable gradients (paper's core claim)."""
    tracker_stable = OptimizationDifficultyTracker(TauConfig(ema_decay=0.9))
    tracker_unstable = OptimizationDifficultyTracker(TauConfig(ema_decay=0.9))

    ids = ["sample_0"]
    stable_grad = 1.0
    for _ in range(30):
        tracker_stable.update(ids, torch.tensor([stable_grad]), torch.tensor([0.5]))

    unstable_grads = [1.0, 3.0, 0.5, 3.5, 0.8, 3.2, 0.6, 3.1] * 4
    for i in range(30):
        g = unstable_grads[i % len(unstable_grads)]
        loss = 0.5 + 0.3 * ((-1) ** i)
        tracker_unstable.update(ids, torch.tensor([g]), torch.tensor([loss]))

    tau_stable = tracker_stable.get_tau(ids).item()
    tau_unstable = tracker_unstable.get_tau(ids).item()

    assert tau_unstable > tau_stable, (
        f"Expected unstable sample to have higher tau ({tau_unstable}) "
        f"than stable sample ({tau_stable})"
    )


def test_tau_ema_smoothing_reduces_transient_noise():
    """A single-iteration noise spike should not dominate the EMA-smoothed tau."""
    tracker = OptimizationDifficultyTracker(TauConfig(ema_decay=0.9))
    ids = ["x"]

    # Steady low gradient
    for _ in range(20):
        tracker.update(ids, torch.tensor([0.1]), torch.tensor([0.1]))

    tau_before_spike = tracker.get_tau(ids).item()

    # One noise spike
    tracker.update(ids, torch.tensor([5.0]), torch.tensor([5.0]))
    tau_after_spike = tracker.get_tau(ids).item()

    # Immediately return to steady state
    for _ in range(15):
        tracker.update(ids, torch.tensor([0.1]), torch.tensor([0.1]))
    tau_after_recovery = tracker.get_tau(ids).item()

    assert tau_after_spike > tau_before_spike  # spike is registered
    # After recovery, tau should have decayed back close to baseline (EMA suppresses transient noise)
    assert tau_after_recovery < tau_after_spike
    assert abs(tau_after_recovery - tau_before_spike) < abs(tau_after_spike - tau_before_spike)


def test_partition_head_tail_splits_correctly():
    tau_batch = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.5])
    result = partition_head_tail(tau_batch)
    assert result["tail_mask"].sum() + result["head_mask"].sum() == len(tau_batch)
    # All tail-mask entries must be strictly above the threshold (mean)
    assert torch.all(tau_batch[result["tail_mask"]] > result["threshold"])


def test_warmup_gate():
    tracker = OptimizationDifficultyTracker(TauConfig(warmup_iters=5))
    assert not tracker.is_warmed_up()
    for _ in range(5):
        tracker.update(["a"], torch.tensor([1.0]), torch.tensor([1.0]))
    assert tracker.is_warmed_up()


def test_feature_level_importance_eq3():
    g = torch.tensor([1.0, 2.0, 0.5])
    s = torch.tensor([0.1, 0.5, 0.0])
    alpha = feature_level_importance(g, s)
    expected = g * (1 + s)
    assert torch.allclose(alpha, expected)


def test_normalize_importance_sums_to_one():
    alpha = torch.tensor([1.0, 2.0, 3.0])
    norm = normalize_importance(alpha)
    assert abs(norm.sum().item() - 1.0) < 1e-4


def test_state_dict_roundtrip():
    tracker = OptimizationDifficultyTracker(TauConfig(ema_decay=0.9))
    tracker.update(["a", "b"], torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.7]))
    state = tracker.state_dict()

    tracker2 = OptimizationDifficultyTracker(TauConfig(ema_decay=0.9))
    tracker2.load_state_dict(state)

    tau1 = tracker.get_tau(["a", "b"])
    tau2 = tracker2.get_tau(["a", "b"])
    assert torch.allclose(tau1, tau2)
