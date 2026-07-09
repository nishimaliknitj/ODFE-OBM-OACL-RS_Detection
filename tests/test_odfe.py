"""Unit tests for src/models/odfe.py -- Optimization-Driven Feature Emphasis."""

import torch

from models.odfe import OptimizationDrivenFeatureEmphasis, register_feature_grad_hooks


def test_odfe_passthrough_before_first_gradient_update():
    """Before any gradient statistics exist, ODFE should return features unchanged."""
    odfe = OptimizationDrivenFeatureEmphasis(num_levels=3)
    features = [torch.randn(2, 8, 16, 16) for _ in range(3)]
    out = odfe(features)
    for f_in, f_out in zip(features, out):
        assert torch.equal(f_in, f_out)


def test_odfe_reweights_after_gradient_update():
    odfe = OptimizationDrivenFeatureEmphasis(num_levels=3)
    grad_norms = torch.tensor([1.0, 5.0, 0.5])
    odfe.update_gradient_stats(grad_norms)

    features = [torch.ones(1, 4, 4, 4) for _ in range(3)]
    out = odfe(features)

    # The feature level with the highest gradient norm should receive
    # the highest relative weight after normalization (Eq. 3-4).
    weights = [out[l].mean().item() for l in range(3)]
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]


def test_odfe_output_shapes_match_input():
    odfe = OptimizationDrivenFeatureEmphasis(num_levels=3)
    odfe.update_gradient_stats(torch.tensor([1.0, 1.0, 1.0]))
    features = [torch.randn(2, 16, 8, 8), torch.randn(2, 16, 4, 4), torch.randn(2, 16, 2, 2)]
    out = odfe(features)
    for f_in, f_out in zip(features, out):
        assert f_in.shape == f_out.shape


def test_odfe_wrong_number_of_levels_raises():
    odfe = OptimizationDrivenFeatureEmphasis(num_levels=3)
    features = [torch.randn(1, 4, 4, 4) for _ in range(2)]  # wrong count
    try:
        odfe(features)
        assert False, "Expected AssertionError for mismatched feature level count"
    except AssertionError:
        pass


def test_importance_weights_sum_to_one_after_update():
    odfe = OptimizationDrivenFeatureEmphasis(num_levels=3)
    odfe.update_gradient_stats(torch.tensor([2.0, 1.0, 3.0]))
    alpha = odfe.compute_importance()
    assert abs(alpha.sum().item() - 1.0) < 1e-4


def test_register_feature_grad_hooks_captures_gradient_norms():
    features = [torch.randn(1, 4, 4, 4, requires_grad=True) for _ in range(3)]
    grad_norms, handles = register_feature_grad_hooks(features)

    loss = sum(f.sum() for f in features)
    loss.backward()

    assert grad_norms.shape == (3,)
    assert torch.all(grad_norms >= 0)

    for h in handles:
        h.remove()
