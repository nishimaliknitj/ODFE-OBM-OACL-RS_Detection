"""Smoke tests for src/models/detector.py -- full model forward pass."""

import torch

from models.detector import ODFEDetector, Backbone, FPNNeck, build_detector


def test_backbone_output_shapes():
    backbone = Backbone(base_channels=16)
    x = torch.randn(1, 3, 128, 128)
    p3, p4, p5 = backbone(x)
    assert p3.shape[2] == 16 and p3.shape[3] == 16  # stride 8
    assert p4.shape[2] == 8 and p4.shape[3] == 8  # stride 16
    assert p5.shape[2] == 4 and p5.shape[3] == 4  # stride 32


def test_fpn_neck_output_channels_consistent():
    neck = FPNNeck(channels=(64, 128, 256), out_channels=32)
    p3 = torch.randn(1, 64, 16, 16)
    p4 = torch.randn(1, 128, 8, 8)
    p5 = torch.randn(1, 256, 4, 4)
    feats = neck(p3, p4, p5)
    assert len(feats) == 3
    for f in feats:
        assert f.shape[1] == 32


def test_full_detector_forward_pass():
    model = ODFEDetector(num_classes=5, base_channels=8, fpn_channels=16, odfe_enabled=True)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert "cls_outputs" in out and "reg_outputs" in out
    assert len(out["cls_outputs"]) == 3
    assert len(out["reg_outputs"]) == 3
    for cls_out in out["cls_outputs"]:
        assert cls_out.shape[1] == 5  # num_classes
    for reg_out in out["reg_outputs"]:
        assert reg_out.shape[1] == 4  # xyxy


def test_detector_backward_pass_produces_gradients():
    model = ODFEDetector(num_classes=3, base_channels=8, fpn_channels=16)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    loss = sum(o.sum() for o in out["cls_outputs"]) + sum(o.sum() for o in out["reg_outputs"])
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_odfe_disabled_still_runs():
    model = ODFEDetector(num_classes=3, base_channels=8, fpn_channels=16, odfe_enabled=False)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert len(out["cls_outputs"]) == 3


def test_build_detector_from_config():
    config = {
        "dataset": {"num_classes": 20},
        "model": {"base_channels": 8, "fpn_channels": 16},
        "odfe": {"enabled": True},
        "tau": {"ema_decay": 0.9},
    }
    model = build_detector(config)
    assert isinstance(model, ODFEDetector)
    assert model.num_classes == 20


def test_num_parameters_is_positive():
    model = ODFEDetector(num_classes=5, base_channels=8, fpn_channels=16)
    assert model.num_parameters() > 0
