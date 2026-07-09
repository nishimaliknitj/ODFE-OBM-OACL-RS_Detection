"""
Baseline Detector: YOLOv8-style single-stage detector
=========================================================

Implements the "Basic Architecture" described in the paper's Methodology
section: a backbone for hierarchical feature extraction, a feature pyramid
neck for multi-scale aggregation (P3/P4/P5), and a detection head for
joint classification and localization.

This module provides a lightweight, self-contained CSP-style backbone/neck/
head implementation so the repository is runnable end-to-end without an
external dependency on a specific YOLO codebase. For full parity with the
paper's exact YOLOv8-CSP configuration, swap `Backbone`/`Neck` below with
the Ultralytics `ultralytics.nn.modules` building blocks (see
`docs/methodology.md` for the mapping).

The ODFE module (src/models/odfe.py) is applied to {P3, P4, P5} before the
detection head, exactly as shown in Figure 1 / Figure 3 of the paper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .odfe import OptimizationDrivenFeatureEmphasis


def conv_bn_act(
    c_in: int, c_out: int, k: int = 3, s: int = 1, p: Optional[int] = None
) -> nn.Sequential:
    p = k // 2 if p is None else p
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(inplace=True),
    )


class CSPBlock(nn.Module):
    """Simplified CSP (Cross Stage Partial) bottleneck block."""

    def __init__(self, c_in: int, c_out: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        c_hidden = c_out // 2
        self.cv1 = conv_bn_act(c_in, c_hidden, 1, 1)
        self.cv2 = conv_bn_act(c_in, c_hidden, 1, 1)
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    conv_bn_act(c_hidden, c_hidden, 3, 1),
                    conv_bn_act(c_hidden, c_hidden, 3, 1),
                )
                for _ in range(n)
            ]
        )
        self.cv3 = conv_bn_act(c_hidden * 2, c_out, 1, 1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class Backbone(nn.Module):
    """CSP-style backbone producing multi-scale features at strides 8, 16, 32 (P3, P4, P5)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.stem = conv_bn_act(in_channels, c, 3, 2)  # stride 2
        self.stage1 = nn.Sequential(
            conv_bn_act(c, c * 2, 3, 2), CSPBlock(c * 2, c * 2, n=1)
        )  # stride 4
        self.stage2 = nn.Sequential(
            conv_bn_act(c * 2, c * 4, 3, 2), CSPBlock(c * 4, c * 4, n=2)
        )  # stride 8  -> P3
        self.stage3 = nn.Sequential(
            conv_bn_act(c * 4, c * 8, 3, 2), CSPBlock(c * 8, c * 8, n=2)
        )  # stride 16 -> P4
        self.stage4 = nn.Sequential(
            conv_bn_act(c * 8, c * 16, 3, 2), CSPBlock(c * 16, c * 16, n=1)
        )  # stride 32 -> P5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class FPNNeck(nn.Module):
    """Simple top-down + bottom-up feature pyramid neck aggregating P3/P4/P5."""

    def __init__(self, channels: Tuple[int, int, int] = (256, 512, 1024), out_channels: int = 256):
        super().__init__()
        c3, c4, c5 = channels
        self.reduce5 = conv_bn_act(c5, out_channels, 1, 1)
        self.reduce4 = conv_bn_act(c4, out_channels, 1, 1)
        self.reduce3 = conv_bn_act(c3, out_channels, 1, 1)

        self.smooth4 = conv_bn_act(out_channels, out_channels, 3, 1)
        self.smooth3 = conv_bn_act(out_channels, out_channels, 3, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> List[torch.Tensor]:
        p5_r = self.reduce5(p5)
        p4_r = self.reduce4(p4) + self.upsample(p5_r)
        p4_r = self.smooth4(p4_r)
        p3_r = self.reduce3(p3) + self.upsample(p4_r)
        p3_r = self.smooth3(p3_r)
        return [p3_r, p4_r, p5_r]


class DetectionHead(nn.Module):
    """Decoupled classification + box regression head, applied per feature level."""

    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.cls_conv = nn.Sequential(
            conv_bn_act(in_channels, in_channels, 3, 1),
            conv_bn_act(in_channels, in_channels, 3, 1),
        )
        self.reg_conv = nn.Sequential(
            conv_bn_act(in_channels, in_channels, 3, 1),
            conv_bn_act(in_channels, in_channels, 3, 1),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)
        return cls_out, reg_out


class ODFEDetector(nn.Module):
    """
    Full detector = Backbone -> FPN Neck -> ODFE -> multi-scale Detection Heads.

    This is the model referenced throughout the paper as "YOLOv8-style
    single-stage detector" with ODFE inserted immediately before the
    detection head (Figure 1).
    """

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 64,
        fpn_channels: int = 256,
        odfe_enabled: bool = True,
        ema_decay: float = 0.9,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Backbone(base_channels=base_channels)
        self.neck = FPNNeck(
            channels=(base_channels * 4, base_channels * 8, base_channels * 16),
            out_channels=fpn_channels,
        )
        self.odfe_enabled = odfe_enabled
        self.odfe = OptimizationDrivenFeatureEmphasis(num_levels=3, ema_decay=ema_decay)

        self.heads = nn.ModuleList([DetectionHead(fpn_channels, num_classes) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        p3, p4, p5 = self.backbone(x)
        features = self.neck(p3, p4, p5)  # [P3', P4', P5']

        if self.odfe_enabled:
            features = self.odfe(features)

        cls_outs, reg_outs = [], []
        for feat, head in zip(features, self.heads):
            cls_out, reg_out = head(feat)
            cls_outs.append(cls_out)
            reg_outs.append(reg_out)

        return {"cls_outputs": cls_outs, "reg_outputs": reg_outs, "features": features}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_detector(config: dict) -> ODFEDetector:
    """Factory function used by train.py / test.py / predict.py, driven by YAML config."""
    model_cfg = config.get("model", {})
    odfe_cfg = config.get("odfe", {})
    tau_cfg = config.get("tau", {})

    return ODFEDetector(
        num_classes=config["dataset"]["num_classes"],
        base_channels=model_cfg.get("base_channels", 64),
        fpn_channels=model_cfg.get("fpn_channels", 256),
        odfe_enabled=odfe_cfg.get("enabled", True),
        ema_decay=tau_cfg.get("ema_decay", 0.9),
    )
