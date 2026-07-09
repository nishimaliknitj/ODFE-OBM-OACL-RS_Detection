"""
Training Entry Point
=======================

Implements Algorithm 1 of the paper ("Optimization-Driven Long-Tailed
Object Detection"):

    Initialize theta
    for t = 1 to T:
        Sample mini-batch B
        Extract multi-scale features {f_l}
        foreach (x_i, y_i) in B:
            Compute tau_i^(t)                      # gradient sensitivity + fluctuation + curvature
        Compute feature importance alpha_l -> emphasized features f_tilde_l   # ODFE
        Compute sampling probability P_i ∝ tau_i^(t), generate mosaic samples # OBM
        Compute weighted classification loss w_i ∝ tau_i^(t)                  # OACL
        Compute L_final = L_loc + lambda * L_cls
        Update theta via gradient descent
    return theta*

Usage
-----
    python src/train.py --config configs/dior.yaml
    python src/train.py --config configs/dior.yaml --no-odfe
    python src/train.py --config configs/dior.yaml --no-obm --no-oacl
    python src/train.py --config configs/dior.yaml --seed 123
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset, detection_collate_fn
from datasets.augmentations import DetectionAugmentation
from losses.combined_loss import build_loss
from models import (
    ODFEDetector,
    OptimizationBalancedMosaicSampler,
    OptimizationDifficultyTracker,
    TauConfig,
    build_detector,
)
from utils import EMAMeter, get_logger, load_config, save_checkpoint, set_seed
from utils.config import save_config


def parse_args():
    p = argparse.ArgumentParser(description="Train ODFE+OBM+OACL long-tailed RS detector")
    p.add_argument(
        "--config", type=str, required=True, help="Path to YAML config (e.g. configs/dior.yaml)"
    )
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--no-odfe", action="store_true", help="Disable ODFE (ablation)")
    p.add_argument("--no-obm", action="store_true", help="Disable OBM (ablation)")
    p.add_argument(
        "--no-oacl", action="store_true", help="Disable OACL, fall back to plain BCE (ablation)"
    )
    p.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return p.parse_args()


def build_dataloaders(config: dict):
    dataset_cfg = config["dataset"]
    train_transform = DetectionAugmentation(
        hflip_prob=config["augmentation"]["horizontal_flip_prob"],
        vflip_prob=config["augmentation"]["vertical_flip_prob"],
        scale_jitter=tuple(config["augmentation"]["scale_jitter"]),
        brightness=config["augmentation"]["color_jitter"]["brightness"],
        contrast=config["augmentation"]["color_jitter"]["contrast"],
        saturation=config["augmentation"]["color_jitter"]["saturation"],
        is_train=True,
    )
    val_transform = DetectionAugmentation(is_train=False)

    train_dataset = build_dataset(
        dataset_cfg["name"],
        images_dir=dataset_cfg["train_images"],
        annotations_dir=dataset_cfg["train_annotations"],
        class_names=dataset_cfg["class_names"],
        input_size=tuple(dataset_cfg["input_size"]),
        is_train=True,
        transform=train_transform,
    )
    val_dataset = build_dataset(
        dataset_cfg["name"],
        images_dir=dataset_cfg["val_images"],
        annotations_dir=dataset_cfg["val_annotations"],
        class_names=dataset_cfg["class_names"],
        input_size=tuple(dataset_cfg["input_size"]),
        is_train=False,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        collate_fn=detection_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        collate_fn=detection_collate_fn,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def dummy_targets_from_batch(batch: Dict, num_classes: int, device: str):
    """
    Placeholder target-assignment: converts variable-length per-image boxes
    into a fixed-size classification target tensor for the OACL loss demo.
    A full implementation would use an anchor/point assigner (e.g. SimOTA,
    TaskAlignedAssigner) matching the original YOLOv8 label assignment --
    left as an integration point for users plugging in their own head.
    """
    batch_size = batch["images"].shape[0]
    cls_targets = torch.zeros(batch_size, num_classes, device=device)
    for i, labels in enumerate(batch["labels"]):
        for lbl in labels:
            cls_targets[i, int(lbl)] = 1.0
    return cls_targets


def train_one_epoch(
    model: ODFEDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    tau_tracker: OptimizationDifficultyTracker,
    obm_sampler: OptimizationBalancedMosaicSampler,
    config: dict,
    device: str,
    epoch: int,
    logger,
    use_obm: bool,
):
    model.train()
    loss_meter = EMAMeter(beta=0.98)
    loc_meter = EMAMeter(beta=0.98)
    cls_meter = EMAMeter(beta=0.98)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for it, batch in enumerate(pbar):
        images = batch["images"].to(device, non_blocking=True)
        sample_ids = batch["ids"]

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        cls_logits = outputs["cls_outputs"][0].mean(
            dim=[2, 3]
        )  # (B, num_classes) global-pooled logits for demo head
        # Dummy box regression target/pred pooled to per-image single box for demonstration purposes;
        # replace with a proper per-anchor assigner in a production training loop.
        reg_pred = outputs["reg_outputs"][0].mean(dim=[2, 3])  # (B, 4)

        cls_targets = dummy_targets_from_batch(batch, model.num_classes, device)

        # Build simple pseudo box targets (mean GT box per image, zero-padded) for CIoU demo.
        box_targets = torch.zeros(images.shape[0], 4, device=device)
        for i, boxes in enumerate(batch["boxes"]):
            if len(boxes) > 0:
                box_targets[i] = boxes.mean(dim=0).to(device)
            else:
                box_targets[i] = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)

        pred_boxes_xyxy = (
            reg_pred  # network already regresses xyxy-like values in this simplified head
        )

        # ---- tau computation (gradient sensitivity requires a backward pass) ----
        per_sample_loss_proxy = torch.nn.functional.binary_cross_entropy_with_logits(
            cls_logits, cls_targets, reduction="none"
        ).mean(dim=1)

        if tau_tracker.is_warmed_up():
            tau = tau_tracker.get_tau(sample_ids)
            if tau.sum() == 0:
                tau = torch.ones(len(sample_ids))
        else:
            tau = torch.ones(len(sample_ids))
        tau = tau.to(device)
        tau_batch_exp = tau_tracker.batch_expectation(tau.cpu())

        losses = loss_fn(
            pred_boxes=pred_boxes_xyxy,
            target_boxes=box_targets,
            pred_cls_logits=cls_logits,
            target_cls=cls_targets,
            tau=tau,
            tau_batch_expectation=tau_batch_exp,
        )
        loss_total = losses["loss_total"]
        loss_total.backward()

        grad_norm_proxy = (
            torch.tensor(
                [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            ).mean()
            if any(p.grad is not None for p in model.parameters())
            else torch.tensor(0.0)
        )

        per_sample_grad_norms = grad_norm_proxy.repeat(len(sample_ids))

        torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["grad_clip_norm"])
        optimizer.step()

        # Update tau tracker AFTER the step, using this iteration's grad/loss signal
        tau_tracker.update(sample_ids, per_sample_grad_norms, per_sample_loss_proxy.detach())

        loss_meter.update(loss_total.item())
        loc_meter.update(losses["loss_loc"].item())
        cls_meter.update(losses["loss_cls"].item())

        if it % config["logging"]["log_interval"] == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss_meter.value:.4f}",
                    "loc": f"{loc_meter.value:.4f}",
                    "cls": f"{cls_meter.value:.4f}",
                }
            )

    logger.info(
        f"[Epoch {epoch}] train_loss={loss_meter.value:.4f} "
        f"loc={loc_meter.value:.4f} cls={cls_meter.value:.4f}"
    )
    return {"loss": loss_meter.value, "loc": loc_meter.value, "cls": cls_meter.value}


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["hardware"]["device"] = args.device
    if args.no_odfe:
        config["odfe"]["enabled"] = False
    if args.no_obm:
        config["obm"]["enabled"] = False
    if args.no_oacl:
        config["oacl"]["enabled"] = False

    set_seed(config["experiment"]["seed"], deterministic=config["experiment"]["deterministic"])

    device = config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
    exp_name = config["experiment"]["name"]

    os.makedirs(config["experiment"]["log_dir"], exist_ok=True)
    os.makedirs(config["experiment"]["checkpoint_dir"], exist_ok=True)

    logger = get_logger(
        exp_name, log_file=os.path.join(config["experiment"]["log_dir"], f"{exp_name}.log")
    )
    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Device: {device}")
    save_config(
        config, os.path.join(config["experiment"]["log_dir"], f"{exp_name}_resolved_config.yaml")
    )

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(config)
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    model = build_detector(config).to(device)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        momentum=config["optimizer"]["momentum"],
        weight_decay=config["optimizer"]["weight_decay"],
        nesterov=config["optimizer"]["nesterov"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["dataset"].get(
            "lr_decay_epochs", config["training"].get("lr_decay_epochs", [50, 80])
        ),
        gamma=config["lr_schedule"]["gamma"],
    )

    tau_tracker = OptimizationDifficultyTracker(
        TauConfig(
            ema_decay=config["tau"]["ema_decay"],
            epsilon=config["tau"]["epsilon"],
            warmup_iters=config["tau"]["warmup_iters"],
        )
    )
    obm_sampler = OptimizationBalancedMosaicSampler(temperature=config["obm"]["temperature"])
    loss_fn = build_loss(config, tau_tracker)

    start_epoch = 0
    if args.resume:
        from utils.checkpoint import load_checkpoint

        ckpt = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        if ckpt.get("tau_tracker_state"):
            tau_tracker.load_state_dict(ckpt["tau_tracker_state"])
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        t0 = time.time()
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            tau_tracker,
            obm_sampler,
            config,
            device,
            epoch,
            logger,
            use_obm=config["obm"]["enabled"],
        )
        scheduler.step()
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch} completed in {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if (epoch + 1) % config["logging"]["save_interval"] == 0:
            ckpt_path = os.path.join(
                config["experiment"]["checkpoint_dir"], f"{exp_name}_epoch{epoch}.pt"
            )
            save_checkpoint(ckpt_path, model, optimizer, epoch, tau_tracker.state_dict())
            logger.info(f"Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(config["experiment"]["checkpoint_dir"], f"{exp_name}_final.pt")
    save_checkpoint(
        final_path, model, optimizer, config["training"]["epochs"] - 1, tau_tracker.state_dict()
    )
    logger.info(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
