"""Train a change detection model on bi-temporal satellite imagery.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --model snunet-ecam
    python train.py --config configs/default.yaml --model biunet --epochs 50
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import (
    ChangeDetectionDataset,
    ChangeDetectionSiameseDataset,
    extract_patches,
    load_geotiff,
    spatial_split,
)
from src.losses import BCEDiceLoss
from src.metrics import Metrics
from src.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train LULC change detection model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    os.makedirs(data_cfg["output_dir"], exist_ok=True)

    is_siamese = model_cfg["name"].lower() in ("snunet-ecam", "siamunet-diff")

    # ── Load data ──
    print("Loading imagery...")
    img_t1, meta_t1 = load_geotiff(data_cfg["t1_path"])
    img_t2, meta_t2 = load_geotiff(data_cfg["t2_path"])
    mask, meta_mask = load_geotiff(data_cfg["mask_path"])

    nb = model_cfg["num_bands"]
    if img_t1.ndim == 3 and img_t1.shape[0] > nb:
        img_t1 = img_t1[:nb]
    if img_t2.ndim == 3 and img_t2.shape[0] > nb:
        img_t2 = img_t2[:nb]
    if mask.ndim == 3:
        mask = mask[0]

    # Clean NoData and normalize
    mask[mask <= -3.4e38] = 0
    mask = np.clip(mask, 0, 1)
    img_t1 = img_t1 / max(img_t1.max(), 1)
    img_t2 = img_t2 / max(img_t2.max(), 1)

    _, H, W = img_t1.shape
    res = cfg["inference"]["resolution_m"]
    print(f"\n  T1: {img_t1.shape}  |  T2: {img_t2.shape}  |  Mask: {mask.shape}")
    print(f"  Coverage: {H * res / 1000:.1f} km × {W * res / 1000:.1f} km")
    print(f"  Change pixels: {mask.sum():.0f} ({mask.mean() * 100:.1f}%)")

    # ── Extract patches ──
    ps = train_cfg["patch_size"]
    stride = train_cfg["stride"]
    print(f"\nExtracting {ps}×{ps} patches (stride={stride})...")
    all_coords = extract_patches(H, W, ps, stride, mask)
    print(f"  Total patches: {len(all_coords)}")
    train_coords, test_coords = spatial_split(all_coords, W, ps)

    # Class balance for pos_weight
    change_px = sum(mask[r : r + ps, c : c + ps].sum() for r, c in train_coords)
    total_px = len(train_coords) * ps ** 2
    change_ratio = change_px / max(total_px, 1)
    pos_weight = (1 - change_ratio) / max(change_ratio, 1e-6)
    print(f"  Change ratio: {change_ratio * 100:.1f}% → pos_weight={pos_weight:.2f}")

    # ── Build dataloaders ──
    DatasetClass = ChangeDetectionSiameseDataset if is_siamese else ChangeDetectionDataset
    train_ds = DatasetClass(img_t1, img_t2, mask, train_coords, ps, augment=True)
    test_ds = DatasetClass(img_t1, img_t2, mask, test_coords, ps, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=True,
    )

    # ── Build model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg["name"], nb, model_cfg["num_classes"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_cfg['name']} | Params: {n_params:,} | Device: {device}")

    # ── Optimizer, scheduler, loss ──
    criterion = BCEDiceLoss(bce_weight=0.5, pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"], eta_min=1e-7)

    train_m, val_m = Metrics(), Metrics()
    best_f1, patience = 0.0, 0
    history = []
    ckpt_path = os.path.join(data_cfg["output_dir"], f"best_{model_cfg['name'].replace('-', '_')}.pth")

    epochs = train_cfg["epochs"]
    print(f"\n{'=' * 70}")
    print(f"  TRAINING — {model_cfg['name']} | {epochs} epochs | batch={train_cfg['batch_size']} | lr={train_cfg['learning_rate']}")
    print(f"{'=' * 70}\n")

    # ── Training loop ──
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_m.reset()
        tloss = 0
        for batch in train_loader:
            if is_siamese:
                xA, xB, y = [t.to(device) for t in batch]
                logits = model(xA, xB)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)

            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.shape[1] == 2:
                logits = logits[:, 1:2]

            optimizer.zero_grad()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tloss += loss.item()
            train_m.update(logits, y)

        tloss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_m.reset()
        vloss = 0
        with torch.no_grad():
            for batch in test_loader:
                if is_siamese:
                    xA, xB, y = [t.to(device) for t in batch]
                    logits = model(xA, xB)
                else:
                    x, y = batch[0].to(device), batch[1].to(device)
                    logits = model(x)

                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits.shape[1] == 2:
                    logits = logits[:, 1:2]

                vloss += criterion(logits, y).item()
                val_m.update(logits, y)

        vloss /= max(len(test_loader), 1)
        scheduler.step()

        # Checkpoint
        elapsed = time.time() - t0
        marker = ""
        if val_m.f1 > best_f1:
            best_f1 = val_m.f1
            patience = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "best_f1": best_f1},
                ckpt_path,
            )
            marker = " ★"
        else:
            patience += 1

        print(
            f"Ep {epoch:03d}/{epochs} | loss={tloss:.4f}/{vloss:.4f} | "
            f"F1={val_m.f1:.4f} IoU={val_m.iou:.4f} κ={val_m.kappa:.4f} | "
            f"{elapsed:.0f}s{marker}"
        )

        history.append({
            "epoch": epoch, "train_loss": round(tloss, 5),
            "val_loss": round(vloss, 5), "val_f1": round(val_m.f1, 4),
            "val_iou": round(val_m.iou, 4), "val_kappa": round(val_m.kappa, 4),
        })

        if patience >= train_cfg["early_stopping"]:
            print(f"\n⏹ Early stopping at epoch {epoch}")
            break

    # Save training history
    history_path = os.path.join(data_cfg["output_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete. Best F1={best_f1:.4f}")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   History:    {history_path}")


if __name__ == "__main__":
    main()
