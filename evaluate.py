"""Generate full-scene change maps and compute final metrics.

Usage:
    python evaluate.py --config configs/default.yaml --checkpoint outputs/best_snunet_ecam.pth
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import cohen_kappa_score

from src.data import load_geotiff, save_geotiff
from src.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate change detection model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["name"] = args.model

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    inf_cfg = cfg["inference"]

    is_siamese = model_cfg["name"].lower() in ("snunet-ecam", "siamunet-diff")
    os.makedirs(data_cfg["output_dir"], exist_ok=True)

    # ── Load data ──
    print("Loading imagery...")
    img_t1, _ = load_geotiff(data_cfg["t1_path"])
    img_t2, _ = load_geotiff(data_cfg["t2_path"])
    mask, meta_mask = load_geotiff(data_cfg["mask_path"])

    nb = model_cfg["num_bands"]
    if img_t1.ndim == 3 and img_t1.shape[0] > nb:
        img_t1 = img_t1[:nb]
    if img_t2.ndim == 3 and img_t2.shape[0] > nb:
        img_t2 = img_t2[:nb]
    if mask.ndim == 3:
        mask = mask[0]

    mask[mask <= -3.4e38] = 0
    mask = np.clip(mask, 0, 1)
    img_t1 = img_t1 / max(img_t1.max(), 1)
    img_t2 = img_t2 / max(img_t2.max(), 1)

    _, H, W = img_t1.shape

    # ── Load model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg["name"], nb, model_cfg["num_classes"]).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, F1={ckpt['best_f1']:.4f})")

    # ── Sliding-window inference ──
    print("Generating change map...")
    ps = cfg["training"]["patch_size"]
    inf_stride = cfg["training"]["stride"] // inf_cfg["stride_divisor"]

    prob_map = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.float64)
    total = len(range(0, H - ps + 1, inf_stride)) * len(range(0, W - ps + 1, inf_stride))
    done = 0

    with torch.no_grad():
        for r in range(0, H - ps + 1, inf_stride):
            for c in range(0, W - ps + 1, inf_stride):
                p1 = img_t1[:, r : r + ps, c : c + ps]
                p2 = img_t2[:, r : r + ps, c : c + ps]

                if is_siamese:
                    xA = torch.from_numpy(p1).float().unsqueeze(0).to(device)
                    xB = torch.from_numpy(p2).float().unsqueeze(0).to(device)
                    logits = model(xA, xB)
                else:
                    x = torch.from_numpy(np.concatenate([p1, p2], 0)).float().unsqueeze(0).to(device)
                    logits = model(x)

                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits.shape[1] == 2:
                    logits = logits[:, 1:2]

                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                prob_map[r : r + ps, c : c + ps] += probs
                count_map[r : r + ps, c : c + ps] += 1.0

                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{total} ({done / total * 100:.0f}%)", end="\r")

    prob_map /= np.maximum(count_map, 1.0)
    thr = inf_cfg["threshold"]
    binary_map = (prob_map > thr).astype(np.uint8)

    # Save GeoTIFFs
    save_geotiff(os.path.join(data_cfg["output_dir"], "change_probability.tif"), prob_map, meta_mask)
    save_geotiff(os.path.join(data_cfg["output_dir"], "change_prediction.tif"), binary_map.astype(np.float32), meta_mask)

    res = inf_cfg["resolution_m"]
    print(f"\n\n  Changed: {binary_map.sum() * res**2 / 10000:.1f} ha ({binary_map.mean() * 100:.1f}%)")

    # ── Compute metrics ──
    gt = mask.flatten().astype(int)
    pred = binary_map.flatten().astype(int)
    v = (gt >= 0) & (gt <= 1)

    tp = np.sum((pred[v] == 1) & (gt[v] == 1))
    fp = np.sum((pred[v] == 1) & (gt[v] == 0))
    fn = np.sum((pred[v] == 0) & (gt[v] == 1))
    tn = np.sum((pred[v] == 0) & (gt[v] == 0))

    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0
    oa = (tp + tn) / v.sum()
    kap = cohen_kappa_score(gt[v], pred[v])

    print(f"\n{'=' * 50}")
    print(f"  RESULTS — {model_cfg['name']}")
    print(f"{'=' * 50}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  IoU        : {iou:.4f}")
    print(f"  OA         : {oa:.4f}")
    print(f"  Kappa      : {kap:.4f}")
    print(f"{'=' * 50}")

    # Save report
    report = {
        "model": model_cfg["name"],
        "checkpoint_epoch": ckpt["epoch"],
        "metrics": {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "iou": round(iou, 4),
            "oa": round(oa, 4), "kappa": round(kap, 4),
        },
        "area": {
            "changed_ha": round(binary_map.sum() * res ** 2 / 10000, 1),
            "total_ha": round(binary_map.size * res ** 2 / 10000, 1),
        },
    }
    report_path = os.path.join(data_cfg["output_dir"], "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Visualization ──
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    ax[0].imshow(mask, cmap="RdYlGn_r")
    ax[0].set_title("Ground Truth", fontsize=14)
    ax[0].axis("off")

    im = ax[1].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    ax[1].set_title(f"{model_cfg['name']} — Probability Map", fontsize=14)
    ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046)

    ax[2].imshow(binary_map, cmap="RdYlGn_r")
    ax[2].set_title(f"Prediction (F1={f1:.3f}, κ={kap:.3f})", fontsize=14)
    ax[2].axis("off")

    plt.tight_layout()
    fig_path = os.path.join(data_cfg["output_dir"], "results_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
