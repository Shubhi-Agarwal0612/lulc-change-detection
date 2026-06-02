"""Loss functions for change detection training."""

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy + Dice loss.

    Balances pixel-level classification (BCE) with region-level overlap (Dice).

    Args:
        bce_weight: Weight for the BCE term (Dice weight = 1 - bce_weight).
        pos_weight: Positive class weight for BCE (handles class imbalance).
        smooth: Smoothing constant for Dice to avoid division by zero.
    """

    def __init__(self, bce_weight: float = 0.5, pos_weight: float = 2.67, smooth: float = 1.0):
        super().__init__()
        self.bce_w = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        dice = 1 - (2 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return self.bce_w * bce + (1 - self.bce_w) * dice
