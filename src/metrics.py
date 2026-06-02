"""Change detection metrics with running accumulation."""

import torch


class Metrics:
    """Accumulates TP/FP/FN/TN counts across batches and computes metrics.

    Usage:
        m = Metrics()
        for logits, targets in loader:
            m.update(logits, targets)
        print(f"F1: {m.f1:.4f}, IoU: {m.iou:.4f}")
        m.reset()
    """

    def __init__(self, threshold: float = 0.5):
        self.thr = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    @torch.no_grad()
    def update(self, logits, targets):
        p = (torch.sigmoid(logits) > self.thr).float()
        t = targets.float()
        self.tp += ((p == 1) & (t == 1)).sum().item()
        self.fp += ((p == 1) & (t == 0)).sum().item()
        self.fn += ((p == 0) & (t == 1)).sum().item()
        self.tn += ((p == 0) & (t == 0)).sum().item()

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0

    @property
    def iou(self):
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else 0

    @property
    def overall_accuracy(self):
        tot = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / tot if tot else 0

    @property
    def kappa(self):
        tot = self.tp + self.fp + self.fn + self.tn
        if not tot:
            return 0
        po = (self.tp + self.tn) / tot
        pe = (
            (self.tp + self.fp) * (self.tp + self.fn)
            + (self.tn + self.fn) * (self.tn + self.fp)
        ) / (tot * tot)
        return (po - pe) / (1 - pe) if (1 - pe) else 0

    def to_dict(self):
        """Return all metrics as a dictionary."""
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "iou": round(self.iou, 4),
            "overall_accuracy": round(self.overall_accuracy, 4),
            "kappa": round(self.kappa, 4),
        }
