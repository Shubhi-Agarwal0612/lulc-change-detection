"""PyTorch datasets for bi-temporal change detection.

Two dataset classes are provided:
  - ChangeDetectionDataset: concatenates T1 + T2 → single tensor (for BiUNet).
  - ChangeDetectionSiameseDataset: returns T1, T2 separately (for Siamese models).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def _augment(t1, t2, m):
    """Apply random flips, rotations, and brightness jitter."""
    # Horizontal flip
    if np.random.rand() > 0.5:
        t1, t2, m = t1[:, :, ::-1], t2[:, :, ::-1], m[:, ::-1]
    # Vertical flip
    if np.random.rand() > 0.5:
        t1, t2, m = t1[:, ::-1, :], t2[:, ::-1, :], m[::-1, :]
    # Random 90° rotation
    k = np.random.randint(0, 4)
    if k:
        t1 = np.rot90(t1, k, (1, 2)).copy()
        t2 = np.rot90(t2, k, (1, 2)).copy()
        m = np.rot90(m, k, (0, 1)).copy()
    # Brightness jitter
    if np.random.rand() > 0.5:
        f = np.random.uniform(0.9, 1.1)
        t1 = np.clip(t1 * f, 0, 1)
        t2 = np.clip(t2 * f, 0, 1)
    return t1, t2, m


class ChangeDetectionDataset(Dataset):
    """Returns concatenated [T1 | T2] input for non-Siamese models (e.g. BiUNet).

    Each sample: (x, y) where x is (2C, H, W) and y is (1, H, W).
    """

    def __init__(self, t1, t2, mask, coords, patch_size, augment=False):
        self.t1 = t1
        self.t2 = t2
        self.mask = mask
        self.coords = coords
        self.ps = patch_size
        self.aug = augment

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        r, c = self.coords[idx]
        ps = self.ps
        p1 = self.t1[:, r : r + ps, c : c + ps].copy()
        p2 = self.t2[:, r : r + ps, c : c + ps].copy()
        m = self.mask[r : r + ps, c : c + ps].copy()

        if self.aug:
            p1, p2, m = _augment(p1, p2, m)

        x = np.concatenate([p1, p2], axis=0)
        return (
            torch.from_numpy(x.copy()).float(),
            torch.from_numpy(m[None].copy()).float(),
        )


class ChangeDetectionSiameseDataset(Dataset):
    """Returns separate T1 and T2 inputs for Siamese models.

    Each sample: (xA, xB, y) where xA, xB are (C, H, W) and y is (1, H, W).
    """

    def __init__(self, t1, t2, mask, coords, patch_size, augment=False):
        self.t1 = t1
        self.t2 = t2
        self.mask = mask
        self.coords = coords
        self.ps = patch_size
        self.aug = augment

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        r, c = self.coords[idx]
        ps = self.ps
        p1 = self.t1[:, r : r + ps, c : c + ps].copy()
        p2 = self.t2[:, r : r + ps, c : c + ps].copy()
        m = self.mask[r : r + ps, c : c + ps].copy()

        if self.aug:
            p1, p2, m = _augment(p1, p2, m)

        return (
            torch.from_numpy(p1.copy()).float(),
            torch.from_numpy(p2.copy()).float(),
            torch.from_numpy(m[None].copy()).float(),
        )
