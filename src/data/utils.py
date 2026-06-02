"""Geospatial data loading, saving, and patch extraction utilities."""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_geotiff(path: str):
    """Load a GeoTIFF as a numpy array with optional metadata.

    Falls back to PIL if rasterio is unavailable.

    Returns:
        (data, meta): data is (C, H, W) float32; meta is rasterio metadata or None.
    """
    meta = None
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            meta = src.meta.copy()
        print(f"  Loaded: {os.path.basename(path)}  shape={data.shape}")
        return data, meta
    except Exception:
        img = Image.open(path)
        data = np.array(img).astype(np.float32)
        if data.ndim == 3:
            data = data.transpose(2, 0, 1)
        print(f"  Loaded (PIL): {os.path.basename(path)}  shape={data.shape}")
        return data, meta


def save_geotiff(path: str, data: np.ndarray, ref_meta=None):
    """Save a 2D array as a single-band GeoTIFF (or PNG fallback)."""
    try:
        import rasterio
        if ref_meta:
            m = ref_meta.copy()
            m.update(count=1, dtype="float32", compress="lzw")
            with rasterio.open(path, "w", **m) as dst:
                dst.write(data.astype(np.float32), 1)
            return
    except Exception:
        pass
    Image.fromarray(data.astype(np.float32)).save(path)


def extract_patches(h: int, w: int, patch_size: int, stride: int, mask=None):
    """Extract grid patch coordinates, skipping all-NaN regions.

    Returns:
        List of (row, col) top-left coordinates.
    """
    coords = []
    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            if mask is not None:
                if not np.isfinite(mask[r : r + patch_size, c : c + patch_size]).any():
                    continue
            coords.append((r, c))
    return coords


def spatial_split(coords, image_width: int, patch_size: int, test_frac: float = 0.2):
    """Split patches geographically (east–west) to avoid spatial leakage.

    Falls back to random split if one partition is too small.

    Returns:
        (train_coords, test_coords)
    """
    split_col = int(image_width * (1 - test_frac))
    train = [(r, c) for r, c in coords if c + patch_size <= split_col]
    test = [(r, c) for r, c in coords if c >= split_col]

    if len(test) < 5 or len(train) < 5:
        train, test = train_test_split(coords, test_size=test_frac, random_state=42)
        print("  ⚠ Fell back to random split (not enough patches for spatial split)")

    print(f"  Train: {len(train)} patches | Test: {len(test)} patches")
    return train, test
