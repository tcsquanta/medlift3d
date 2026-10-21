"""Image and segmentation metrics."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def psnr(x: np.ndarray, ref: np.ndarray, data_range: float) -> float:
    mse = float(np.mean((np.asarray(x, np.float64) - np.asarray(ref, np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(data_range ** 2 / mse)


def ssim3d(x: np.ndarray, ref: np.ndarray, data_range: float) -> float:
    from skimage.metrics import structural_similarity

    win = min(7, *[s if s % 2 == 1 else s - 1 for s in x.shape])
    return float(structural_similarity(np.asarray(x, np.float64), np.asarray(ref, np.float64),
                                       data_range=data_range, win_size=max(3, win)))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def hd95(a: np.ndarray, b: np.ndarray, spacing_zyx) -> float:
    """95th percentile symmetric surface distance (mm)."""
    a = a.astype(bool)
    b = b.astype(bool)
    if not a.any() or not b.any():
        return float("nan")
    sa = a & ~ndimage.binary_erosion(a)
    sb = b & ~ndimage.binary_erosion(b)
    da = ndimage.distance_transform_edt(~sb, sampling=spacing_zyx)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=spacing_zyx)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def rel_error(est: float, ref: float) -> float:
    return float((est - ref) / ref) if ref != 0 else float("nan")


def local_fwhm(profile, center_idx: int, spacing: float, half_window: int = 20) -> float:
    """FWHM (mm) of the peak at ``center_idx`` of a 1-D profile, measured locally:
    baseline = 10th percentile within +-half_window samples, walk outwards from
    the peak until the profile drops below half maximum."""
    p = np.asarray(profile, float)
    lo = max(0, int(center_idx) - half_window)
    hi = min(len(p), int(center_idx) + half_window + 1)
    w = p[lo:hi] - np.percentile(p[lo:hi], 10)
    c = int(center_idx) - lo
    a, b = max(0, c - 2), min(len(w), c + 3)
    c = a + int(np.argmax(w[a:b]))
    half = w[c] / 2.0
    if half <= 0:
        return float("nan")
    l = c
    while l > 0 and w[l - 1] >= half:
        l -= 1
    r = c
    while r < len(w) - 1 and w[r + 1] >= half:
        r += 1
    return float((r - l + 1) * spacing)
