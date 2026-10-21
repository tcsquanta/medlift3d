"""Nodule growth: doubling times, risk categories and controlled virtual growth.

Doubling time  DT = dt * ln 2 / ln(Q2 / Q1)  with Q = volume (VDT) or mass (MDT).
Categories follow the NELSON / European Position Statement VDT cut-offs:
<400 days "fast (suspicious)", 400-600 "indeterminate", >600 "slow / stable".

Virtual growth: LIDC-IDRI has no follow-up scans, so a follow-up CT is made by
scaling the nodule's excess density about its centroid by a known factor s
(volume and mass both multiply by s^3). The true doubling time is then known
exactly. Because projections are linear, only the local change has to be
projected (see ``simulate.project_local_change``).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from .geometry import VolumeGeometry
from .measure import dilate_mm

FAST, INDET, SLOW = "fast (<400 d)", "indeterminate (400-600 d)", "slow/stable (>600 d)"


def doubling_time(q1: float, q2: float, dt_days: float) -> float:
    """Doubling time in days (inf if no growth)."""
    if not (q1 > 0 and q2 > 0) or q2 <= q1:
        return float("inf")
    return float(dt_days * math.log(2.0) / math.log(q2 / q1))


def growth_category(dt_days: float) -> str:
    if dt_days < 400:
        return FAST
    if dt_days <= 600:
        return INDET
    return SLOW


def scale_for_doubling(dt_days: float, doubling_days: float) -> float:
    """Linear scale factor s so that the volume grows by 2^(dt/DT) = s^3."""
    dt_days, doubling_days = float(dt_days), float(doubling_days)
    if not np.isfinite(doubling_days) or doubling_days >= 1e8:
        return 1.0
    return float(2.0 ** (dt_days / doubling_days / 3.0))


def virtual_growth(hu: np.ndarray, vg: VolumeGeometry, mask: np.ndarray, scale: float,
                   baseline_hu: float, soft_mm: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Grow (scale > 1) a nodule inside a CT patch. Returns (new_hu, new_mask).

    excess = clip(HU - baseline, 0) inside the softly dilated mask; the patch is
    first "emptied" (excess removed), then the excess resampled with zoom
    ``scale`` about the mask centroid is added back.
    """
    if abs(scale - 1.0) < 1e-6:
        return hu.copy(), mask.copy()
    region = dilate_mm(mask, vg, (soft_mm,) * 3)
    w = ndimage.gaussian_filter(region.astype(np.float32), sigma=0.7)
    excess = np.clip(hu - baseline_hu, 0.0, None) * w
    emptied = hu - excess
    c = np.array(ndimage.center_of_mass(mask))                       # (iz, iy, ix) index space
    # output index p maps to input index c + (p - c) / scale (same factor on every axis)
    matrix = np.full(3, 1.0 / scale)
    offset = c - matrix * c
    grown = ndimage.affine_transform(excess, matrix, offset=offset, order=1, mode="constant", cval=0.0)
    gmask = ndimage.affine_transform(mask.astype(np.float32), matrix, offset=offset, order=1,
                                     mode="constant", cval=0.0) >= 0.5
    return (emptied + grown).astype(np.float32), gmask
