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


def _grow_fine(hu: np.ndarray, sp: np.ndarray, mask: np.ndarray, scale: float, baseline_hu: float,
               soft_mm: float) -> Tuple[np.ndarray, np.ndarray, float]:
    region = dilate_mm(mask, VolumeGeometry(hu.shape, tuple(sp[::-1]), (0.0, 0.0, 0.0)), (soft_mm,) * 3)
    w = ndimage.gaussian_filter(region.astype(np.float32), sigma=0.5 / sp)      # ~0.5 mm taper
    excess = np.clip(hu - baseline_hu, 0.0, None) * w
    emptied = hu - excess
    c = np.array(ndimage.center_of_mass(mask))                                   # (iz, iy, ix)
    matrix = np.full(3, 1.0 / scale)                                             # output p <- input c + (p - c) / s
    offset = c - matrix * c
    grown = ndimage.affine_transform(excess, matrix, offset=offset, order=1, mode="constant", cval=0.0)
    gmask = ndimage.affine_transform(mask.astype(np.float32), matrix, offset=offset, order=1,
                                     mode="constant", cval=0.0)
    return emptied + grown, gmask, float(excess.sum())


def virtual_growth(hu: np.ndarray, vg: VolumeGeometry, mask: np.ndarray, scale: float,
                   baseline_hu: float, soft_mm: float = 2.0, fine_mm: float = 0.5,
                   return_info: bool = False):
    """Grow (scale > 1) a nodule inside a CT patch. Returns (new_hu, new_mask[, info]).

    excess = clip(HU - baseline, 0) inside the softly dilated mask; the patch is
    "emptied" (excess removed) and the excess, zoomed by ``scale`` about the mask
    centroid, is added back. To be exact for thick CT slices the operation is done
    on a crop upsampled to ``fine_mm`` and averaged back to the CT grid, so the
    added mass is (s^3 - 1) x the nodule's excess mass at any voxel size.
    """
    if abs(scale - 1.0) < 1e-6:
        out = (hu.copy(), mask.copy())
        return out + ({"excess_mass_mg": float("nan"), "added_mass_mg": 0.0},) if return_info else out
    sp = np.asarray(vg.spacing, float)[::-1]                                     # (z, y, x) mm
    f = np.maximum(1, np.ceil(sp / fine_mm - 1e-6)).astype(int)
    if np.all(f == 1):
        new, gm, exc = _grow_fine(hu.astype(np.float32), sp, mask, scale, baseline_hu, soft_mm)
        gmask = gm >= 0.5
        new_hu = new.astype(np.float32)
    else:
        hu_f = np.repeat(np.repeat(np.repeat(hu.astype(np.float32), f[0], 0), f[1], 1), f[2], 2)
        m_f = ndimage.zoom(mask.astype(np.float32), f, order=1) >= 0.5           # smooth-ish fine mask
        m_f = m_f[: hu_f.shape[0], : hu_f.shape[1], : hu_f.shape[2]]
        new_f, gm_f, exc = _grow_fine(hu_f, sp / f, m_f, scale, baseline_hu, soft_mm)
        shp = (hu.shape[0], f[0], hu.shape[1], f[1], hu.shape[2], f[2])
        delta = (new_f - hu_f).reshape(shp).mean(axis=(1, 3, 5))                # change, averaged back
        new_hu = (hu + delta).astype(np.float32)
        gmask = gm_f.reshape(shp).mean(axis=(1, 3, 5)) >= 0.5
        exc = exc / float(np.prod(f))
    vv = float(np.prod(vg.spacing))
    info = {"excess_mass_mg": exc / 1000.0 * vv, "added_mass_mg": float((new_hu - hu).sum()) / 1000.0 * vv}
    return (new_hu, gmask, info) if return_info else (new_hu, gmask)
