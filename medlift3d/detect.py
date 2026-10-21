"""Candidate nodule suggestions on the coarse DTS volume (training-free).

Multi-scale, scale-normalised Laplacian-of-Gaussian blob detection on the
excess-density image, restricted to the lung fields. The depth (y) blur of
DTS is accounted for by stretching the kernel along y. Candidates must be
surrounded (in-plane) by lung-like tissue, which removes rib / chest-wall
blobs. The doctor then clicks a candidate (or anywhere else) -- the coronal
slice of the click gives the depth.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage

from .geometry import VolumeGeometry


def lung_field_mask(vol_hu: np.ndarray, vg: VolumeGeometry = None, body_thr: float = -800.0,
                    lung_thr: float = -600.0, edge_mm: float = 15.0, min_width_mm: float = 20.0) -> np.ndarray:
    """Coronal (z, x) lung-field map broadcast along depth.

    DTS smears everything along depth, so lung fields are found in the coronal
    plane: columns whose mean HU over body voxels is lung-like, at least
    ``edge_mm`` inside the coronal body outline (removes the chest-wall strip)
    and part of regions wider than ``min_width_mm``.
    """
    vox = min(vg.spacing[0], vg.spacing[2]) if vg is not None else 2.5
    body = vol_hu > body_thr
    depth = body.sum(axis=1).astype(float)                         # [z, x]
    mean_body = np.where(depth > 0, (vol_hu * body).sum(axis=1) / np.maximum(depth, 1), 0.0)
    outline = ndimage.binary_fill_holes(depth > 0)
    inside = ndimage.distance_transform_edt(outline) * vox >= edge_mm
    field = inside & (mean_body < lung_thr)
    k = max(1, int(round(0.5 * min_width_mm / vox)))
    field = ndimage.binary_opening(field, structure=np.ones((3, 3), bool), iterations=k)
    return np.broadcast_to(field[:, None, :], vol_hu.shape).copy() & body


def detect_candidates(vol_hu: np.ndarray, vg: VolumeGeometry, sigmas_mm: Sequence[float] = (2.0, 3.0, 4.5, 6.0),
                      depth_stretch: float = 3.0, top_k: int = 10, min_dist_mm: float = 12.0,
                      baseline_hu: float = -850.0, ring_mm: float = 12.0, ring_hu: float = -500.0) -> List[Dict]:
    """Return up to ``top_k`` candidates sorted by LoG response."""
    excess = np.clip(vol_hu - baseline_hu, 0.0, None).astype(np.float32)
    sx, sy, sz = vg.spacing
    vox = min(sx, sz)
    sigmas = sorted({max(float(s), vox) for s in sigmas_mm})
    best = np.full(vol_hu.shape, -np.inf, np.float32)
    best_s = np.zeros(vol_hu.shape, np.float32)
    for s in sigmas:
        sig = (s / sz, s * depth_stretch / sy, s / sx)
        resp = -ndimage.gaussian_laplace(excess, sigma=sig) * (s ** 2)
        upd = resp > best
        best[upd] = resp[upd]
        best_s[upd] = s
    region = lung_field_mask(vol_hu, vg)
    # in-plane ring just outside the blob (radius ~ 1.4 sigma) must look like lung:
    # removes ribs / chest wall / mediastinum edges; ring size follows the blob scale
    ring_ok = np.zeros(vol_hu.shape, bool)
    for s in sigmas:
        r_in = max(0.5 * ring_mm, 1.5 * 1.41 * s)
        r_out = r_in + max(6.0, 1.41 * s)
        r = max(1, int(round(r_out / vox)))
        zz, xx = np.mgrid[-r:r + 1, -r:r + 1]
        d2 = (zz * sz) ** 2 + (xx * sx) ** 2
        fp = np.zeros((2 * r + 1, 1, 2 * r + 1), np.float32)
        fp[:, 0, :] = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
        ring_mean = ndimage.correlate(vol_hu.astype(np.float32), fp / fp.sum(), mode="nearest")
        ring_ok |= (best_s == s) & (ring_mean < ring_hu)
    region &= ring_ok
    best = np.where(region, best, -np.inf)
    md = max(1, int(round(min_dist_mm / vox)))
    footprint = np.ones((2 * md + 1, max(3, 2 * int(round(md * vox / sy)) + 1), 2 * md + 1), bool)
    local_max = (best == ndimage.maximum_filter(best, footprint=footprint, mode="constant",
                                                cval=-np.inf)) & np.isfinite(best) & (best > 0)
    idx = np.argwhere(local_max)
    if len(idx) == 0:
        return []
    vals = best[local_max]
    order = np.argsort(-vals)[: max(5 * top_k, 50)]
    # blobness in the coronal plane: both Hessian eigenvalues negative and similar (round),
    # vessels running in-plane are tube-like (one eigenvalue ~ 0) and get down-weighted
    smoothed = {s: ndimage.gaussian_filter(excess, sigma=(s / sz, s * depth_stretch / sy, s / sx)) for s in sigmas}
    scored = []
    for o in order:
        iz, iy, ix = idx[o]
        g = smoothed[float(best_s[iz, iy, ix])]
        z0, z1 = max(iz - 1, 0), min(iz + 2, g.shape[0])
        x0, x1 = max(ix - 1, 0), min(ix + 2, g.shape[2])
        sl = g[z0:z1, iy, x0:x1]
        if sl.shape != (3, 3):
            continue
        hzz = (sl[2, 1] - 2 * sl[1, 1] + sl[0, 1]) / sz ** 2
        hxx = (sl[1, 2] - 2 * sl[1, 1] + sl[1, 0]) / sx ** 2
        hzx = (sl[2, 2] - sl[2, 0] - sl[0, 2] + sl[0, 0]) / (4 * sx * sz)
        ev = np.linalg.eigvalsh(np.array([[hzz, hzx], [hzx, hxx]]))       # ascending: ev[0] most negative
        if ev[1] >= 0:
            continue                                                       # not a bright blob
        blob = float(ev[1] / ev[0])                                        # 1 = round, 0 = tube
        scored.append((float(vals[o]) * (0.25 + blob), o, blob))
    scored.sort(key=lambda t: -t[0])
    out = []
    for score, o, blob in scored[:top_k]:
        iz, iy, ix = idx[o]
        out.append({"xyz": [float(v) for v in vg.index_to_world(np.array([ix, iy, iz], float))],
                    "score": score, "blobness": blob, "sigma_mm": float(best_s[iz, iy, ix]),
                    "index_zyx": [int(iz), int(iy), int(ix)]})
    return out


def candidate_hits(cands: List[Dict], nodule_xyz, nodule_diam_mm: float, depth_tol_mm: float = 25.0) -> int:
    """Rank (1-based) of the first candidate hitting the nodule, 0 if none."""
    c = np.asarray(nodule_xyz, float)
    r_in = max(6.0, 0.5 * nodule_diam_mm + 3.0)
    for k, cd in enumerate(cands, start=1):
        p = np.asarray(cd["xyz"], float)
        if np.hypot(p[0] - c[0], p[2] - c[2]) <= r_in and abs(p[1] - c[1]) <= depth_tol_mm:
            return k
    return 0
