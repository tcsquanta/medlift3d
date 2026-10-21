"""Nodule measurements: segmentation, volume, excess mass, 2-D diameter.

Definitions (applied identically to CT ground truth and every reconstruction)
-----------------------------------------------------------------------------
* volume  V = (#voxels in nodule mask) x voxel volume                [mm^3]
* mass    M = sum over (mask dilated by ``mass_dilate_mm``; the depth
              dilation grows with nodule size, ``mass_depth_factor``) of
              (HU - HU_lung) / 1000 x voxel volume                   [mg]
  i.e. the *excess* water-equivalent mass of the nodule over local lung
  (1 mg/mm^3 water density). Integrating over a dilated region makes M
  insensitive to blur / partial volume / depth smear -- that is the point.
* d2D     longest in-plane (coronal, x-z) diameter of the mask through its
          peak slice, the measurement a radiologist makes on DTS slices;
          V2D = pi/6 * d^3 (sphere assumption).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .geometry import VolumeGeometry


def ball(radius_vox: int) -> np.ndarray:
    r = int(radius_vox)
    z, y, x = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= r * r


def sphere_mask(vg: VolumeGeometry, center, radius_mm) -> np.ndarray:
    """Ellipsoid mask; ``radius_mm`` is a scalar or per-axis (x, y, z) radii."""
    r = np.broadcast_to(np.asarray(radius_mm, float), (3,))
    x, y, z = vg.axes()
    Z, Y, X = np.meshgrid((z - center[2]) / r[2], (y - center[1]) / r[1], (x - center[0]) / r[0], indexing="ij")
    return (X ** 2 + Y ** 2 + Z ** 2) <= 1.0


def dilate_mm(mask: np.ndarray, vg: VolumeGeometry, radii_mm: Sequence[float]) -> np.ndarray:
    """Ellipsoidal dilation with per-axis radii (x, y, z) in mm."""
    if not mask.any():
        return mask.copy()
    rx, ry, rz = (max(float(r), 1e-3) for r in radii_mm)
    sx, sy, sz = vg.spacing
    dist = ndimage.distance_transform_edt(~mask, sampling=(sz / rz, sy / ry, sx / rx))
    return dist <= 1.0


def segment_nodule(vol_hu: np.ndarray, vg: VolumeGeometry, seed_xyz, baseline_hu: float,
                   mode: str = "halfmax", fixed_hu: float = -400.0, floor_hu: float = -1000.0,
                   search_radius_mm=(6.0, 12.0, 6.0), open_radius_vox: int = 1,
                   min_contrast_hu: float = 50.0, max_extent_mm=(25.0, 50.0, 25.0)) -> Tuple[np.ndarray, Dict]:
    """Seeded nodule segmentation.

    halfmax: threshold = baseline + 0.5 (peak - baseline), peak = max HU within
    the (depth-elongated) search ellipsoid around the seed; fixed: threshold = ``fixed_hu``.
    The threshold is at least ``min_contrast_hu`` above the local lung (so faint
    ground-glass nodules still segment) and the result is confined to an
    ellipsoid of radii ``max_extent_mm`` around the peak (no leaks into the
    chest wall / heart). A morphological opening detaches thin vessels; the
    connected component at the peak is kept and its eroded rim restored.
    """
    near = sphere_mask(vg, seed_xyz, search_radius_mm)
    info: Dict = {"threshold_hu": float("nan"), "peak_hu": float("nan"), "peak_xyz": [float("nan")] * 3,
                  "touches_limit": False}
    if not near.any():
        return np.zeros(vol_hu.shape, bool), info
    masked = np.where(near, vol_hu, -np.inf)
    pk = np.unravel_index(int(np.argmax(masked)), vol_hu.shape)
    peak = float(vol_hu[pk])
    thr = baseline_hu + 0.5 * (peak - baseline_hu) if mode == "halfmax" else float(fixed_hu)
    thr = max(thr, float(floor_hu), baseline_hu + min_contrast_hu)
    peak_xyz = vg.index_to_world(np.array(pk[::-1], float))
    info.update(threshold_hu=float(thr), peak_hu=peak, peak_xyz=[float(v) for v in peak_xyz])
    if peak <= thr or peak - baseline_hu < 2.0 * min_contrast_hu:
        return np.zeros(vol_hu.shape, bool), info
    limit = sphere_mask(vg, peak_xyz, max_extent_mm)
    binary = (vol_hu > thr) & limit
    comp = None
    if open_radius_vox > 0:
        opened = ndimage.binary_opening(binary, structure=ball(open_radius_vox))
        if opened[pk]:
            lab, _ = ndimage.label(opened)
            comp = lab == lab[pk]
            comp = ndimage.binary_dilation(comp, structure=ball(open_radius_vox)) & binary
    if comp is None:
        lab, _ = ndimage.label(binary)
        comp = lab == lab[pk]
    edge = limit & ~ndimage.binary_erosion(limit)
    info["touches_limit"] = bool((comp & edge).sum() > 0.05 * max(1, edge.sum()))
    return comp, info


def volume_mm3(mask: np.ndarray, vg: VolumeGeometry) -> float:
    return float(mask.sum()) * vg.voxel_volume


def mass_region(mask: np.ndarray, vg: VolumeGeometry, dilate: Sequence[float] = (2.0, 6.0, 2.0),
                depth_factor: float = 0.0) -> np.ndarray:
    """Mask dilated by (x, y, z) mm; the depth (y) dilation grows with nodule size:
    max(dilate_y, depth_factor * in-plane long diameter) -- DTS smears ~ size / tan(15 deg)."""
    if not mask.any():
        return mask.copy()
    dil = list(dilate)
    if depth_factor > 0:
        d_long, _ = coronal_diameters(mask, vg)
        dil[1] = max(dil[1], depth_factor * d_long)
    return dilate_mm(mask, vg, dil)


def excess_mass_mg(vol_hu: np.ndarray, vg: VolumeGeometry, mask: np.ndarray, baseline_hu: float,
                   dilate: Sequence[float] = (2.0, 6.0, 2.0), depth_factor: float = 0.0) -> float:
    if not mask.any():
        return 0.0
    region = mass_region(mask, vg, dilate, depth_factor)
    return float(((vol_hu[region] - baseline_hu) / 1000.0).sum() * vg.voxel_volume)


def coronal_diameters(mask: np.ndarray, vg: VolumeGeometry) -> Tuple[float, float]:
    """(long, short) axis in mm on the coronal (x-z) slice with the largest area."""
    if not mask.any():
        return 0.0, 0.0
    areas = mask.sum(axis=(0, 2))                                   # per y index
    iy = int(np.argmax(areas))
    sl = mask[:, iy, :]                                             # [z, x]
    zz, xx = np.nonzero(sl)
    pts = np.stack([xx * vg.spacing[0], zz * vg.spacing[2]], -1).astype(float)
    if len(pts) == 1:
        return float(vg.spacing[0]), float(vg.spacing[0])
    # include voxel extent: use the 4 corners of every boundary voxel
    half = np.array([vg.spacing[0], vg.spacing[2]]) / 2.0
    corners = np.concatenate([pts + half * np.array(s) for s in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
    try:
        from scipy.spatial import ConvexHull

        hull = corners[ConvexHull(corners).vertices]
    except Exception:
        hull = corners
    d = np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    long_ax = float(d[i, j])
    u = (hull[j] - hull[i]) / max(long_ax, 1e-9)
    perp = np.array([-u[1], u[0]])
    proj = hull @ perp
    return long_ax, float(proj.max() - proj.min())


def sphere_volume(d_mm: float) -> float:
    return math.pi / 6.0 * d_mm ** 3


def equivalent_diameter(volume_mm3: float) -> float:
    return (6.0 * max(volume_mm3, 0.0) / math.pi) ** (1.0 / 3.0)


def measure_reconstruction(vol_hu: np.ndarray, vg: VolumeGeometry, seed_xyz, baseline_hu: float,
                           mcfg: dict) -> Tuple[Dict, np.ndarray]:
    """All measurements for one reconstructed ROI (HU). Returns (dict, mask)."""
    mask, info = segment_nodule(vol_hu, vg, seed_xyz, baseline_hu, mode=mcfg["mode"],
                                fixed_hu=mcfg["fixed_hu"], floor_hu=mcfg["floor_hu"],
                                search_radius_mm=mcfg["search_radius_mm"],
                                open_radius_vox=mcfg["open_radius_vox"],
                                min_contrast_hu=float(mcfg.get("min_contrast_hu", 50.0)),
                                max_extent_mm=mcfg.get("max_extent_mm", (25.0, 50.0, 25.0)))
    v = volume_mm3(mask, vg)
    m = excess_mass_mg(vol_hu, vg, mask, baseline_hu, mcfg["mass_dilate_mm"], float(mcfg.get("mass_depth_factor", 0.0)))
    d_long, d_short = coronal_diameters(mask, vg)
    out = {"volume_mm3": v, "mass_mg": m, "d2d_long_mm": d_long, "d2d_short_mm": d_short,
           "volume2d_mm3": sphere_volume(d_long), "volume2d_mean_mm3": sphere_volume(0.5 * (d_long + d_short)),
           "baseline_hu": float(baseline_hu), **info}
    return out, mask


def measure_ground_truth(vol_hu: np.ndarray, vg: VolumeGeometry, mask: np.ndarray, mcfg: dict,
                         baseline_hu: Optional[float] = None) -> Dict:
    """Ground-truth measurements from CT + radiologist mask (same definitions)."""
    if baseline_hu is None:
        shell = dilate_mm(mask, vg, (15.0, 15.0, 15.0)) & ~dilate_mm(mask, vg, (5.0, 5.0, 5.0))
        sel = shell & (vol_hu > -1000) & (vol_hu < -500)
        baseline_hu = float(np.median(vol_hu[sel])) if sel.sum() > 30 else -850.0
    d_long, d_short = coronal_diameters(mask, vg)
    return {"volume_mm3": volume_mm3(mask, vg),
            "mass_mg": excess_mass_mg(vol_hu, vg, mask, baseline_hu, mcfg["mass_dilate_mm"],
                                      float(mcfg.get("mass_depth_factor", 0.0))),
            "d2d_long_mm": d_long, "d2d_short_mm": d_short,
            "eq_diameter_mm": equivalent_diameter(volume_mm3(mask, vg)),
            "baseline_hu": float(baseline_hu),
            "mean_hu": float(vol_hu[mask].mean()) if mask.any() else float("nan")}


def gaussian_mass_mg(gf, region: np.ndarray, vg: VolumeGeometry, mu_water: float = 0.02) -> float:
    """Closed-form mass: sum of exact Gaussian integrals whose centres lie in ``region``."""
    from .gaussians import gaussian_centers_in_mask

    inside = gaussian_centers_in_mask(gf, region, vg)
    if not inside.any():
        return 0.0
    integ = gf.integrals().detach().cpu().numpy()
    return float(integ[inside].sum() / mu_water)
