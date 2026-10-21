"""Synthetic chest phantom with nodules of known size and mass.

Used by the unit tests and the end-to-end smoke test so every stage can be
checked against an exact ground truth before touching real LIDC-IDRI data.
Anatomy (all in HU): body (soft tissue), two lungs with smooth texture,
heart/mediastinum, spine, ribs, branching-ish vessels, and nodules (solid or
ground-glass, spherical or ellipsoidal, with a partial-volume soft edge).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .geometry import VolumeGeometry


@dataclass
class NoduleSpec:
    center: Tuple[float, float, float]          # world (x, y, z) mm
    radii: Tuple[float, float, float]           # semi-axes (x, y, z) mm
    hu: float = 20.0                            # solid ~0..40 HU, GGO ~-600
    name: str = ""


@dataclass
class Phantom:
    hu: np.ndarray                              # [z, y, x] float32
    vgeom: VolumeGeometry
    nodules: List[NoduleSpec]
    masks: List[np.ndarray]                     # boolean [z, y, x], one per nodule
    lung_mask: np.ndarray
    body_mask: np.ndarray
    lung_hu: float = -850.0
    meta: dict = field(default_factory=dict)


def _grid(vg: VolumeGeometry):
    x, y, z = vg.axes()
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)


def _smooth_noise(shape, sigma_vox, amp, rng) -> np.ndarray:
    n = rng.standard_normal(shape).astype(np.float32)
    n = ndimage.gaussian_filter(n, sigma_vox)
    n /= n.std() + 1e-8
    return amp * n


def _soft_ellipsoid(X, Y, Z, c, r, edge_mm: float):
    """Soft indicator (0..1) of an ellipsoid, edge width ~ edge_mm (partial volume)."""
    q = np.sqrt(((X - c[0]) / r[0]) ** 2 + ((Y - c[1]) / r[1]) ** 2 + ((Z - c[2]) / r[2]) ** 2)
    sd = (q - 1.0) * float(min(r))                       # approx signed distance (mm)
    return 1.0 / (1.0 + np.exp(np.clip(sd / max(edge_mm, 1e-3), -50, 50))), sd


def _tube(X, Y, Z, p0, p1, radius):
    """Indicator of a capsule between p0 and p1."""
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    v = p1 - p0
    L2 = float(v @ v) + 1e-6
    t = ((X - p0[0]) * v[0] + (Y - p0[1]) * v[1] + (Z - p0[2]) * v[2]) / L2
    t = np.clip(t, 0, 1)
    dx = X - (p0[0] + t * v[0])
    dy = Y - (p0[1] + t * v[1])
    dz = Z - (p0[2] + t * v[2])
    return (dx * dx + dy * dy + dz * dz) <= radius * radius


def random_nodule_specs(rng, n: int, lung_centers, lung_radii, diam_range=(4.0, 18.0),
                        ggo_fraction: float = 0.2, min_sep: float = 30.0) -> List[NoduleSpec]:
    specs: List[NoduleSpec] = []
    tries = 0
    while len(specs) < n and tries < 2000:
        tries += 1
        side = rng.integers(0, 2)
        c0 = np.asarray(lung_centers[side], float)
        r0 = np.asarray(lung_radii[side], float)
        u = rng.uniform(-1, 1, 3)
        if np.sum(u ** 2) > 0.45:                                   # stay well inside the lung
            continue
        c = c0 + u * r0
        if any(np.linalg.norm(c - np.asarray(s.center)) < min_sep for s in specs):
            continue
        d = rng.uniform(*diam_range)
        elong = rng.uniform(0.8, 1.25, 3)
        radii = tuple(float(v) for v in (d / 2.0) * elong / np.prod(elong) ** (1 / 3))
        is_ggo = rng.uniform() < ggo_fraction
        hu = float(rng.uniform(-650, -500)) if is_ggo else float(rng.uniform(-20, 60))
        specs.append(NoduleSpec(tuple(float(v) for v in c), radii, hu, f"N{len(specs)}{'g' if is_ggo else 's'}"))
    return specs


def make_chest_phantom(spacing: Sequence[float] = (2.0, 2.0, 2.0),
                       fov_mm: Sequence[float] = (360.0, 260.0, 280.0),
                       nodules: Optional[List[NoduleSpec]] = None, n_random_nodules: int = 4,
                       seed: int = 0, n_vessels: int = 40, with_ribs: bool = True,
                       edge_mm: float = 0.6) -> Phantom:
    """Build the phantom on a grid centred at the origin.

    fov_mm = (x, y, z) extent. Default 2 mm voxels -> 180 x 130 x 140 voxels.
    """
    rng = np.random.default_rng(seed)
    sp = np.asarray(spacing, float)
    n_xyz = np.maximum(8, np.round(np.asarray(fov_mm, float) / sp)).astype(int)
    vg = VolumeGeometry.centered((n_xyz[2], n_xyz[1], n_xyz[0]), sp, (0.0, 0.0, 0.0))
    X, Y, Z = _grid(vg)
    vox = float(min(sp))

    hu = np.full(vg.shape, -1000.0, dtype=np.float32)
    # body: elliptic cylinder along z, slightly tapered
    a, b = 165.0, 115.0
    taper = 1.0 - 0.10 * (Z / (fov_mm[2] / 2.0)) ** 2
    body = (X / (a * taper)) ** 2 + (Y / (b * taper)) ** 2 <= 1.0
    hu[body] = 30.0 + _smooth_noise(vg.shape, 3.0 / vox, 8.0, rng)[body]

    lung_centers = [(-72.0, 0.0, 15.0), (72.0, 0.0, 15.0)]
    lung_radii = [(58.0, 82.0, 108.0), (60.0, 84.0, 110.0)]
    lung_hu = -850.0
    lung = np.zeros(vg.shape, bool)
    for c, r in zip(lung_centers, lung_radii):
        lung |= ((X - c[0]) / r[0]) ** 2 + ((Y - c[1]) / r[1]) ** 2 + ((Z - c[2]) / r[2]) ** 2 <= 1.0
    lung &= body
    texture = _smooth_noise(vg.shape, 2.5 / vox, 25.0, rng)
    hu[lung] = lung_hu + texture[lung]

    # heart / mediastinum (anterior, between lungs, sits on top of lungs)
    heart = ((X - 12.0) / 55.0) ** 2 + ((Y + 35.0) / 48.0) ** 2 + ((Z + 45.0) / 62.0) ** 2 <= 1.0
    hu[heart & body] = 40.0
    # aorta / trachea-ish column
    hu[_tube(X, Y, Z, (5.0, 10.0, -60.0), (5.0, 10.0, 130.0), 13.0) & body] = 45.0
    trachea = _tube(X, Y, Z, (0.0, -5.0, 60.0), (0.0, -5.0, 140.0), 8.0) & body
    hu[trachea] = -950.0

    # vessels: tubes radiating from the hila, clipped to the lungs
    for side in (0, 1):
        hilum = np.array([(-30.0 if side == 0 else 30.0), 5.0, 20.0])
        for _ in range(n_vessels // 2):
            direction = rng.normal(size=3)
            direction[0] = abs(direction[0]) * (-1 if side == 0 else 1)
            direction /= np.linalg.norm(direction)
            length = rng.uniform(40.0, 95.0)
            p1 = hilum + direction * length
            r_v = rng.uniform(1.0, 2.6)
            tube = _tube(X, Y, Z, hilum, p1, r_v) & lung
            hu[tube] = 30.0 + rng.uniform(-10, 10)

    # spine (posterior) and ribs
    hu[_tube(X, Y, Z, (0.0, 88.0, -150.0), (0.0, 88.0, 150.0), 17.0) & body] = 650.0
    if with_ribs:
        for zr in np.arange(-110.0, 125.0, 24.0):
            for t in np.linspace(0, 2 * np.pi, 60, endpoint=False):
                p0 = (152.0 * np.cos(t), 102.0 * np.sin(t), zr + 6.0 * np.sin(t))
                t2 = t + 2 * np.pi / 60
                p1 = (152.0 * np.cos(t2), 102.0 * np.sin(t2), zr + 6.0 * np.sin(t2))
                if np.sin(t) < -0.85:          # leave a gap anteriorly (sternum region)
                    continue
                seg = _tube(X, Y, Z, p0, p1, 4.0) & body
                hu[seg] = 550.0

    # nodules
    if nodules is None:
        nodules = random_nodule_specs(rng, n_random_nodules, lung_centers, lung_radii)
    masks = []
    for s in nodules:
        soft, sd = _soft_ellipsoid(X, Y, Z, s.center, s.radii, edge_mm)
        region = soft > 1e-3
        hu[region] = hu[region] * (1.0 - soft[region]) + s.hu * soft[region]
        masks.append(sd <= 0.0)

    return Phantom(hu=hu.astype(np.float32), vgeom=vg, nodules=list(nodules), masks=masks,
                   lung_mask=lung, body_mask=body, lung_hu=lung_hu,
                   meta={"seed": seed, "lung_centers": lung_centers, "lung_radii": lung_radii})


def sphere_volume(hu_inside: float, radius_mm: float, spacing: float = 0.5, size_mm: float = 40.0,
                  background_hu: float = -1000.0, edge_mm: float = 0.0):
    """Small test volume with one sphere at the origin (for analytic checks)."""
    n = int(round(size_mm / spacing))
    vg = VolumeGeometry.centered((n, n, n), (spacing,) * 3, (0.0, 0.0, 0.0))
    X, Y, Z = _grid(vg)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    if edge_mm > 0:
        w = 1.0 / (1.0 + np.exp((r - radius_mm) / edge_mm))
    else:
        # exact partial-volume fraction by 4x supersampling of each voxel
        w = np.zeros_like(r)
        offs = (np.arange(4) + 0.5) / 4 - 0.5
        for ox in offs:
            for oy in offs:
                for oz in offs:
                    rr = np.sqrt((X + ox * spacing) ** 2 + (Y + oy * spacing) ** 2 + (Z + oz * spacing) ** 2)
                    w += (rr <= radius_mm)
        w /= 64.0
    hu = background_hu + (hu_inside - background_hu) * w
    return hu.astype(np.float32), vg
