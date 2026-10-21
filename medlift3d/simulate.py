"""CT -> low-dose chest DTS projection simulation.

The CT (HU) is converted to linear attenuation and projected along the rays
of every view with a fine step (default 0.5 mm, at the CT's native voxel
grid). Reconstruction later uses different grids and coarser sampling, which
avoids the "inverse crime" of simulating and reconstructing with the same
discretisation. Clean line integrals are stored; Poisson noise is added on
demand with a seed, so the same case can be re-simulated at any dose.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .geometry import DTSGeometry, VolumeGeometry, estimate_detector_to_iso
from .physics import add_poisson_noise, hu_to_mu
from .projector import Projector


def body_extent(hu: np.ndarray, vg: VolumeGeometry, thr: float = -500.0) -> Tuple[np.ndarray, np.ndarray]:
    """World-space bounding box (min_xyz, max_xyz) of voxels above ``thr`` HU."""
    idx = np.argwhere(hu > thr)                              # rows: (iz, iy, ix)
    if idx.size == 0:
        return vg.box_min, vg.box_max
    lo = idx.min(0)[::-1].astype(float)
    hi = idx.max(0)[::-1].astype(float)
    return vg.index_to_world(lo), vg.index_to_world(hi)


def geometry_for_volume(hu: np.ndarray, vg: VolumeGeometry, gcfg: dict) -> DTSGeometry:
    """Place the DTS system around a CT: isocentre at the body centre (x, y)
    and volume centre (z); detector 10 mm in front of the anterior surface."""
    lo, hi = body_extent(hu, vg)
    iso = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), float(vg.center[2]))
    d2i = estimate_detector_to_iso(lo[1], iso[1], default=float(gcfg["detector_to_iso"]))
    return DTSGeometry.from_config(gcfg, iso=iso, detector_to_iso=d2i)


@torch.no_grad()
def simulate_clean(hu: np.ndarray, vg: VolumeGeometry, geom: DTSGeometry, mu_water: float = 0.02,
                   step_mm: float = 0.5, device="cpu", views=None) -> torch.Tensor:
    """Noise-free line integrals, shape [V, rows, cols] (float32, on ``device``)."""
    mu = torch.as_tensor(hu_to_mu(hu, mu_water), dtype=torch.float32, device=device)
    P = Projector(vg, step_mm=step_mm, use_checkpoint=False)
    nr, nc = geom.det_shape
    views = list(range(geom.n_views)) if views is None else list(views)
    out = []
    for v in views:
        o, d = geom.rays([v], device=device)
        out.append(P(mu, o, d).reshape(nr, nc))
    return torch.stack(out, 0)


def add_noise(clean: torch.Tensor, i0: float, seed: int = 0, electronic_sigma: float = 0.0) -> torch.Tensor:
    g = torch.Generator(device=clean.device)
    g.manual_seed(int(seed))
    return add_poisson_noise(clean, i0, generator=g, electronic_sigma=electronic_sigma)


@torch.no_grad()
def project_local_change(delta_mu: np.ndarray, vg_local: VolumeGeometry, geom: DTSGeometry,
                         step_mm: float = 0.5, device="cpu") -> torch.Tensor:
    """Line integrals of a small local attenuation change (e.g. nodule growth).

    Projections are linear, so a follow-up scan is ``clean_baseline + this``.
    Only pixels whose rays can hit the local box are traced.
    """
    nr, nc = geom.det_shape
    out = torch.zeros((geom.n_views, nr, nc), dtype=torch.float32, device=device)
    dmu = torch.as_tensor(delta_mu, dtype=torch.float32, device=device)
    P = Projector(vg_local, step_mm=step_mm, use_checkpoint=False)
    lo, hi = vg_local.box_min, vg_local.box_max
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    rows, cols = geom.project_points(corners)
    for v in range(geom.n_views):
        r0 = int(max(0, np.floor(rows[v].min()) - 2))
        r1 = int(min(nr, np.ceil(rows[v].max()) + 3))
        c0 = int(max(0, np.floor(cols[v].min()) - 2))
        c1 = int(min(nc, np.ceil(cols[v].max()) + 3))
        if r1 <= r0 or c1 <= c0:
            continue
        o, d = geom.window_rays([(r0, r1, c0, c1)], device=device, views=[v])
        out[v, r0:r1, c0:c1] = P(dmu, o, d).reshape(r1 - r0, c1 - c0)
    return out
