"""ROI reconstruction problem around a clicked nodule.

Design ("zoom in-plane, not in depth")
--------------------------------------
DTS is sharp in-plane (x, z) and blind in depth (y). A small 3-D box around
the nodule does not work: each ray's attenuation mostly comes from outside
the box, and any error in that background swamps the nodule. We therefore
reconstruct a *column* ("prism"): small in-plane (default 64 x 64 mm at 1 mm)
but spanning the whole chest depth (default 128 x 2 mm, starting just behind
the detector). Rays are kept only if most of their attenuation lies inside
the prism, so the out-of-prism background (taken from the coarse
reconstruction) is a small correction:

    measured = background (outside prism, coarse SIRT) + prism content (unknown)

Inside the prism, a smaller *nodule box* around the click holds the
3-D Gaussians; the rest of the prism is kept fixed (see ``SubProblem``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .geometry import DTSGeometry, VolumeGeometry, depth_support
from .physics import hu_to_mu, mu_to_hu
from .projector import Projector, ray_aabb
from .sirt import SIRT


@dataclass
class ROI:
    click: Tuple[float, float, float]                 # world (x, y, z) mm: nodule seed
    size_vox: Tuple[int, int, int] = (64, 128, 64)    # prism (nx, ny, nz)
    voxel_mm: Tuple[float, float, float] = (1.0, 2.0, 1.0)
    y_start: float = float("nan")                     # world y of the prism's anterior face
    box_mm: Tuple[float, float, float] = (40.0, 50.0, 40.0)

    @classmethod
    def from_config(cls, click, rcfg: dict, geom: Optional[DTSGeometry] = None) -> "ROI":
        size = tuple(int(s) for s in rcfg["size_vox"])
        vox = tuple(float(v) for v in rcfg["voxel_mm"])
        box = tuple(float(b) for b in rcfg.get("box_mm", (40.0, 50.0, 40.0)))
        if geom is not None:
            y0 = geom.det_y + float(rcfg.get("depth_start_mm", 5.0))
            # deep (posterior) nodule in a thick chest: shift the prism back so the nodule box
            # (+10 mm) stays inside; the anterior tissue it leaves out becomes background
            y0 = max(y0, float(click[1]) + 0.5 * box[1] + 10.0 - size[1] * vox[1])
        else:                                           # no geometry: centre the prism on the click
            y0 = float(click[1]) - 0.5 * size[1] * vox[1]
        return cls(tuple(float(c) for c in click), size, vox, float(y0), box)

    # ---------------------------------------------------------------- prism
    @property
    def vgeom(self) -> VolumeGeometry:
        nx, ny, nz = self.size_vox
        sx, sy, sz = self.voxel_mm
        cy = self.y_start + 0.5 * ny * sy
        return VolumeGeometry.centered((nz, ny, nx), self.voxel_mm, (self.click[0], cy, self.click[2]))

    @property
    def box_min(self) -> np.ndarray:
        return self.vgeom.box_min

    @property
    def box_max(self) -> np.ndarray:
        return self.vgeom.box_max

    # ------------------------------------------------------------ nodule box
    def box_slices(self) -> Tuple[slice, slice, slice]:
        """Index slices (z, y, x) of the nodule box inside the prism arrays."""
        vg = self.vgeom
        c = vg.world_to_index(np.asarray(self.click))                  # (ix, iy, iz)
        half = np.asarray(self.box_mm) / vg.spacing_arr / 2.0
        n = vg.n_xyz
        lo = np.clip(np.round(c - half).astype(int), 0, n - 1)
        hi = np.clip(np.round(c + half).astype(int) + 1, lo + 1, n)
        return slice(lo[2], hi[2]), slice(lo[1], hi[1]), slice(lo[0], hi[0])

    @property
    def box_vgeom(self) -> VolumeGeometry:
        vg = self.vgeom
        sz, sy, sx = self.box_slices()
        origin = vg.index_to_world(np.array([sx.start, sy.start, sz.start], float))
        return VolumeGeometry((sz.stop - sz.start, sy.stop - sy.start, sx.stop - sx.start), vg.spacing, tuple(origin))


def footprint_windows(roi: ROI, geom: DTSGeometry, margin_px: int = 3) -> List[Tuple[int, int, int, int]]:
    """Equal-size detector windows (r0, r1, c0, c1) covering the prism in every view."""
    lo, hi = roi.box_min, roi.box_max
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    rows, cols = geom.project_points(corners)                       # V,8
    nr, nc = geom.det_shape
    rmin = np.floor(rows.min(1)) - margin_px
    rmax = np.ceil(rows.max(1)) + margin_px
    cmin = np.floor(cols.min(1)) - margin_px
    cmax = np.ceil(cols.max(1)) + margin_px
    h = int(min(nr, (rmax - rmin).max() + 1))
    w = int(min(nc, (cmax - cmin).max() + 1))
    wins = []
    for v in range(geom.n_views):
        rc = 0.5 * (rmin[v] + rmax[v])
        cc = 0.5 * (cmin[v] + cmax[v])
        r0 = int(np.clip(round(rc - h / 2.0), 0, nr - h))
        c0 = int(np.clip(round(cc - w / 2.0), 0, nc - w))
        wins.append((r0, r0 + h, c0, c0 + w))
    return wins


class ROIProblem:
    """Measured data + forward model for one prism ROI.

    proj      : measured (noisy) line integrals [V, rows, cols] (torch)
    coarse_mu : coarse whole-chest reconstruction (attenuation) and its grid
    min_ray_fraction : keep rays whose attenuation (per the coarse model) lies
                       at least this much inside the prism.
    """

    def __init__(self, roi: ROI, geom: DTSGeometry, proj: torch.Tensor, coarse_mu: torch.Tensor,
                 coarse_vg: VolumeGeometry, step_mm: float = 1.0, coarse_step_mm: Optional[float] = None,
                 margin_px: int = 3, device="cpu", mu_water: float = 0.02, min_ray_fraction: float = 0.5):
        self.roi = roi
        self.vg = roi.vgeom
        self.geom = geom
        self.device = device
        self.mu_water = mu_water
        self.step = float(step_mm)
        self.windows = footprint_windows(roi, geom, margin_px)
        self.win_shape = (self.windows[0][1] - self.windows[0][0], self.windows[0][3] - self.windows[0][2])
        o, d = geom.window_rays(self.windows, device=device)
        meas = torch.stack([proj[v, r0:r1, c0:c1] for v, (r0, r1, c0, c1) in enumerate(self.windows)], 0)
        meas = meas.reshape(-1).to(device).float()
        bgP = Projector(coarse_vg, step_mm=coarse_step_mm or 0.5 * float(min(coarse_vg.spacing)), use_checkpoint=False)
        # attenuation can only sit in front of the detector, inside the patient: a coarse volume
        # reconstructed without that support carries depth smear there, which must not count as background
        cm = coarse_mu.to(device).float() * torch.as_tensor(depth_support(coarse_vg, geom), device=device)
        with torch.no_grad():
            bg_out = bgP(cm, o, d, exclude_box=(roi.box_min, roi.box_max))
            bg_tot = bgP(cm, o, d)
        t0, t1 = ray_aabb(o, d, roi.box_min, roi.box_max)
        frac = 1.0 - bg_out / bg_tot.clamp(min=1e-6)
        keep = (t1 > t0 + 1e-3) & (bg_tot > 1e-3) & (frac >= min_ray_fraction)
        if int(keep.sum()) < 50:                     # degenerate case: relax the criterion
            keep = (t1 > t0 + 1e-3) & (bg_tot > 1e-3) & (frac >= 0.5 * min_ray_fraction)
        self.keep = keep
        self.ray_fraction = frac
        self.o, self.d = o[keep], d[keep]
        self.meas = meas[keep]
        self.background = bg_out[keep]
        self.residual = self.meas - self.background
        self.P = Projector(self.vg, step_mm=step_mm)
        self.support = torch.as_tensor(depth_support(self.vg, geom), device=device)   # [1, H, 1]
        self._sirt: Optional[SIRT] = None

    # ------------------------------------------------------------------ model
    def forward(self, mu_roi: torch.Tensor) -> torch.Tensor:
        return self.P(mu_roi, self.o, self.d)

    def data_loss(self, mu_roi: torch.Tensor) -> torch.Tensor:
        return ((self.forward(mu_roi) - self.residual) ** 2).mean(dim=-1)

    def sirt_solver(self) -> SIRT:
        if self._sirt is None:
            self._sirt = SIRT(self.P, self.o, self.d, device=self.device, support=self.support)
        return self._sirt

    def sirt(self, n_iter: int = 40, x0_mu: Optional[torch.Tensor] = None, init_hu: float = -850.0,
             relax: float = 1.0) -> Tuple[torch.Tensor, List[float]]:
        """Physics-only prism reconstruction (attenuation), starting from uniform lung."""
        if x0_mu is None:
            D, H, W = self.vg.shape
            x0_mu = torch.full((D, H, W), float(hu_to_mu(np.float32(init_hu), self.mu_water)), device=self.device)
        return self.sirt_solver().run(self.residual, n_iter=n_iter, x0=x0_mu, relax=relax)

    def sirt_steps(self, mu: torch.Tensor, n_iter: int = 2, relax: float = 1.0) -> torch.Tensor:
        """A few SIRT updates from ``mu`` ([D,H,W] or [B,D,H,W]) -- diffusion data consistency."""
        x, _ = self.sirt_solver().run(self.residual, n_iter=n_iter, x0=mu, relax=relax)
        return x

    def residual_image(self, values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Ray values (default: residual) on the full windows [V, h, w]; unused rays = NaN."""
        v = self.residual if values is None else values
        full = torch.full((self.keep.numel(),), float("nan"), device=v.device)
        full[self.keep] = v
        return full.reshape(self.geom.n_views, *self.win_shape)


class SubProblem:
    """Nodule box inside a prism, with the rest of the prism fixed.

    residual_box = prism residual - projection of the fixed prism content
                   outside the box (rays that miss the box are dropped).
    Exposes ``vg``, ``device``, ``residual`` and ``forward`` like ROIProblem,
    so ``gaussians.refine_with_projections`` works on it unchanged.
    """

    def __init__(self, parent: ROIProblem, fixed_mu_prism: torch.Tensor):
        self.parent = parent
        self.roi = parent.roi
        self.vg = parent.roi.box_vgeom
        self.device = parent.device
        bmin, bmax = self.vg.box_min, self.vg.box_max
        t0, t1 = ray_aabb(parent.o, parent.d, bmin, bmax)
        hit = t1 > t0 + 1e-3
        self.o, self.d = parent.o[hit], parent.d[hit]
        with torch.no_grad():
            outside = parent.P(fixed_mu_prism.to(parent.device).float(), self.o, self.d, exclude_box=(bmin, bmax))
        self.residual = parent.residual[hit] - outside
        self.P = Projector(self.vg, step_mm=parent.step)

    def forward(self, mu_box: torch.Tensor) -> torch.Tensor:
        return self.P(mu_box, self.o, self.d)


def estimate_baseline_hu(vol_hu: np.ndarray, vg: VolumeGeometry, center, inner_radius_mm: float = 15.0,
                         hu_range=(-1000.0, -500.0), default: float = -850.0, max_radius_mm: float = 45.0) -> float:
    """Local lung baseline: median HU of lung-range voxels near, but not at, the nodule."""
    x, y, z = vg.axes()
    Z, Y, X = np.meshgrid(z - center[2], y - center[1], x - center[0], indexing="ij")
    r2 = X ** 2 + Y ** 2 + Z ** 2
    sel = (r2 > inner_radius_mm ** 2) & (r2 < max_radius_mm ** 2) & (vol_hu > hu_range[0]) & (vol_hu < hu_range[1])
    if sel.sum() < 50:
        return float(default)
    return float(np.median(vol_hu[sel]))


def resample_to_grid(vol: np.ndarray, src_vg: VolumeGeometry, dst_vg: VolumeGeometry, order: int = 1,
                     cval: float = -1000.0) -> np.ndarray:
    """Resample a volume onto another (axis-aligned) grid with scipy map_coordinates."""
    from scipy.ndimage import map_coordinates

    x, y, z = dst_vg.axes()
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    ix = (X - src_vg.origin[0]) / src_vg.spacing[0]
    iy = (Y - src_vg.origin[1]) / src_vg.spacing[1]
    iz = (Z - src_vg.origin[2]) / src_vg.spacing[2]
    out = map_coordinates(vol.astype(np.float32), [iz, iy, ix], order=order, mode="constant", cval=cval)
    return out.astype(np.float32)


def resample_mask(mask: np.ndarray, src_vg: VolumeGeometry, dst_vg: VolumeGeometry) -> np.ndarray:
    return resample_to_grid(mask.astype(np.float32), src_vg, dst_vg, order=1, cval=0.0) >= 0.5
