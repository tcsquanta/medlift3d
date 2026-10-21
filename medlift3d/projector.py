"""Differentiable X-ray projector (pure PyTorch, no custom CUDA).

Computes line integrals  p = integral f(r) dt  along arbitrary rays through a
voxel volume by ray marching with trilinear interpolation
(``torch.nn.functional.grid_sample``). Because the operator is linear in the
volume, autograd gives its exact adjoint (back-projection) for free.

Memory is controlled by processing rays in chunks; when gradients are needed
each chunk is wrapped in ``torch.utils.checkpoint`` so only one chunk's
intermediates live at a time.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import VolumeGeometry


def ray_aabb(o: torch.Tensor, d: torch.Tensor, bmin, bmax) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slab-method ray/box intersection. Returns (t_near, t_far); miss if t_far <= t_near."""
    bmin = torch.as_tensor(np.asarray(bmin, dtype=np.float64), dtype=o.dtype, device=o.device)
    bmax = torch.as_tensor(np.asarray(bmax, dtype=np.float64), dtype=o.dtype, device=o.device)
    eps = 1e-12
    dd = torch.where(d.abs() < eps, torch.full_like(d, eps), d)
    inv = 1.0 / dd
    t1 = (bmin - o) * inv
    t2 = (bmax - o) * inv
    tn = torch.minimum(t1, t2).amax(dim=-1).clamp(min=0.0)
    tf = torch.maximum(t1, t2).amin(dim=-1)
    return tn, tf


def _as_5d(vol: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Return (vol as [B,1,D,H,W], original ndim)."""
    nd = vol.dim()
    if nd == 3:
        return vol[None, None], nd
    if nd == 4:
        return vol[:, None], nd
    if nd == 5:
        return vol, nd
    raise ValueError(f"volume must be 3-, 4- or 5-D, got {tuple(vol.shape)}")


class Projector:
    """Line-integral projector for one voxel grid.

    Parameters
    ----------
    vgeom : VolumeGeometry of the volumes that will be projected.
    step_mm : maximum ray-marching step (default: half the smallest voxel).
    samples_per_chunk : approx. number of (ray, sample) pairs per chunk.
    use_checkpoint : recompute chunks in backward to save memory.
    """

    def __init__(self, vgeom: VolumeGeometry, step_mm: Optional[float] = None,
                 samples_per_chunk: Optional[int] = None, use_checkpoint: bool = True):
        self.vg = vgeom
        self.step = float(step_mm) if step_mm else 0.5 * float(min(vgeom.spacing))
        self.samples_per_chunk = samples_per_chunk
        self.use_checkpoint = use_checkpoint
        n = vgeom.n_xyz.astype(float)
        sp = vgeom.spacing_arr
        # integrate over the voxel-centre hull extended by one voxel (zeros outside)
        self.bmin = vgeom.origin_arr - sp
        self.bmax = vgeom.origin_arr + n * sp
        denom = np.where(n > 1, (n - 1.0) * sp, 1.0)
        self._scale = 2.0 / denom
        self._offset = vgeom.origin_arr

    # ------------------------------------------------------------------ core
    def _chunk(self, vol5, o, d, t0, t1, n_s: int, ex_min, ex_max):
        frac = (torch.arange(n_s, device=o.device, dtype=o.dtype) + 0.5) / n_s
        seg = (t1 - t0).clamp(min=0.0)
        t = t0[:, None] + seg[:, None] * frac[None, :]
        pts = o[:, None, :] + d[:, None, :] * t[..., None]                 # r,S,3
        scale = torch.as_tensor(self._scale, dtype=o.dtype, device=o.device)
        offset = torch.as_tensor(self._offset, dtype=o.dtype, device=o.device)
        g = (pts - offset) * scale - 1.0
        g5 = g[None, None]
        if vol5.shape[0] > 1:
            g5 = g5.expand(vol5.shape[0], -1, -1, -1, -1)
        v = F.grid_sample(vol5, g5.to(vol5.dtype), mode="bilinear", padding_mode="zeros",
                          align_corners=True)[:, 0, 0]                       # B,r,S
        if ex_min is not None:
            inside = ((pts > ex_min) & (pts < ex_max)).all(dim=-1)          # r,S
            v = v * (~inside).to(v.dtype)[None]
        return v.sum(dim=-1) * (seg / n_s).to(v.dtype)[None, :]

    def __call__(self, vol: torch.Tensor, origins: torch.Tensor, dirs: torch.Tensor,
                 exclude_box: Optional[Tuple[Sequence[float], Sequence[float]]] = None) -> torch.Tensor:
        """Project ``vol`` along rays.

        vol : [D,H,W], [B,D,H,W] or [B,1,D,H,W]
        origins, dirs : [R,3] (dirs must be unit vectors)
        exclude_box : optional (min_xyz, max_xyz); samples inside it are ignored
                      (used to integrate the background *outside* an ROI).
        Returns [R] for a 3-D volume, else [B,R].
        """
        vol5, nd = _as_5d(vol)
        o = origins.to(device=vol5.device, dtype=torch.float32)
        d = dirs.to(device=vol5.device, dtype=torch.float32)
        t0, t1 = ray_aabb(o, d, self.bmin, self.bmax)
        if exclude_box is not None:
            ex_min = torch.as_tensor(np.asarray(exclude_box[0], float), dtype=o.dtype, device=o.device)
            ex_max = torch.as_tensor(np.asarray(exclude_box[1], float), dtype=o.dtype, device=o.device)
        else:
            ex_min = ex_max = None
        R = o.shape[0]
        chord = (t1 - t0).clamp(min=0.0)
        max_chord = float(chord.max()) if R > 0 else 0.0
        n_s = max(1, int(math.ceil(max_chord / self.step)))
        spc = self.samples_per_chunk or (2 ** 25 if vol5.is_cuda else 2 ** 23)
        chunk = max(256, spc // (n_s * max(1, vol5.shape[0])))
        need_ckpt = self.use_checkpoint and torch.is_grad_enabled() and vol5.requires_grad
        outs = []
        for s in range(0, R, chunk):
            e = min(R, s + chunk)
            c_chord = float(chord[s:e].max()) if e > s else 0.0
            ns = max(1, int(math.ceil(c_chord / self.step)))
            args = (vol5, o[s:e], d[s:e], t0[s:e], t1[s:e], ns, ex_min, ex_max)
            if need_ckpt:
                outs.append(checkpoint(self._chunk, *args, use_reentrant=False))
            else:
                outs.append(self._chunk(*args))
        out = torch.cat(outs, dim=1) if outs else torch.zeros(vol5.shape[0], 0, device=vol5.device)
        return out[0] if nd == 3 else out

    def backproject(self, proj: torch.Tensor, origins: torch.Tensor, dirs: torch.Tensor,
                    exclude_box=None) -> torch.Tensor:
        """Exact adjoint of :meth:`__call__` (via autograd). proj: [R] or [B,R]."""
        squeeze = proj.dim() == 1
        p = proj[None] if squeeze else proj
        D, H, W = self.vg.shape
        x = torch.zeros((p.shape[0], 1, D, H, W), device=p.device, dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            y = self(x, origins, dirs, exclude_box)
            (g,) = torch.autograd.grad(y, x, grad_outputs=p.to(y.dtype))
        g = g[:, 0]
        return g[0] if squeeze else g

    def forward_and_adjoint(self, vol: torch.Tensor, origins, dirs, residual_fn):
        """One pass computing y = A x and A^T residual_fn(y) (for SIRT-type updates)."""
        x = vol.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            y = self(x, origins, dirs)
            r = residual_fn(y.detach())
            (g,) = torch.autograd.grad(y, x, grad_outputs=r)
        return y.detach(), g

    def chord_lengths(self, origins, dirs, device="cpu") -> torch.Tensor:
        """Length (mm) of each ray inside this grid's integration box (no sampling)."""
        o = origins.to(device=device, dtype=torch.float32)
        d = dirs.to(device=device, dtype=torch.float32)
        t0, t1 = ray_aabb(o, d, self.bmin, self.bmax)
        return (t1 - t0).clamp(min=0.0)
