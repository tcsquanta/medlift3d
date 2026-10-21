"""3D Gaussian representation of the nodule ROI (excess attenuation over lung).

    f(r) = mu_base + sum_i a_i * exp(-1/2 (r - m_i)^T S_i^-1 (r - m_i)),
    S_i = R_i diag(s_i^2) R_i^T,  a_i > 0

Why Gaussians here:
* the integral of each Gaussian is closed form, a_i (2 pi)^{3/2} s1 s2 s3,
  so nodule *mass* is an exact sum over the Gaussians that form the nodule;
* the field is continuous -> volume can be evaluated on any (finer) grid;
* positivity + sparsity is a good prior for limited-angle data.

Voxelisation evaluates each Gaussian on a fixed local neighbourhood of
voxels and scatters the values (``index_add``), which is differentiable with
respect to every Gaussian parameter and needs no custom CUDA.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import VolumeGeometry

TWO_PI_15 = (2.0 * math.pi) ** 1.5


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) quaternions (w, x, y, z) -> (N,3,3) rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x.clamp(min=1e-8)))


class GaussianField(torch.nn.Module):
    """A set of anisotropic 3D Gaussians with bounded scales and positive amplitudes."""

    def __init__(self, means: torch.Tensor, scales: torch.Tensor, amps: torch.Tensor,
                 quats: Optional[torch.Tensor] = None, min_scale: float = 0.6, max_scale: float = 1.2):
        super().__init__()
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        n = means.shape[0]
        if quats is None:
            quats = torch.zeros((n, 4), dtype=means.dtype, device=means.device)
            quats[:, 0] = 1.0
        frac = ((scales - self.min_scale) / (self.max_scale - self.min_scale)).clamp(1e-4, 1 - 1e-4)
        self.means = torch.nn.Parameter(means.detach().clone().float())
        self.scale_raw = torch.nn.Parameter(torch.logit(frac).detach().clone().float())
        self.quats = torch.nn.Parameter(quats.detach().clone().float())
        self.amp_raw = torch.nn.Parameter(_inv_softplus(amps.detach().clone().float().clamp(min=1e-8)))

    # ------------------------------------------------------------- properties
    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    @property
    def scales(self) -> torch.Tensor:
        return self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(self.scale_raw)

    @property
    def amps(self) -> torch.Tensor:
        return F.softplus(self.amp_raw)

    def integrals(self) -> torch.Tensor:
        """Exact integral of every Gaussian over R^3 (attenuation x mm^3)."""
        return self.amps * TWO_PI_15 * self.scales.prod(dim=-1)

    def clone_detached(self) -> "GaussianField":
        g = GaussianField(self.means.detach(), self.scales.detach(), self.amps.detach(),
                          self.quats.detach(), self.min_scale, self.max_scale)
        return g

    def subset(self, keep: torch.Tensor) -> "GaussianField":
        keep = keep.to(self.means.device)
        return GaussianField(self.means.detach()[keep], self.scales.detach()[keep], self.amps.detach()[keep],
                             self.quats.detach()[keep], self.min_scale, self.max_scale)

    def state(self) -> Dict[str, np.ndarray]:
        return {"means": self.means.detach().cpu().numpy(), "scales": self.scales.detach().cpu().numpy(),
                "amps": self.amps.detach().cpu().numpy(), "quats": self.quats.detach().cpu().numpy()}

    # ------------------------------------------------------------- voxelise
    @staticmethod
    def neighbourhood(vg: VolumeGeometry, max_scale: float, device) -> torch.Tensor:
        sp = vg.spacing_arr
        r = np.ceil(3.0 * max_scale / sp + 0.5 - 1e-6).astype(int)          # per axis (x, y, z)
        ox, oy, oz = (np.arange(-ri, ri + 1) for ri in r)
        Oz, Oy, Ox = np.meshgrid(oz, oy, ox, indexing="ij")
        offs = np.stack([Ox.ravel(), Oy.ravel(), Oz.ravel()], axis=-1)      # M,3 (x,y,z)
        return torch.as_tensor(offs, dtype=torch.long, device=device)

    def _voxelize_chunk(self, means, scales, quats, amps, offs, origin, spacing, shape_t):
        D, H, W = shape_t
        idx = torch.round((means - origin) / spacing).long()                # n,3 nearest voxel
        base = idx.float() * spacing + origin - means                       # n,3 (mm)
        d = base[:, None, :] + offs.float()[None] * spacing                 # n,M,3
        R = quat_to_rotmat(quats)                                           # n,3,3
        A = R / scales[:, None, :]                                          # columns scaled by 1/s
        y = torch.einsum("nmi,nij->nmj", d, A)
        q = (y * y).sum(-1)
        val = amps[:, None] * torch.exp(-0.5 * q)                           # n,M
        vox = idx[:, None, :] + offs[None]                                  # n,M,3 (x,y,z)
        valid = ((vox[..., 0] >= 0) & (vox[..., 0] < W) & (vox[..., 1] >= 0) & (vox[..., 1] < H)
                 & (vox[..., 2] >= 0) & (vox[..., 2] < D))
        vx = vox[..., 0].clamp(0, W - 1)
        vy = vox[..., 1].clamp(0, H - 1)
        vz = vox[..., 2].clamp(0, D - 1)
        flat = (vz * H + vy) * W + vx
        val = val * valid.to(val.dtype)
        out = torch.zeros(D * H * W, dtype=val.dtype, device=val.device)
        return out.index_add(0, flat.reshape(-1), val.reshape(-1))

    def voxelize(self, vg: VolumeGeometry, chunk: int = 8192, use_checkpoint: bool = False) -> torch.Tensor:
        """Sum of all Gaussians sampled at the voxel centres of ``vg`` -> [D,H,W]."""
        dev = self.means.device
        offs = self.neighbourhood(vg, self.max_scale, dev)
        origin = torch.as_tensor(vg.origin_arr, dtype=torch.float32, device=dev)
        spacing = torch.as_tensor(vg.spacing_arr, dtype=torch.float32, device=dev)
        D, H, W = vg.shape
        means, scales, quats, amps = self.means, self.scales, self.quats, self.amps
        total = torch.zeros(D * H * W, dtype=torch.float32, device=dev)
        for s in range(0, self.n, chunk):
            e = min(self.n, s + chunk)
            args = (means[s:e], scales[s:e], quats[s:e], amps[s:e], offs, origin, spacing, (D, H, W))
            if use_checkpoint and torch.is_grad_enabled():
                part = checkpoint(self._voxelize_chunk, *args, use_reentrant=False)
            else:
                part = self._voxelize_chunk(*args)
            total = total + part
        return total.reshape(D, H, W)


# ---------------------------------------------------------------------------
# construction and optimisation
# ---------------------------------------------------------------------------
def scale_bounds(vg: VolumeGeometry, radius_vox: int = 4):
    sp = float(min(vg.spacing))
    return 0.6 * sp, max(0.7 * sp, (radius_vox - 0.5) * sp / 3.0)


def gaussians_from_volume(excess_mu: torch.Tensor, vg: VolumeGeometry, stride: int = 2, sigma_factor: float = 0.6,
                          threshold_mu: float = 1e-3, max_gaussians: int = 30000, radius_vox: int = 4,
                          jitter_mm: float = 0.0, generator: Optional[torch.Generator] = None) -> GaussianField:
    """Place one isotropic Gaussian per ``stride^3`` voxel block above ``threshold_mu``.

    Amplitudes are chosen so that the lattice of Gaussians reproduces the block
    means (lattice sum of Gaussians with sigma = f * stride * spacing is
    a * (2 pi)^{3/2} * f^3).
    """
    dev = excess_mu.device
    x = excess_mu.float()[None, None]
    blocks = F.avg_pool3d(x, kernel_size=stride, stride=stride)[0, 0]       # d,h,w (z,y,x)
    min_s, max_s = scale_bounds(vg, radius_vox)
    sp = torch.as_tensor(vg.spacing_arr, dtype=torch.float32, device=dev)
    sigma = (sigma_factor * stride * sp).clamp(min_s, max_s)                # (x,y,z)
    density = TWO_PI_15 * torch.prod(sigma / (stride * sp))
    sel = torch.nonzero(blocks > threshold_mu, as_tuple=False)              # k,3 (bz,by,bx)
    vals = blocks[sel[:, 0], sel[:, 1], sel[:, 2]]
    if sel.shape[0] > max_gaussians:
        top = torch.topk(vals, max_gaussians).indices
        sel, vals = sel[top], vals[top]
    if sel.shape[0] == 0:                                                   # keep at least one (tiny) Gaussian
        sel = torch.tensor([[blocks.shape[0] // 2, blocks.shape[1] // 2, blocks.shape[2] // 2]], device=dev)
        vals = torch.tensor([threshold_mu * 0.1], device=dev)
    centers_idx = sel.flip(-1).float() * stride + (stride - 1) / 2.0        # (x,y,z) voxel index
    origin = torch.as_tensor(vg.origin_arr, dtype=torch.float32, device=dev)
    means = origin + centers_idx * sp
    if jitter_mm > 0:
        means = means + jitter_mm * torch.randn(means.shape, generator=generator, device=dev)
    scales = sigma[None].expand(means.shape[0], 3).clone()
    amps = vals / density
    return GaussianField(means, scales, amps, None, min_s, max_s)


def tv3d(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dz = v[1:, :-1, :-1] - v[:-1, :-1, :-1]
    dy = v[:-1, 1:, :-1] - v[:-1, :-1, :-1]
    dx = v[:-1, :-1, 1:] - v[:-1, :-1, :-1]
    return torch.sqrt(dx * dx + dy * dy + dz * dz + eps).mean()


def _optimizer(gf: GaussianField, lr_means: float, lr_scales: float, lr_amps: float, lr_rot: float = 0.01):
    return torch.optim.Adam([
        {"params": [gf.means], "lr": lr_means},
        {"params": [gf.scale_raw], "lr": lr_scales},
        {"params": [gf.amp_raw], "lr": lr_amps},
        {"params": [gf.quats], "lr": lr_rot},
    ])


def fit_to_volume(gf: GaussianField, target: torch.Tensor, vg: VolumeGeometry, n_iter: int = 150,
                  lr_means: float = 0.05, lr_scales: float = 0.1, lr_amps: float = 0.05) -> List[float]:
    """Least-squares fit of the voxelised Gaussians to a target volume (relative MSE history)."""
    opt = _optimizer(gf, lr_means, lr_scales, lr_amps)
    denom = float((target ** 2).mean()) + 1e-12
    hist = []
    for _ in range(int(n_iter)):
        opt.zero_grad(set_to_none=True)
        loss = ((gf.voxelize(vg) - target) ** 2).mean() / denom
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return hist


def refine_with_projections(gf: GaussianField, problem, base_mu: float, n_iter: int = 150,
                            lr_means: float = 0.05, lr_scales: float = 0.1, lr_amps: float = 0.05,
                            tv_weight: float = 1e-4, prior: Optional[torch.Tensor] = None,
                            prior_weight: float = 0.0) -> Dict[str, List[float]]:
    """Make the Gaussians agree with the measured X-rays (ROI residual).

    loss = ||A(mu_base + V) - r||^2 / ||r||^2  +  tv * TV(V)  +  w_prior * ||V - V0||^2 / ||V0||^2
    """
    vg = problem.vg
    with torch.no_grad():
        ones = torch.ones(vg.shape, device=problem.device)
        base_proj = base_mu * problem.forward(ones)
    target = problem.residual - base_proj
    denom = float((target ** 2).mean()) + 1e-12
    pden = float((prior ** 2).mean()) + 1e-12 if prior is not None else 1.0
    opt = _optimizer(gf, lr_means, lr_scales, lr_amps)
    hist = {"data": [], "total": []}
    for _ in range(int(n_iter)):
        opt.zero_grad(set_to_none=True)
        V = gf.voxelize(vg)
        data = ((problem.forward(V) - target) ** 2).mean() / denom
        loss = data
        if tv_weight > 0:
            loss = loss + tv_weight * tv3d(V) / (float(base_mu) + 1e-3)
        if prior is not None and prior_weight > 0:
            loss = loss + prior_weight * ((V - prior) ** 2).mean() / pden
        loss.backward()
        opt.step()
        hist["data"].append(data.item())
        hist["total"].append(loss.item())
    return hist


def gaussian_centers_in_mask(gf: GaussianField, mask: np.ndarray, vg: VolumeGeometry) -> np.ndarray:
    """Boolean per Gaussian: is its centre inside ``mask`` (on grid ``vg``)?"""
    m = gf.means.detach().cpu().numpy()
    idx = np.round(vg.world_to_index(m)).astype(int)                        # (ix,iy,iz)
    D, H, W = vg.shape
    ok = (idx[:, 0] >= 0) & (idx[:, 0] < W) & (idx[:, 1] >= 0) & (idx[:, 1] < H) & (idx[:, 2] >= 0) & (idx[:, 2] < D)
    inside = np.zeros(len(m), bool)
    inside[ok] = mask[idx[ok, 2], idx[ok, 1], idx[ok, 0]]
    return inside
