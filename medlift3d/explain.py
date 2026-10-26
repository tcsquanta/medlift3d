"""Explainability: what in a reconstruction was *measured* by the X-rays and what was *filled in*.

Seen / unseen split (range / null space)
    15 DTS views constrain only part of a 3D nodule: every volume x splits exactly into
        x = R(x) + N(x),   A N(x) = 0,   R(x) = the smallest volume giving the same projections A x
    R(x) is what the projections determine ("seen"); N(x) changes nothing in the projections
    ("unseen": no measurement can confirm or reject it). A physics-only reconstruction (SIRT from a
    flat start) contains almost no N; whatever N a diffusion reconstruction contains was supplied by the
    learned prior. Comparing it with N(truth) shows whether the prior filled in the right thing:
        unseen skill = 1 - |N(ours) - N(truth)|^2 / |N(truth)|^2
    (1 = the invisible part recovered exactly, 0 = no better than leaving it out, < 0 = worse:
    invented structure). R is computed with CGLS from zero, which stays in the row space of A and so
    converges to the minimum-norm solution; ``rel_residual`` = |A N| / |A x| checks the split.
    The split is done inside the nodule box (only rays through the box, only the box content), on the
    excess over the box's lung level: a local analysis that takes the prism outside the box as known.

Detector saliency
    input x gradient of the heatmap logit at a detection: which voxels of the DTS volume made the
    detector fire there, and how much of it lies on the nodule itself.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .projector import Projector, ray_aabb


def cgls_split(x: torch.Tensor, fwd: Callable, adj: Callable, n_iter: int = 60, tol: float = 1e-5,
               support: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Split x into (seen, unseen) for the linear operator ``fwd`` with adjoint ``adj``.

    seen = minimum-norm z with fwd(z) = fwd(x) (CGLS from zero), unseen = x - seen. With ``support``
    (0/1 mask) the operator is fwd(support * .), i.e. the split happens among volumes inside the support.
    """
    S = (lambda v: v) if support is None else (lambda v: v * support)
    A = lambda v: fwd(S(v))                                   # noqa: E731
    At = lambda r: S(adj(r))                                  # noqa: E731
    x = S(x.float())
    b = A(x)
    z = torch.zeros_like(x)
    r = b.clone()
    s = At(r)
    p = s.clone()
    gamma = float((s.double() ** 2).sum())
    g0 = gamma
    it = 0
    for it in range(1, n_iter + 1):
        q = A(p)
        qq = float((q.double() ** 2).sum())
        if qq <= 0 or gamma <= 0:
            break
        a = gamma / qq
        z = z + a * p
        r = r - a * q
        s = At(r)
        g_new = float((s.double() ** 2).sum())
        if g_new <= (tol ** 2) * g0:
            break
        p = s + (g_new / gamma) * p
        gamma = g_new
    n = x - z
    bn = float(b.norm()) + 1e-12
    info = {"iterations": it, "rel_residual": float(A(n).norm()) / bn,
            "cos_seen_unseen": float((z * n).sum()) / (float(z.norm()) * float(n.norm()) + 1e-12)}
    return z, n, info


class BoxOperator:
    """Projections of the nodule box alone, along the prism's measured rays that cross it."""

    def __init__(self, prob, step_mm: Optional[float] = None):
        self.vg = prob.roi.box_vgeom
        t0, t1 = ray_aabb(prob.o, prob.d, self.vg.box_min, self.vg.box_max)
        hit = t1 > t0 + 1e-3
        self.o, self.d = prob.o[hit], prob.d[hit]
        self.P = Projector(self.vg, step_mm=step_mm or prob.step)
        self.device = prob.device

    def fwd(self, x: torch.Tensor) -> torch.Tensor:
        return self.P(x, self.o, self.d)

    def adj(self, y: torch.Tensor) -> torch.Tensor:
        return self.P.backproject(y, self.o, self.d)

    def split_hu(self, box_hu: np.ndarray, baseline_hu: float, mu_water: float, n_iter: int = 60):
        """Seen / unseen parts of the excess over ``baseline_hu`` (both in HU), plus the split info."""
        k = mu_water / 1000.0
        x = torch.as_tensor((np.asarray(box_hu, np.float32) - baseline_hu) * k, device=self.device)
        with torch.no_grad():
            seen, unseen, info = cgls_split(x, self.fwd, self.adj, n_iter=n_iter)
        return seen.cpu().numpy() / k, unseen.cpu().numpy() / k, info


def split_metrics(excess: np.ndarray, seen: np.ndarray, unseen: np.ndarray) -> Dict[str, float]:
    """How much of a volume's excess attenuation is unseen: energy share and share of the box mass."""
    e2 = float((excess.astype(np.float64) ** 2).sum()) + 1e-12
    tot = float(excess.sum())
    return {"unseen_energy_share": float((unseen.astype(np.float64) ** 2).sum()) / e2,
            "unseen_mass_share": float(unseen.sum()) / tot if abs(tot) > 1e-6 else float("nan")}


def skill(pred: np.ndarray, ref: np.ndarray) -> float:
    """1 - |pred - ref|^2 / |ref|^2: 1 perfect, 0 = as good as predicting 0, negative = worse."""
    den = float((ref.astype(np.float64) ** 2).sum())
    return 1.0 - float(((pred - ref).astype(np.float64) ** 2).sum()) / den if den > 0 else float("nan")


def depth_profile(vol: np.ndarray, center_zyx, half: int = 1) -> np.ndarray:
    """Mean along depth (axis 1) over a (2 half + 1)^2 column through the centre."""
    z, _, x = (int(round(c)) for c in center_zyx)
    sl = vol[max(0, z - half):z + half + 1, :, max(0, x - half):x + half + 1]
    return sl.mean(axis=(0, 2))


# ---------------------------------------------------------------------------
# detector saliency
# ---------------------------------------------------------------------------
def detector_saliency(model, vol_hu: np.ndarray, index_zyx, patch: int = 64, device="cpu"):
    """|input x gradient| of the detector logit at ``index_zyx`` over a ``patch``^3 window around it.
    Returns (window HU, saliency, centre index inside the window)."""
    from .detector import normalize_hu

    c = np.asarray(index_zyx, int)
    lo = c - patch // 2
    air = float(normalize_hu(np.float32(-1000.0)))
    win = np.full((patch,) * 3, -1000.0, np.float32)
    src_lo = np.maximum(lo, 0)
    src_hi = np.minimum(lo + patch, vol_hu.shape)
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    win[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        vol_hu[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    x = torch.as_tensor(normalize_hu(win), device=device)[None, None].float()
    x = torch.where(torch.isfinite(x), x, torch.full_like(x, air)).requires_grad_(True)
    model.eval()
    with torch.enable_grad():
        out = model(x)
        cc = tuple(int(v) for v in c - lo)
        logit = out[0, 0][cc]
        (g,) = torch.autograd.grad(logit, x)
    sal = (g * x).abs()[0, 0].detach().cpu().numpy()
    return win, sal, np.asarray(cc)


def saliency_share_within(sal: np.ndarray, center_zyx, radius_mm: float, spacing_zyx) -> float:
    """Share of the total saliency inside a sphere (radius in mm) around the centre."""
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in sal.shape], indexing="ij")
    sp = np.asarray(spacing_zyx, float)
    d2 = ((zz - center_zyx[0]) * sp[0]) ** 2 + ((yy - center_zyx[1]) * sp[1]) ** 2 + ((xx - center_zyx[2]) * sp[2]) ** 2
    tot = float(sal.sum())
    return float(sal[d2 <= radius_mm ** 2].sum()) / tot if tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# global: which nodule properties go with large errors
# ---------------------------------------------------------------------------
def spearman_ci(x, y, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float, int]:
    """Spearman correlation with a percentile bootstrap 95% interval (rho, lo, hi, n)."""
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), float("nan"), n
    rho = float(spearmanr(x, y).correlation)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            bs.append(spearmanr(x[i], y[i]).correlation)
    lo, hi = (float(v) for v in np.percentile(bs, [2.5, 97.5])) if bs else (float("nan"), float("nan"))
    return rho, lo, hi, n
