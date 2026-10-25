"""Learned nodule detector on the standard (coarse) DTS reconstruction.

A small 3D U-Net predicts a nodule-centre heatmap from the coarse SIRT volume (2.5 mm
voxels, the image the radiologist scrolls through). It is trained on the *train*-split
patients with the LIDC radiologists' nodules as labels (Gaussian blobs at the consensus
centres, longer in depth where DTS cannot localise), using the penalty-reduced focal loss
of CenterNet; checkpoints are selected on the *cal* patients and the detector is evaluated
on the *test* patients only (FROC / CPM as in LUNA16).

Candidates = local maxima of the heatmap (non-maximum suppression with a footprint longer
in depth), in the same format as the training-free detector (:mod:`medlift3d.detect`), so
the demo and the pipeline use either one.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from .geometry import VolumeGeometry

HU_LO, HU_HI = -1000.0, 400.0
FROC_FPS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)          # false positives per scan (LUNA16 CPM)


def normalize_hu(hu):
    """HU -> about [-1, 1] (same window as the diffusion model), clipped."""
    u = (hu - (HU_LO + HU_HI) / 2.0) / ((HU_HI - HU_LO) / 2.0)
    return u.clamp(-1.5, 1.5) if isinstance(u, torch.Tensor) else np.clip(u, -1.5, 1.5)


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
def _block(cin: int, cout: int) -> nn.Sequential:
    g = min(8, cout)
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.SiLU(),
                         nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.SiLU())


class DetNet(nn.Module):
    """3-level 3D U-Net: 1-channel volume -> 1-channel heatmap logits (same size).
    Input sizes must be multiples of 4 (see :func:`predict_heatmap`)."""

    def __init__(self, base_ch: int = 16):
        super().__init__()
        c1, c2, c3 = base_ch, 2 * base_ch, 4 * base_ch
        self.e1, self.e2, self.e3 = _block(1, c1), _block(c1, c2), _block(c2, c3)
        self.u2, self.d2 = nn.ConvTranspose3d(c3, c2, 2, stride=2), _block(2 * c2, c2)
        self.u1, self.d1 = nn.ConvTranspose3d(c2, c1, 2, stride=2), _block(2 * c1, c1)
        self.head = nn.Conv3d(c1, 1, 1)
        nn.init.constant_(self.head.bias, -4.6)            # start at p ~ 0.01 (focal-loss practice)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(F.max_pool3d(x1, 2))
        x3 = self.e3(F.max_pool3d(x2, 2))
        y = self.d2(torch.cat([self.u2(x3), x2], 1))
        y = self.d1(torch.cat([self.u1(y), x1], 1))
        return self.head(y)


# ---------------------------------------------------------------------------
# targets and loss
# ---------------------------------------------------------------------------
def heatmap_target(vg: VolumeGeometry, centers_xyz: Sequence, diameters_mm: Sequence[float],
                   min_sigma_mm: float = 2.5, depth_factor: float = 2.0) -> np.ndarray:
    """[z, y, x] float32 heatmap: a Gaussian per nodule (in-plane sigma = max(min_sigma, d/4),
    ``depth_factor`` x longer in depth), exactly 1 at the voxel nearest each centre."""
    out = np.zeros(vg.shape, np.float32)
    xs, ys, zs = vg.axes()
    for c, d in zip(centers_xyz, diameters_mm):
        s = max(float(min_sigma_mm), 0.25 * float(d))
        sy = depth_factor * s
        gx = np.exp(-(xs - c[0]) ** 2 / (2 * s * s))
        gy = np.exp(-(ys - c[1]) ** 2 / (2 * sy * sy))
        gz = np.exp(-(zs - c[2]) ** 2 / (2 * s * s))
        g = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
        ix, iy, iz = np.round(vg.world_to_index(np.asarray(c, float))).astype(int)
        if 0 <= iz < vg.shape[0] and 0 <= iy < vg.shape[1] and 0 <= ix < vg.shape[2]:
            g[iz, iy, ix] = 1.0                              # the one positive voxel (off-grid centres)
        out = np.maximum(out, np.minimum(g, 1.0).astype(np.float32))
    return out


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced focal loss (CenterNet): positives are the centre voxels (target = 1);
    negatives near a centre are down-weighted by (1 - target)^beta. Normalised by #centres."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target >= 0.999
    pos_loss = -((1 - p) ** alpha * torch.log(p))[pos].sum()
    neg_loss = -((1 - target) ** beta * p ** alpha * torch.log(1 - p))[~pos].sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_heatmap(model: nn.Module, vol_hu: np.ndarray, device="cpu") -> np.ndarray:
    """Sigmoid heatmap [z, y, x] for a whole coarse DTS volume (padded to multiples of 4)."""
    x = torch.as_tensor(normalize_hu(np.asarray(vol_hu, np.float32)), device=device)[None, None]
    pads = [(-s) % 4 for s in x.shape[2:]]                   # z, y, x
    x = F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0]), value=float(normalize_hu(np.float32(-1000.0))))
    model.eval()
    h = torch.sigmoid(model(x).float())[0, 0]
    nz, ny, nx = vol_hu.shape
    return h[:nz, :ny, :nx].cpu().numpy()


def heatmap_peaks(heat: np.ndarray, vg: VolumeGeometry, top_k: int = 10, min_dist_mm: float = 12.0,
                  depth_factor: float = 2.0, min_score: float = 0.0) -> List[Dict]:
    """Local maxima of the heatmap (NMS footprint ``min_dist_mm`` in-plane, x ``depth_factor`` in
    depth), highest first, in the candidate format of :func:`medlift3d.detect.detect_candidates`."""
    sx, sy, sz = vg.spacing
    rz, ry, rx = (max(1, int(round(min_dist_mm / sz))), max(1, int(round(depth_factor * min_dist_mm / sy))),
                  max(1, int(round(min_dist_mm / sx))))
    mx = ndimage.maximum_filter(heat, size=(2 * rz + 1, 2 * ry + 1, 2 * rx + 1), mode="constant", cval=0.0)
    idx = np.argwhere((heat == mx) & (heat > max(min_score, 1e-6)))
    if len(idx) == 0:
        return []
    vals = heat[tuple(idx.T)]
    out = []
    for o in np.argsort(-vals)[:top_k]:
        iz, iy, ix = (int(v) for v in idx[o])
        out.append({"xyz": [float(v) for v in vg.index_to_world(np.array([ix, iy, iz], float))],
                    "score": float(vals[o]), "sigma_mm": 3.0, "index_zyx": [iz, iy, ix]})
    return out


def detect_learned(model: nn.Module, vol_hu: np.ndarray, vg: VolumeGeometry, device="cpu", top_k: int = 10,
                   **peak_kw) -> List[Dict]:
    return heatmap_peaks(predict_heatmap(model, vol_hu, device), vg, top_k=top_k, **peak_kw)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def is_hit(cand_xyz, nodule_xyz, diameter_mm: float, depth_tol_mm: float = 12.0) -> bool:
    """A suggestion counts as finding a nodule if a click there lets the pipeline find it:
    in-plane within max(6 mm, radius + 3 mm), in depth within ``depth_tol_mm`` + radius
    (the measurement searches the nodule peak 12 mm deep around the click)."""
    p, c = np.asarray(cand_xyz, float), np.asarray(nodule_xyz, float)
    r = 0.5 * float(diameter_mm)
    return bool(np.hypot(p[0] - c[0], p[2] - c[2]) <= max(6.0, r + 3.0) and abs(p[1] - c[1]) <= depth_tol_mm + r)


def match_candidates(cands: List[Dict], nodules: List[Dict], depth_tol_mm: float = 12.0) -> List[Optional[int]]:
    """For each candidate (score order) the index of the nodule it finds, or None. ``nodules``:
    dicts with ``xyz`` and ``diameter_mm``. A nodule found by a higher-scored candidate is not
    counted again (the later candidate gets the same index; it is neither a hit nor a false positive)."""
    out = []
    for c in cands:
        hit = [j for j, n in enumerate(nodules) if is_hit(c["xyz"], n["xyz"], n["diameter_mm"], depth_tol_mm)]
        out.append(min(hit, key=lambda j: np.linalg.norm(np.asarray(c["xyz"]) - np.asarray(nodules[j]["xyz"])))
                   if hit else None)
    return out


def froc(scans: List[Dict], fps: Sequence[float] = FROC_FPS, depth_tol_mm: float = 12.0) -> Dict:
    """FROC over scans (each ``{"cands": [...], "nodules": [...]}``): sensitivity at the given
    average false positives per scan, CPM (their mean, LUNA16) and recall within the top 10."""
    events, n_nod = [], 0                                   # (score, is_true_positive)
    top10 = 0
    for s in scans:
        nods = s["nodules"]
        n_nod += len(nods)
        m = match_candidates(s["cands"], nods, depth_tol_mm)
        seen = set()
        for k, (c, j) in enumerate(zip(s["cands"], m)):
            if j is None:
                events.append((c["score"], False))
            elif j not in seen:
                seen.add(j)
                events.append((c["score"], True))
                top10 += int(k < 10)
    n_scans = max(len(scans), 1)
    events.sort(key=lambda e: -e[0])
    tp = np.cumsum([e[1] for e in events]) if events else np.zeros(1)
    fp = np.cumsum([not e[1] for e in events]) if events else np.zeros(1)
    sens_at = {}
    for f in fps:
        ok = fp / n_scans <= f
        sens_at[f] = float(tp[ok].max() / max(n_nod, 1)) if ok.any() else 0.0
    return {"sensitivity": sens_at, "cpm": float(np.mean(list(sens_at.values()))),
            "recall_top10": top10 / max(n_nod, 1), "n_nodules": n_nod, "n_scans": len(scans),
            "curve_fp_per_scan": (fp / n_scans).tolist(), "curve_sensitivity": (tp / max(n_nod, 1)).tolist()}


def score_threshold(scans: List[Dict], fp_per_scan: float = 2.0, depth_tol_mm: float = 12.0) -> float:
    """Score at which the detector makes ``fp_per_scan`` false positives per scan on ``scans``
    (an operating point chosen on the calibration patients)."""
    fps = []
    for s in scans:
        m = match_candidates(s["cands"], s["nodules"], depth_tol_mm)
        fps += [c["score"] for c, j in zip(s["cands"], m) if j is None]
    if not fps:
        return 0.0
    fps.sort(reverse=True)
    k = int(round(fp_per_scan * max(len(scans), 1)))
    return float(fps[k]) if k < len(fps) else 0.0


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def save_detector(path: str, model: DetNet, base_ch: int, extra: Optional[dict] = None) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "base_ch": int(base_ch), "extra": extra or {}}, path)


def load_detector(path: str, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = DetNet(int(ck["base_ch"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck.get("extra", {})
