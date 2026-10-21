"""Figures used by the verification scripts (matplotlib, Agg backend)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .geometry import VolumeGeometry  # noqa: E402

HU_WIN = (-1000, 400)


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _idx(vg: VolumeGeometry, xyz) -> np.ndarray:
    i = np.round(vg.world_to_index(np.asarray(xyz, float))).astype(int)
    return np.clip(i, 0, vg.n_xyz - 1)                          # (ix, iy, iz)


def orthoslices(vol: np.ndarray, vg: VolumeGeometry, xyz, mask: Optional[np.ndarray] = None, ax=None,
                window=HU_WIN, title: str = "", cmap="gray", zoom_mm: Optional[float] = None):
    """Axial | coronal | sagittal through ``xyz`` on three axes (returns the axes).

    ``zoom_mm`` shows only a ``zoom_mm`` wide window around ``xyz`` (to check small nodules)."""
    ix, iy, iz = _idx(vg, xyz)
    sx, sy, sz = vg.spacing
    nz = vol.shape[0]
    if ax is None:
        _, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    # (image, mask, aspect, name, (column of xyz, row of xyz), (column, row) spacing)
    views = [(vol[iz, :, :], None if mask is None else mask[iz, :, :], sy / sx, "axial (x-y)", (ix, iy), (sx, sy)),
             (vol[:, iy, :][::-1], None if mask is None else mask[:, iy, :][::-1], sz / sx, "coronal (x-z) = DTS slice",
              (ix, nz - 1 - iz), (sx, sz)),
             (vol[:, :, ix][::-1], None if mask is None else mask[:, :, ix][::-1], sz / sy, "sagittal (y-z), depth ->",
              (iy, nz - 1 - iz), (sy, sz))]
    for a, (img, m, aspect, name, (c, r), (sc, sr)) in zip(ax, views):
        a.imshow(img, cmap=cmap, vmin=window[0], vmax=window[1], aspect=aspect)
        if m is not None and m.any():
            a.contour(m.astype(float), levels=[0.5], colors="r", linewidths=0.8)
        if zoom_mm:
            hc, hr = zoom_mm / 2 / sc, zoom_mm / 2 / sr
            a.set_xlim(c - hc, c + hc)
            a.set_ylim(r + hr, r - hr)
            name = f"{name}, {zoom_mm:.0f} mm"
        a.set_title(f"{title} {name}".strip(), fontsize=8)
        a.axis("off")
    return ax


def save_orthoslices(path: str, vol, vg, xyz, mask=None, title="", window=HU_WIN,
                     zoom_mm: Optional[float] = None) -> str:
    """Three orthogonal slices; with ``zoom_mm`` a second row zooms in on ``xyz``."""
    rows = 2 if zoom_mm else 1
    fig, ax = plt.subplots(rows, 3, figsize=(11, 3.8 * rows), squeeze=False)
    orthoslices(vol, vg, xyz, mask, ax[0], window)
    if zoom_mm:
        orthoslices(vol, vg, xyz, mask, ax[1], window, zoom_mm=zoom_mm)
    if title:
        fig.suptitle(title, fontsize=9)
    return _save(fig, path)


def save_projection_montage(path: str, proj: np.ndarray, marks: Optional[np.ndarray] = None, title: str = "",
                            cols: int = 5) -> str:
    """proj [V, rows, cols]; marks [V, 2] (row, col) predicted nodule positions."""
    V = proj.shape[0]
    rows = int(np.ceil(V / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    axs = np.atleast_1d(axs).ravel()
    lo, hi = np.percentile(proj, [1, 99.5])
    for v in range(len(axs)):
        a = axs[v]
        a.axis("off")
        if v >= V:
            continue
        a.imshow(proj[v], cmap="gray", vmin=lo, vmax=hi)       # radiograph convention: bone white
        if marks is not None:
            a.plot(marks[v, 1], marks[v, 0], "o", mfc="none", mec="r", ms=10, mew=1.2)
        a.set_title(f"view {v}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def save_curve(path: str, ys: Dict[str, Sequence[float]], title: str, xlabel: str = "iteration",
               ylabel: str = "", logy: bool = False) -> str:
    fig, a = plt.subplots(figsize=(5.5, 3.6))
    for k, y in ys.items():
        a.plot(np.arange(len(y)), y, label=k)
    if logy:
        a.set_yscale("log")
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.set_title(title, fontsize=10)
    a.grid(alpha=0.3)
    if len(ys) > 1:
        a.legend(fontsize=8)
    return _save(fig, path)


def save_roi_panel(path: str, vols: Dict[str, np.ndarray], vg: VolumeGeometry, xyz, masks: Optional[Dict] = None,
                   std: Optional[Dict[str, np.ndarray]] = None, title: str = "") -> str:
    """One row per method: coronal (what DTS shows) and sagittal (depth) slices."""
    names = list(vols.keys())
    ncol = 4 if std else 2
    fig, axs = plt.subplots(len(names), ncol, figsize=(2.8 * ncol, 2.6 * len(names)), squeeze=False)
    ix, iy, iz = _idx(vg, xyz)
    sx, sy, sz = vg.spacing
    for r, n in enumerate(names):
        v = vols[n]
        m = masks.get(n) if masks else None
        for c, (img, mm, asp, lab) in enumerate([(v[:, iy, :][::-1], None if m is None else m[:, iy, :][::-1], sz / sx, "coronal"),
                                                 (v[:, :, ix][::-1], None if m is None else m[:, :, ix][::-1], sz / sy, "sagittal (depth ->)")]):
            a = axs[r, c]
            a.imshow(img, cmap="gray", vmin=HU_WIN[0], vmax=HU_WIN[1], aspect=asp)
            if mm is not None and mm.any():
                a.contour(mm.astype(float), levels=[0.5], colors="r", linewidths=0.8)
            a.set_title(f"{n}: {lab}", fontsize=7)
            a.axis("off")
        if std:
            s = std.get(n)
            for c, (img, asp, lab) in enumerate([(None if s is None else s[:, iy, :][::-1], sz / sx, "std coronal"),
                                                 (None if s is None else s[:, :, ix][::-1], sz / sy, "std sagittal")], start=2):
                a = axs[r, c]
                a.axis("off")
                if img is not None:
                    a.imshow(img, cmap="magma", vmin=0, vmax=max(1.0, float(np.percentile(s, 99.5))), aspect=asp)
                    a.set_title(f"{n}: {lab}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def save_scatter(path: str, x, y, labels: Sequence[str], title: str, xlabel: str, ylabel: str, log: bool = True) -> str:
    fig, a = plt.subplots(figsize=(4.8, 4.4))
    allv = []
    for xi, yi, lab in zip(x, y, labels):
        a.scatter(xi, yi, s=12, alpha=0.7, label=lab)
        allv += list(xi) + list(yi)
    allv = np.asarray([v for v in allv if np.isfinite(v) and v > 0])
    if len(allv):
        lo, hi = allv.min() * 0.8, allv.max() * 1.25
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    if log:
        a.set_xscale("log")
        a.set_yscale("log")
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.set_title(title, fontsize=10)
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    return _save(fig, path)


def save_boxplot(path: str, data: Dict[str, Sequence[float]], title: str, ylabel: str) -> str:
    fig, a = plt.subplots(figsize=(1.2 * len(data) + 2.5, 3.8))
    keys = list(data.keys())
    a.boxplot([np.asarray(data[k], float)[np.isfinite(data[k])] for k in keys], showfliers=False)
    a.set_xticks(range(1, len(keys) + 1))
    a.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    a.axhline(0, color="k", lw=0.6)
    a.set_ylabel(ylabel)
    a.set_title(title, fontsize=10)
    a.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def save_bland_altman(path: str, est, ref, title: str, label: str = "") -> str:
    est = np.asarray(est, float)
    ref = np.asarray(ref, float)
    ok = (est > 0) & (ref > 0)
    mean = 0.5 * (np.log(est[ok]) + np.log(ref[ok]))
    diff = np.log(est[ok]) - np.log(ref[ok])
    fig, a = plt.subplots(figsize=(5, 3.6))
    a.scatter(np.exp(mean), 100 * (np.exp(diff) - 1), s=10, alpha=0.7)
    md, sd = diff.mean() if len(diff) else 0.0, diff.std() if len(diff) else 0.0
    for v, ls in ((md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")):
        a.axhline(100 * (np.exp(v) - 1), color="r", ls=ls, lw=0.8)
    a.set_xscale("log")
    a.set_xlabel(f"mean of estimate and truth {label}")
    a.set_ylabel("error (%)")
    a.set_title(title, fontsize=10)
    a.grid(alpha=0.3)
    return _save(fig, path)
