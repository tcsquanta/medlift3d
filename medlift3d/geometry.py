"""Imaging geometry for linear-sweep chest digital tomosynthesis (DTS).

Conventions (used everywhere in the package)
--------------------------------------------
* World coordinates are millimetres, ordered (x, y, z) = DICOM LPS:
  x: patient right -> left, y: anterior -> posterior, z: inferior -> superior.
* Volumes are arrays indexed ``[z, y, x]`` (same as SimpleITK / numpy).
* PA chest DTS: the anterior chest faces a stationary flat-panel detector
  (on the -y side); the X-ray tube sits behind the patient (+y side) and
  translates along z, so the 15 views span +-15 degrees in the y-z plane.
  Rays travel mostly along -y, which makes y the poorly resolved *depth* axis.
* Detector pixel (row r, col c): rows run from superior (row 0) to inferior,
  columns from patient-right (col 0) to patient-left -- the usual way a PA
  radiograph is displayed with ``imshow``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class VolumeGeometry:
    """Regular voxel grid.

    shape   : (nz, ny, nx)  array shape, indexed [z, y, x]
    spacing : (sx, sy, sz)  voxel size in mm along x, y, z
    origin  : (ox, oy, oz)  world position (mm) of the *centre* of voxel [0, 0, 0]
    """

    shape: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]

    def __post_init__(self):
        self.shape = tuple(int(s) for s in self.shape)
        self.spacing = tuple(float(s) for s in self.spacing)
        self.origin = tuple(float(s) for s in self.origin)

    # ------------------------------------------------------------------ build
    @classmethod
    def centered(cls, shape_zyx: Sequence[int], spacing_xyz: Sequence[float],
                 center_xyz: Sequence[float] = (0.0, 0.0, 0.0)) -> "VolumeGeometry":
        nz, ny, nx = (int(s) for s in shape_zyx)
        n_xyz = np.array([nx, ny, nz], dtype=float)
        sp = np.asarray(spacing_xyz, dtype=float)
        origin = np.asarray(center_xyz, dtype=float) - 0.5 * (n_xyz - 1.0) * sp
        return cls((nz, ny, nx), tuple(sp), tuple(origin))

    @classmethod
    def from_dict(cls, d: dict) -> "VolumeGeometry":
        return cls(tuple(d["shape"]), tuple(d["spacing"]), tuple(d["origin"]))

    def to_dict(self) -> dict:
        return {"shape": list(self.shape), "spacing": list(self.spacing), "origin": list(self.origin)}

    # ------------------------------------------------------------- properties
    @property
    def n_xyz(self) -> np.ndarray:
        nz, ny, nx = self.shape
        return np.array([nx, ny, nz])

    @property
    def spacing_arr(self) -> np.ndarray:
        return np.asarray(self.spacing, dtype=float)

    @property
    def origin_arr(self) -> np.ndarray:
        return np.asarray(self.origin, dtype=float)

    @property
    def center(self) -> np.ndarray:
        return self.origin_arr + 0.5 * (self.n_xyz - 1) * self.spacing_arr

    @property
    def box_min(self) -> np.ndarray:
        """Physical lower corner (outer face of the first voxel)."""
        return self.origin_arr - 0.5 * self.spacing_arr

    @property
    def box_max(self) -> np.ndarray:
        return self.origin_arr + (self.n_xyz - 0.5) * self.spacing_arr

    @property
    def size_mm(self) -> np.ndarray:
        return self.n_xyz * self.spacing_arr

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.spacing))

    # ------------------------------------------------------------ conversions
    def world_to_index(self, xyz) -> np.ndarray:
        """Continuous voxel index (ix, iy, iz) for world points (..., 3)."""
        return (np.asarray(xyz, dtype=float) - self.origin_arr) / self.spacing_arr

    def index_to_world(self, ixyz) -> np.ndarray:
        return self.origin_arr + np.asarray(ixyz, dtype=float) * self.spacing_arr

    def axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1-D world coordinates of voxel centres along x, y, z."""
        nx, ny, nz = self.n_xyz
        ox, oy, oz = self.origin
        sx, sy, sz = self.spacing
        return ox + sx * np.arange(nx), oy + sy * np.arange(ny), oz + sz * np.arange(nz)

    def contains(self, xyz) -> np.ndarray:
        p = np.asarray(xyz, dtype=float)
        return np.all((p >= self.box_min) & (p <= self.box_max), axis=-1)


@dataclass
class DTSGeometry:
    """Linear-sweep tomosynthesis with a stationary flat detector.

    Source k sits at ``iso + (0, SOD, SOD * tan(theta_k))`` with
    ``theta_k`` evenly spaced in ``[-sweep/2, +sweep/2]`` and
    ``SOD = sid - detector_to_iso``. The detector plane is
    ``y = iso_y - detector_to_iso``.
    """

    n_views: int = 15
    sweep_deg: float = 30.0
    sid: float = 1800.0
    detector_to_iso: float = 150.0
    det_shape: Tuple[int, int] = (536, 536)          # (rows ~ z, cols ~ x)
    det_pitch: Tuple[float, float] = (0.8, 0.8)      # (row pitch, col pitch) mm
    iso: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    det_offset: Tuple[float, float] = (0.0, 0.0)     # detector centre shift (dz, dx) mm

    def __post_init__(self):
        self.det_shape = tuple(int(s) for s in self.det_shape)
        self.det_pitch = tuple(float(s) for s in self.det_pitch)
        self.iso = tuple(float(s) for s in self.iso)
        self.det_offset = tuple(float(s) for s in self.det_offset)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, gcfg: dict, iso=(0.0, 0.0, 0.0), detector_to_iso: Optional[float] = None) -> "DTSGeometry":
        return cls(
            n_views=int(gcfg["n_views"]),
            sweep_deg=float(gcfg["sweep_deg"]),
            sid=float(gcfg["sid"]),
            detector_to_iso=float(detector_to_iso if detector_to_iso is not None else gcfg["detector_to_iso"]),
            det_shape=tuple(gcfg["det_shape"]),
            det_pitch=tuple(gcfg["det_pitch"]),
            iso=tuple(iso),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "DTSGeometry":
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

    # ------------------------------------------------------------- properties
    @property
    def sod(self) -> float:
        """Distance from the source line to the isocentre (along y)."""
        return self.sid - self.detector_to_iso

    @property
    def det_y(self) -> float:
        return self.iso[1] - self.detector_to_iso

    def angles(self) -> np.ndarray:
        if self.n_views == 1:
            return np.zeros(1)
        return np.deg2rad(np.linspace(-self.sweep_deg / 2.0, self.sweep_deg / 2.0, self.n_views))

    def source_positions(self) -> np.ndarray:
        th = self.angles()
        x = np.full_like(th, self.iso[0])
        y = np.full_like(th, self.iso[1] + self.sod)
        z = self.iso[2] + self.sod * np.tan(th)
        return np.stack([x, y, z], axis=-1)

    # ---------------------------------------------------------------- detector
    def pixel_centers(self, rows=None, cols=None) -> np.ndarray:
        """World xyz of detector pixel centres, shape (n_rows, n_cols, 3)."""
        nr, nc = self.det_shape
        dv, du = self.det_pitch
        r = np.arange(nr, dtype=float) if rows is None else np.asarray(rows, dtype=float)
        c = np.arange(nc, dtype=float) if cols is None else np.asarray(cols, dtype=float)
        x = self.iso[0] + self.det_offset[1] + (c - (nc - 1) / 2.0) * du
        z = self.iso[2] + self.det_offset[0] - (r - (nr - 1) / 2.0) * dv
        zz, xx = np.meshgrid(z, x, indexing="ij")
        yy = np.full_like(xx, self.det_y)
        return np.stack([xx, yy, zz], axis=-1)

    def world_to_pixel(self, xyz_on_detector) -> Tuple[np.ndarray, np.ndarray]:
        """Continuous (row, col) of points lying on the detector plane."""
        p = np.asarray(xyz_on_detector, dtype=float)
        nr, nc = self.det_shape
        dv, du = self.det_pitch
        col = (p[..., 0] - self.iso[0] - self.det_offset[1]) / du + (nc - 1) / 2.0
        row = -(p[..., 2] - self.iso[2] - self.det_offset[0]) / dv + (nr - 1) / 2.0
        return row, col

    def project_points(self, pts) -> Tuple[np.ndarray, np.ndarray]:
        """Project world points (N, 3) into every view.

        Returns continuous (rows, cols), each of shape (n_views, N).
        """
        p = np.atleast_2d(np.asarray(pts, dtype=float))[None, :, :]       # 1,N,3
        s = self.source_positions()[:, None, :]                              # V,1,3
        t = (self.det_y - s[..., 1]) / (p[..., 1] - s[..., 1])              # V,N
        hit = s + t[..., None] * (p - s)
        return self.world_to_pixel(hit)

    def magnification(self, pts) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        src_y = self.iso[1] + self.sod
        return (src_y - self.det_y) / (src_y - p[:, 1])

    # -------------------------------------------------------------------- rays
    def rays(self, views: Optional[Sequence[int]] = None, device="cpu",
             dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rays for every pixel of the selected views.

        Returns (origins, dirs), each (V * rows * cols, 3), ordered [view, row, col].
        """
        views = range(self.n_views) if views is None else views
        src = self.source_positions()[list(views)]                           # V,3
        pix = self.pixel_centers().reshape(-1, 3)                            # P,3
        o = np.repeat(src[:, None, :], pix.shape[0], axis=1)                 # V,P,3
        d = pix[None, :, :] - o
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return (torch.as_tensor(o.reshape(-1, 3), dtype=dtype, device=device),
                torch.as_tensor(d.reshape(-1, 3), dtype=dtype, device=device))

    def window_rays(self, windows: Sequence[Tuple[int, int, int, int]], device="cpu",
                    dtype=torch.float32, views: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rays for a rectangular pixel window per view.

        ``windows[i] = (r0, r1, c0, c1)`` (half-open) belongs to view
        ``views[i]`` (default: view i). Rays are ordered [window, row, col].
        """
        src = self.source_positions()
        views = list(range(len(windows))) if views is None else list(views)
        os_, ds_ = [], []
        for v, (r0, r1, c0, c1) in zip(views, windows):
            pix = self.pixel_centers(np.arange(r0, r1), np.arange(c0, c1)).reshape(-1, 3)
            o = np.repeat(src[v][None, :], pix.shape[0], axis=0)
            d = pix - o
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            os_.append(o)
            ds_.append(d)
        return (torch.as_tensor(np.concatenate(os_), dtype=dtype, device=device),
                torch.as_tensor(np.concatenate(ds_), dtype=dtype, device=device))

    # ------------------------------------------------------------------ binning
    def binned(self, b: int) -> "DTSGeometry":
        """Geometry of the detector after ``b x b`` binning (see bin_projections)."""
        if b == 1:
            return self
        nr, nc = self.det_shape
        dv, du = self.det_pitch
        nrb, ncb = nr // b, nc // b
        off_z = self.det_offset[0] + (nr - nrb * b) / 2.0 * dv
        off_x = self.det_offset[1] - (nc - ncb * b) / 2.0 * du
        return DTSGeometry(self.n_views, self.sweep_deg, self.sid, self.detector_to_iso,
                           (nrb, ncb), (dv * b, du * b), self.iso, (off_z, off_x))


def bin_projections(proj, b: int):
    """Average ``b x b`` pixel blocks of projections (..., rows, cols).

    Trailing rows/cols that do not fill a block are dropped, matching
    :meth:`DTSGeometry.binned`. Works for numpy arrays and torch tensors.
    """
    if b == 1:
        return proj
    nr, nc = proj.shape[-2], proj.shape[-1]
    nrb, ncb = nr // b, nc // b
    p = proj[..., : nrb * b, : ncb * b]
    shape = tuple(p.shape[:-2]) + (nrb, b, ncb, b)
    p = p.reshape(shape)
    return p.mean(axis=(-3, -1)) if isinstance(p, np.ndarray) else p.mean(dim=(-3, -1))


def estimate_detector_to_iso(body_y_min: float, iso_y: float, gap_mm: float = 10.0,
                             default: float = 150.0) -> float:
    """Detector sits ``gap_mm`` in front of the anterior body surface."""
    d = iso_y - (body_y_min - gap_mm)
    return float(d) if np.isfinite(d) and d > 20.0 else float(default)
