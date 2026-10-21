"""A ``Case`` = one CT volume (HU) + its nodules with radiologist masks.

Saved as a single compressed ``.npz`` so later steps never need DICOM or
pylidc again (handy on Vast.ai / Kaggle).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..geometry import VolumeGeometry


@dataclass
class Nodule:
    nid: str
    center_xyz: List[float]                    # world mm (centroid of the consensus mask)
    bbox_zyx: List[int]                        # z0, z1, y0, y1, x0, x1 (half-open) in the case volume
    mask: np.ndarray                           # bool crop of the consensus mask (>= 50% readers)
    reader_masks: Optional[np.ndarray] = None  # bool [n_readers, *crop]
    diameter_mm: float = float("nan")          # mean reader diameter (pylidc)
    n_readers: int = 0
    malignancy: float = float("nan")           # mean radiologist rating 1..5
    texture: float = float("nan")              # 1 = ground glass ... 5 = solid
    meta: Dict = field(default_factory=dict)

    def full_mask(self, shape) -> np.ndarray:
        m = np.zeros(shape, bool)
        z0, z1, y0, y1, x0, x1 = self.bbox_zyx
        m[z0:z1, y0:y1, x0:x1] = self.mask
        return m

    def header(self) -> Dict:
        return {"nid": self.nid, "center_xyz": [float(v) for v in self.center_xyz],
                "bbox_zyx": [int(v) for v in self.bbox_zyx], "diameter_mm": float(self.diameter_mm),
                "n_readers": int(self.n_readers), "malignancy": float(self.malignancy),
                "texture": float(self.texture), "meta": self.meta}


@dataclass
class Case:
    case_id: str
    hu: np.ndarray                             # [z, y, x]
    vgeom: VolumeGeometry
    nodules: List[Nodule]
    source: str = "lidc"
    meta: Dict = field(default_factory=dict)

    @property
    def patient_id(self) -> str:
        return str(self.meta.get("patient_id", self.case_id))


def save_case(case: Case, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {"hu": np.clip(case.hu, -1024, 3071).astype(np.int16)}
    for i, n in enumerate(case.nodules):
        arrays[f"mask_{i}"] = np.packbits(n.mask.astype(np.uint8), axis=None)
        if n.reader_masks is not None:
            arrays[f"readers_{i}"] = np.packbits(n.reader_masks.astype(np.uint8), axis=None)
    header = {"case_id": case.case_id, "source": case.source, "meta": case.meta,
              "vgeom": case.vgeom.to_dict(),
              "nodules": [dict(n.header(), mask_shape=list(n.mask.shape),
                               readers_shape=(list(n.reader_masks.shape) if n.reader_masks is not None else None))
                          for n in case.nodules]}
    np.savez_compressed(path, header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8), **arrays)


def _unpack(bits: np.ndarray, shape) -> np.ndarray:
    n = int(np.prod(shape))
    return np.unpackbits(bits)[:n].reshape(shape).astype(bool)


def load_case(path: str, load_readers: bool = True) -> Case:
    z = np.load(path, allow_pickle=False)
    header = json.loads(bytes(z["header"]).decode())
    nods = []
    for i, h in enumerate(header["nodules"]):
        mask = _unpack(z[f"mask_{i}"], h["mask_shape"])
        readers = None
        if load_readers and h.get("readers_shape") and f"readers_{i}" in z:
            readers = _unpack(z[f"readers_{i}"], h["readers_shape"])
        nods.append(Nodule(h["nid"], h["center_xyz"], h["bbox_zyx"], mask, readers, h["diameter_mm"],
                           h["n_readers"], h["malignancy"], h["texture"], h.get("meta", {})))
    return Case(header["case_id"], z["hu"].astype(np.float32), VolumeGeometry.from_dict(header["vgeom"]),
                nods, header["source"], header.get("meta", {}))


def nodule_from_full_mask(nid: str, mask_full: np.ndarray, vg: VolumeGeometry, pad: int = 2, **kw) -> Nodule:
    idx = np.argwhere(mask_full)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask_full.shape)
    crop = mask_full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    com = idx.mean(0)                                                # (iz, iy, ix)
    center = vg.index_to_world(com[::-1])
    return Nodule(nid, [float(v) for v in center], [int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]),
                                                     int(lo[2]), int(hi[2])], crop, **kw)


def case_from_phantom(ph, case_id: str) -> Case:
    nods = []
    for i, (spec, m) in enumerate(zip(ph.nodules, ph.masks)):
        if not m.any():
            continue
        d = 2.0 * float(np.prod(spec.radii)) ** (1.0 / 3.0)
        nods.append(nodule_from_full_mask(f"{case_id}_n{i}", m, ph.vgeom, diameter_mm=d, n_readers=4,
                                          texture=(1.0 if spec.hu < -300 else 5.0),
                                          meta={"phantom_hu": spec.hu, "radii": list(spec.radii)}))
    return Case(case_id, ph.hu.astype(np.float32), ph.vgeom, nods, "phantom",
                {"patient_id": case_id, "lung_hu": ph.lung_hu})


def body_mask(hu: np.ndarray, thr: float = -500.0, open_iter: int = 2) -> np.ndarray:
    """Largest connected body component, lungs included.

    Removes the CT table / couch and loose objects, which would otherwise add
    attenuation to every simulated DTS ray (a standing DTS patient has no table).
    Order matters: fill the lungs per axial slice first, then open the *solid*
    body (detaches thin table / cushion bridges without breaking the chest wall).
    """
    from scipy import ndimage

    m = hu > thr
    filled = np.zeros_like(m)
    for k in range(m.shape[0]):
        filled[k] = ndimage.binary_fill_holes(m[k])
    opened = ndimage.binary_opening(filled, structure=np.ones((1, 3, 3), bool), iterations=open_iter) if open_iter else filled
    lab, n = ndimage.label(opened)
    if n == 0:
        return np.ones_like(m)
    sizes = ndimage.sum(opened, lab, range(1, n + 1))
    keep = lab == (1 + int(np.argmax(sizes)))
    if open_iter:
        keep = ndimage.binary_dilation(keep, structure=np.ones((1, 3, 3), bool), iterations=open_iter)
    return keep & filled


def remove_table(hu: np.ndarray, thr: float = -500.0) -> np.ndarray:
    out = hu.copy()
    out[~body_mask(hu, thr)] = -1000.0
    return out
