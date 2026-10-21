"""ROI pair dataset: (condition = ROI-SIRT in HU, target = CT ROI in HU).

Each pair is a ``.npz`` with float16 arrays ``cond`` and ``target`` [D,H,W]
plus metadata. Augmentation: random flips along x, y and z (the DTS geometry
is symmetric under these flips, so the missing-wedge pattern stays valid).
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .ddpm import hu_to_unit


class RoiPairDataset(Dataset):
    """Returns (cond, target) unit-scaled tensors [1,D,H,W]; with return_meta=True also the
    nodule-centre index (iz, iy, ix) in the prism (-1s if unknown)."""

    def __init__(self, root_or_files, augment: bool = True, max_items: Optional[int] = None,
                 return_meta: bool = False):
        if isinstance(root_or_files, (list, tuple)):
            files = list(root_or_files)
        else:
            files = sorted(glob.glob(os.path.join(root_or_files, "*.npz")))
        if max_items:
            files = files[:max_items]
        if not files:
            raise FileNotFoundError(f"no ROI pairs found in {root_or_files}")
        self.files = files
        self.augment = augment
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        z = np.load(self.files[i])
        cond = hu_to_unit(z["cond"].astype(np.float32)).clip(-1.5, 1.5)
        tgt = hu_to_unit(z["target"].astype(np.float32)).clip(-1.2, 1.2)
        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    cond = np.flip(cond, ax)
                    tgt = np.flip(tgt, ax)
        out = (torch.from_numpy(np.ascontiguousarray(cond))[None],
               torch.from_numpy(np.ascontiguousarray(tgt))[None])
        if self.return_meta:
            cidx = z["center_idx"].astype(float) if "center_idx" in z else np.full(3, -1.0)
            return out + (cidx,)
        return out


def save_pair(path: str, cond_hu: np.ndarray, target_hu: np.ndarray, **meta) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, cond=cond_hu.astype(np.float16), target=target_hu.astype(np.float16),
                        **{k: np.asarray(v) for k, v in meta.items()})
