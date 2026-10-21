"""File layout of the ``work/`` directory and small persistence helpers.

work/
  lidc_selection.csv        nodules chosen from the pylidc database (step 02)
  lidc_subset.tcia          NBIA manifest for the chosen series (step 02)
  cases/<case>.npz          CT + radiologist masks (step 04)
  splits.json               patient-level train / cal / test split (step 04)
  dts/<case>.npz            clean DTS line integrals + geometry (step 05)
  sirt/<case>.npz           coarse SIRT volume (step 06)
  roi_pairs/{train,val}/    diffusion training pairs (step 07)
  models/diffusion/         checkpoints + logs (step 08)
  results/*.jsonl           per-nodule results (steps 09, 11)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .geometry import DTSGeometry, VolumeGeometry


class Paths:
    def __init__(self, cfg: dict):
        self.work = cfg["paths"]["work"]
        self.outputs = cfg["paths"]["outputs"]
        self.lidc_root = cfg["paths"]["lidc_root"]

    def p(self, *parts) -> str:
        path = os.path.join(self.work, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def out(self, step: str) -> str:
        d = os.path.join(self.outputs, step)
        os.makedirs(d, exist_ok=True)
        return d

    def case(self, cid: str) -> str:
        return self.p("cases", f"{cid}.npz")

    def dts(self, cid: str) -> str:
        return self.p("dts", f"{cid}.npz")

    def sirt(self, cid: str) -> str:
        return self.p("sirt", f"{cid}.npz")

    def case_ids(self) -> List[str]:
        return sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(self.work, "cases", "*.npz")))


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def make_splits(patient_ids: Iterable[str], seed: int = 42, fractions=(0.6, 0.2, 0.2)) -> Dict[str, str]:
    """Patient-level split -> {'patient': 'train'|'cal'|'test'} (deterministic)."""
    pats = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    n_tr = int(round(fractions[0] * n))
    n_cal = int(round(fractions[1] * n))
    if n >= 3:
        n_tr = max(1, min(n_tr, n - 2))
        n_cal = max(1, min(n_cal, n - n_tr - 1))
    out = {}
    for i, p in enumerate(pats):
        out[p] = "train" if i < n_tr else ("cal" if i < n_tr + n_cal else "test")
    return out


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def append_jsonl(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (VolumeGeometry, DTSGeometry)):
        return o.to_dict()
    return str(o)


def save_dts(path: str, clean: torch.Tensor, geom: DTSGeometry, meta: Optional[dict] = None) -> None:
    np.savez_compressed(path, clean=clean.detach().cpu().numpy().astype(np.float16),
                        geom=np.frombuffer(json.dumps(geom.to_dict()).encode(), dtype=np.uint8),
                        meta=np.frombuffer(json.dumps(meta or {}, default=_json_default).encode(), dtype=np.uint8))


def load_dts(path: str, device="cpu") -> Tuple[torch.Tensor, DTSGeometry, dict]:
    z = np.load(path)
    geom = DTSGeometry.from_dict(json.loads(bytes(z["geom"]).decode()))
    meta = json.loads(bytes(z["meta"]).decode()) if "meta" in z else {}
    return torch.as_tensor(z["clean"].astype(np.float32), device=device), geom, meta


def save_sirt(path: str, mu: torch.Tensor, vg: VolumeGeometry, meta: Optional[dict] = None) -> None:
    np.savez_compressed(path, mu=mu.detach().cpu().numpy().astype(np.float32),
                        vgeom=np.frombuffer(json.dumps(vg.to_dict()).encode(), dtype=np.uint8),
                        meta=np.frombuffer(json.dumps(meta or {}, default=_json_default).encode(), dtype=np.uint8))


def load_sirt(path: str, device="cpu") -> Tuple[torch.Tensor, VolumeGeometry, dict]:
    z = np.load(path)
    vg = VolumeGeometry.from_dict(json.loads(bytes(z["vgeom"]).decode()))
    meta = json.loads(bytes(z["meta"]).decode()) if "meta" in z else {}
    return torch.as_tensor(z["mu"], device=device), vg, meta
