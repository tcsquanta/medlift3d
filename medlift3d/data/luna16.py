"""Optional image source: LUNA16 ``.mhd/.raw`` (the same LIDC-IDRI CT scans,
pre-converted; handy if you already have them on Kaggle). Annotations still
come from pylidc, matched by SeriesInstanceUID (= the .mhd file name)."""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from ..geometry import VolumeGeometry
from .cases import Case

_DTYPES = {"MET_SHORT": np.int16, "MET_USHORT": np.uint16, "MET_FLOAT": np.float32,
           "MET_UCHAR": np.uint8, "MET_CHAR": np.int8, "MET_INT": np.int32, "MET_DOUBLE": np.float64}


def read_mhd(path: str) -> Tuple[np.ndarray, VolumeGeometry, Dict]:
    """Minimal MetaImage reader (no SimpleITK needed). Returns HU [z,y,x]."""
    hdr: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                hdr[k.strip()] = v.strip()
    dims = [int(v) for v in hdr["DimSize"].split()]                    # x, y, z
    spacing = [float(v) for v in hdr["ElementSpacing"].split()]
    offset = [float(v) for v in hdr.get("Offset", hdr.get("Origin", "0 0 0")).split()]
    tm = [float(v) for v in hdr.get("TransformMatrix", "1 0 0 0 1 0 0 0 1").split()]
    dtype = _DTYPES[hdr["ElementType"]]
    raw = hdr["ElementDataFile"]
    raw_path = raw if os.path.isabs(raw) else os.path.join(os.path.dirname(path), raw)
    data = np.fromfile(raw_path, dtype=dtype)
    if hdr.get("BinaryDataByteOrderMSB", "False").lower() == "true":
        data = data.byteswap()
    vol = data.reshape(dims[2], dims[1], dims[0]).astype(np.float32)
    if not np.allclose(tm, [1, 0, 0, 0, 1, 0, 0, 0, 1], atol=1e-3):
        # flip axes with negative direction cosines so arrays stay LPS-ascending
        d = np.array(tm).reshape(3, 3)
        for ax_xyz in range(3):
            if d[ax_xyz, ax_xyz] < 0:
                arr_axis = 2 - ax_xyz
                vol = np.flip(vol, axis=arr_axis)
                offset[ax_xyz] = offset[ax_xyz] - (dims[ax_xyz] - 1) * spacing[ax_xyz]
        vol = np.ascontiguousarray(vol)
    vol = np.clip(vol, -1000.0, 3071.0)
    return vol, VolumeGeometry(vol.shape, tuple(spacing), tuple(offset)), hdr


def build_case_luna16(mhd_path: str, min_readers: int = 3, clevel: float = 0.5,
                      min_diameter: float = 4.0, max_diameter: float = 30.0):
    from .lidc import import_pylidc, nodules_for_scan

    pl = import_pylidc()
    uid = os.path.splitext(os.path.basename(mhd_path))[0]
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == uid).first()
    if scan is None:
        raise KeyError(f"{uid} not in pylidc database")
    hu, vg, _ = read_mhd(mhd_path)
    from .cases import remove_table

    hu = remove_table(hu)
    zs = vg.origin[2] + vg.spacing[2] * np.arange(hu.shape[0])
    nods, info = nodules_for_scan(scan, zs, hu.shape, vg, min_readers, clevel, min_diameter, max_diameter)
    case = Case(scan.patient_id + "_" + uid[-6:], hu, vg, nods, "luna16",
                {"patient_id": scan.patient_id, "series_uid": uid, "slice_thickness": float(scan.slice_thickness)})
    info.update(n_nodules=len(nods), shape=list(hu.shape), spacing=list(vg.spacing))
    return case, info
