"""LIDC-IDRI: radiologist annotations (pylidc) + DICOM images (TCIA).

* The 8.62 MB "Radiologist Annotations/Segmentations" XML from the TCIA page
  is exactly what pylidc ships inside its SQLite database, so nodule masks,
  diameters and malignancy ratings are available offline.
* Images: instead of the 133 GB bundle, ``select_nodules`` picks good cases
  and ``download_series`` fetches only those series through TCIA's public
  NBIA REST API (or use the written ``.tcia`` manifest with NBIA Data Retriever).
* ``load_dicom_series`` mirrors pylidc's slice handling (duplicate z removed,
  sorted by ascending z), and ``nodules_for_scan`` maps pylidc's (row, col,
  slice) masks into our [z, y, x] volume with an explicit alignment check.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import VolumeGeometry
from .cases import Case, Nodule

NBIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def import_pylidc():
    """Import pylidc 0.2.3 on a modern stack.

    It uses np.int / np.float / np.bool (removed in NumPy >= 1.24) and
    ``pkg_resources.resource_filename`` (removed from recent setuptools).
    """
    import sys
    import types

    os.environ.setdefault("MPLBACKEND", "Agg")
    for name, typ in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, typ)
    try:
        import pkg_resources  # noqa: F401
    except Exception:
        import importlib.util

        shim = types.ModuleType("pkg_resources")

        def resource_filename(package, resource):
            spec = importlib.util.find_spec(package)
            return os.path.join(os.path.dirname(spec.origin), resource)

        shim.resource_filename = resource_filename
        sys.modules["pkg_resources"] = shim
    import pylidc as pl  # noqa: E402

    return pl


# ---------------------------------------------------------------------------
# subset selection (offline, from the pylidc annotation database)
# ---------------------------------------------------------------------------
def select_nodules(max_slice_thickness: float = 2.5, min_readers: int = 3, min_diameter: float = 4.0,
                   max_diameter: float = 30.0, max_scans: Optional[int] = None, verbose: bool = True) -> List[Dict]:
    """One row per physical nodule (cluster of reader annotations)."""
    pl = import_pylidc()
    q = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= max_slice_thickness)
    scans = q.all()
    if max_scans:
        scans = scans[:max_scans]
    rows: List[Dict] = []
    for si, scan in enumerate(scans):
        try:
            clusters = scan.cluster_annotations(verbose=False)
        except Exception as e:  # pragma: no cover - rare DB quirks
            if verbose:
                print(f"[warn] {scan.patient_id}: cluster_annotations failed ({e})")
            continue
        for ci, anns in enumerate(clusters):
            if len(anns) < min_readers:
                continue
            diam = float(np.mean([a.diameter for a in anns]))
            if not (min_diameter <= diam <= max_diameter):
                continue
            cen = np.mean([a.centroid for a in anns], axis=0)
            rows.append({
                "patient_id": scan.patient_id, "series_uid": scan.series_instance_uid,
                "study_uid": scan.study_instance_uid, "cluster": ci, "n_readers": len(anns),
                "diameter_mm": diam, "volume_mm3_pylidc": float(np.mean([a.volume for a in anns])),
                "malignancy": float(np.mean([a.malignancy for a in anns])),
                "texture": float(np.mean([a.texture for a in anns])),
                "subtlety": float(np.mean([a.subtlety for a in anns])),
                "centroid_ijk": [float(c) for c in cen], "slice_thickness": float(scan.slice_thickness),
                "pixel_spacing": float(scan.pixel_spacing), "n_slices": int(len(scan.slice_zvals)),
            })
        if verbose and (si + 1) % 100 == 0:
            print(f"  scanned {si + 1}/{len(scans)} scans, {len(rows)} nodules so far")
    return rows


def choose_patients(rows: List[Dict], n_patients: int, seed: int = 42) -> List[str]:
    """Pick patients with the most qualifying nodules first (more data per download)."""
    from collections import Counter

    cnt = Counter(r["patient_id"] for r in rows)
    rng = np.random.default_rng(seed)
    pats = list(cnt.keys())
    rng.shuffle(pats)
    pats.sort(key=lambda p: -cnt[p])
    return pats[:n_patients]


def write_tcia_manifest(series_uids: Iterable[str], path: str) -> None:
    """Manifest for NBIA Data Retriever (fallback to the REST downloader)."""
    uids = list(dict.fromkeys(series_uids))
    lines = ["downloadServerUrl=https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet",
             "includeAnnotation=true", "noOfrRetry=4", "databasketId=manifest-medlift3d.tcia",
             "manifestVersion=3.0", "ListOfSeriesToDownload="] + uids
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def download_series(series_uid: str, out_root: str, retries: int = 4, timeout: int = 300,
                    base_url: Optional[str] = None) -> str:
    """Download one series (ZIP of DICOMs) via the public NBIA REST API.

    Files land in ``out_root/<series_uid>/``. The ZIP is unpacked into
    ``<series_uid>.part`` and renamed only when complete, so an interrupted
    download is never mistaken for a finished one. Skips series already present.
    """
    import shutil
    import urllib.request

    out_dir = os.path.join(out_root, series_uid)
    if os.path.isdir(out_dir) and any(f.endswith(".dcm") for f in os.listdir(out_dir)):
        return out_dir
    url = f"{base_url or NBIA_API}/getImage?NewFileNames=Yes&SeriesInstanceUID={series_uid}"
    req = urllib.request.Request(url, headers={"User-Agent": "medlift3d/1.0"})
    tmp_dir = out_dir + ".part"
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            os.makedirs(tmp_dir)
            n = 0
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    blob = zf.read(info)
                    name = os.path.basename(info.filename)
                    if not (name.lower().endswith(".dcm") or blob[128:132] == b"DICM"):
                        continue                                  # e.g. a LICENSE file
                    if not name.lower().endswith(".dcm"):
                        name += ".dcm"
                    with open(os.path.join(tmp_dir, name), "wb") as f:
                        f.write(blob)
                    n += 1
            if n == 0:
                raise RuntimeError("ZIP contains no DICOM files (wrong SeriesInstanceUID?)")
            shutil.rmtree(out_dir, ignore_errors=True)
            os.replace(tmp_dir, out_dir)
            return out_dir
        except Exception as e:  # network hiccups -> exponential backoff
            last = e
            if attempt + 1 < retries:
                time.sleep(2 ** (attempt + 1))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    raise RuntimeError(f"download failed for {series_uid}: {last}")


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------
def find_series_dir(root: str, series_uid: str, patient_id: Optional[str] = None) -> Optional[str]:
    """Locate the folder holding a series under the common NBIA layouts."""
    import pydicom

    direct = os.path.join(root, series_uid)
    if os.path.isdir(direct):
        return direct
    bases = [os.path.join(root, patient_id)] if patient_id and os.path.isdir(os.path.join(root, patient_id)) else [root]
    for base in bases:
        for dirpath, dirnames, filenames in os.walk(base):
            if os.path.basename(dirpath) == series_uid:
                return dirpath
            dcms = [f for f in filenames if f.lower().endswith(".dcm")]
            if dcms:
                try:
                    ds = pydicom.dcmread(os.path.join(dirpath, dcms[0]), stop_before_pixels=True)
                    if str(ds.SeriesInstanceUID).strip() == series_uid:
                        return dirpath
                except Exception:
                    continue
    return None


def load_dicom_series(folder: str, series_uid: Optional[str] = None) -> Tuple[np.ndarray, VolumeGeometry, np.ndarray]:
    """Read an axial CT series -> (HU [z,y,x] float32, geometry, sorted slice z positions)."""
    import pydicom

    items = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".dcm") or f.startswith("."):
            continue
        ds = pydicom.dcmread(os.path.join(folder, f))
        if series_uid and str(ds.SeriesInstanceUID).strip() != series_uid:
            continue
        if not hasattr(ds, "ImagePositionPatient"):
            continue
        items.append((float(ds.ImagePositionPatient[2]), float(getattr(ds, "InstanceNumber", 0)), ds))
    if not items:
        raise FileNotFoundError(f"no CT slices in {folder}")
    # pylidc rule: duplicate z -> keep lower InstanceNumber; then sort by z ascending
    best: Dict[float, Tuple[float, object]] = {}
    for z, inum, ds in items:
        if z not in best or inum < best[z][0]:
            best[z] = (inum, ds)
    zs = np.array(sorted(best.keys()))
    slices = [best[z][1] for z in zs]
    ds0 = slices[0]
    iop = np.array([float(v) for v in getattr(ds0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])])
    if not np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=1e-3):
        print(f"[warn] non-standard orientation {iop.tolist()} in {folder}; assuming axial LPS")
    vol = np.stack([s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1.0))
                    + float(getattr(s, "RescaleIntercept", 0.0)) for s in slices], axis=0)
    vol = np.clip(vol, -1000.0, 3071.0)                         # padding (-2000/-3024) -> air
    ps = [float(v) for v in ds0.PixelSpacing]                   # (row spacing = y, col spacing = x)
    dz = float(np.median(np.diff(zs))) if len(zs) > 1 else float(getattr(ds0, "SliceThickness", 1.0))
    ipp = [float(v) for v in ds0.ImagePositionPatient]
    vg = VolumeGeometry(vol.shape, (ps[1], ps[0], dz), (ipp[0], ipp[1], zs[0]))
    return vol, vg, zs


# ---------------------------------------------------------------------------
# radiologist masks -> our volume
# ---------------------------------------------------------------------------
def slice_index_map(scan_zvals: np.ndarray, our_z: np.ndarray) -> Tuple[np.ndarray, float]:
    """For every pylidc slice index k, the index of our slice with the nearest z.
    Returns (map, max |dz|) -- max |dz| ~ 0 means perfect alignment."""
    sz = np.asarray(scan_zvals, float)
    oz = np.asarray(our_z, float)
    idx = np.abs(sz[:, None] - oz[None, :]).argmin(axis=1)
    return idx, float(np.abs(oz[idx] - sz).max())


def nodules_for_scan(scan, our_z: np.ndarray, vol_shape, vg: VolumeGeometry, min_readers: int = 3,
                     clevel: float = 0.5, min_diameter: float = 4.0, max_diameter: float = 30.0,
                     keep_reader_masks: bool = True) -> Tuple[List[Nodule], Dict]:
    """Consensus masks (>= clevel of readers) for every qualifying nodule of a pylidc Scan."""
    import_pylidc()
    from pylidc.utils import consensus

    kmap, max_dz = slice_index_map(scan.slice_zvals, our_z)
    info = {"max_slice_dz_mm": max_dz, "n_clusters": 0}
    tol = 0.5 * float(np.median(np.diff(our_z))) if len(our_z) > 1 else 0.5
    if max_dz > tol:
        raise ValueError(f"slice misalignment {max_dz:.3f} mm between DICOM and pylidc for {scan.patient_id}")
    out: List[Nodule] = []
    clusters = scan.cluster_annotations(verbose=False)
    info["n_clusters"] = len(clusters)
    for ci, anns in enumerate(clusters):
        if len(anns) < min_readers:
            continue
        diam = float(np.mean([a.diameter for a in anns]))
        if not (min_diameter <= diam <= max_diameter):
            continue
        cmask, cbbox, masks = consensus(anns, clevel=clevel, pad=2, ret_masks=True)
        i0, i1 = cbbox[0].start, cbbox[0].stop
        j0, j1 = cbbox[1].start, cbbox[1].stop
        ks = kmap[cbbox[2].start:cbbox[2].stop]
        if not np.all(np.diff(ks) == 1):
            raise ValueError("non-contiguous slice mapping")
        z0, z1 = int(ks[0]), int(ks[-1]) + 1
        crop = np.transpose(cmask, (2, 0, 1))                    # (i,j,k) -> (k,i,j) = [z,y,x]
        if not crop.any():
            continue
        readers = np.stack([np.transpose(m, (2, 0, 1)) for m in masks]) if keep_reader_masks else None
        idx = np.argwhere(crop) + np.array([z0, i0, j0])
        center = vg.index_to_world(idx.mean(0)[::-1])
        out.append(Nodule(f"{scan.patient_id}_c{ci}", [float(v) for v in center],
                          [z0, z1, int(i0), int(i1), int(j0), int(j1)], crop, readers, diam, len(anns),
                          float(np.mean([a.malignancy for a in anns])), float(np.mean([a.texture for a in anns])),
                          {"volume_mm3_pylidc": float(np.mean([a.volume for a in anns])),
                           "subtlety": float(np.mean([a.subtlety for a in anns]))}))
    return out, info


def build_case(series_uid: str, lidc_root: str, min_readers: int = 3, clevel: float = 0.5,
               min_diameter: float = 4.0, max_diameter: float = 30.0) -> Tuple[Case, Dict]:
    """DICOM + annotations -> Case (raises with a clear message if anything is off)."""
    pl = import_pylidc()
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).first()
    if scan is None:
        raise KeyError(f"series {series_uid} not in pylidc database")
    folder = find_series_dir(lidc_root, series_uid, scan.patient_id)
    if folder is None:
        raise FileNotFoundError(f"DICOM folder for {scan.patient_id} / {series_uid} not found under {lidc_root}")
    hu, vg, zs = load_dicom_series(folder, series_uid)
    from .cases import remove_table

    hu = remove_table(hu)                                   # no CT table in a standing DTS exam
    nods, info = nodules_for_scan(scan, zs, hu.shape, vg, min_readers, clevel, min_diameter, max_diameter)
    info.update(folder=folder, n_nodules=len(nods), shape=list(hu.shape), spacing=list(vg.spacing))
    case = Case(scan.patient_id + "_" + series_uid[-6:], hu, vg, nods, "lidc",
                {"patient_id": scan.patient_id, "series_uid": series_uid,
                 "slice_thickness": float(scan.slice_thickness)})
    return case, info
