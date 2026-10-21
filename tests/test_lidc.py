"""LIDC-IDRI annotation handling with the real pylidc database (offline, no DICOM needed)."""
import os

import numpy as np
import pytest

pl = pytest.importorskip("pylidc", reason="pip install pylidc") if False else None


def _pl():
    try:
        from medlift3d.data.lidc import import_pylidc

        return import_pylidc()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"pylidc unavailable: {e}")


def test_select_nodules_from_database():
    _pl()
    from medlift3d.data.lidc import select_nodules

    rows = select_nodules(max_scans=15, verbose=False)
    assert len(rows) > 0
    r = rows[0]
    for k in ("patient_id", "series_uid", "n_readers", "diameter_mm", "malignancy", "centroid_ijk"):
        assert k in r
    assert all(r["n_readers"] >= 3 and 4.0 <= r["diameter_mm"] <= 30.0 for r in rows)


def test_consensus_masks_map_into_volume():
    pl = _pl()
    from medlift3d.data.lidc import nodules_for_scan
    from medlift3d.geometry import VolumeGeometry

    scan = None
    for s in pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 2.5).limit(30):
        if any(len(c) >= 3 for c in s.cluster_annotations(verbose=False)):
            scan = s
            break
    assert scan is not None
    zs = np.asarray(scan.slice_zvals)
    shape = (len(zs), 512, 512)
    ps = float(scan.pixel_spacing)
    vg = VolumeGeometry(shape, (ps, ps, float(np.median(np.diff(zs)))), (0.0, 0.0, float(zs[0])))
    nods, info = nodules_for_scan(scan, zs, shape, vg, min_readers=3, min_diameter=3.0)
    assert info["max_slice_dz_mm"] < 1e-6 and len(nods) > 0
    for n in nods:
        assert n.mask.any()
        full = n.full_mask(shape)
        com = np.argwhere(full).mean(0)                     # (k, i, j) in our [z, y, x]
        z0, z1, y0, y1, x0, x1 = n.bbox_zyx
        assert z0 <= com[0] <= z1 and y0 <= com[1] <= y1 and x0 <= com[2] <= x1
        # consensus volume is in the same ballpark as pylidc's own reader volumes
        v = full.sum() * vg.voxel_volume
        assert 0.3 < v / n.meta["volume_mm3_pylidc"] < 2.0


def test_slice_misalignment_is_detected():
    pl = _pl()
    from medlift3d.data.lidc import nodules_for_scan
    from medlift3d.geometry import VolumeGeometry

    scan = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 2.5).first()
    zs = np.asarray(scan.slice_zvals)
    dz = float(np.median(np.diff(zs)))
    shifted = zs + 0.7 * dz
    vg = VolumeGeometry((len(zs), 512, 512), (0.7, 0.7, dz), (0.0, 0.0, float(shifted[0])))
    with pytest.raises(ValueError):
        nodules_for_scan(scan, shifted, (len(zs), 512, 512), vg)


def test_manifest(tmp_path):
    from medlift3d.data.lidc import write_tcia_manifest

    p = tmp_path / "m.tcia"
    write_tcia_manifest(["1.2.3", "1.2.4", "1.2.3"], str(p))
    lines = p.read_text().strip().splitlines()
    assert lines[5] == "ListOfSeriesToDownload=" and lines[6:] == ["1.2.3", "1.2.4"]
