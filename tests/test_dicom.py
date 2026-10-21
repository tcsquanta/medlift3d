"""DICOM loading on a synthetic CT series (shuffled files, duplicate z, rescale)."""
import os

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")


def _write_series(folder, uid, zs, inums, rows=16, cols=20, ps=(0.7, 0.8), slope=1.0, intercept=-1024.0):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    os.makedirs(folder, exist_ok=True)
    for n, (z, inum) in enumerate(zip(zs, inums)):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = uid
        ds.StudyInstanceUID = "1.2.3.4"
        ds.PatientID = "LIDC-IDRI-9999"
        ds.Modality = "CT"
        ds.InstanceNumber = int(inum)
        ds.ImagePositionPatient = [-100.0, -120.0, float(z)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [ps[0], ps[1]]
        ds.SliceThickness = 1.25
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept
        # pixel value encodes the slice z (so ordering can be checked) and the column index
        arr = (np.full((rows, cols), 1024 + int(round(z * 10)), np.int16) + np.arange(cols, dtype=np.int16)[None])
        ds.PixelData = arr.tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(os.path.join(folder, f"img_{n:03d}.dcm"), enforce_file_format=True)


def test_load_sorted_dedup_rescaled(tmp_path):
    from medlift3d.data.lidc import find_series_dir, load_dicom_series

    uid = "1.2.826.0.1.3680043.2.1125.1"
    zs = [-30.0, -27.5, -35.0, -32.5, -30.0]                  # shuffled, one duplicate z (-30.0)
    inums = [7, 6, 9, 8, 3]                                   # duplicate: keep InstanceNumber 3
    root = tmp_path / "LIDC-IDRI"
    _write_series(str(root / "LIDC-IDRI-9999" / "study" / "series"), uid, zs, inums)
    _write_series(str(root / "LIDC-IDRI-9999" / "study" / "other"), "9.9.9", [0.0, 2.5], [1, 2])
    folder = find_series_dir(str(root), uid, "LIDC-IDRI-9999")
    assert folder is not None and folder.endswith("series")
    hu, vg, z_sorted = load_dicom_series(folder, uid)
    assert list(z_sorted) == [-35.0, -32.5, -30.0, -27.5]     # ascending, duplicate removed
    assert hu.shape == (4, 16, 20)
    # HU = pixel * slope + intercept = z*10 + column index
    assert np.allclose(hu[:, 0, 0], np.array(z_sorted) * 10)
    assert np.allclose(hu[0, 0, :], -350 + np.arange(20))
    assert vg.spacing == (0.8, 0.7, 2.5)                      # (x = column spacing, y = row spacing, z)
    assert vg.origin == (-100.0, -120.0, -35.0)
