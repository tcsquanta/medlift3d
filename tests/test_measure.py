"""Measurement definitions, incl. the core claim: depth blur changes volume, not mass."""
import math

import numpy as np
from scipy import ndimage

from medlift3d.config import load_config
from medlift3d.measure import (coronal_diameters, excess_mass_mg, measure_ground_truth,
                               measure_reconstruction, segment_nodule, volume_mm3)
from medlift3d.phantom import sphere_volume

MCFG = load_config()["measure"]


def _nodule(r=5.0, hu=20.0, lung=-850.0, spacing=0.5, size=48.0):
    vol, vg = sphere_volume(hu, r, spacing=spacing, size_mm=size, background_hu=lung)
    return vol, vg


def test_sphere_volume_and_mass():
    r, hu, lung = 5.0, 20.0, -850.0
    vol, vg = _nodule(r, hu, lung)
    v_true = 4 / 3 * math.pi * r ** 3
    m_true = v_true * (hu - lung) / 1000.0
    out, mask = measure_reconstruction(vol, vg, (0, 0, 0), lung, MCFG)
    assert abs(out["volume_mm3"] - v_true) / v_true < 0.06
    assert abs(out["mass_mg"] - m_true) / m_true < 0.03


def test_ground_truth_measure_uses_mask():
    r, hu, lung = 6.0, 0.0, -850.0
    vol, vg = _nodule(r, hu, lung)
    x, y, z = vg.axes()
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    mask = X ** 2 + Y ** 2 + Z ** 2 <= r * r
    gt = measure_ground_truth(vol, vg, mask, MCFG)
    v_true = 4 / 3 * math.pi * r ** 3
    assert abs(gt["volume_mm3"] - v_true) / v_true < 0.05
    assert abs(gt["baseline_hu"] - lung) < 1.0


def test_depth_blur_changes_volume_not_mass():
    """Simulated missing-wedge smear along y (depth): mass stays, volume does not."""
    r, hu, lung = 4.0, 20.0, -850.0
    vol, vg = _nodule(r, hu, lung, size=64.0)
    m0, _ = measure_reconstruction(vol, vg, (0, 0, 0), lung, MCFG)
    excess = vol - lung
    smeared = lung + ndimage.gaussian_filter1d(excess, sigma=6.0 / vg.spacing[1], axis=1)   # y axis
    m1, _ = measure_reconstruction(smeared, vg, (0, 0, 0), lung, dict(MCFG, mass_dilate_mm=[2.0, 14.0, 2.0]))
    mass_change = abs(m1["mass_mg"] - m0["mass_mg"]) / m0["mass_mg"]
    vol_change = abs(m1["volume_mm3"] - m0["volume_mm3"]) / m0["volume_mm3"]
    assert mass_change < 0.05, mass_change
    assert vol_change > 0.3, vol_change


def test_coronal_diameter_of_sphere():
    r = 6.0
    vol, vg = _nodule(r)
    mask, _ = segment_nodule(vol, vg, (0, 0, 0), -850.0)
    d_long, d_short = coronal_diameters(mask, vg)
    assert abs(d_long - 2 * r) < 1.5 and abs(d_short - 2 * r) < 1.5


def test_segmentation_detaches_from_adjacent_structure_or_fails():
    """Nodule touching a wall: raise the threshold until it detaches; if it never does
    (wall as dense as the nodule), report nothing instead of the whole wall."""
    from medlift3d.geometry import VolumeGeometry
    from medlift3d.measure import segment_nodule

    vg = VolumeGeometry((80, 80, 80), (1.0, 1.0, 1.0), (-40.0, -40.0, -40.0))
    z, y, x = np.meshgrid(*(np.arange(80) - 40.0,) * 3, indexing="ij")
    sphere = x ** 2 + y ** 2 + z ** 2 <= 4.0 ** 2
    wall = x >= 3.5                                          # touches the nodule, extends past the search limit
    for wall_hu, separable in ((-300.0, True), (0.0, False)):
        vol = np.full((80, 80, 80), -850.0, np.float32)
        vol[wall] = wall_hu
        vol[sphere] = 0.0
        mask, info = segment_nodule(vol, vg, (0.0, 0.0, 0.0), baseline_hu=-850.0)
        if separable:
            assert info["threshold_raised"] >= 1 and not info["touches_limit"]
            assert 0.6 < mask.sum() / sphere.sum() < 1.4, mask.sum() / sphere.sum()
        else:
            assert info["touches_limit"] and not mask.any()


def test_fwhm_and_depth_smear_metrics():
    from medlift3d.geometry import VolumeGeometry
    from medlift3d.measure import depth_smear_metrics, fwhm_1d

    x = np.arange(101) - 50.0
    g = np.exp(-x ** 2 / (2 * 4.0 ** 2))
    assert abs(fwhm_1d(g, 50, 0.5) - 2.3548 * 4.0 * 0.5) < 0.05
    assert math.isnan(fwhm_1d(np.ones(10), 5, 1.0))                  # never drops to half
    # a Gaussian blob 3x longer in depth (y) than in-plane, on a lung background
    vg = VolumeGeometry.centered((65, 65, 65), (1.0, 1.0, 1.0))           # voxel centre at 0
    xs, ys, zs = vg.axes()
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    vol = -850.0 + 300.0 * np.exp(-(X ** 2 + Z ** 2) / (2 * 2.0 ** 2) - Y ** 2 / (2 * 6.0 ** 2))
    m = depth_smear_metrics(vol, vg, (0.0, 3.0, 0.0), -850.0)
    assert abs(m["contrast_hu"] - 300.0 * 0.85) < 5.0                # 3 x 3 in-plane average: x (1 + 2 e^-1/8)^2 / 9
    assert abs(m["depth_fwhm_mm"] / m["inplane_fwhm_mm"] - 3.0) < 0.3
    assert m["clutter_hu"] < 1.0                                      # no anatomy around it
