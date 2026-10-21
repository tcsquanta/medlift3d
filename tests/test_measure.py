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
