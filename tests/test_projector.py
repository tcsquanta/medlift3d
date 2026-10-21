"""Physics checks for the projector: analytic chords, adjoint, mass conservation, parallax."""
import math

import numpy as np
import pytest
import torch

from medlift3d.geometry import DTSGeometry, VolumeGeometry, bin_projections
from medlift3d.phantom import sphere_volume
from medlift3d.physics import hu_to_mu
from medlift3d.projector import Projector, ray_aabb


def _small_geom(n=96, pitch=0.5, sid=1800.0, d2i=150.0, views=15):
    return DTSGeometry(n_views=views, sweep_deg=30.0, sid=sid, detector_to_iso=d2i,
                       det_shape=(n, n), det_pitch=(pitch, pitch))


def test_ray_aabb_basic():
    o = torch.tensor([[0.0, 10.0, 0.0]])
    d = torch.tensor([[0.0, -1.0, 0.0]])
    t0, t1 = ray_aabb(o, d, [-1, -1, -1], [1, 1, 1])
    assert abs(float(t0) - 9.0) < 1e-5 and abs(float(t1) - 11.0) < 1e-5


def test_uniform_cube_chord():
    vg = VolumeGeometry.centered((20, 20, 20), (1.0, 1.0, 1.0))
    vol = torch.full(vg.shape, 0.02)
    P = Projector(vg, step_mm=0.25)
    o = torch.tensor([[0.3, 100.0, -0.2], [5.0, 100.0, 5.0]])
    d = torch.tensor([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    p = P(vol, o, d)
    # physical extent along y is 20 mm -> 0.4
    assert torch.allclose(p, torch.full_like(p, 0.4), rtol=0.01)


@pytest.mark.parametrize("offset", [0.0, 4.0, 8.0])
def test_sphere_chord_analytic(offset):
    R, mu = 10.0, 0.02
    hu, vg = sphere_volume(0.0, R, spacing=0.5, size_mm=26.0, background_hu=-1000.0)
    vol = torch.as_tensor(hu_to_mu(hu, mu))
    P = Projector(vg, step_mm=0.1)
    o = torch.tensor([[offset, 500.0, 0.0]])
    d = torch.tensor([[0.0, -1.0, 0.0]])
    expected = 2 * math.sqrt(R * R - offset * offset) * mu
    got = float(P(vol, o, d))
    assert abs(got - expected) / expected < 0.03, (got, expected)


def test_adjoint_dot_product():
    torch.manual_seed(0)
    vg = VolumeGeometry.centered((16, 18, 20), (1.0, 1.2, 0.9))
    geom = _small_geom(n=24, pitch=1.0, views=3)
    o, d = geom.rays()
    P = Projector(vg)
    x = torch.rand(vg.shape)
    y = torch.rand(o.shape[0])
    lhs = float((P(x, o, d) * y).sum())
    rhs = float((x * P.backproject(y, o, d)).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-4


def test_checkpoint_gradients_match():
    torch.manual_seed(1)
    vg = VolumeGeometry.centered((12, 12, 12), (1.0, 1.0, 1.0))
    geom = _small_geom(n=16, pitch=1.0, views=2)
    o, d = geom.rays()
    x = torch.rand(vg.shape)
    grads = []
    for ck in (True, False):
        xx = x.clone().requires_grad_(True)
        P = Projector(vg, use_checkpoint=ck, samples_per_chunk=2000)
        (P(xx, o, d) ** 2).sum().backward()
        grads.append(xx.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_exclude_box():
    vg = VolumeGeometry.centered((20, 20, 20), (1.0, 1.0, 1.0))
    vol = torch.zeros(vg.shape)
    vol[:, 10:, :] = 0.02                              # voxel centres y >= 0.5 filled
    P = Projector(vg, step_mm=0.1)
    o = torch.tensor([[0.0, 100.0, 0.0]])
    d = torch.tensor([[0.0, -1.0, 0.0]])
    full = float(P(vol, o, d))
    # trilinear interpolation ramps from 0 at y=-0.5 to full at y=+0.5 -> exclude from -0.5
    excl = float(P(vol, o, d, exclude_box=([-50, -0.5, -50], [50, 50, 50])))
    assert full > 0.15 and abs(excl) < 0.01 * full


def _mass_and_detector_integral(radii, center, geom, spacing=0.75):
    """Blob of known mass; return (mass, per-view detector integrals / M^2)."""
    n = 64
    vg = VolumeGeometry.centered((n, n, n), (spacing,) * 3, center)
    x, y, z = vg.axes()
    Z, Y, X = np.meshgrid(z - center[2], y - center[1], x - center[0], indexing="ij")
    f = 0.017 * np.exp(-0.5 * ((X / radii[0]) ** 2 + (Y / radii[1]) ** 2 + (Z / radii[2]) ** 2))
    mass = f.sum() * vg.voxel_volume
    P = Projector(vg, step_mm=0.25)
    vol = torch.as_tensor(f, dtype=torch.float32)
    integ = []
    mag = float(geom.magnification(np.array([center]))[0])
    cos_g = np.cos(geom.angles())                 # obliquity of the ray through the blob (blob at isocentre)
    for v in range(geom.n_views):
        o, d = geom.rays([v])
        p = P(vol, o, d)
        # detector integral = mass * M^2 / cos(gamma)
        integ.append(float(p.sum()) * geom.det_pitch[0] * geom.det_pitch[1] * cos_g[v] / mag ** 2)
    return mass, np.array(integ)


def test_mass_is_measured_by_every_view():
    """Zeroth moment: detector integral * cos(gamma) / M^2 == object mass in all 15 views."""
    geom = _small_geom(n=280, pitch=0.5)          # 140 mm: the blob stays on the detector in every view
    mass, integ = _mass_and_detector_integral((4.0, 4.0, 4.0), (0.0, 0.0, 0.0), geom)
    rel = np.abs(integ - mass) / mass
    assert rel.max() < 0.02, rel


def test_depth_extent_is_invisible_to_mass():
    """Two blobs with equal mass but 3x different depth extent give the same detector integral."""
    geom = _small_geom(n=280, pitch=0.5)
    m1, i1 = _mass_and_detector_integral((4.0, 3.0, 4.0), (0.0, 0.0, 0.0), geom)
    m2, i2 = _mass_and_detector_integral((4.0, 9.0, 4.0), (0.0, 0.0, 0.0), geom)
    # normalise to equal mass
    i2 = i2 * (m1 / m2)
    assert np.max(np.abs(i1 - i2) / i1) < 0.02


def test_parallax_matches_geometry():
    """A small blob above the isocentre moves across views exactly as predicted."""
    geom = _small_geom(n=400, pitch=0.5)          # 200 mm detector
    c = (3.0, 40.0, -5.0)
    vg = VolumeGeometry.centered((32, 32, 32), (0.5,) * 3, c)
    x, y, z = vg.axes()
    Z, Y, X = np.meshgrid(z - c[2], y - c[1], x - c[0], indexing="ij")
    f = torch.as_tensor(0.02 * np.exp(-0.5 * (X ** 2 + Y ** 2 + Z ** 2) / 1.5 ** 2), dtype=torch.float32)
    P = Projector(vg)
    rows_pred, cols_pred = geom.project_points(np.array([c]))
    nr, nc = geom.det_shape
    rr, cc = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    for v in (0, 7, 14):
        o, d = geom.rays([v])
        img = P(f, o, d).reshape(nr, nc).numpy()
        r_c = (img * rr).sum() / img.sum()
        c_c = (img * cc).sum() / img.sum()
        assert abs(r_c - rows_pred[v, 0]) < 0.3 and abs(c_c - cols_pred[v, 0]) < 0.3


def test_binning_consistency():
    geom = _small_geom(n=63, pitch=0.5, views=2)
    gb = geom.binned(3)
    pix = geom.pixel_centers()
    pb = bin_projections(pix.transpose(2, 0, 1), 3).transpose(1, 2, 0)
    assert np.allclose(pb, gb.pixel_centers(), atol=1e-6)


def test_local_change_projection_matches_full_projection():
    """project_local_change (window per view) == projecting the whole volume (regression test)."""
    from medlift3d.simulate import project_local_change

    geom = _small_geom(n=120, pitch=1.0)
    c = (5.0, 20.0, -8.0)
    vg = VolumeGeometry.centered((16, 16, 16), (1.0, 1.0, 1.0), c)
    f = np.zeros(vg.shape, np.float32)
    f[5:11, 6:10, 4:12] = 0.015
    local = project_local_change(f, vg, geom).numpy()
    P = Projector(vg)
    full = np.stack([P(torch.as_tensor(f), *geom.rays([v])).reshape(120, 120).numpy() for v in range(geom.n_views)])
    assert np.abs(local - full).max() < 1e-5 and full.max() > 0.05


def test_prism_covers_deep_nodules():
    """The prism starts at the detector, but is shifted back so a deep nodule's box stays inside."""
    from medlift3d.geometry import DTSGeometry
    from medlift3d.roi import ROI

    geom = DTSGeometry(detector_to_iso=150.0)                  # detector plane at y = -150
    rcfg = {"size_vox": [16, 128, 16], "voxel_mm": [1.0, 2.0, 1.0], "box_mm": [40.0, 50.0, 40.0]}
    for depth in (60.0, 200.0, 260.0, 300.0):                  # nodule depth in front of the detector (mm)
        click = (0.0, geom.det_y + depth, 0.0)
        roi = ROI.from_config(click, rcfg, geom)
        lo, hi = roi.vgeom.box_min[1], roi.vgeom.box_max[1]
        assert lo >= geom.det_y                                # never behind the detector
        assert lo <= click[1] - 25.0 and hi >= click[1] + 25.0  # nodule box inside the prism
        if depth <= 200.0:
            assert abs(roi.y_start - (geom.det_y + 5.0)) < 1e-9  # usual case: unchanged
