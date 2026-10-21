"""Gaussian representation: closed-form integral, gradients, fitting."""
import math

import numpy as np
import torch

from medlift3d.gaussians import (GaussianField, fit_to_volume, gaussians_from_volume, quat_to_rotmat,
                                 scale_bounds)
from medlift3d.geometry import VolumeGeometry


def test_rotation_is_orthonormal():
    q = torch.randn(10, 4)
    R = quat_to_rotmat(q)
    eye = torch.eye(3).expand(10, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5)


def test_voxel_sum_equals_closed_form_integral():
    vg = VolumeGeometry.centered((40, 40, 40), (1.0, 1.0, 1.0))
    lo, hi = scale_bounds(vg, 4)
    torch.manual_seed(0)
    n = 20
    means = torch.rand(n, 3) * 16 - 8
    scales = lo + (hi - lo) * torch.rand(n, 3)
    amps = torch.rand(n) * 0.02 + 0.001
    quats = torch.randn(n, 4)
    gf = GaussianField(means, scales, amps, quats, lo, hi)
    vox = gf.voxelize(vg)
    total_vox = vox.detach().sum().item() * vg.voxel_volume
    total_cf = float(gf.integrals().sum())
    assert abs(total_vox - total_cf) / total_cf < 0.01, (total_vox, total_cf)


def test_gradients_reach_all_parameters():
    vg = VolumeGeometry.centered((16, 16, 16), (1.0, 1.0, 1.0))
    lo, hi = scale_bounds(vg, 4)
    gf = GaussianField(torch.zeros(3, 3) + 0.3, torch.full((3, 3), 1.0), torch.full((3,), 0.01), None, lo, hi)
    (gf.voxelize(vg) ** 2).sum().backward()
    for p in (gf.means, gf.scale_raw, gf.amp_raw):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_init_and_fit_reproduce_blob():
    vg = VolumeGeometry.centered((32, 32, 32), (1.0, 1.0, 1.0))
    x, y, z = vg.axes()
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    blob = 0.017 / (1 + np.exp((np.sqrt(X ** 2 + (Y / 1.3) ** 2 + Z ** 2) - 6.0) / 0.7))
    target = torch.as_tensor(blob, dtype=torch.float32)
    gf = gaussians_from_volume(target, vg, stride=2, threshold_mu=1e-4)
    init_err = float(((gf.voxelize(vg) - target) ** 2).mean() / (target ** 2).mean())
    hist = fit_to_volume(gf, target, vg, n_iter=60)
    assert hist[-1] < init_err and hist[-1] < 0.03, (init_err, hist[-1])
    mass_true = float(target.sum()) * vg.voxel_volume
    mass_cf = float(gf.integrals().sum())
    assert abs(mass_cf - mass_true) / mass_true < 0.03
