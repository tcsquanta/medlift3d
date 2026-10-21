#!/usr/bin/env python
"""Step 1 - verify the X-ray physics before using any real data.

Checks (each prints PASS/FAIL, figures go to outputs/01_physics/):
  1. projector line integrals match the analytic chord through a sphere
  2. back-projection is the exact adjoint (dot-product test)
  3. Poisson noise has the theoretical standard deviation
  4. parallax: a nodule moves across the 15 views exactly as geometry predicts
  5. KEY CLAIM: every view measures the nodule MASS (detector integral / M^2),
     and a 3x longer depth extent (bigger volume, same mass) is invisible to it
  6. limited-angle DTS reconstruction smears nodules in depth (missing wedge)

    python scripts/01_verify_physics.py            # ~1-3 min on CPU
"""
import math
import os

from _common import base_parser, setup

import numpy as np
import torch

from medlift3d.geometry import DTSGeometry, VolumeGeometry, bin_projections
from medlift3d.metrics import local_fwhm
from medlift3d.phantom import NoduleSpec, make_chest_phantom, sphere_volume
from medlift3d.physics import add_poisson_noise, hu_to_mu, mu_to_hu
from medlift3d.projector import Projector
from medlift3d.simulate import geometry_for_volume, simulate_clean
from medlift3d.sirt import SIRT
from medlift3d.utils import Checker
from medlift3d import viz


def blob(center, radii, n=64, spacing=0.75, peak=0.017):
    vg = VolumeGeometry.centered((n, n, n), (spacing,) * 3, center)
    x, y, z = vg.axes()
    Z, Y, X = np.meshgrid(z - center[2], y - center[1], x - center[0], indexing="ij")
    f = peak * np.exp(-0.5 * ((X / radii[0]) ** 2 + (Y / radii[1]) ** 2 + (Z / radii[2]) ** 2))
    return f.astype(np.float32), vg


def detector_integrals(f, vg, geom, center, dev):
    P = Projector(vg, step_mm=0.25)
    vol = torch.as_tensor(f, device=dev)
    mag = float(geom.magnification(np.array([center]))[0])
    cos_g = np.cos(geom.angles())             # ray obliquity (object at the isocentre)
    out = []
    for v in range(geom.n_views):
        o, d = geom.rays([v], device=dev)
        # detector integral = mass * M^2 / cos(gamma)  (inverse-square magnification + obliquity)
        out.append(float(P(vol, o, d).sum()) * geom.det_pitch[0] * geom.det_pitch[1] * cos_g[v] / mag ** 2)
    return np.array(out)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--fast", action="store_true", help="smaller phantom for slow CPUs")
    args = ap.parse_args()
    cfg, dev = setup(args)
    out = os.path.join(cfg["paths"]["outputs"], "01_physics")
    os.makedirs(out, exist_ok=True)
    ck = Checker("Step 1: X-ray physics")
    mu_w = cfg["physics"]["mu_water"]

    # 1. analytic sphere chords --------------------------------------------------
    R = 10.0
    hu, vg = sphere_volume(0.0, R, spacing=0.5, size_mm=26.0)
    vol = torch.as_tensor(hu_to_mu(hu, mu_w), device=dev)
    P = Projector(vg, step_mm=0.1)
    errs = []
    for off in (0.0, 4.0, 8.0):
        o = torch.tensor([[off, 500.0, 0.0]], device=dev)
        d = torch.tensor([[0.0, -1.0, 0.0]], device=dev)
        exp = 2 * math.sqrt(R * R - off * off) * mu_w
        errs.append(abs(float(P(vol, o, d)) - exp) / exp)
    ck.check("1 sphere chord vs analytic (max rel. error)", max(errs) < 0.03, max(errs), "< 0.03")

    # 2. adjoint ---------------------------------------------------------------
    g_small = DTSGeometry(n_views=3, det_shape=(24, 24), det_pitch=(1.0, 1.0))
    vg2 = VolumeGeometry.centered((16, 18, 20), (1.0, 1.2, 0.9))
    o, d = g_small.rays(device=dev)
    P2 = Projector(vg2)
    torch.manual_seed(0)
    x = torch.rand(vg2.shape, device=dev)
    yv = torch.rand(o.shape[0], device=dev)
    lhs = float((P2(x, o, d) * yv).sum())
    rhs = float((x * P2.backproject(yv, o, d)).sum())
    ck.check("2 adjoint dot-product test |<Ax,y>-<x,A'y>|/|<Ax,y>|", abs(lhs - rhs) / abs(lhs) < 1e-4,
             abs(lhs - rhs) / abs(lhs), "< 1e-4")

    # 3. noise --------------------------------------------------------------------
    i0 = cfg["simulation"]["i0"]
    p = torch.full((200000,), 2.0, device=dev)
    noisy = add_poisson_noise(p, i0, torch.Generator(device=dev).manual_seed(0))
    theory = 1.0 / math.sqrt(i0 * math.exp(-2.0))
    emp = float(noisy.std())
    ck.check("3 Poisson noise std of line integrals (empirical/theory)", abs(emp / theory - 1) < 0.05,
             f"{emp:.4f} / {theory:.4f}", "ratio within 5%")

    # 4. parallax -------------------------------------------------------------------
    geom = DTSGeometry(n_views=15, sweep_deg=30.0, det_shape=(400, 400), det_pitch=(0.5, 0.5))
    c = (3.0, 40.0, -5.0)
    f, vgb = blob(c, (1.5, 1.5, 1.5), n=32, spacing=0.5)
    Pb = Projector(vgb)
    rows_p, cols_p = geom.project_points(np.array([c]))
    nr, nc = geom.det_shape
    rr, cc = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    worst = 0.0
    imgs = []
    for v in range(geom.n_views):
        o, d = geom.rays([v], device=dev)
        img = Pb(torch.as_tensor(f, device=dev), o, d).reshape(nr, nc).cpu().numpy()
        imgs.append(img)
        r_c = (img * rr).sum() / img.sum()
        c_c = (img * cc).sum() / img.sum()
        worst = max(worst, abs(r_c - rows_p[v, 0]), abs(c_c - cols_p[v, 0]))
    ck.check("4 parallax: measured vs predicted blob position (max px error)", worst < 0.3, worst, "< 0.3 px")
    shift = float(rows_p[-1, 0] - rows_p[0, 0])
    ck.info("   blob 40 mm above isocentre shifts across the sweep by", f"{shift:.1f} px")
    viz.save_projection_montage(os.path.join(out, "parallax_blob.png"), np.stack(imgs),
                                np.stack([rows_p[:, 0], cols_p[:, 0]], -1), "blob 40 mm above isocentre: red = predicted")

    # 5. mass is measured by every view ---------------------------------------------------
    geom5 = DTSGeometry(n_views=15, sweep_deg=30.0, det_shape=(280, 280), det_pitch=(0.5, 0.5))
    f1, vg1 = blob((0, 0, 0), (4.0, 3.0, 4.0))
    f2, vg2b = blob((0, 0, 0), (4.0, 9.0, 4.0))
    m1 = f1.sum() * vg1.voxel_volume
    m2 = f2.sum() * vg2b.voxel_volume
    i1 = detector_integrals(f1, vg1, geom5, (0, 0, 0), dev)
    i2 = detector_integrals(f2, vg2b, geom5, (0, 0, 0), dev) * (m1 / m2)
    ck.check("5a every view measures the mass (max |detector integral*cos(g)/M^2 - mass|/mass)",
             np.max(np.abs(i1 - m1) / m1) < 0.02, float(np.max(np.abs(i1 - m1) / m1)), "< 0.02")
    ck.check("5b 3x depth extent (3x volume, same mass) changes detector integral by",
             np.max(np.abs(i1 - i2) / i1) < 0.02, float(np.max(np.abs(i1 - i2) / i1)), "< 0.02")
    s = math.sin(math.radians(cfg["geometry"]["sweep_deg"] / 2))
    ck.info("   moment conditioning: depth position / depth extent are worse than in-plane by",
            f"1/sin(15deg) = {1 / s:.1f}x and 1/sin^2(15deg) = {1 / s ** 2:.1f}x; mass: 1x")

    # 6. DTS reconstruction of a phantom: depth smear ---------------------------------------------
    sp = 3.0 if args.fast else 2.0
    ph = make_chest_phantom(spacing=(sp, sp, sp), seed=1, n_vessels=16 if args.fast else 40,
                            nodules=[NoduleSpec((-70.0, -5.0, 25.0), (6.0, 6.0, 6.0), 20.0, "test")])
    gcfg = dict(cfg["geometry"])
    if args.fast:
        gcfg.update(det_shape=[180, 180], det_pitch=[2.4, 2.4])
    else:
        gcfg.update(det_shape=[268, 268], det_pitch=[1.6, 1.6])
    g = geometry_for_volume(ph.hu, ph.vgeom, gcfg)
    clean = simulate_clean(ph.hu, ph.vgeom, g, mu_w, step_mm=sp / 2, device=dev)
    noisy = add_poisson_noise(clean, i0, torch.Generator(device=dev).manual_seed(1))
    nod = ph.nodules[0]
    r_p, c_p = g.project_points(np.array([nod.center]))
    viz.save_projection_montage(os.path.join(out, "phantom_dts_views.png"), noisy.cpu().numpy(),
                                np.stack([r_p[:, 0], c_p[:, 0]], -1), "phantom: 15 low-dose DTS views (red = nodule)")
    b = 2
    gb = g.binned(b)
    cvg = VolumeGeometry.centered((ph.vgeom.shape[0] // 1, ph.vgeom.shape[1] // 1, ph.vgeom.shape[2] // 1),
                                  ph.vgeom.spacing, ph.vgeom.center)
    o, d = gb.rays(device=dev)
    solver = SIRT(Projector(cvg, step_mm=sp), o, d, device=dev)
    xs, hist = solver.run(bin_projections(noisy, b).reshape(-1), n_iter=30)
    rec_hu = mu_to_hu(xs.cpu().numpy(), mu_w)
    ck.check("6a SIRT data residual decreases", hist[-1] < 0.6 * hist[0], f"{hist[0]:.3f} -> {hist[-1]:.3f}")
    viz.save_curve(os.path.join(out, "sirt_residual.png"), {"relative residual": hist}, "SIRT convergence",
                   ylabel="||b - Ax|| / ||b||")
    ix, iy, iz = np.round(cvg.world_to_index(np.array(nod.center))).astype(int)
    hw = int(round(40.0 / sp))                                   # +-40 mm local window
    w_x = local_fwhm(rec_hu[iz, iy, :], ix, cvg.spacing[0], hw)
    w_y = local_fwhm(rec_hu[iz, :, ix], iy, cvg.spacing[1], hw)
    w_z = local_fwhm(rec_hu[:, iy, ix], iz, cvg.spacing[2], hw)
    ck.check("6b missing wedge: depth (y) FWHM / in-plane FWHM of the nodule", w_y > 1.5 * max(w_x, w_z),
             f"y={w_y:.1f} mm vs x={w_x:.1f}, z={w_z:.1f} mm (true diameter 12 mm)", "ratio > 1.5")
    viz.save_orthoslices(os.path.join(out, "phantom_ct.png"), ph.hu, ph.vgeom, nod.center, ph.masks[0], "CT")
    viz.save_orthoslices(os.path.join(out, "phantom_dts_sirt.png"), rec_hu, cvg, nod.center, None, "DTS-SIRT")
    ck.finish(out)


if __name__ == "__main__":
    main()
