#!/usr/bin/env python
"""Step 5 - simulate 15-view low-dose chest DTS from every CT case.

Stores clean line integrals (noise is added later with a fixed per-case seed,
so any dose can be re-simulated). Verification: physically plausible line
integrals, every nodule's projected position, nodule contrast-to-noise.

    python scripts/05_simulate_dts.py            # GPU: a few s/case, CPU: minutes
"""
import os

from _common import base_parser, setup

import numpy as np
import torch

from medlift3d.data.cases import load_case
from medlift3d.geometry import VolumeGeometry
from medlift3d.io import Paths, save_dts, stable_seed
from medlift3d.physics import add_poisson_noise, hu_to_mu
from medlift3d.simulate import geometry_for_volume, project_local_change, simulate_clean
from medlift3d.utils import Checker
from medlift3d import viz


def nodule_cnr(case, nod, geom, clean, i0, mu_w, dev) -> float:
    """Peak line-integral increment of the nodule / Poisson noise there (central view)."""
    z0, z1, y0, y1, x0, x1 = nod.bbox_zyx
    crop = case.hu[z0:z1, y0:y1, x0:x1]
    lung = np.median(crop[~nod.mask]) if (~nod.mask).any() else -850.0
    exc = np.where(nod.mask, np.clip(crop - lung, 0, None), 0.0)
    dmu = hu_to_mu(exc - 1000.0, mu_w)                                    # excess HU -> attenuation
    vg = VolumeGeometry(crop.shape, case.vgeom.spacing, tuple(case.vgeom.index_to_world(np.array([x0, y0, z0], float))))
    dp = project_local_change(dmu, vg, geom, device=dev)
    v = geom.n_views // 2
    r, c = np.unravel_index(int(torch.argmax(dp[v])), dp[v].shape)
    sigma = 1.0 / np.sqrt(i0 * np.exp(-float(clean[v, r, c])))
    return float(dp[v, r, c]) / sigma


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--figures", type=int, default=4)
    args = ap.parse_args()
    cfg, dev = setup(args)
    paths = Paths(cfg)
    out = paths.out("05_dts")
    ck = Checker("Step 5: DTS simulation")
    mu_w = cfg["physics"]["mu_water"]
    i0 = cfg["simulation"]["i0"]
    ids = paths.case_ids()[: args.limit] if args.limit else paths.case_ids()
    pmax, starve, cnrs, done = [], [], [], 0
    for k, cid in enumerate(ids, 1):
        if os.path.exists(paths.dts(cid)) and not args.overwrite:
            continue
        case = load_case(paths.case(cid), load_readers=False)
        geom = geometry_for_volume(case.hu, case.vgeom, cfg["geometry"])
        clean = simulate_clean(case.hu, case.vgeom, geom, mu_w, cfg["simulation"]["step_mm"], device=dev)
        save_dts(paths.dts(cid), clean, geom, {"case_id": cid, "mu_water": mu_w})
        done += 1
        pmax.append(float(clean.max()))
        starve.append(float((i0 * torch.exp(-clean) < 10).float().mean()))
        for n in case.nodules:
            try:
                cnrs.append(nodule_cnr(case, n, geom, clean, i0, mu_w, dev))
            except Exception as e:  # pragma: no cover
                print("cnr failed:", e)
        print(f"[{k}/{len(ids)}] {cid}: max line integral {pmax[-1]:.2f}, d2i {geom.detector_to_iso:.0f} mm")
        if done <= args.figures:
            noisy = add_poisson_noise(clean, i0, torch.Generator(device=dev).manual_seed(stable_seed(cid)))
            pts = np.array([n.center_xyz for n in case.nodules])
            rows, cols = geom.project_points(pts)
            viz.save_projection_montage(os.path.join(out, f"{cid}_views.png"), noisy.cpu().numpy(),
                                        np.stack([rows[:, 0], cols[:, 0]], -1),
                                        f"{cid}: 15 low-dose views (red: nodule 1 predicted position)")
    if done == 0:
        ck.info("nothing to do", "all cases already simulated (use --overwrite)")
    else:
        ck.check("line integrals finite and physical (max < 15)", max(pmax) < 15 and np.isfinite(pmax).all(),
                 f"max {max(pmax):.2f}")
        ck.info("fraction of photon-starved pixels (< 10 counts)", f"{np.mean(starve) * 100:.2f}%")
        if cnrs:
            c = np.array(cnrs)
            ck.info("nodule CNR in central view (median, 10%)", f"{np.median(c):.1f}, {np.percentile(c, 10):.1f}")
            ck.check("most nodules are above the noise in a single view (median CNR > 1)", np.median(c) > 1.0,
                     f"{np.median(c):.1f}")
    ck.finish(out)


if __name__ == "__main__":
    main()
