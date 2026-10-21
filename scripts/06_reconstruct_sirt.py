#!/usr/bin/env python
"""Step 6 - standard DTS reconstruction (coarse whole-chest SIRT) + candidates.

This is what a DTS scanner shows the radiologist (coronal slices). It also
supplies the out-of-ROI background for step 9. Verification: the data
residual falls, nodules are smeared in depth (missing wedge), and the
training-free candidate detector finds the radiologists' nodules.

    python scripts/06_reconstruct_sirt.py
"""
import os

from _common import base_parser, setup

import numpy as np
import torch

from medlift3d.data.cases import load_case
from medlift3d.detect import candidate_hits, detect_candidates
from medlift3d.io import Paths, load_dts, save_json, save_sirt, stable_seed
from medlift3d.metrics import local_fwhm
from medlift3d.physics import mu_to_hu
from medlift3d.pipeline import coarse_sirt
from medlift3d.simulate import add_noise
from medlift3d.utils import Checker
from medlift3d import viz


def depth_ratio(rec_hu, cvg, xyz):
    """Depth FWHM / in-plane FWHM of the nodule in the coarse reconstruction."""
    ix, iy, iz = np.clip(np.round(cvg.world_to_index(np.asarray(xyz))).astype(int), 0, cvg.n_xyz - 1)
    hw = int(round(40.0 / min(cvg.spacing)))
    wx = local_fwhm(rec_hu[iz, iy, :], ix, cvg.spacing[0], hw)
    wy = local_fwhm(rec_hu[iz, :, ix], iy, cvg.spacing[1], hw)
    wz = local_fwhm(rec_hu[:, iy, ix], iz, cvg.spacing[2], hw)
    return wy / max(wx, wz, 1e-6)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--figures", type=int, default=4)
    args = ap.parse_args()
    cfg, dev = setup(args)
    paths = Paths(cfg)
    out = paths.out("06_sirt")
    ck = Checker("Step 6: coarse DTS reconstruction (SIRT) + candidate detection")
    mu_w = cfg["physics"]["mu_water"]
    ids = paths.case_ids()[: args.limit] if args.limit else paths.case_ids()
    res_drop, ratios, ranks, done = [], [], [], 0
    cand_all = {}
    for k, cid in enumerate(ids, 1):
        if not os.path.exists(paths.dts(cid)):
            print(f"[{k}] {cid}: no DTS (run step 5)")
            continue
        if os.path.exists(paths.sirt(cid)) and not args.overwrite:
            continue
        case = load_case(paths.case(cid), load_readers=False)
        clean, geom, _ = load_dts(paths.dts(cid), device=dev)
        seed = stable_seed(cid)
        noisy = add_noise(clean, cfg["simulation"]["i0"], seed=seed, electronic_sigma=cfg["simulation"]["electronic_sigma"])
        mu, cvg, hist = coarse_sirt(noisy, geom, case.vgeom, cfg["sirt"], device=dev)
        save_sirt(paths.sirt(cid), mu, cvg, {"noise_seed": seed, "i0": cfg["simulation"]["i0"], "history": hist})
        done += 1
        res_drop.append(hist[-1] / hist[0])
        rec = mu_to_hu(mu.cpu().numpy(), mu_w)
        lung = float(np.median(rec[(rec > -1000) & (rec < -600)])) if np.any((rec > -1000) & (rec < -600)) else -850.0
        cands = detect_candidates(rec, cvg, baseline_hu=lung, top_k=10)
        cand_all[cid] = cands
        for n in case.nodules:
            ratios.append(depth_ratio(rec, cvg, n.center_xyz))
            ranks.append(candidate_hits(cands, n.center_xyz, n.diameter_mm))
        print(f"[{k}/{len(ids)}] {cid}: residual {hist[0]:.3f}->{hist[-1]:.3f}, grid {cvg.shape}, "
              f"nodule ranks among candidates {ranks[-len(case.nodules):]}")
        if done <= args.figures:
            n = case.nodules[0]
            viz.save_orthoslices(os.path.join(out, f"{cid}_sirt.png"), rec, cvg, n.center_xyz, None, f"{cid} DTS-SIRT")
            viz.save_orthoslices(os.path.join(out, f"{cid}_ct.png"), case.hu, case.vgeom, n.center_xyz,
                                 n.full_mask(case.hu.shape), f"{cid} CT")
            viz.save_curve(os.path.join(out, f"{cid}_residual.png"), {"residual": hist}, f"{cid} SIRT")
    save_json(cand_all, paths.p("results", "candidates.json")) if cand_all else None
    if done:
        ck.check("SIRT residual drops in every case (worst final/initial)", max(res_drop) < 0.9, f"{max(res_drop):.3f}", "< 0.9")
        r = np.array([x for x in ratios if np.isfinite(x)])
        if len(r):
            ck.check("missing wedge visible: depth/in-plane FWHM ratio (median)", np.median(r) > 1.3, f"{np.median(r):.2f}", "> 1.3")
        rk = np.array(ranks)
        ck.info("candidate detector recall@10 (nodule among top-10 suggestions)", f"{np.mean(rk > 0) * 100:.0f}%")
        ck.info("candidate detector recall@3", f"{np.mean((rk > 0) & (rk <= 3)) * 100:.0f}%")
    else:
        ck.info("nothing to do", "all cases reconstructed (use --overwrite)")
    ck.finish(out)


if __name__ == "__main__":
    main()
