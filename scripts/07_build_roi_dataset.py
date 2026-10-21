#!/usr/bin/env python
"""Step 7 - training pairs for the diffusion prior (train-split patients only).

pair = (condition: physics-only ROI-SIRT from the simulated DTS,
        target:    CT resampled into the same 64^3 ROI)
* nodule ROIs with random centre jitter (so the model never assumes the
  nodule is exactly centred), and
* random lung ROIs without nodules (so the prior learns NOT to invent nodules).
10% of train patients are held out as a validation set for training previews.

Verification: pair counts, condition-vs-target PSNR (the bar the diffusion
model has to beat), preview montage.

    python scripts/07_build_roi_dataset.py --jitter 3 --negatives 2
"""
import os

from _common import base_parser, pick_cases, setup

import numpy as np
import torch

from medlift3d.data.cases import load_case
from medlift3d.diffusion.dataset import save_pair
from medlift3d.io import Paths, load_dts, load_json, load_sirt, stable_seed
from medlift3d.metrics import psnr
from medlift3d.physics import hu_to_mu, mu_to_hu
from medlift3d.roi import ROI, ROIProblem, resample_to_grid
from medlift3d.simulate import add_noise
from medlift3d.utils import Checker
from medlift3d import viz


def lung_points(case, rng, n, avoid, min_dist=25.0):
    hu = case.hu
    vg = case.vgeom
    pts = []
    tries = 0
    while len(pts) < n and tries < 5000:
        tries += 1
        iz, iy, ix = (rng.integers(int(0.15 * s), int(0.85 * s)) for s in hu.shape)
        if hu[iz, iy, ix] > -700:
            continue
        w = vg.index_to_world(np.array([ix, iy, iz], float))
        if any(np.linalg.norm(w - np.asarray(a)) < min_dist for a in avoid):
            continue
        lo = np.maximum([iz - 4, iy - 4, ix - 4], 0)
        if hu[lo[0]:iz + 5, lo[1]:iy + 5, lo[2]:ix + 5].mean() > -600:
            continue
        pts.append(w)
    return pts


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--jitter", type=int, default=3, help="jittered copies per nodule")
    ap.add_argument("--jitter-mm", type=float, default=6.0)
    ap.add_argument("--negatives", type=int, default=2, help="random lung ROIs per case")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg, dev = setup(args)
    paths = Paths(cfg)
    out = paths.out("07_roi_dataset")
    ck = Checker("Step 7: diffusion training pairs")
    splits = load_json(paths.p("splits.json"))
    train_pats = sorted(p for p, s in splits["patients"].items() if s == "train")
    rng_v = np.random.default_rng(cfg["seed"])
    val_pats = set(rng_v.choice(train_pats, size=max(1, int(round(args.val_frac * len(train_pats)))), replace=False))
    mu_w = cfg["physics"]["mu_water"]
    ids = pick_cases([c for c, s in splits["cases"].items() if s == "train"], args)
    ids = ids[: args.limit] if args.limit else ids
    counts = {"train": 0, "val": 0}
    ps_c, previews, att_ratio = [], [], []
    for k, cid in enumerate(sorted(ids), 1):
        if not (os.path.exists(paths.dts(cid)) and os.path.exists(paths.sirt(cid))):
            print(f"[{k}] {cid}: missing DTS/SIRT, skipped")
            continue
        case = load_case(paths.case(cid), load_readers=False)
        split = "val" if case.patient_id in val_pats else "train"
        clean, geom, _ = load_dts(paths.dts(cid), device=dev)
        cmu, cvg, meta = load_sirt(paths.sirt(cid), device=dev)
        noisy = add_noise(clean, cfg["simulation"]["i0"], seed=int(meta.get("noise_seed", stable_seed(cid))),
                          electronic_sigma=cfg["simulation"]["electronic_sigma"])
        rng = np.random.default_rng(stable_seed(cid, "roi"))
        centers = []
        for j, n in enumerate(case.nodules):
            for r in range(args.jitter):
                jit = rng.uniform(-args.jitter_mm, args.jitter_mm, 3) if r > 0 else np.zeros(3)
                centers.append(("nodule", j, np.asarray(n.center_xyz) + jit))
        for p in lung_points(case, rng, args.negatives, [n.center_xyz for n in case.nodules]):
            centers.append(("negative", -1, p))
        for i, (kind, j, c) in enumerate(centers):
            roi = ROI.from_config(c, cfg["roi"], geom)
            prob = ROIProblem(roi, geom, noisy, cmu, cvg, step_mm=cfg["roi"]["step_mm"],
                              coarse_step_mm=cfg["sirt"]["step_mm"], margin_px=cfg["roi"]["margin_px"],
                              device=dev, mu_water=mu_w, min_ray_fraction=float(cfg["roi"]["min_ray_fraction"]))
            mu, _ = prob.sirt(n_iter=int(cfg["roi"]["sirt_iter"]), init_hu=float(cfg["roi"]["lung_hu"]))
            cond = mu_to_hu(mu.cpu().numpy(), mu_w)
            target = resample_to_grid(case.hu, case.vgeom, roi.vgeom, order=1)
            cidx = roi.vgeom.world_to_index(np.asarray(c, float))[::-1]          # (iz, iy, ix) in the prism
            save_pair(paths.p("roi_pairs", split, f"{cid}_{i:03d}.npz"), cond, target, case_id=cid, kind=kind,
                      nodule=j, center=c, center_idx=cidx)
            counts[split] += 1
            ps_c.append(psnr(np.clip(cond, -1000, 400), np.clip(target, -1000, 400), 1400.0))
            # DTS measures each column's total attenuation well: the condition must keep it
            att_ratio.append(float(hu_to_mu(cond, mu_w).sum() / max(hu_to_mu(target, mu_w).sum(), 1e-9)))
            if len(previews) < 4 and kind == "nodule":
                previews.append((cond, target, roi.vgeom, c))
        print(f"[{k}/{len(ids)}] {cid} -> {split}: {len(centers)} pairs")
    ck.check("training pairs written", counts["train"] > 0, counts)
    if att_ratio:
        ck.check("condition keeps the prism's attenuation (sum mu cond / CT, median)",
                 0.85 < np.median(att_ratio) < 1.15, f"{np.median(att_ratio):.3f} "
                 f"(10-90%: {np.percentile(att_ratio, 10):.2f}-{np.percentile(att_ratio, 90):.2f})", "0.85 - 1.15")
    if ps_c:
        ck.info("PSNR of condition (ROI-SIRT) vs CT target, median dB", f"{np.median(ps_c):.2f}  <- the diffusion must beat this")
    for i, (cnd, tgt, vg, c) in enumerate(previews):
        viz.save_roi_panel(os.path.join(out, f"pair_{i}.png"), {"condition ROI-SIRT": cnd, "target CT": tgt}, vg, c)
    ck.finish(out)


if __name__ == "__main__":
    main()
