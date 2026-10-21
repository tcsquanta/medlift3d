#!/usr/bin/env python
"""Step 11 - growth tracking with controlled virtual growth (known truth).

For each test nodule: a follow-up CT is created by growing the nodule by a
known factor (doubling time DT over an interval dt, or no growth), the
follow-up DTS is simulated (baseline projections + projection of the local
change, new noise), reconstructed, and every method's VDT (volume) and MDT
(mass) are compared with the true doubling time and NELSON category.

  H4  mass doubling time (MDT) from DTS is more accurate than VDT

    python scripts/11_growth_experiment.py --limit 40 --pairs-per-nodule 2
"""
import copy
import math
import os

from _common import base_parser, setup

import numpy as np
import torch

from medlift3d.data.cases import load_case, nodule_from_full_mask
from medlift3d.geometry import VolumeGeometry
from medlift3d.growth import SLOW, doubling_time, growth_category, scale_for_doubling, virtual_growth
from medlift3d.io import Paths, append_jsonl, load_dts, load_json, load_sirt, read_jsonl, stable_seed
from medlift3d.measure import measure_ground_truth
from medlift3d.physics import hu_to_mu
from medlift3d.pipeline import CaseContext, coarse_sirt, ground_truth_for_roi, process_nodule
from medlift3d.roi import ROI
from medlift3d.simulate import add_noise, project_local_change
from medlift3d.utils import Checker
from medlift3d import viz


def grow_case(case, j, scale, mcfg):
    """Return (follow-up case, crop geometry, delta-mu crop, gt ratio check)."""
    nod = case.nodules[j]
    full = nod.full_mask(case.hu.shape)
    r_mm = 0.5 * max(nod.diameter_mm, 4.0)
    pad_mm = (scale - 1.0) * r_mm + 12.0
    sp_zyx = np.asarray(case.vgeom.spacing)[::-1]
    pad = np.ceil(pad_mm / sp_zyx).astype(int)
    z0, z1, y0, y1, x0, x1 = nod.bbox_zyx
    lo = np.maximum(np.array([z0, y0, x0]) - pad, 0)
    hi = np.minimum(np.array([z1, y1, x1]) + pad, case.hu.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    crop = case.hu[sl]
    cmask = full[sl]
    cvg = VolumeGeometry(crop.shape, case.vgeom.spacing, tuple(case.vgeom.index_to_world(lo[::-1].astype(float))))
    base = measure_ground_truth(crop, cvg, cmask, mcfg)["baseline_hu"]
    grown, gmask = virtual_growth(crop, cvg, cmask, scale, base)
    dmu = hu_to_mu(grown, 0.02) - hu_to_mu(crop, 0.02)
    case_f = copy.copy(case)
    case_f.hu = case.hu.copy()
    case_f.hu[sl] = grown
    full_f = np.zeros(case.hu.shape, bool)
    full_f[sl] = gmask
    nf = nodule_from_full_mask(nod.nid, full_f, case.vgeom, diameter_mm=nod.diameter_mm * scale,
                               n_readers=nod.n_readers, malignancy=nod.malignancy, texture=nod.texture)
    nf.center_xyz = list(nod.center_xyz)
    case_f.nodules = list(case.nodules)
    case_f.nodules[j] = nf
    return case_f, cvg, dmu


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--limit", type=int, default=30, help="max nodules")
    ap.add_argument("--pairs-per-nodule", type=int, default=2)
    ap.add_argument("--warm-sirt-iter", type=int, default=10)
    args = ap.parse_args()
    cfg, dev = setup(args)
    paths = Paths(cfg)
    out = paths.out("11_growth")
    ck = Checker("Step 11: virtual growth -> VDT vs MDT")
    methods = args.methods or list(cfg["pipeline"]["methods"])
    model = diffusion = None
    ckpt = args.ckpt or os.path.join(cfg["paths"]["work"], "models", "diffusion", "best.pt")
    if any(m.startswith("diffusion") for m in methods):
        if os.path.exists(ckpt):
            from medlift3d.diffusion.checkpoint import load_for_sampling

            model, diffusion, _ = load_for_sampling(ckpt, device=dev)
        else:
            methods = [m for m in methods if not m.startswith("diffusion")]
            ck.info("no diffusion checkpoint -> physics methods only", ckpt)
    res_path = paths.p("results", "growth.jsonl")
    done = {(r["case_id"], r["nodule_index"], r["interval_days"], r["true_dt_days"]) for r in read_jsonl(res_path)}
    splits = load_json(paths.p("splits.json"))
    ids = sorted(c for c, s in splits["cases"].items() if s in args.splits)
    combos = [(float(dt), float(DT)) for dt in cfg["growth"]["interval_days"] for DT in cfg["growth"]["doubling_days"]]
    n_nod = 0
    gt_ratio_err = []
    for cid in ids:
        if n_nod >= args.limit:
            break
        if not (os.path.exists(paths.dts(cid)) and os.path.exists(paths.sirt(cid))):
            continue
        case = load_case(paths.case(cid), load_readers=False)
        clean, geom, _ = load_dts(paths.dts(cid), device=dev)
        cmu, cvg, meta = load_sirt(paths.sirt(cid), device=dev)
        seed0 = int(meta.get("noise_seed", stable_seed(cid)))
        noisy = add_noise(clean, cfg["simulation"]["i0"], seed=seed0)
        ctx_b = CaseContext(cid, geom, noisy, cmu, cvg, cfg["physics"]["mu_water"], str(dev))
        for j, nod in enumerate(case.nodules):
            if n_nod >= args.limit:
                break
            rng = np.random.default_rng(stable_seed(cid, j, "growth"))
            pick = [combos[i] for i in rng.choice(len(combos), size=min(args.pairs_per_nodule, len(combos)), replace=False)]
            pick = [p for p in pick if (cid, j, p[0], p[1]) not in done]
            if not pick:
                continue
            roi = ROI.from_config(nod.center_xyz, cfg["roi"])
            gt_b = ground_truth_for_roi(case, j, roi, cfg["measure"])
            res_b = process_nodule(ctx_b, nod.center_xyz, cfg, model, diffusion, methods, args.members,
                                   seed=stable_seed(cid, j), gt=gt_b)
            n_nod += 1
            for dt, DT in pick:
                s = scale_for_doubling(dt, DT)
                case_f, gvg, dmu = grow_case(case, j, s, cfg["measure"])
                clean_f = clean + project_local_change(dmu, gvg, geom, device=dev)
                noisy_f = add_noise(clean_f, cfg["simulation"]["i0"], seed=stable_seed(cid, j, dt, DT))
                cmu_f, _, _ = coarse_sirt(noisy_f, geom, case.vgeom, cfg["sirt"], device=dev, x0=cmu,
                                          n_iter=args.warm_sirt_iter)
                ctx_f = CaseContext(cid, geom, noisy_f, cmu_f, cvg, cfg["physics"]["mu_water"], str(dev))
                gt_f = ground_truth_for_roi(case_f, j, roi, cfg["measure"])
                res_f = process_nodule(ctx_f, nod.center_xyz, cfg, model, diffusion, methods, args.members,
                                       seed=stable_seed(cid, j, "fu"), gt=gt_f)
                true_dt = DT if DT < 1e8 else float("inf")
                g_ratio = gt_f["meas"]["volume_mm3"] / gt_b["meas"]["volume_mm3"]
                m_ratio = gt_f["meas"]["mass_mg"] / max(gt_b["meas"]["mass_mg"], 1e-6)
                if s > 1.0:                                       # stable pairs are trivially exact
                    gt_ratio_err.append((abs(m_ratio / s ** 3 - 1), abs(g_ratio / s ** 3 - 1)))
                row = {"case_id": cid, "nodule_index": j, "diameter_mm": nod.diameter_mm, "interval_days": dt,
                       "true_dt_days": DT, "scale": s, "gt_volume_ratio": g_ratio,
                       "gt_mass_ratio": gt_f["meas"]["mass_mg"] / max(gt_b["meas"]["mass_mg"], 1e-6),
                       "true_category": growth_category(true_dt), "methods": {}}
                for m in res_b["methods"]:
                    if m not in res_f["methods"]:
                        continue
                    eb, ef = res_b["methods"][m], res_f["methods"][m]
                    entry = {}
                    for qty, key in (("volume", "volume_mm3"), ("mass", "mass_mg")):
                        dtm = doubling_time(eb[key], ef[key], dt)
                        mb = [p[key] for p in eb["members"]]
                        mf = [p[key] for p in ef["members"]]
                        pairs = [doubling_time(a, b, dt) for a in mb for b in mf]
                        fin = np.array([p if np.isfinite(p) else 1e6 for p in pairs])
                        entry[qty] = {"dt_days": dtm, "category": growth_category(dtm),
                                      "dt_p05": float(np.percentile(fin, 5)), "dt_p95": float(np.percentile(fin, 95)),
                                      "log_ratio": math.log(max(ef[key], 1e-6) / max(eb[key], 1e-6))}
                    row["methods"][m] = entry
                append_jsonl(row, res_path)
                mm = "diffusion_gs" if "diffusion_gs" in row["methods"] else list(row["methods"])[-1]
                print(f"{cid} n{j} dt={dt} DT={DT:g} s={s:.3f}: true {row['true_category']} | {mm}: "
                      f"VDT {row['methods'][mm]['volume']['dt_days']:.0f} d, MDT {row['methods'][mm]['mass']['dt_days']:.0f} d")
    rows = read_jsonl(res_path)
    ck.check("growth pairs available", len(rows) > 0, len(rows))
    if gt_ratio_err:
        ge = np.array(gt_ratio_err)
        ck.check("virtual growth on CT: |GT mass ratio / s^3 - 1| (median)", np.median(ge[:, 0]) < 0.1,
                 f"{100 * np.median(ge[:, 0]):.1f}%", "< 10%")
        ck.info("   GT voxel-volume ratio vs s^3 (median abs dev; discretisation of small masks)",
                f"{100 * np.median(ge[:, 1]):.1f}%")
    if not rows:
        ck.finish(out)
        return
    lines = ["| method | quantity | category accuracy | median abs log2(DT_est/DT_true) (growing) | false growth alarms (stable) |",
             "|---|---|---|---|---|"]
    acc = {}
    for m in sorted({k for r in rows for k in r["methods"]}):
        for qty in ("volume", "mass"):
            R = [r for r in rows if m in r["methods"]]
            if not R:
                continue
            cat = np.mean([r["methods"][m][qty]["category"] == r["true_category"] for r in R])
            grow = [r for r in R if r["true_dt_days"] < 1e8]
            le = [abs(math.log2(max(r["methods"][m][qty]["dt_days"], 1e-6) / r["true_dt_days"]))
                  for r in grow if np.isfinite(r["methods"][m][qty]["dt_days"])]
            stab = [r for r in R if r["true_dt_days"] >= 1e8]
            fa = np.mean([r["methods"][m][qty]["dt_days"] < 600 for r in stab]) if stab else float("nan")
            acc[(m, qty)] = (cat, np.median(le) if le else float("nan"))
            lines.append(f"| {m} | {qty} ({'VDT' if qty == 'volume' else 'MDT'}) | {100 * cat:.0f}% | "
                         f"{np.median(le) if le else float('nan'):.2f} | {100 * fa if stab else float('nan'):.0f}% |")
    print("\n" + "\n".join(lines) + "\n")
    with open(os.path.join(out, "table_growth.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    best = "diffusion_gs" if any("diffusion_gs" in r["methods"] for r in rows) else ("roi_gs" if any("roi_gs" in r["methods"] for r in rows) else "roi_sirt")
    if (best, "mass") in acc and (best, "volume") in acc:
        ck.check(f"H4 MDT category accuracy >= VDT ({best})", acc[(best, "mass")][0] >= acc[(best, "volume")][0],
                 f"MDT {100 * acc[(best, 'mass')][0]:.0f}% vs VDT {100 * acc[(best, 'volume')][0]:.0f}%")
    xs, ys, labs = [], [], []
    for qty in ("volume", "mass"):
        R = [r for r in rows if best in r["methods"] and r["true_dt_days"] < 1e8 and np.isfinite(r["methods"][best][qty]["dt_days"])]
        if R:
            xs.append([r["true_dt_days"] for r in R])
            ys.append([r["methods"][best][qty]["dt_days"] for r in R])
            labs.append("VDT" if qty == "volume" else "MDT")
    if xs:
        viz.save_scatter(os.path.join(out, f"doubling_time_{best}.png"), xs, ys, labs,
                         f"{best}: estimated vs true doubling time", "true (days)", "estimated (days)")
    ck.finish(out)


if __name__ == "__main__":
    main()
