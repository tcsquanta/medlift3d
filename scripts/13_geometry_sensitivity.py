#!/usr/bin/env python
"""Step 13 - sensitivity to the DTS acquisition geometry (number of views, sweep angle).

Our default is 15 views over a 30 deg linear sweep (the angular range of the GE VolumeRAD
chest DTS; real units take ~60-74 views, Shimadzu sweeps up to 40 deg). This step re-simulates
the same patients with other geometries and runs the physics-only methods on the same
nodules, so the report can say how much the results depend on that choice:

  views  (e.g. 15 vs 60 views over 30 deg)   -- same total dose, so mostly streaks / noise
  angle  (e.g. 30 vs 40 deg sweep)          -- the missing wedge, i.e. the depth smear

Total dose is kept equal (photons per view = i0 x reference views / views; use --dose per-view
to keep i0 per view instead). Only physics methods: the diffusion model was trained on the
reference geometry, so other geometries would be out of distribution for it.

Per nodule and geometry: nodule contrast and clutter in the prism ROI-SIRT, depth smear
(depth FWHM / in-plane FWHM of the nodule peak; no segmentation involved), and every method's
volume / mass error. Results: work/results/geometry.jsonl (resumable),
outputs/13_geometry/table_geometry.md and figures.

    python scripts/13_geometry_sensitivity.py --limit 12                       # ~1 h on one T4
    python scripts/13_geometry_sensitivity.py --variants 15x30 60x30 15x40 74x40 --cases LIDC-IDRI-0012_457842
"""
import copy
import os
import time

from _common import base_parser, pick_cases, setup

import numpy as np
import torch

from medlift3d.data.cases import load_case
from medlift3d.growth import MIN_MEASURABLE
from medlift3d.io import Paths, append_jsonl, load_dts, load_json, read_jsonl, stable_seed
from medlift3d.measure import depth_smear_metrics
from medlift3d.pipeline import CaseContext, coarse_sirt, ground_truth_for_roi, process_nodule
from medlift3d.roi import ROI
from medlift3d.simulate import add_noise, geometry_for_volume, simulate_clean
from medlift3d.utils import Checker
from medlift3d import viz

PHYSICS = ("sirt_coarse", "roi_sirt", "roi_gs")


def parse_variant(s: str):
    """'60x30' -> (60 views, 30.0 deg sweep)."""
    n, deg = s.lower().split("x")
    return int(n), float(deg)


def label(v: str) -> str:
    n, deg = parse_variant(v)
    return f"{n} views / {deg:g} deg (+-{deg / 2:g})"


def median(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--variants", nargs="+", default=["15x30", "60x30", "15x40", "74x40"],
                    help="views x total sweep (deg); the first one that matches the config is the reference")
    ap.add_argument("--splits", nargs="+", default=["test", "cal"])
    ap.add_argument("--methods", nargs="+", default=list(PHYSICS))
    ap.add_argument("--members", type=int, default=None, help="roi_gs ensemble size (default: config)")
    ap.add_argument("--limit", type=int, default=12, help="nodules (finished ones count)")
    ap.add_argument("--min-diameter", type=float, default=5.0, help="skip smaller nodules (mm)")
    ap.add_argument("--dose", choices=["matched", "per-view"], default="matched")
    ap.add_argument("--figures", type=int, default=4)
    args = ap.parse_args()
    cfg, dev = setup(args)
    paths = Paths(cfg)
    out = paths.out("13_geometry")
    ck = Checker("Step 13: acquisition geometry sensitivity (views, sweep angle)")
    bad = [m for m in args.methods if m not in PHYSICS]
    if bad:
        raise SystemExit(f"physics methods only ({', '.join(PHYSICS)}); not: {bad}")
    variants = list(dict.fromkeys(args.variants))
    g0 = cfg["geometry"]
    ref_views = int(g0["n_views"])
    ref = next((v for v in variants if parse_variant(v) == (ref_views, float(g0["sweep_deg"]))), variants[0])
    i0 = float(cfg["simulation"]["i0"])
    ck.info("reference geometry (config; the diffusion model's training geometry)", label(ref))
    ck.info("dose", f"total photons equal to {ref_views} views x {i0:g} per pixel" if args.dose == "matched"
            else f"{i0:g} photons per pixel per view (total dose grows with the views)")

    res_path = paths.p("results", "geometry.jsonl")
    done = {(r["case_id"], r["nodule_index"], r["variant"]) for r in read_jsonl(res_path)}
    splits = load_json(paths.p("splits.json"))
    ids = pick_cases(sorted(c for c, s in splits["cases"].items() if s in args.splits), args)
    n_nod, n_fail, figs, unseen = 0, 0, 0, []
    for cid in ids:
        if n_nod >= args.limit:
            break
        if not os.path.exists(paths.case(cid)):
            continue
        case = load_case(paths.case(cid), load_readers=False)
        nods = [j for j, n in enumerate(case.nodules) if n.diameter_mm >= args.min_diameter][: args.limit - n_nod]
        if not nods:
            continue
        n_nod += len(nods)
        todo = [v for v in variants if any((cid, j, v) not in done for j in nods)]
        gt_cache, fig_vols = {}, {j: {} for j in nods[: max(0, args.figures - figs)]}
        for v in todo:
            n, deg = parse_variant(v)
            gcfg = dict(copy.deepcopy(g0), n_views=n, sweep_deg=deg)
            geom = geometry_for_volume(case.hu, case.vgeom, gcfg)
            t0 = time.perf_counter()
            clean = None
            if os.path.exists(paths.dts(cid)):                    # step 5 already simulated this geometry
                c, gs, _ = load_dts(paths.dts(cid), device=dev)
                if gs.n_views == n and abs(gs.sweep_deg - deg) < 1e-6:
                    clean, geom = c, gs
            if clean is None:
                clean = simulate_clean(case.hu, case.vgeom, geom, cfg["physics"]["mu_water"],
                                       cfg["simulation"]["step_mm"], device=dev)
            t_sim = time.perf_counter() - t0
            i0_v = i0 * ref_views / n if args.dose == "matched" else i0
            noisy = add_noise(clean, i0_v, seed=stable_seed(cid, "geometry", v),
                              electronic_sigma=cfg["simulation"]["electronic_sigma"])
            del clean
            t0 = time.perf_counter()
            cmu, cvg, _ = coarse_sirt(noisy, geom, case.vgeom, cfg["sirt"], device=dev)
            t_sirt = time.perf_counter() - t0
            ctx = CaseContext(cid, geom, noisy, cmu, cvg, cfg["physics"]["mu_water"], str(dev))
            for j in nods:
                if (cid, j, v) in done and j not in fig_vols:
                    continue
                nod = case.nodules[j]
                rows, cols = geom.project_points(np.asarray(nod.center_xyz)[None])
                nr, nc = geom.det_shape
                if not np.all((rows >= 0) & (rows < nr) & (cols >= 0) & (cols < nc)):
                    unseen.append(f"{cid} n{j} {v}")
                roi = ROI.from_config(nod.center_xyz, cfg["roi"], geom)
                try:
                    if j not in gt_cache:
                        gt = ground_truth_for_roi(case, j, roi, cfg["measure"])
                        gt_cache[j] = (gt, depth_smear_metrics(gt["roi_hu"], gt["roi_vg"], nod.center_xyz,
                                                               gt["meas"]["baseline_hu"]))
                    gt, gt_smear = gt_cache[j]
                    res = process_nodule(ctx, nod.center_xyz, cfg, None, None, args.methods, None, seed=stable_seed(cid, j),
                                         gt=gt, keep_volumes=True, n_members_physics=args.members)
                except Exception as e:
                    n_fail += 1
                    print(f"{cid} n{j} {v}: FAILED {e}")
                    continue
                smear = depth_smear_metrics(res["roi_sirt_hu"], res["roi_vg"], nod.center_xyz, res["baseline_hu"])
                if j in fig_vols:
                    fig_vols[j][label(v)] = res["roi_sirt_hu"]
                if (cid, j, v) in done:
                    continue
                row = {"case_id": cid, "nodule_index": j, "diameter_mm": nod.diameter_mm, "variant": v,
                       "n_views": n, "sweep_deg": deg, "i0_per_view": i0_v, "reference": v == ref,
                       "sim_s": t_sim, "coarse_sirt_s": t_sirt, "gt": gt["meas"], "gt_smear": gt_smear,
                       "roi_sirt_smear": smear, "methods": {}}
                for m, e in res["methods"].items():
                    row["methods"][m] = {k: e[k] for k in ("volume_mm3", "mass_mg", "n_found", "n_members", "dice")}
                append_jsonl(row, res_path)
                done.add((cid, j, v))
                g = gt["meas"]
                line = "  ".join(f"{m}: V{e['volume_mm3']:.0f}/M{e['mass_mg']:.0f}" for m, e in row["methods"].items())
                print(f"{cid} n{j} d={nod.diameter_mm:.1f}mm {v}: contrast {smear['contrast_hu']:.0f} HU, "
                      f"clutter {smear['clutter_hu']:.0f} HU, depth/in-plane FWHM "
                      f"{smear['depth_fwhm_mm']:.1f}/{smear['inplane_fwhm_mm']:.1f} mm | GT V{g['volume_mm3']:.0f}/"
                      f"M{g['mass_mg']:.0f} | {line}", flush=True)
            del ctx, noisy, cmu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for j, vols in fig_vols.items():
            if len(vols) < 2 or j not in gt_cache:
                continue
            gt = gt_cache[j][0]
            panel = {"CT (truth)": gt["roi_hu"], **vols}
            viz.save_roi_panel(os.path.join(out, f"{cid}_n{j}_roi_sirt.png"), panel, gt["roi_vg"],
                               case.nodules[j].center_xyz, {"CT (truth)": gt["roi_mask"]}, None,
                               f"{cid} nodule {j} (d={case.nodules[j].diameter_mm:.1f} mm): physics-only ROI-SIRT "
                               "per DTS geometry")
            figs += 1

    rows = read_jsonl(res_path)
    have = {v: [r for r in rows if r["variant"] == v] for v in variants}
    ck.check("nodules reconstructed without errors", n_fail == 0, f"{n_fail} failed")
    ck.check("every geometry has results", all(have[v] for v in variants),
             ", ".join(f"{v}: {len(have[v])}" for v in variants))
    ck.check("nodule centre on the detector in every view of every geometry", not unseen,
             f"{len(unseen)} misses" + (f" e.g. {unseen[:3]}" if unseen else ""))
    if not rows:
        ck.finish(out)
        return

    def err(r, m, qty):
        e = r["methods"].get(m)
        if e is None or not r["gt"][qty] > 0:
            return float("nan")
        return abs(max(e[qty], 0.0) / r["gt"][qty] - 1.0)

    keys = {(r["case_id"], r["nodule_index"]) for r in rows}
    common = [k for k in keys if all(any((r["case_id"], r["nodule_index"]) == k for r in have[v]) for v in variants)]
    ck.info("nodules with results for every geometry (the tables use only these)", len(common))
    cs = set(common)
    sel = {v: [r for r in have[v] if (r["case_id"], r["nodule_index"]) in cs] for v in variants}
    t1 = ["| geometry | nodules | contrast (HU above lung) | clutter (HU) | contrast / clutter "
          "| depth smear (depth FWHM / in-plane FWHM) | simulation + coarse SIRT per case |",
          "|---|---|---|---|---|---|---|"]
    ct = [r["gt_smear"] for r in sel[ref]]
    t1.append(f"| CT (truth) | {len(ct)} | {median([s['contrast_hu'] for s in ct]):.0f} | "
              f"{median([s['clutter_hu'] for s in ct]):.0f} | {median([s['cnr'] for s in ct]):.1f} | "
              f"{median([s['depth_fwhm_mm'] / s['inplane_fwhm_mm'] for s in ct]):.2f} | - |")
    summary = {}
    for v in variants:
        S = [r["roi_sirt_smear"] for r in sel[v]]
        ratio = [s["depth_fwhm_mm"] / s["inplane_fwhm_mm"] for s in S]
        per_case = {r["case_id"]: r["sim_s"] + r["coarse_sirt_s"] for r in sel[v]}
        summary[v] = {"contrast": median([s["contrast_hu"] for s in S]), "cnr": median([s["cnr"] for s in S]),
                      "smear": median(ratio)}
        t1.append(f"| {label(v)}{' (ours)' if v == ref else ''} | {len(S)} | {summary[v]['contrast']:.0f} | "
                  f"{median([s['clutter_hu'] for s in S]):.0f} | {summary[v]['cnr']:.1f} | {summary[v]['smear']:.2f} "
                  f"(n={int(np.isfinite(ratio).sum())}) | {median(list(per_case.values())):.0f} s |")
    t2 = ["| geometry | method | volume MdAPE | mass MdAPE | nodule found |", "|---|---|---|---|---|"]
    for v in variants:
        for m in args.methods:
            R = [r for r in sel[v] if m in r["methods"]]
            if not R:
                continue
            ev, em = median([err(r, m, "volume_mm3") for r in R]), median([err(r, m, "mass_mg") for r in R])
            found = np.mean([r["methods"][m]["volume_mm3"] > MIN_MEASURABLE for r in R])
            summary[v][m] = (ev, em)
            t2.append(f"| {label(v)} | {m} | {100 * ev:.0f}% | {100 * em:.0f}% | {100 * found:.0f}% |")
    note = ("Physics-only methods, same nodules in every row. MdAPE = median |estimate / truth - 1| "
            "(a nodule not found counts as 100%). Depth smear has no segmentation: widths of the nodule's "
            "peak in the ROI-SIRT. Dose: " + ("equal total dose per scan." if args.dose == "matched"
                                               else "equal dose per view."))
    text = "\n".join(t1) + "\n\n" + "\n".join(t2) + "\n\n" + note + "\n"
    print("\n" + text)
    with open(os.path.join(out, "table_geometry.md"), "w") as f:
        f.write(text)
    for v in variants:
        if v == ref:
            continue
        a, b = summary[ref], summary[v]
        parts = [f"depth smear {a['smear']:.2f} -> {b['smear']:.2f}", f"contrast {a['contrast']:.0f} -> {b['contrast']:.0f} HU",
                 f"contrast/clutter {a['cnr']:.1f} -> {b['cnr']:.1f}"]
        parts += [f"{m} volume/mass MdAPE {100 * a[m][0]:.0f}/{100 * a[m][1]:.0f}% -> {100 * b[m][0]:.0f}/{100 * b[m][1]:.0f}%"
                  for m in args.methods if m in a and m in b]
        ck.info(f"{label(v)} vs {label(ref)}", "; ".join(parts))
    ck.info("table", os.path.join(out, "table_geometry.md"))
    ck.finish(out)


if __name__ == "__main__":
    main()
