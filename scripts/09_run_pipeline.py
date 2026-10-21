#!/usr/bin/env python
"""Step 9 - run every method on the held-out nodules (cal + test patients).

For each nodule: ROI around the radiologists' nodule centre (= the doctor's
click), all methods, measurements, ground truth. Results are appended to
work/results/pipeline.jsonl (safe to interrupt and resume).

    python scripts/09_run_pipeline.py --members 8
    python scripts/09_run_pipeline.py --methods sirt_coarse roi_sirt roi_gs   # no diffusion model yet
"""
import os

from _common import base_parser, setup

import numpy as np
import torch

from medlift3d.data.cases import load_case
from medlift3d.io import Paths, append_jsonl, load_dts, load_json, load_sirt, read_jsonl, stable_seed
from medlift3d.pipeline import CaseContext, ground_truth_for_roi, process_nodule
from medlift3d.roi import ROI
from medlift3d.simulate import add_noise
from medlift3d.utils import Checker
from medlift3d import viz


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--splits", nargs="+", default=["cal", "test"])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--ckpt", default=None, help="diffusion checkpoint (default work/models/diffusion/best.pt)")
    ap.add_argument("--results", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max nodules")
    ap.add_argument("--figures", type=int, default=8)
    args = ap.parse_args()
    cfg, dev = setup(args)
    paths = Paths(cfg)
    out = paths.out("09_pipeline")
    ck = Checker("Step 9: per-nodule reconstruction + measurement")
    methods = args.methods or list(cfg["pipeline"]["methods"])
    model = diffusion = None
    ckpt = args.ckpt or os.path.join(cfg["paths"]["work"], "models", "diffusion", "best.pt")
    if any(m.startswith("diffusion") for m in methods):
        if os.path.exists(ckpt):
            from medlift3d.diffusion.checkpoint import load_for_sampling

            model, diffusion, dcfg_ck = load_for_sampling(ckpt, device=dev)
            ck.info("diffusion checkpoint", ckpt)
        else:
            methods = [m for m in methods if not m.startswith("diffusion")]
            ck.info("diffusion checkpoint not found -> running without diffusion methods", ckpt)
    res_path = args.results or paths.p("results", "pipeline.jsonl")
    done = {(r["case_id"], r["nodule_index"]) for r in read_jsonl(res_path)}
    splits = load_json(paths.p("splits.json"))
    ids = sorted(c for c, s in splits["cases"].items() if s in args.splits)
    n_done, n_fail, times, figs = 0, 0, [], 0
    for cid in ids:
        if not (os.path.exists(paths.dts(cid)) and os.path.exists(paths.sirt(cid))):
            continue
        case = load_case(paths.case(cid))
        todo = [j for j in range(len(case.nodules)) if (cid, j) not in done]
        if not todo:
            continue
        clean, geom, _ = load_dts(paths.dts(cid), device=dev)
        cmu, cvg, meta = load_sirt(paths.sirt(cid), device=dev)
        noisy = add_noise(clean, cfg["simulation"]["i0"], seed=int(meta.get("noise_seed", stable_seed(cid))),
                          electronic_sigma=cfg["simulation"]["electronic_sigma"])
        ctx = CaseContext(cid, geom, noisy, cmu, cvg, cfg["physics"]["mu_water"], str(dev))
        for j in todo:
            if args.limit and n_done >= args.limit:
                break
            nod = case.nodules[j]
            roi = ROI.from_config(nod.center_xyz, cfg["roi"], geom)
            try:
                gt = ground_truth_for_roi(case, j, roi, cfg["measure"])
                keep = figs < args.figures
                res = process_nodule(ctx, nod.center_xyz, cfg, model, diffusion, methods, args.members,
                                     seed=stable_seed(cid, j), gt=gt, keep_volumes=keep)
            except Exception as e:
                n_fail += 1
                print(f"{cid} nodule {j}: FAILED {e}")
                continue
            if keep:
                vols = {"CT (truth)": gt["roi_hu"], "ROI-SIRT input": res["roi_sirt_hu"]}
                masks = {"CT (truth)": gt["roi_mask"]}
                stds = {}
                for name, e in res["methods"].items():
                    vols[name] = e.pop("mean_volume")
                    masks[name] = e.pop("vote_mask")
                    s = e.pop("std_volume")
                    if s is not None:
                        stds[name] = s
                viz.save_roi_panel(os.path.join(out, f"{cid}_n{j}.png"), vols, res["roi_vg"], nod.center_xyz, masks,
                                   stds or None, f"{cid} nodule {j}: d={nod.diameter_mm:.1f} mm")
                figs += 1
                for k in ("roi_sirt_hu", "roi_vg", "fields"):
                    res.pop(k, None)
            res.update(case_id=cid, patient_id=case.patient_id, split=splits["cases"][cid], nodule_index=j,
                       nid=nod.nid, diameter_mm=nod.diameter_mm, malignancy=nod.malignancy, texture=nod.texture,
                       n_readers=nod.n_readers, methods_run=list(res["methods"].keys()))
            append_jsonl(res, res_path)
            n_done += 1
            times.append(res["timing_s"]["total"])
            g = res["gt"]
            line = "  ".join(f"{m}: V{e['volume_mm3']:.0f}/M{e['mass_mg']:.0f}" for m, e in res["methods"].items())
            print(f"{cid} n{j} d={nod.diameter_mm:.1f}mm  GT V{g['volume_mm3']:.0f} mm3 / M{g['mass_mg']:.0f} mg | {line}"
                  f"  ({times[-1]:.1f}s)")
        if args.limit and n_done >= args.limit:
            break
    ck.check("nodules processed without errors", n_fail == 0, f"{n_done} ok, {n_fail} failed")
    if times:
        ck.info("time per nodule (median, all methods)", f"{np.median(times):.1f} s")
    rows = read_jsonl(res_path)
    for m in methods:
        e_v = [abs(r["methods"][m]["volume_mm3"] / r["gt"]["volume_mm3"] - 1) for r in rows if m in r["methods"]]
        e_m = [abs(r["methods"][m]["mass_mg"] / r["gt"]["mass_mg"] - 1) for r in rows if m in r["methods"] and r["gt"]["mass_mg"] > 0]
        if e_v:
            ck.info(f"{m}: median |volume error| / |mass error|", f"{np.median(e_v) * 100:.0f}% / {np.median(e_m) * 100:.0f}%  (n={len(e_v)})")
    ck.info("results file", res_path)
    ck.finish(out)


if __name__ == "__main__":
    main()
