#!/usr/bin/env python
"""Step 17 - explainability: what the X-rays measured, what the diffusion prior filled in, what the
detector looked at, and which nodules are hard.

Runs on Kaggle (GPU T4) from the bundle: models + the bundled test patients + pipeline.jsonl.

    python scripts/17_explain.py                     # all three parts, 12 nodules, 4 diffusion members
    python scripts/17_explain.py --limit 20 --members 8
    python scripts/17_explain.py --skip-maps         # only the fast parts (error drivers, detector)

A. Error drivers (CPU, from pipeline.jsonl): which nodule properties go with large volume / mass
   errors of our method (Spearman with bootstrap 95% CI; size and solidity groups).
B. Seen / unseen split (needs the diffusion model, ~1-3 min per nodule on a T4): every nodule box of
   the true CT, the physics reconstruction and ours is split into the part the 15 views measure and the
   part they cannot see (medlift3d/explain.py). Shows how much of a nodule is invisible to DTS, that
   the mass is (almost) entirely seen (why H1 holds), how much the prior adds and whether what it adds
   matches the truth ("unseen skill"). Figures: depth view of the excess, seen, unseen parts.
C. Detector saliency: input x gradient around each suggestion (true vs false positives): does the
   detector look at the nodule or at its surroundings?

Outputs: outputs/17_explain/{table_error_drivers.md, table_explain.md, table_saliency.md,
explain_nodules.jsonl, <nodule>_seen_unseen.png, saliency_examples.png}.
"""
import os

from _common import base_parser, setup

import numpy as np
import torch

from medlift3d.explain import (BoxOperator, depth_profile, detector_saliency, saliency_share_within, skill,
                               spearman_ci, split_metrics)
from medlift3d.io import Paths, append_jsonl, load_dts, load_json, load_sirt, read_jsonl, stable_seed
from medlift3d.utils import Checker

LABEL = {"truth": "CT truth (what DTS cannot see)", "roi_sirt": "ROI-SIRT (physics only)",
         "diffusion": "Diffusion (ours, mean of members)", "diffusion_gs": "Diffusion+Gaussians (ours)"}


def med_iqr(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "-"
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f"{q2:.2f} [{q1:.2f}, {q3:.2f}]"


# ---------------------------------------------------------------------------------------------- A
def error_drivers(rows, method, out, ck):
    rows = [r for r in rows if method in r["methods"] and r["gt"]["volume_mm3"] > 0 and r["gt"]["mass_mg"] > 0]
    if len(rows) < 5:
        ck.info("A. error drivers", f"too few nodules with {method} ({len(rows)})")
        return
    e = [r["methods"][method] for r in rows]
    ev = np.abs(np.log(np.maximum([x["volume_mm3"] for x in e], 1e-3) / np.array([r["gt"]["volume_mm3"] for r in rows])))
    em = np.abs(np.log(np.maximum([x["mass_mg"] for x in e], 1e-3) / np.array([r["gt"]["mass_mg"] for r in rows])))
    g = lambda k: [r["gt"].get(k) for r in rows]                       # noqa: E731
    feats = [("nodule diameter (radiologists, mm)", [r.get("diameter_mm") for r in rows]),
             ("solidity (LIDC texture, 1 = ground glass ... 5 = solid)", [r.get("texture") for r in rows]),
             ("density (mean HU inside the nodule)", g("mean_hu")),
             ("radiologists' own volume disagreement (CV)", g("reader_volume_cv")),
             ("malignancy rating (1-5)", [r.get("malignancy") for r in rows]),
             ("ensemble spread (volume log-std)", [x.get("volume_log_std") for x in e]),
             ("physics data misfit (ROI-SIRT residual)", [r.get("roi_sirt_residual") for r in rows])]
    lines = ["| nodule property | Spearman with abs(volume error) [95% CI] | Spearman with abs(mass error) [95% CI] | n |",
             "|---|---|---|---|"]
    for name, x in feats:
        x = np.array([np.nan if v is None else float(v) for v in x])
        rv, lv, hv, n = spearman_ci(x, ev)
        rm, lm, hm, _ = spearman_ci(x, em)
        f = lambda r, a, b: "-" if not np.isfinite(r) else f"{r:+.2f} [{a:+.2f}, {b:+.2f}]"   # noqa: E731
        lines.append(f"| {name} | {f(rv, lv, hv)} | {f(rm, lm, hm)} | {n} |")
    d = np.array([r["gt"].get("eq_diameter_mm", np.nan) for r in rows], float)
    edges = np.nanpercentile(d, [33.3, 66.7])
    grp = np.digitize(d, edges)
    tex = np.array([np.nan if r.get("texture") is None else float(r["texture"]) for r in rows])
    lines += ["", "| group | n | volume MdAPE | mass MdAPE |", "|---|---|---|---|"]
    rv = np.maximum([x["volume_mm3"] for x in e], 0) / np.array([r["gt"]["volume_mm3"] for r in rows]) - 1
    rm = np.maximum([x["mass_mg"] for x in e], 0) / np.array([r["gt"]["mass_mg"] for r in rows]) - 1
    pct = lambda v: f"{100 * np.median(np.abs(v)):.0f}%"            # noqa: E731
    for k, name in enumerate([f"small (< {edges[0]:.1f} mm)", f"medium ({edges[0]:.1f}-{edges[1]:.1f} mm)",
                              f"large (>= {edges[1]:.1f} mm)"]):
        m = grp == k
        if m.any():
            lines.append(f"| {name} | {int(m.sum())} | {pct(rv[m])} | {pct(rm[m])} |")
    for name, m in (("solid (texture >= 4)", tex >= 4), ("part-solid / ground glass (texture < 4)", tex < 4)):
        if m.any():
            lines.append(f"| {name} | {int(m.sum())} | {pct(rv[m])} | {pct(rm[m])} |")
    note = ("\nSpearman > 0: larger values of the property go with larger errors. A CI that contains 0 means no "
            "clear relation at this sample size. Correlations use abs(log(estimate / truth)); the groups show the "
            "median absolute % error (MdAPE, as in step 10; a failed measurement counts as 100%). Sizes: equivalent "
            "diameter of the radiologists' consensus. All measured nodules (cal + test).")
    with open(os.path.join(out, "table_error_drivers.md"), "w") as f:
        f.write(f"Method: {method}\n\n" + "\n".join(lines) + "\n" + note + "\n")
    ck.info("A. error drivers", f"{len(rows)} nodules -> table_error_drivers.md")


# ---------------------------------------------------------------------------------------------- B
def seen_unseen(cfg, dev, paths, args, out, ck):
    from medlift3d.data.cases import load_case
    from medlift3d.measure import measure_reconstruction
    from medlift3d.pipeline import CaseContext, build_problem, ground_truth_for_roi, process_nodule
    from medlift3d.roi import ROI, estimate_baseline_hu
    from medlift3d.simulate import add_noise

    ckpt = args.ckpt or os.path.join(cfg["paths"]["work"], "models", "diffusion", "best.pt")
    model = diffusion = None
    methods = ["roi_sirt"]
    if os.path.exists(ckpt):
        from medlift3d.diffusion.checkpoint import load_for_sampling
        model, diffusion, _ = load_for_sampling(ckpt, device=dev)
        methods += ["diffusion", "diffusion_gs"]
    else:
        ck.info("B. no diffusion checkpoint -> truth and physics only", ckpt)
    splits = load_json(paths.p("splits.json"))
    ids = sorted(c for c, s in splits["cases"].items() if s in args.splits
                 and all(os.path.exists(p) for p in (paths.case(c), paths.dts(c), paths.sirt(c))))
    res_path = os.path.join(out, "explain_nodules.jsonl")
    if os.path.exists(res_path):
        os.remove(res_path)
    mu_w = float(cfg["physics"]["mu_water"])
    recs, n_fig = [], 0
    for cid in ids:
        if len(recs) >= args.limit:
            break
        case = load_case(paths.case(cid))
        clean, geom, _ = load_dts(paths.dts(cid), device=dev)
        cmu, cvg, meta = load_sirt(paths.sirt(cid), device=dev)
        noisy = add_noise(clean, cfg["simulation"]["i0"], seed=int(meta.get("noise_seed", stable_seed(cid))),
                          electronic_sigma=cfg["simulation"]["electronic_sigma"])
        ctx = CaseContext(cid, geom, noisy, cmu, cvg, mu_w, str(dev))
        for j, nod in enumerate(case.nodules):
            if len(recs) >= args.limit:
                break
            click = nod.center_xyz
            roi = ROI.from_config(click, cfg["roi"], geom)
            try:
                gt = ground_truth_for_roi(case, j, roi, cfg["measure"])
                res = process_nodule(ctx, click, cfg, model, diffusion, methods, args.members,
                                     seed=stable_seed(cid, j), gt=gt, keep_volumes=True)
                prob = build_problem(ctx, click, cfg)
            except Exception as e:                                      # one bad nodule must not stop the rest
                print(f"{cid} n{j}: FAILED {e}")
                continue
            op = BoxOperator(prob)
            sl = roi.box_slices()
            vg = roi.vgeom
            vols = {"truth": gt["roi_hu"], "roi_sirt": res["roi_sirt_hu"]}
            for m in ("diffusion", "diffusion_gs"):
                if m in res["methods"]:
                    vols[m] = res["methods"][m]["mean_volume"]
            parts, rec = {}, {"case_id": cid, "nodule_index": j, "nid": nod.nid, "diameter_mm": nod.diameter_mm,
                              "gt_volume_mm3": res["gt"]["volume_mm3"], "gt_mass_mg": res["gt"]["mass_mg"]}
            bvg = roi.box_vgeom
            for name, v in vols.items():
                box = v[sl]                                             # lung level *inside* the box (a Gaussian
                base = float(estimate_baseline_hu(box, bvg, click, inner_radius_mm=10.0,   # box sits on its own)
                                                  default=float(res["baseline_hu"])))
                seen, unseen, info = op.split_hu(box, base, mu_w, n_iter=args.cg_iter)
                parts[name] = (box - base, seen, unseen)
                r = split_metrics(box - base, seen, unseen)
                r.update(baseline_hu=base, rel_residual=info["rel_residual"], iterations=info["iterations"])
                seen_only = np.array(v, copy=True)
                seen_only[sl] = base + seen
                full_m, _ = measure_reconstruction(np.asarray(v, np.float32), vg, click, base, cfg["measure"])
                seen_m, _ = measure_reconstruction(seen_only.astype(np.float32), vg, click, base, cfg["measure"])
                r.update(volume_full_mm3=full_m["volume_mm3"], volume_seen_only_mm3=seen_m["volume_mm3"],
                         mass_full_mg=full_m["mass_mg"], mass_seen_only_mg=seen_m["mass_mg"])
                if name != "truth":
                    r["unseen_skill"] = skill(unseen, parts["truth"][2])
                    r["seen_skill"] = skill(seen, parts["truth"][1])
                    e = res["methods"].get(name) or res["methods"].get("roi_sirt")
                    r["abs_log_volume_error"] = float(abs(np.log(max(e["volume_mm3"], 1e-3) / res["gt"]["volume_mm3"])))
                rec[name] = r
            append_jsonl(rec, res_path)
            recs.append(rec)
            ours = "diffusion_gs" if "diffusion_gs" in rec else "roi_sirt"
            print(f"{cid} n{j} d={nod.diameter_mm:.1f} mm: unseen share truth {rec['truth']['unseen_energy_share']:.2f}, "
                  f"{ours} {rec[ours]['unseen_energy_share']:.2f}, unseen skill {rec[ours].get('unseen_skill', np.nan):+.2f}, "
                  f"split residual {rec['truth']['rel_residual']:.3f}", flush=True)
            if n_fig < args.figures:
                stds = {m: res["methods"][m]["std_volume"][sl] for m in ("diffusion", "diffusion_gs")
                        if m in res["methods"] and res["methods"][m].get("std_volume") is not None}
                save_split_figure(os.path.join(out, f"{cid}_n{j}_seen_unseen.png"), parts, stds, roi.box_vgeom,
                                  click, nod)
                n_fig += 1
    if not recs:
        ck.check("B. seen / unseen split ran on at least one nodule", False, "no test case with case + dts + sirt")
        return
    names = [n for n in ("truth", "roi_sirt", "diffusion", "diffusion_gs") if n in recs[0]]
    lines = ["| volume | unseen share of the box signal | unseen share of the box mass | volume from the seen part "
             "only / full | unseen skill vs truth | seen skill vs truth |", "|---|---|---|---|---|---|"]
    for n in names:
        rr = [r[n] for r in recs]
        ratio = [x["volume_seen_only_mm3"] / x["volume_full_mm3"] if x["volume_full_mm3"] > 0 else np.nan for x in rr]
        lines.append(f"| {LABEL[n]} | {med_iqr([x['unseen_energy_share'] for x in rr])} | "
                     f"{med_iqr([x['unseen_mass_share'] for x in rr])} | {med_iqr(ratio)} | "
                     f"{med_iqr([x.get('unseen_skill') for x in rr]) if n != 'truth' else '-'} | "
                     f"{med_iqr([x.get('seen_skill') for x in rr]) if n != 'truth' else '-'} |")
    res_med = float(np.median([r[n]["rel_residual"] for r in recs for n in names]))
    lines += ["", f"Median [IQR] over {len(recs)} nodules. Split inside the nodule box (40 x 50 x 40 mm), excess over "
              f"the local lung level. Split accuracy: median |A unseen| / |A x| = {res_med:.3f} (0 = exact).",
              "unseen share: how much of the signal the 15 views cannot see. unseen skill: 1 = the invisible part "
              "matches the truth, 0 = no better than leaving it out, < 0 = invented structure. A physics-only "
              "reconstruction adds (almost) no unseen part: its unseen skill is about 0 by construction."]
    if "diffusion_gs" in names:
        rho, lo, hi, n = spearman_ci([r["truth"]["unseen_energy_share"] for r in recs],
                                     [r["diffusion_gs"]["abs_log_volume_error"] for r in recs])
        lines.append(f"\nDo nodules that are less visible to DTS get larger volume errors (ours)? Spearman "
                     f"{rho:+.2f} [{lo:+.2f}, {hi:+.2f}], n = {n} (exploratory).")
    with open(os.path.join(out, "table_explain.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    ck.check("B. seen / unseen split is accurate (median |A unseen| / |A x| < 0.05)", res_med < 0.05, f"{res_med:.3f}")
    tm = float(np.median([abs(r["truth"]["unseen_mass_share"]) for r in recs]))
    ck.check("B. the true nodule's mass is seen by the X-rays (unseen share of the box mass < 10%; explains H1)",
             tm < 0.10, f"median {100 * tm:.1f}%")
    ck.info("B. share of the true nodule-box signal invisible to 15-view DTS (median)",
            f"{100 * np.median([r['truth']['unseen_energy_share'] for r in recs]):.0f}%")
    if "diffusion_gs" in names:
        ck.info("B. unseen skill of ours (median: 1 = recovers the invisible part, < 0 = invents)",
                f"{np.median([r['diffusion_gs']['unseen_skill'] for r in recs]):+.2f}")


def save_split_figure(path, parts, stds, bvg, click, nod):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ix, iy, iz = (int(round(v)) for v in bvg.world_to_index(np.asarray(click)))
    rows = [n for n in ("truth", "roi_sirt", "diffusion", "diffusion_gs") if n in parts]
    sx, sy, sz = bvg.spacing
    nz, ny, nx = parts["truth"][0].shape
    ext = [0, ny * sy, nz * sz, 0]                                         # depth (y) horizontal, z vertical
    vmax = max(float(np.percentile(np.abs(parts["truth"][0]), 99.5)), 50.0)
    fig = plt.figure(figsize=(12, 3.0 * len(rows) + 2.8))
    gsp = fig.add_gridspec(len(rows) + 1, 4, height_ratios=[1] * len(rows) + [0.9])
    titles = ["excess over lung (HU)", "SEEN: measured by the 15 views", "UNSEEN: invisible to the views", ""]
    for r, n in enumerate(rows):
        for c, img in enumerate(parts[n]):
            a = fig.add_subplot(gsp[r, c])
            a.imshow(img[:, :, min(ix, nx - 1)], cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext, aspect="equal")
            a.set_title(titles[c] if r == 0 else "", fontsize=9)
            a.set_xticks([]), a.set_yticks([])
            if c == 0:
                a.set_ylabel(LABEL[n].split(" (")[0], fontsize=9)
        a = fig.add_subplot(gsp[r, 3])
        if n == "truth":
            a.axis("off")
            a.text(0.0, 0.5, f"{nod.nid}\nd = {nod.diameter_mm:.1f} mm\n\nside view: depth ->\n(the direction DTS\n"
                   "resolves worst)", fontsize=9, va="center")
        elif n in stds:
            a.imshow(stds[n][:, :, min(ix, nx - 1)], cmap="magma", extent=ext, aspect="equal")
            a.set_title("disagreement of the members (std, HU)", fontsize=8)
            a.set_xticks([]), a.set_yticks([])
        else:
            err = parts[n][2] - parts["truth"][2]
            a.imshow(err[:, :, min(ix, nx - 1)], cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext, aspect="equal")
            a.set_title("unseen part minus truth's unseen part", fontsize=8)
            a.set_xticks([]), a.set_yticks([])
    a = fig.add_subplot(gsp[len(rows), :])
    y = np.arange(ny) * sy
    cz = (min(iz, nz - 1), 0, min(ix, nx - 1))
    for n, col in zip(rows, ("black", "tab:orange", "tab:green", "tab:blue")):
        ex, se, un = parts[n]
        a.plot(y, depth_profile(ex, cz), color=col, lw=2, label=f"{LABEL[n].split(' (')[0]}: excess")
        a.plot(y, depth_profile(se, cz), color=col, ls="--", label="seen part")
        a.plot(y, depth_profile(un, cz), color=col, ls=":", label="unseen part")
    a.axhline(0, color="gray", lw=0.6)
    a.set_xlabel("depth through the nodule centre (mm)"), a.set_ylabel("HU over lung")
    a.legend(fontsize=7, ncol=3), a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- C
def detector_part(cfg, dev, paths, args, out, ck):
    from medlift3d.data.cases import load_case
    from medlift3d.detector import detect_learned, load_detector, match_candidates
    from medlift3d.physics import mu_to_hu

    path = os.path.join(cfg["paths"]["work"], "models", "detector", "best.pt")
    if not os.path.exists(path):
        ck.info("C. detector saliency", f"no detector at {path}")
        return
    model, extra = load_detector(path, dev)
    thr = extra.get("threshold")
    splits = load_json(paths.p("splits.json"))
    ids = sorted(c for c, s in splits["cases"].items() if s in args.splits
                 and os.path.exists(paths.sirt(c)) and os.path.exists(paths.case(c)))[: args.det_scans]
    mu_w = float(cfg["physics"]["mu_water"])
    items = []
    for cid in ids:
        mu, vg, _ = load_sirt(paths.sirt(cid))
        hu = mu_to_hu(mu.numpy(), mu_w).astype(np.float32)
        case = load_case(paths.case(cid), load_readers=False)
        nods = [{"xyz": [float(v) for v in n.center_xyz], "diameter_mm": float(n.diameter_mm)} for n in case.nodules]
        with torch.no_grad():
            cands = detect_learned(model, hu, vg, dev, top_k=args.det_top)
        if thr is not None:
            cands = [c for c in cands if c["score"] >= thr] or cands[:1]
        seen = set()
        for c, m in zip(cands, match_candidates(cands, nods)):
            if m is not None and m in seen:
                continue                                                 # a second hit on the same nodule
            kind = "true positive" if m is not None else "false positive"
            if m is not None:
                seen.add(m)
            win, sal, cc = detector_saliency(model, hu, c["index_zyx"], patch=64, device=dev)
            sp = np.asarray(vg.spacing, float)[::-1]
            items.append({"case_id": cid, "kind": kind, "score": c["score"], "win": win, "sal": sal, "cc": cc, "sp": sp,
                          "share10": saliency_share_within(sal, cc, 10.0, sp),
                          "share_nod": saliency_share_within(sal, cc, max(6.0, nods[m]["diameter_mm"] / 2 + 3), sp)
                          if m is not None else np.nan})
    if not items:
        ck.info("C. detector saliency", "no detections")
        return
    lines = ["| suggestions | n | median score | saliency within 10 mm of the suggestion [IQR] |", "|---|---|---|---|"]
    for kind in ("true positive", "false positive"):
        it = [i for i in items if i["kind"] == kind]
        if it:
            lines.append(f"| {kind} | {len(it)} | {np.median([i['score'] for i in it]):.2f} | "
                         f"{med_iqr([i['share10'] for i in it])} |")
    tp = [i["share_nod"] for i in items if i["kind"] == "true positive"]
    lines += ["", f"True positives: share of the saliency on the nodule itself (radius + 3 mm): {med_iqr(tp)}.",
              f"Saliency = |input x gradient| of the detector output at the suggestion, over a 64^3 window (160 mm). "
              f"{len(ids)} {'/'.join(args.splits)} scans, suggestions above the operating threshold."]
    with open(os.path.join(out, "table_saliency.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    save_saliency_examples(os.path.join(out, "saliency_examples.png"), items)
    ck.info("C. detector saliency (true / false positives)",
            f"{sum(i['kind'] == 'true positive' for i in items)} / {sum(i['kind'] == 'false positive' for i in items)}"
            " -> table_saliency.md")


def save_saliency_examples(path, items, n_each=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pick = ([i for i in items if i["kind"] == "true positive"][:n_each]
            + [i for i in items if i["kind"] == "false positive"][:n_each])
    fig, ax = plt.subplots(2, len(pick), figsize=(2.9 * len(pick), 6), squeeze=False)
    for k, it in enumerate(pick):
        cz, cy, cx = it["cc"]
        sz, sy, sx = it["sp"]
        for r, (img, sal, ext, lab) in enumerate((
                (it["win"][:, cy, :], it["sal"][:, cy, :], [0, img_w(it, 2) * sx, img_w(it, 0) * sz, 0], "coronal"),
                (it["win"][:, :, cx], it["sal"][:, :, cx], [0, img_w(it, 1) * sy, img_w(it, 0) * sz, 0], "side (depth ->)"))):
            a = ax[r][k]
            a.imshow(img, cmap="gray", vmin=-1000, vmax=200, extent=ext)
            a.imshow(np.ma.masked_less(sal, np.percentile(sal, 97)), cmap="autumn", alpha=0.6, extent=ext)
            a.set_xticks([]), a.set_yticks([])
            a.set_title(f"{it['kind']}\nscore {it['score']:.2f}, {lab}" if r == 0 else lab, fontsize=8)
    fig.suptitle("detector saliency (top 3% of |input x gradient| in colour) around each suggestion", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def img_w(it, axis):
    return it["win"].shape[axis]


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--results", default=None)
    ap.add_argument("--method", default="diffusion_gs", help="method for the error drivers (A)")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--limit", type=int, default=12, help="nodules for the seen / unseen split (B)")
    ap.add_argument("--members", type=int, default=4, help="diffusion members per nodule (B)")
    ap.add_argument("--figures", type=int, default=6)
    ap.add_argument("--cg-iter", type=int, default=60, help="CGLS iterations of the split")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--det-scans", type=int, default=15)
    ap.add_argument("--det-top", type=int, default=5)
    ap.add_argument("--skip-maps", action="store_true")
    ap.add_argument("--skip-detector", action="store_true")
    args = ap.parse_args()
    cfg, dev = setup(args)
    paths = Paths(cfg)
    out = paths.out("17_explain")
    ck = Checker("Step 17: explainability (error drivers, seen / unseen split, detector saliency)")
    rp = args.results or paths.p("results", "pipeline.jsonl")
    rows = read_jsonl(rp) if os.path.exists(rp) else []
    method = args.method if any(args.method in r["methods"] for r in rows) else \
        next((m for m in ("roi_gs", "roi_sirt") if any(m in r["methods"] for r in rows)), args.method)
    error_drivers(rows, method, out, ck)
    if not args.skip_maps:
        seen_unseen(cfg, dev, paths, args, out, ck)
    if not args.skip_detector:
        detector_part(cfg, dev, paths, args, out, ck)
    for name in ("table_error_drivers.md", "table_explain.md", "table_saliency.md"):
        p = os.path.join(out, name)
        if os.path.exists(p):
            print(f"\n## {name}\n" + open(p).read())
    ck.finish(out)


if __name__ == "__main__":
    main()
