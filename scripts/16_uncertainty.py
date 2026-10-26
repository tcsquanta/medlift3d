#!/usr/bin/env python
"""Step 16 - uncertainty quantification with conformal predictive systems (crepes).

Reads work/results/pipeline.jsonl (step 9); calibrates on the *cal* patients, evaluates on the *test*
patients. CPU only, seconds: runs on Kaggle from the bundle.

    pip install crepes scikit-learn
    python scripts/16_uncertainty.py                  # our method (diffusion + Gaussians)
    python scripts/16_uncertainty.py --method roi_gs  # any other method

Questions it answers (outputs/16_uncertainty/):
  table_uq_intervals.md  90% intervals: standard vs normalised by the diffusion-ensemble spread vs
                         Mondrian (separate calibration per estimated-size group). Is the ensemble
                         spread informative (narrower intervals, same coverage)?
  table_uq_groups.md     coverage inside each size group: plain conformal is only right *on average*;
                         small nodules can be under-covered. Mondrian fixes that.
  table_uq_risk.md       calibrated probabilities of the EUPS volume categories (< 100 / 100-300 / >= 300
                         mm3, the categories the demo shows) vs the uncalibrated ensemble vote: Brier score, calibration error,
                         category sets that contain the true category 90% of the time.
  table_uq_methods.md    every method: interval coverage / width, CRPS, Brier scores.
  uq_test_nodules.csv    per test nodule: estimate, interval, P(>= 100), P(>= 300), category set, truth.
  reliability.png, cpd_examples.png, group_coverage.png
"""
import csv
import math
import os

from _common import base_parser, setup

import numpy as np

from medlift3d.io import Paths, read_jsonl, save_json
from medlift3d.uncertainty import ConformalCalibrator, coverage, relative_width, spearman
from medlift3d.uq import (CATEGORIES, VOLUME_THRESHOLDS_MM3, LogCPS, brier, category_of, category_probs, category_sets,
                          coverage_by_group, ece, eq_diameter_mm, lognormal_probability, measurements,
                          reliability_curve, size_groups, vote_probability)
from medlift3d.utils import Checker

ORDER = ["sirt_coarse", "roi_sirt", "roi_gs", "diffusion", "diffusion_gs"]
LABEL = {"sirt_coarse": "DTS SIRT (standard)", "roi_sirt": "ROI-SIRT (physics)", "roi_gs": "ROI-Gaussians (physics)",
         "diffusion": "Diffusion only", "diffusion_gs": "Diffusion+Gaussians (ours)"}


def split_rows(rows, ck):
    cal = [r for r in rows if r["split"] == "cal"]
    test = [r for r in rows if r["split"] == "test"]
    if not cal or not test:                                   # tiny runs: split by patient halves (as step 10)
        pats = sorted({r["patient_id"] for r in rows})
        half = set(pats[: max(1, len(pats) // 2)])
        cal = [r for r in rows if r["patient_id"] in half]
        test = [r for r in rows if r["patient_id"] not in half] or cal
        ck.info("cal/test split", "not enough patients in each split -> split results by patient halves")
    return cal, test


def pct(x):
    return f"{100 * x:.0f}%" if np.isfinite(x) else "-"


def width_txt(w):
    return f"{100 * w:.0f}%" if np.isfinite(w) else "inf (too few cal.)"


def evaluate_variants(C, T, alpha, floor, n_groups):
    """All interval variants for one method / quantity. Returns rows, the fitted models and groups."""
    gc = gt = None
    edges = []
    if n_groups >= 2:
        gc, gt, edges = size_groups(np.log(C["pred"]), np.log(T["pred"]), n_groups)
    eval_groups = gt if gt is not None else np.zeros(len(T["pred"]), int)
    variants = [("standard", False, False), ("normalised by ensemble spread", True, False)]
    if gc is not None:
        variants += [("Mondrian size groups", False, True), ("Mondrian + normalised", True, True)]
    out, models = [], {}
    for name, norm, mond in variants:
        m = LogCPS(norm, floor).fit(C["pred"], C["sigma"], C["truth"], bins=gc if mond else None)
        lo, hi = m.interval(T["pred"], T["sigma"], alpha, bins=gt if mond else None)
        byg = coverage_by_group(lo, hi, T["truth"], eval_groups)
        out.append({"variant": name, "coverage": coverage(lo, hi, T["truth"]), "width": relative_width(lo, hi, T["pred"]),
                    "worst_group": min(byg.values()), "by_group": byg,
                    "crps": m.crps(T["pred"], T["sigma"], T["truth"], bins=gt if mond else None), "lo": lo, "hi": hi})
        models[name] = (m, gt if mond else None)
    return out, models, eval_groups, edges


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--results", default=None)
    ap.add_argument("--method", default=None, help="default: diffusion_gs, else the best method present")
    ap.add_argument("--groups", type=int, default=3, help="Mondrian size groups (fewer if cal is small)")
    args = ap.parse_args()
    cfg, _ = setup(args)
    paths = Paths(cfg)
    out = paths.out("16_uncertainty")
    ck = Checker("Step 16: uncertainty quantification (conformal predictive systems, crepes)")
    try:
        import crepes  # noqa: F401
    except ImportError:
        ck.check("crepes installed", False, "pip install crepes scikit-learn")
        ck.finish(out)
        return
    rows = read_jsonl(args.results or paths.p("results", "pipeline.jsonl"))
    ck.check("results available", len(rows) > 0, f"{len(rows)} nodules")
    if not rows:
        ck.finish(out)
        return
    cal, test = split_rows(rows, ck)
    present = [m for m in ORDER if any(m in r["methods"] for r in rows)]
    method = args.method or next((m for m in ("diffusion_gs", "roi_gs", "roi_sirt") if m in present), present[-1])
    alpha = float(cfg["uncertainty"]["alpha"])
    floor = float(cfg["uncertainty"]["sigma_floor"])
    lev = int(round(100 * (1 - alpha)))
    n_min = int(math.ceil(2.0 / alpha - 1.0))              # finite asymmetric interval needs >= this many per group
    ck.info("method / nodules cal / test", f"{LABEL.get(method, method)} / {len(cal)} / {len(test)}")

    # ---- intervals and groups ------------------------------------------------------------------------
    int_lines = [f"| quantity | variant | coverage of {lev}% interval (test) | median width (% of estimate) | "
                 f"worst size group coverage | CRPS (log) |", "|---|---|---|---|---|---|"]
    grp_lines = []
    keep = {}
    for qty in ("volume", "mass"):
        C, T = measurements(cal, method, qty), measurements(test, method, qty)
        if len(C["pred"]) < 3 or len(T["pred"]) == 0:
            continue
        n_groups = min(args.groups, len(C["pred"]) // n_min)
        res, models, groups, edges = evaluate_variants(C, T, alpha, floor, n_groups)
        keep[qty] = (C, T, res, models, groups, edges)
        for v in res:
            int_lines.append(f"| {qty} | {v['variant']} | {pct(v['coverage'])} | {width_txt(v['width'])} | "
                             f"{pct(v['worst_group'])} | {v['crps']:.3f} |")
        calib = ConformalCalibrator(alpha=alpha, mode=cfg["uncertainty"]["mode"], sigma_floor=floor).fit(
            C["pred"], C["sigma"], C["truth"])
        lo, hi = calib.interval(T["pred"], T["sigma"])
        int_lines.append(f"| {qty} | (cross-check: step 10's own implementation) | {pct(coverage(lo, hi, T['truth']))} | "
                         f"{width_txt(relative_width(lo, hi, T['pred']))} | - | - |")
        rho = spearman(T["sigma"], np.abs(np.log(T["pred"] / T["truth"])))
        ck.info(f"{qty}: ensemble spread vs actual error, Spearman (test)", f"{rho:.2f}")
        std, nrm = res[0], res[1]
        ck.check(f"{qty}: CPS {lev}% intervals cover about {lev}% of test nodules (normalised)",
                 abs(nrm["coverage"] - (1 - alpha)) <= 0.1 or len(T["pred"]) < 20,
                 f"{pct(nrm['coverage'])} (n={len(T['pred'])})")
        ck.info(f"{qty}: is the ensemble spread informative? width standard -> normalised",
                f"{width_txt(std['width'])} -> {width_txt(nrm['width'])} (coverage {pct(std['coverage'])} / "
                f"{pct(nrm['coverage'])})")
        if len(edges):
            unit = "mm (equivalent diameter)" if qty == "volume" else "mg"
            shown = [eq_diameter_mm(e) if qty == "volume" else e for e in edges]
            names = [f"< {shown[0]:.1f}"] + [f"{a:.1f}-{b:.1f}" for a, b in zip(shown[:-1], shown[1:])] + [f">= {shown[-1]:.1f}"]
            grp_lines += [f"\n**{qty}**: groups by the *estimated* {qty} ({unit}), equal-count on the cal nodules\n",
                          "| variant | " + " | ".join(f"group {n} (n={int(np.sum(groups == g))})"
                                                       for g, n in enumerate(names)) + " |",
                          "|---|" + "---|" * len(names)]
            for v in res:
                grp_lines.append(f"| {v['variant']} | " + " | ".join(pct(v["by_group"].get(g, np.nan))
                                                                   for g in range(len(names))) + " |")
            mond = [v for v in res if v["variant"] == "Mondrian + normalised"][0]
            ck.info(f"{qty}: worst size-group coverage, normalised -> Mondrian + normalised",
                    f"{pct(nrm['worst_group'])} -> {pct(mond['worst_group'])}")
    with open(os.path.join(out, "table_uq_intervals.md"), "w") as f:
        f.write("\n".join(int_lines) + f"\n\nMethod: {LABEL.get(method, method)}. Calibrated on {len(cal)} cal nodules, "
                f"tested on {len(test)} test nodules. Width = (upper - lower) / estimate. CRPS = how close the whole "
                "calibrated distribution is to the truth (log units, lower = better).\n")
    with open(os.path.join(out, "table_uq_groups.md"), "w") as f:
        f.write((f"Coverage of the {lev}% intervals inside each size group (test nodules); target {lev}%."
                 + "\n".join(grp_lines) if grp_lines else
                 f"Too few calibration nodules for size groups (need >= {2 * n_min}).") + "\n")

    # ---- decision-level uncertainty: EUPS volume categories ------------------------------------
    risk_lines = []
    if "volume" in keep:
        C, T, res, models, groups, edges = keep["volume"]
        name = "normalised by ensemble spread"          # fixed in advance: all cal nodules, finest probabilities
        m, gtest = models[name]
        truth_cat = category_of(T["truth"])
        probs = {f"conformal ({name})": category_probs([m.cdf(T["pred"], T["sigma"], t) for t in VOLUME_THRESHOLDS_MM3])}
        if "Mondrian + normalised" in models:
            mm, gm = models["Mondrian + normalised"]
            probs["conformal (Mondrian + normalised)"] = category_probs(
                [mm.cdf(T["pred"], T["sigma"], t, bins=gm) for t in VOLUME_THRESHOLDS_MM3])
        vote = [vote_probability(T["members"], t) for t in VOLUME_THRESHOLDS_MM3]
        probs["ensemble vote (uncalibrated)"] = category_probs([1 - v for v in vote])
        logn = [lognormal_probability(T["pred"], T["sigma"], t, floor) for t in VOLUME_THRESHOLDS_MM3]
        probs["ensemble spread as log-normal (uncalibrated)"] = category_probs([1 - p for p in logn])
        point = [(T["pred"] > t).astype(float) for t in VOLUME_THRESHOLDS_MM3]
        probs["point estimate only (0 / 1)"] = category_probs([1 - p for p in point])
        risk_lines = ["| probabilities from | Brier P(V >= 100) | ECE P(V >= 100) | Brier P(V >= 300) | ECE P(V >= 300) | "
                      "most likely category correct |", "|---|---|---|---|---|---|"]
        rel = {}
        for k, pr in probs.items():
            p_lo, p_hi = pr[:, 1] + pr[:, 2], pr[:, 2]
            e_lo, e_hi = T["truth"] >= VOLUME_THRESHOLDS_MM3[0], T["truth"] >= VOLUME_THRESHOLDS_MM3[1]
            risk_lines.append(f"| {k} | {brier(p_lo, e_lo):.3f} | {ece(p_lo, e_lo):.3f} | {brier(p_hi, e_hi):.3f} | "
                              f"{ece(p_hi, e_hi):.3f} | {pct(np.mean(np.argmax(pr, 1) == truth_cat))} |")
            rel[k] = (reliability_curve(p_lo, e_lo), reliability_curve(p_hi, e_hi))
        lo, hi = m.interval(T["pred"], T["sigma"], alpha, bins=gtest)
        sets = category_sets(lo, hi)
        set_cov = float(np.mean([c in s for c, s in zip(truth_cat, sets)]))
        set_size = float(np.mean([len(s) for s in sets]))
        risk_lines += ["", f"Category sets from the {lev}% conformal interval: contain the true category for "
                       f"{pct(set_cov)} of test nodules, average size {set_size:.2f} of 3 categories "
                       f"(1 = a definite category; {pct(np.mean([len(s) == 1 for s in sets]))} of nodules).",
                       "", "Brier score: mean squared difference between the probability and what happened (lower = "
                       "better). ECE: average gap between predicted probability and observed frequency (0 = perfectly "
                       "calibrated). Truth: radiologist consensus volume. Thresholds: European Position Statement "
                       "(EUPS, Oudkerk et al. 2017) volume categories for solid nodules."]
        ck.info("EUPS category set: contains the true category / average size",
                f"{pct(set_cov)} / {set_size:.2f}")
        b_conf = brier(probs[f"conformal ({name})"][:, 2], T["truth"] >= 300)
        b_vote = brier(probs["ensemble vote (uncalibrated)"][:, 2], T["truth"] >= 300)
        ck.info("Brier P(V >= 300 mm3): conformal vs ensemble vote (lower = better)", f"{b_conf:.3f} vs {b_vote:.3f}")
        save_reliability(os.path.join(out, "reliability.png"), rel)
        save_cpd_examples(os.path.join(out, "cpd_examples.png"), m, T, gtest)
        save_group_coverage(os.path.join(out, "group_coverage.png"), res, 1 - alpha)
        pr = probs[f"conformal ({name})"]
        with open(os.path.join(out, "uq_test_nodules.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["nid", "case_id", "estimate_mm3", f"lo{lev}_mm3", f"hi{lev}_mm3", "P_ge_100", "P_ge_300",
                        "category_set", "truth_mm3", "true_category"])
            for i, r in enumerate(T["rows"]):
                w.writerow([r.get("nid"), r["case_id"], f"{T['pred'][i]:.1f}", f"{lo[i]:.1f}", f"{hi[i]:.1f}",
                            f"{pr[i, 1] + pr[i, 2]:.3f}", f"{pr[i, 2]:.3f}", "|".join(CATEGORIES[c] for c in sets[i]),
                            f"{T['truth'][i]:.1f}", CATEGORIES[truth_cat[i]]])
    with open(os.path.join(out, "table_uq_risk.md"), "w") as f:
        f.write("\n".join(risk_lines or ["No volume results."]) + "\n")

    # ---- all methods -----------------------------------------------------------------------------------
    demo_cal = {"main_method": method, "variant": "normalised by ensemble spread (pooled)", "alpha": alpha,
                "thresholds_mm3": list(VOLUME_THRESHOLDS_MM3), "methods": {}}      # the demo's category probabilities
    meth_lines = [f"| method | volume: coverage / width | mass: coverage / width | CRPS volume / mass (log) | "
                  f"Brier P(V >= 100) / P(V >= 300) |", "|---|---|---|---|---|"]
    for mth in present:
        cells, crps = [], []
        for qty in ("volume", "mass"):
            C, T = measurements(cal, mth, qty), measurements(test, mth, qty)
            if len(C["pred"]) < 3 or len(T["pred"]) == 0:
                cells.append("-")
                crps.append(float("nan"))
                continue
            m = LogCPS(True, floor).fit(C["pred"], C["sigma"], C["truth"])
            if qty == "volume":
                demo_cal["methods"][mth] = {"sigma_floor": floor, "residuals": (np.log(C["truth"]) - np.log(C["pred"])).tolist(),
                                            "sigma": C["sigma"].tolist()}
            lo, hi = m.interval(T["pred"], T["sigma"], alpha)
            cells.append(f"{pct(coverage(lo, hi, T['truth']))} / {width_txt(relative_width(lo, hi, T['pred']))}")
            crps.append(m.crps(T["pred"], T["sigma"], T["truth"]))
            if qty == "volume":
                bs = [brier(1 - m.cdf(T["pred"], T["sigma"], t), T["truth"] >= t) for t in VOLUME_THRESHOLDS_MM3]
        meth_lines.append(f"| {LABEL.get(mth, mth)} | {cells[0]} | {cells[1]} | {crps[0]:.3f} / {crps[1]:.3f} | "
                          + (f"{bs[0]:.3f} / {bs[1]:.3f} |" if cells[0] != "-" else "- |"))
    save_json(demo_cal, os.path.join(out, "cps_calibration.json"))
    with open(os.path.join(out, "table_uq_methods.md"), "w") as f:
        f.write("\n".join(meth_lines) + "\n\nAll methods with the normalised conformal predictive system "
                "(calibrated on cal, evaluated on test).\n")
    for name in ("table_uq_intervals.md", "table_uq_groups.md", "table_uq_risk.md", "table_uq_methods.md"):
        print(f"\n## {name}\n" + open(os.path.join(out, name)).read())
    ck.finish(out)


# ---- figures ---------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_reliability(path, rel):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for i, title in enumerate(("P(volume >= 100 mm3)", "P(volume >= 300 mm3)")):
        a = ax[i]
        a.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", label="perfectly calibrated")
        for k, curves in rel.items():
            x, y, n = curves[i]
            if x:
                a.plot(x, y, "o-", label=k, ms=4)
        a.set_xlim(0, 1), a.set_ylim(-0.02, 1.02)
        a.set_xlabel("predicted probability"), a.set_ylabel("observed frequency (test)")
        a.set_title(title), a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_cpd_examples(path, m, T, groups, n=4):
    plt = _plt()
    order = np.argsort(T["truth"])
    pick = order[np.linspace(0, len(order) - 1, min(n, len(order))).astype(int)]
    fig, ax = plt.subplots(1, len(pick), figsize=(3.6 * len(pick), 3.4), squeeze=False)
    for a, i in zip(ax[0], pick):
        lo, hi = max(T["truth"][i], T["pred"][i]) * 0.1, max(T["truth"][i], T["pred"][i]) * 4
        grid = np.geomspace(max(lo, 1.0), hi, 200)
        cdf = m.cdf_curve(T["pred"][i], T["sigma"][i], grid, None if groups is None else groups[i])
        a.plot(grid, cdf, color="tab:blue", label="calibrated P(V <= v)")
        a.axvline(T["truth"][i], color="tab:green", label="truth")
        a.axvline(T["pred"][i], color="tab:blue", ls=":", label="estimate")
        for t in VOLUME_THRESHOLDS_MM3:
            a.axvline(t, color="tab:red", ls="--", lw=0.8)
        a.set_xscale("log"), a.set_ylim(0, 1), a.set_xlabel("volume (mm3)")
        a.set_title(str(T["rows"][i].get("nid", T["rows"][i]["case_id"]))[:28], fontsize=8)
        a.grid(alpha=0.3)
    ax[0][0].set_ylabel("probability")
    ax[0][0].legend(fontsize=7)
    fig.suptitle("conformal predictive distributions of 4 test nodules (red: EUPS 100 / 300 mm3)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_group_coverage(path, res, target):
    plt = _plt()
    groups = sorted(res[0]["by_group"])
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(groups), 3.6))
    w = 0.8 / len(res)
    for k, v in enumerate(res):
        ax.bar(np.arange(len(groups)) + k * w, [v["by_group"][g] for g in groups], w, label=v["variant"])
    ax.axhline(target, color="black", ls="--", lw=0.8, label=f"target {100 * target:.0f}%")
    ax.set_xticks(np.arange(len(groups)) + w * (len(res) - 1) / 2)
    ax.set_xticklabels([f"size group {g + 1}\n(small -> large)" if len(groups) > 1 else "all" for g in groups], fontsize=8)
    ax.set_ylim(0, 1.05), ax.set_ylabel("coverage (test)"), ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("volume intervals: coverage per estimated-size group", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
