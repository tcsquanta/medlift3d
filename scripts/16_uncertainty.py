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
  reliability.png, group_coverage.png
  {volume,mass}_*.png    uncertainty plots of the test nodules (conformal, normalised by the ensemble spread):
                         cpd_examples / cpd_median_nodule (conformal predictive distributions), intervals
                         (prediction intervals), regression (estimate vs truth with intervals), residuals,
                         residual_hist (distribution of residuals), qq (Q-Q of residuals), interval_widths
                         (distribution of interval widths), coverage_levels (coverage at every confidence
                         level); {qty}_uq_overview.png has all eight panels
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
    for qty, (C, T, res, models, _, _) in keep.items():           # the uncertainty plots, one file each
        save_diagnostics(out, qty, C, T, res, models, floor, lev)
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


# ---- diagnostic plots per quantity (main variant: normalised by the ensemble spread) -----------------
UNIT = {"volume": "mm3", "mass": "mg"}
MAIN = "normalised by ensemble spread"


def _finite(lo, hi):
    lo, hi = np.asarray(lo, float).copy(), np.asarray(hi, float).copy()
    bad = ~(np.isfinite(lo) & np.isfinite(hi) & (lo > 0))
    lo[bad], hi[bad] = np.nan, np.nan
    return lo, hi


def ax_intervals(a, T, lo, hi, qty, lev):
    """Prediction intervals of all test nodules, sorted by the estimate; truth inside green, outside red."""
    o = np.argsort(T["pred"])
    x = np.arange(len(o))
    lo, hi = _finite(lo[o], hi[o])
    inside = (T["truth"][o] >= lo) & (T["truth"][o] <= hi)
    a.vlines(x, lo, hi, color="tab:blue", alpha=0.35, lw=3, label=f"{lev}% interval")
    a.plot(x, T["pred"][o], "_", color="black", ms=7, label="estimate")
    a.plot(x[inside], T["truth"][o][inside], "o", color="tab:green", ms=3.5, label="truth inside")
    a.plot(x[~inside], T["truth"][o][~inside], "o", color="tab:red", ms=3.5, label="truth outside")
    a.set_yscale("log"), a.set_xlabel("test nodules, sorted by estimate"), a.set_ylabel(f"{qty} ({UNIT[qty]})")
    a.set_title(f"prediction intervals (coverage {100 * np.mean(inside):.0f}%)", fontsize=9)
    a.legend(fontsize=6, loc="upper left"), a.grid(alpha=0.3)


def ax_regression(a, T, lo, hi, qty, lev):
    """Estimate vs truth with each nodule's interval as a vertical bar: covered if the bar crosses y = x."""
    lo, hi = _finite(lo, hi)
    inside = (T["truth"] >= lo) & (T["truth"] <= hi)
    for m, col in ((inside, "tab:green"), (~inside, "tab:red")):
        a.vlines(T["truth"][m], lo[m], hi[m], color=col, alpha=0.4, lw=1.5)
        a.plot(T["truth"][m], T["pred"][m], "o", color=col, ms=3)
    v = np.concatenate([T["truth"], T["pred"]])
    v = v[v > 0]
    lim = [v.min() / 1.5, v.max() * 1.5]
    a.plot(lim, lim, color="gray", lw=0.8, ls="--", label="estimate = truth")
    a.set_xscale("log"), a.set_yscale("log"), a.set_xlim(lim), a.set_ylim(lim)
    a.set_xlabel(f"truth ({UNIT[qty]})"), a.set_ylabel(f"estimate and {lev}% interval ({UNIT[qty]})")
    a.set_title("regression with uncertainty (bar crosses the line = covered)", fontsize=9)
    a.legend(fontsize=6, loc="upper left"), a.grid(alpha=0.3)


def ax_residuals(a, C, T, lo, hi, qty):
    """log(truth / estimate) against the estimate; the bars are the intervals in the same units."""
    lo, hi = _finite(lo, hi)
    r_t = np.log(T["truth"] / T["pred"])
    inside = (T["truth"] >= lo) & (T["truth"] <= hi)
    a.plot(C["pred"], np.log(C["truth"] / C["pred"]), ".", color="gray", ms=4, alpha=0.6, label="cal nodules")
    a.vlines(T["pred"], np.log(lo / T["pred"]), np.log(hi / T["pred"]), color="tab:blue", alpha=0.25, lw=2,
             label="interval")
    a.plot(T["pred"][inside], r_t[inside], "o", color="tab:green", ms=3, label="test, inside")
    a.plot(T["pred"][~inside], r_t[~inside], "o", color="tab:red", ms=3, label="test, outside")
    a.axhline(0, color="black", lw=0.8)
    a.set_xscale("log"), a.set_xlabel(f"estimate ({UNIT[qty]})"), a.set_ylabel("residual log(truth / estimate)")
    a.set_title("residuals vs estimate (0.1 = 10% too low)", fontsize=9)
    a.legend(fontsize=6), a.grid(alpha=0.3)


def ax_residual_hist(a, C, T):
    """Histogram of the residuals (cal and test) with the normal curve a Gaussian method would assume."""
    from scipy.stats import norm

    rc, rt = np.log(C["truth"] / C["pred"]), np.log(T["truth"] / T["pred"])
    rc, rt = rc[np.isfinite(rc)], rt[np.isfinite(rt)]
    bins = np.linspace(min(rc.min(), rt.min()), max(rc.max(), rt.max()), 30)
    a.hist(rc, bins, density=True, color="gray", alpha=0.5, label=f"cal (n={len(rc)})")
    a.hist(rt, bins, density=True, histtype="step", color="tab:blue", lw=1.5, label=f"test (n={len(rt)})")
    x = np.linspace(bins[0], bins[-1], 200)
    a.plot(x, norm.pdf(x, rc.mean(), rc.std()), "k--", lw=1, label="normal fit (cal)")
    for q in np.quantile(rc, [0.05, 0.95]):
        a.axvline(q, color="tab:orange", lw=1)
    a.axvline(0, color="black", lw=0.6)
    a.set_xlabel("residual log(truth / estimate)"), a.set_ylabel("density")
    a.set_title(f"distribution of residuals (mean {rc.mean():+.2f} = bias; orange: cal 5% / 95%)", fontsize=9)
    a.legend(fontsize=6), a.grid(alpha=0.3)


def ax_qq(a, C, floor):
    """Normal Q-Q plot of the calibration residuals: points off the line = not Gaussian (tails, skew)."""
    from scipy.stats import probplot

    r = np.log(C["truth"] / C["pred"])
    s = r / (C["sigma"] + floor)
    for v, col, lab in ((r, "tab:gray", "residuals"), (s, "tab:blue", "residuals / ensemble spread (the scores)")):
        v = v[np.isfinite(v)]
        if len(v) < 3 or np.std(v) == 0:
            continue
        (q, z), _ = probplot((v - v.mean()) / v.std(), dist="norm")
        a.plot(q, z, ".", color=col, ms=4, label=lab)
    lim = [-3, 3]
    a.plot(lim, lim, color="black", lw=0.8, ls="--", label="normal")
    a.set_xlabel("normal quantiles"), a.set_ylabel("standardised residual quantiles")
    a.set_title("Q-Q of calibration residuals (conformal does not need normality)", fontsize=9)
    a.legend(fontsize=6, loc="upper left"), a.grid(alpha=0.3)


def ax_widths(a, res, T, lev):
    """Distribution of the relative interval widths per variant (standard: one width for everyone)."""
    w_all = {}
    for v in res:
        lo, hi = _finite(v["lo"], v["hi"])
        w = (hi - lo) / T["pred"]
        w_all[v["variant"]] = w[np.isfinite(w) & (w > 0)]
    vals = np.concatenate([w for w in w_all.values() if len(w)]) if any(len(w) for w in w_all.values()) else np.array([1.0])
    bins = np.geomspace(max(vals.min() * 0.8, 1e-3), vals.max() * 1.25, 30)
    for name, w in w_all.items():
        if len(w):
            a.hist(w * 1.0, bins, histtype="step", lw=1.5, label=f"{name} (median {100 * np.median(w):.0f}%)")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    a.set_xscale("log")
    a.xaxis.set_major_locator(FixedLocator([t for t in (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16) if bins[0] <= t <= bins[-1]]))
    a.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{100 * t:.0f}%"))
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel(f"{lev}% interval width / estimate"), a.set_ylabel("test nodules")
    a.set_title("distribution of interval widths (narrow AND covering = good)", fontsize=9)
    a.legend(fontsize=6), a.grid(alpha=0.3)


def ax_coverage_levels(a, C, T, floor):
    """Empirical test coverage at every confidence level: conformal vs the raw ensemble spread."""
    from scipy.stats import norm

    levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    std = LogCPS(False, floor).fit(C["pred"], C["sigma"], C["truth"])
    nrm = LogCPS(True, floor).fit(C["pred"], C["sigma"], C["truth"])
    cov = {"conformal, standard": [], "conformal, normalised": [], "ensemble spread as +-z sd (uncalibrated)": []}
    for L in levels:
        for name, m in (("conformal, standard", std), ("conformal, normalised", nrm)):
            lo, hi = m.interval(T["pred"], T["sigma"], 1 - L)
            cov[name].append(coverage(lo, hi, T["truth"]))
        z = norm.ppf(0.5 + L / 2)
        lo, hi = T["pred"] * np.exp(-z * (T["sigma"] + floor)), T["pred"] * np.exp(z * (T["sigma"] + floor))
        cov["ensemble spread as +-z sd (uncalibrated)"].append(coverage(lo, hi, T["truth"]))
    a.plot([0.45, 1], [0.45, 1], color="gray", lw=0.8, ls="--", label="ideal")
    for name, c in cov.items():
        a.plot(levels, c, "o-", ms=4, label=name)
    a.set_xlim(0.45, 1), a.set_ylim(0, 1.02)
    a.set_xlabel("nominal confidence level"), a.set_ylabel("observed coverage (test)")
    a.set_title("coverage at every level (on the diagonal = calibrated)", fontsize=9)
    a.legend(fontsize=6, loc="lower right"), a.grid(alpha=0.3)


def ax_cpd(a, m, T, i, qty, groups=None):
    """One nodule's conformal predictive distribution."""
    top = max(T["truth"][i], T["pred"][i])
    grid = np.geomspace(max(top * 0.05, 0.1), top * 5, 200)
    a.plot(grid, m.cdf_curve(T["pred"][i], T["sigma"][i], grid, None if groups is None else groups[i]),
           color="tab:blue", label=f"calibrated P({qty} <= v)")
    a.axvline(T["truth"][i], color="tab:green", label="truth")
    a.axvline(T["pred"][i], color="tab:blue", ls=":", label="estimate")
    if qty == "volume":
        for t in VOLUME_THRESHOLDS_MM3:
            if grid[0] < t < grid[-1]:
                a.axvline(t, color="tab:red", ls="--", lw=0.8)
    a.set_xscale("log"), a.set_ylim(0, 1), a.set_xlabel(f"{qty} ({UNIT[qty]})"), a.set_ylabel("probability")
    a.set_title(str(T["rows"][i].get("nid", T["rows"][i]["case_id"]))[:28], fontsize=8)
    a.legend(fontsize=6), a.grid(alpha=0.3)


def save_cpd_grid(path, m, T, qty, n=4):
    plt = _plt()
    order = np.argsort(T["truth"])
    pick = order[np.linspace(0, len(order) - 1, min(n, len(order))).astype(int)]
    fig, ax = plt.subplots(1, len(pick), figsize=(3.6 * len(pick), 3.4), squeeze=False)
    for a, i in zip(ax[0], pick):
        ax_cpd(a, m, T, i, qty)
    fig.suptitle(f"conformal predictive distributions of {len(pick)} test nodules"
                 + (" (red: EUPS 100 / 300 mm3)" if qty == "volume" else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_diagnostics(out, qty, C, T, res, models, floor, lev):
    """Each uncertainty plot as its own file ({qty}_*.png) plus an overview ({qty}_uq_overview.png)."""
    plt = _plt()
    v = next(x for x in res if x["variant"] == MAIN)
    lo, hi = v["lo"], v["hi"]
    m = models[MAIN][0]
    panels = [("intervals", lambda a: ax_intervals(a, T, lo, hi, qty, lev)),
              ("regression", lambda a: ax_regression(a, T, lo, hi, qty, lev)),
              ("residuals", lambda a: ax_residuals(a, C, T, lo, hi, qty)),
              ("residual_hist", lambda a: ax_residual_hist(a, C, T)),
              ("qq", lambda a: ax_qq(a, C, floor)),
              ("interval_widths", lambda a: ax_widths(a, res, T, lev)),
              ("coverage_levels", lambda a: ax_coverage_levels(a, C, T, floor)),
              ("cpd_median_nodule", lambda a: ax_cpd(a, m, T, int(np.argsort(T["truth"])[len(T["truth"]) // 2]), qty))]
    for name, draw in panels:
        fig, a = plt.subplots(figsize=(5.2, 4.2))
        draw(a)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{qty}_{name}.png"), dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(2, 4, figsize=(21, 9.5))
    for a, (_, draw) in zip(ax.ravel(), panels):
        draw(a)
    fig.suptitle(f"{qty}: uncertainty diagnostics on the test nodules (conformal, {MAIN})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{qty}_uq_overview.png"), dpi=110)
    plt.close(fig)
    save_cpd_grid(os.path.join(out, f"{qty}_cpd_examples.png"), m, T, qty)


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
