#!/usr/bin/env python
"""Step 10 - evaluation: accuracy, calibrated intervals, hypothesis checks.

Reads work/results/pipeline.jsonl (step 9). Conformal intervals are fitted
on the *cal* patients and evaluated on the *test* patients.

Outputs (outputs/10_evaluation/): table_accuracy.md, table_intervals.md,
summary.csv, conformal.json (used by the demo), figures.

The H1-H3 lines are scientific results, not code checks: report them
honestly whichever way they come out.
  H1  mass error < volume error for every DTS method
  H2  diffusion+Gaussians (ours) beats physics-only methods on volume / Dice
  H3  90% conformal intervals cover ~90% of test nodules; mass intervals narrower

    python scripts/10_evaluate.py
"""
import csv
import os

from _common import base_parser, setup

import numpy as np

from medlift3d.io import Paths, read_jsonl, save_json
from medlift3d.uncertainty import ConformalCalibrator, coverage, relative_width, spearman
from medlift3d.utils import Checker
from medlift3d import viz

ORDER = ["volume2d", "sirt_coarse", "roi_sirt", "roi_gs", "diffusion", "diffusion_gs"]
LABEL = {"volume2d": "2D diameter (sphere)", "sirt_coarse": "DTS SIRT (standard)", "roi_sirt": "ROI-SIRT (physics)",
         "roi_gs": "ROI-Gaussians (physics)", "diffusion": "Diffusion only", "diffusion_gs": "Diffusion+Gaussians (ours)"}


def collect(rows, method, qty):
    """(pred, sigma_log, truth, rows) for one method and quantity ('volume' | 'mass')."""
    P, S, T, R = [], [], [], []
    for r in rows:
        gt = r["gt"]
        if method == "volume2d":
            src = r["methods"].get("roi_sirt") or r["methods"].get("sirt_coarse")
            if src is None or qty != "volume":
                continue
            p, s = src["volume2d_mm3"], 0.0
        else:
            e = r["methods"].get(method)
            if e is None:
                continue
            p = e["volume_mm3"] if qty == "volume" else e["mass_mg"]
            s = e["volume_log_std"] if qty == "volume" else e["mass_log_std"]
        t = gt["volume_mm3"] if qty == "volume" else gt["mass_mg"]
        if not (np.isfinite(p) and np.isfinite(t) and t > 0):
            continue
        P.append(max(p, 1e-3))
        S.append(s if np.isfinite(s) else 0.0)
        T.append(t)
        R.append(r)
    return np.array(P), np.array(S), np.array(T), R


def err_stats(p, t):
    """Errors over all nodules (a failed measurement, p ~ 0, counts as -100%);
    Bland-Altman limits over successful measurements only; failure rate."""
    e = p / t - 1.0
    ok = p > 1e-2
    le = np.log(p[ok] / t[ok]) if ok.any() else np.array([np.nan])
    return {"n": len(p), "failed_pct": 100 * float(np.mean(~ok)), "median_signed_pct": 100 * float(np.median(e)),
            "mdape_pct": 100 * float(np.median(np.abs(e))), "within20_pct": 100 * float(np.mean(np.abs(e) <= 0.2)),
            "loa_low_pct": 100 * float(np.exp(np.mean(le) - 1.96 * np.std(le)) - 1),
            "loa_high_pct": 100 * float(np.exp(np.mean(le) + 1.96 * np.std(le)) - 1)}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    cfg, _ = setup(args)
    paths = Paths(cfg)
    out = paths.out("10_evaluation")
    ck = Checker("Step 10: evaluation")
    rows = read_jsonl(args.results or paths.p("results", "pipeline.jsonl"))
    ck.check("results available", len(rows) > 0, f"{len(rows)} nodules")
    if not rows:
        ck.finish(out)
        return
    cal = [r for r in rows if r["split"] == "cal"]
    test = [r for r in rows if r["split"] == "test"]
    if not cal or not test:                                   # tiny runs: split rows in half by patient
        pats = sorted({r["patient_id"] for r in rows})
        half = set(pats[: max(1, len(pats) // 2)])
        cal = [r for r in rows if r["patient_id"] in half]
        test = [r for r in rows if r["patient_id"] not in half] or cal
        ck.info("cal/test split", "not enough patients in each split -> split results by patient halves")
    ck.info("nodules cal / test", f"{len(cal)} / {len(test)}")
    methods = [m for m in ORDER if m == "volume2d" or any(m in r["methods"] for r in rows)]
    alpha = float(cfg["uncertainty"]["alpha"])
    summary, conf_out = [], {}
    acc_lines = ["| method | quantity | n | failed | median signed err % | MdAPE % | within +-20% | 95% limits of agreement % (successes) | Dice | HD95 mm | PSNR dB |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
    int_lines = [f"| method | quantity | coverage of {int(100 * (1 - alpha))}% interval (test) | median width (% of estimate) | Spearman(sigma, abs err) |",
                 "|---|---|---|---|---|"]
    box_v, box_m = {}, {}
    mdape = {}
    for m in methods:
        for qty in ("volume", "mass"):
            P, S, T, R = collect(test, m, qty)
            if len(P) == 0:
                continue
            st = err_stats(P, T)
            mdape[(m, qty)] = st["mdape_pct"]
            seg = {k: float(np.nanmedian([r["methods"][m].get(k, np.nan) for r in R])) if m in R[0]["methods"] else float("nan")
                   for k in ("dice", "hd95_mm", "psnr")} if m != "volume2d" else {"dice": np.nan, "hd95_mm": np.nan, "psnr": np.nan}
            acc_lines.append(f"| {LABEL[m]} | {qty} | {st['n']} | {st['failed_pct']:.0f}% | {st['median_signed_pct']:+.1f} | {st['mdape_pct']:.1f} | "
                             f"{st['within20_pct']:.0f}% | [{st['loa_low_pct']:+.0f}, {st['loa_high_pct']:+.0f}] | "
                             f"{seg['dice']:.3f} | {seg['hd95_mm']:.2f} | {seg['psnr']:.2f} |")
            (box_v if qty == "volume" else box_m)[LABEL[m]] = list(100 * (P / T - 1))
            row = {"method": m, "quantity": qty, **st, **seg}
            Pc, Sc, Tc, _ = collect(cal, m, qty)
            if len(Pc) >= 3:
                calib = ConformalCalibrator(alpha=alpha, mode=cfg["uncertainty"]["mode"],
                                            sigma_floor=float(cfg["uncertainty"]["sigma_floor"])).fit(Pc, Sc, Tc)
                lo, hi = calib.interval(P, S)
                cov = coverage(lo, hi, T)
                wid = relative_width(lo, hi, P)
                rho = spearman(S, np.abs(np.log(P / T))) if np.std(S) > 0 else float("nan")
                wtxt = f"{100 * wid:.0f}%" if np.isfinite(wid) else "inf (too few cal. nodules)"
                int_lines.append(f"| {LABEL[m]} | {qty} | {100 * cov:.0f}% | {wtxt} | {rho:.2f} |")
                row.update(coverage=cov, rel_width=wid, spearman_sigma_err=rho)
                conf_out[f"{m}:{qty}"] = calib.to_dict()
            summary.append(row)
    with open(os.path.join(out, "table_accuracy.md"), "w") as f:
        f.write("\n".join(acc_lines) + "\n")
    n_min_asym = int(np.ceil(2.0 / alpha - 1.0))
    int_lines.append(f"\nFinite {int(100 * (1 - alpha))}% conformal intervals need >= {int(np.ceil(1.0 / alpha - 1.0))} calibration "
                     f"nodules (symmetric) or >= {n_min_asym} (asymmetric); this run had {len(cal)}.")
    with open(os.path.join(out, "table_intervals.md"), "w") as f:
        f.write("\n".join(int_lines) + "\n")
    save_json(conf_out, os.path.join(out, "conformal.json"))
    if summary:
        keys = sorted({k for s in summary for k in s})
        with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary)
    print("\n" + "\n".join(acc_lines) + "\n\n" + "\n".join(int_lines) + "\n")
    viz.save_boxplot(os.path.join(out, "volume_errors.png"), box_v, "volume error vs radiologist consensus (test)", "error (%)")
    viz.save_boxplot(os.path.join(out, "mass_errors.png"), box_m, "mass error vs CT (test)", "error (%)")
    best = "diffusion_gs" if "diffusion_gs" in methods else ("roi_gs" if "roi_gs" in methods else "roi_sirt")
    for qty in ("volume", "mass"):
        P, S, T, _ = collect(test, best, qty)
        if len(P):
            viz.save_bland_altman(os.path.join(out, f"bland_altman_{qty}_{best}.png"), P, T, f"{LABEL[best]}: {qty}",
                                  "(mm3)" if qty == "volume" else "(mg)")
    xs, ys, labs = [], [], []
    for m in methods:
        P, _, T, _ = collect(test, m, "volume")
        if len(P):
            xs.append(T), ys.append(P), labs.append(LABEL[m])
    viz.save_scatter(os.path.join(out, "volume_scatter.png"), xs, ys, labs, "volume: estimate vs truth", "truth (mm3)", "estimate (mm3)")
    # inter-reader context
    cv = [r["gt"].get("reader_volume_cv") for r in rows if r["gt"].get("reader_volume_cv") is not None]
    if cv:
        ck.info("radiologists' own volume disagreement (median CV across readers)", f"{100 * np.median(cv):.0f}%")
    # hypotheses ------------------------------------------------------------------------
    dts_methods = [m for m in methods if m != "volume2d" and (m, "mass") in mdape]
    if dts_methods:
        h1 = [mdape[(m, "mass")] < mdape[(m, "volume")] for m in dts_methods]
        ck.check("H1 mass error < volume error (methods where true / all)", all(h1),
                 f"{sum(h1)}/{len(h1)}: " + ", ".join(f"{m} M{mdape[(m, 'mass')]:.0f}% vs V{mdape[(m, 'volume')]:.0f}%" for m in dts_methods))
    phys = [m for m in ("roi_sirt", "roi_gs") if (m, "volume") in mdape]
    if "diffusion_gs" in methods and phys and ("diffusion_gs", "volume") in mdape:
        ours = mdape[("diffusion_gs", "volume")]
        ck.check("H2 ours has lower volume MdAPE than physics-only", all(ours < mdape[(m, "volume")] for m in phys),
                 f"ours {ours:.0f}% vs " + ", ".join(f"{m} {mdape[(m, 'volume')]:.0f}%" for m in phys))
    for s in summary:
        if s["method"] == best and "coverage" in s:
            ck.check(f"H3 {s['quantity']} interval coverage close to {int(100 * (1 - alpha))}% ({LABEL[best]})",
                     abs(s["coverage"] - (1 - alpha)) <= 0.1 or len(test) < 20, f"{100 * s['coverage']:.0f}% (n={len(test)})")
    wid = {s["quantity"]: s.get("rel_width") for s in summary if s["method"] == best and "rel_width" in s}
    if "volume" in wid and "mass" in wid:
        ck.check("H3b mass intervals narrower than volume intervals", wid["mass"] < wid["volume"],
                 f"mass {100 * wid['mass']:.0f}% vs volume {100 * wid['volume']:.0f}%")
    ck.finish(out)


if __name__ == "__main__":
    main()
