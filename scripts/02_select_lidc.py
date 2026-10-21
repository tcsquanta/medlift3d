#!/usr/bin/env python
"""Step 2 - choose LIDC-IDRI cases from the radiologist annotations (offline).

Uses pylidc's built-in database (= the 8.62 MB "Radiologist Annotations"
download on the TCIA LIDC-IDRI page). No images needed for this step.

Selection: slice thickness <= 2.5 mm, nodule marked by >= 3 of 4 radiologists,
mean diameter 4-30 mm. Patients with more qualifying nodules come first.

Outputs:
  work/lidc_selection.csv      one row per nodule (all qualifying nodules)
  work/lidc_patients.txt       the chosen patients
  work/lidc_series.txt         their series UIDs (input to step 3)
  work/lidc_subset.tcia        manifest for NBIA Data Retriever (alternative to step 3)
  outputs/02_selection/        size / malignancy histograms + verification

    python scripts/02_select_lidc.py --n-patients 200
"""
import os

from _common import base_parser, setup

import numpy as np

from medlift3d.data.lidc import choose_patients, select_nodules, write_tcia_manifest
from medlift3d.io import Paths
from medlift3d.utils import Checker


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n-patients", type=int, default=200)
    ap.add_argument("--min-readers", type=int, default=3)
    ap.add_argument("--min-diameter", type=float, default=4.0)
    ap.add_argument("--max-diameter", type=float, default=30.0)
    ap.add_argument("--max-scans", type=int, default=None, help="debug: only look at the first N scans")
    args = ap.parse_args()
    cfg, _ = setup(args)
    paths = Paths(cfg)
    out = paths.out("02_selection")
    ck = Checker("Step 2: LIDC-IDRI selection from radiologist annotations")
    rows = select_nodules(min_readers=args.min_readers, min_diameter=args.min_diameter,
                          max_diameter=args.max_diameter, max_scans=args.max_scans)
    ck.check("qualifying nodules found", len(rows) > 0, len(rows))
    pats = choose_patients(rows, args.n_patients, seed=cfg["seed"])
    chosen = [r for r in rows if r["patient_id"] in set(pats)]
    series = list(dict.fromkeys(r["series_uid"] for r in chosen))
    import csv

    keys = list(rows[0].keys())
    with open(paths.p("lidc_selection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["chosen"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, chosen=r["patient_id"] in set(pats)))
    with open(paths.p("lidc_patients.txt"), "w") as f:
        f.write("\n".join(pats) + "\n")
    with open(paths.p("lidc_series.txt"), "w") as f:
        f.write("\n".join(series) + "\n")
    write_tcia_manifest(series, paths.p("lidc_subset.tcia"))
    d = np.array([r["diameter_mm"] for r in chosen])
    ck.info("chosen patients / series / nodules", f"{len(pats)} / {len(series)} / {len(chosen)}")
    ck.info("diameter mm (median, 10-90%)", f"{np.median(d):.1f} ({np.percentile(d, 10):.1f}-{np.percentile(d, 90):.1f})")
    ck.check("every chosen nodule has >= min readers", all(r["n_readers"] >= args.min_readers for r in chosen), "")
    est_gb = 0.11 * len(series)
    ck.info("approx. download size", f"{est_gb:.1f} GB (~110 MB per CT series)")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
        ax[0].hist(d, bins=25)
        ax[0].set_xlabel("mean reader diameter (mm)")
        ax[1].hist([r["malignancy"] for r in chosen], bins=np.arange(0.75, 5.5, 0.5))
        ax[1].set_xlabel("mean malignancy rating (1-5)")
        ax[2].hist([r["texture"] for r in chosen], bins=np.arange(0.75, 5.5, 0.5))
        ax[2].set_xlabel("texture (1 GGO ... 5 solid)")
        fig.suptitle(f"{len(chosen)} nodules in {len(pats)} chosen patients")
        fig.savefig(os.path.join(out, "selection_histograms.png"), dpi=110, bbox_inches="tight")
    except Exception as e:  # pragma: no cover
        print("plot failed:", e)
    print(f"\nNext: python scripts/03_download_lidc.py   (or open {paths.p('lidc_subset.tcia')} in NBIA Data Retriever)")
    ck.finish(out)


if __name__ == "__main__":
    main()
