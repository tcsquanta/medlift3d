#!/usr/bin/env python
"""Archive everything the project produced, for keeping (report, thesis appendix, re-analysis).

Default contents (a few GB): outputs/ (all tables, figures, PASS/FAIL reports), logs/ (one log per
step), work/results/ (per-nodule results, detections), work/models/ (best.pt + last.pt + training logs
and previews of both models), the small files in work/ (splits, LIDC selection, series lists) and
configs/. With --with-data also the prepared data (cases, DTS, coarse SIRT, training pairs: ~45 GB for
583 patients) -- only worth it if you want to re-run steps 9-14 elsewhere without re-downloading.

    python scripts/archive_results.py --out /workspace/medlift_full_results.zip
    python scripts/archive_results.py --out /workspace/medlift_everything.zip --with-data
"""
import argparse
import glob
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORED = (".npz", ".pt", ".png", ".jpg", ".zip", ".gz")          # already compressed: store as is


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/workspace/medlift_full_results.zip")
    ap.add_argument("--with-data", action="store_true", help="also cases / dts / sirt / roi_pairs (~45 GB)")
    ap.add_argument("--work", default="work")
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()
    os.chdir(ROOT)
    w = args.work
    trees = [args.outputs, "logs", f"{w}/results", f"{w}/models", "configs"]
    if args.with_data:
        trees += [f"{w}/cases", f"{w}/dts", f"{w}/sirt", f"{w}/roi_pairs"]
    files = [f for f in glob.glob(f"{w}/*") if os.path.isfile(f)]            # splits.json, selection csv, ...
    if args.with_data and os.path.exists(f"{w}/.cases_done"):                # restored data: runpod_run.sh skips steps 2-4
        files.append(f"{w}/.cases_done")
    for t in trees:
        files += [f for f in glob.glob(f"{t}/**/*", recursive=True) if os.path.isfile(f)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for f in files:
            z.write(f, compress_type=zipfile.ZIP_STORED if f.lower().endswith(STORED) else zipfile.ZIP_DEFLATED)
    tops = sorted({f.split("/")[0] if not f.startswith(f"{w}/") else "/".join(f.split("/")[:2]) for f in files})
    print(f"wrote {args.out}: {len(files)} files, {os.path.getsize(args.out) / 1e9:.2f} GB")
    print("contains:", ", ".join(tops))


if __name__ == "__main__":
    main()
