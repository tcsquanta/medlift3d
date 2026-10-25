#!/usr/bin/env python
"""Pack what Kaggle needs after the rented-GPU run into one zip (step 10, report, demo).

Contents: both trained models, splits.json, work/results (per-nodule results of steps 9 / 11,
detections), outputs/ (all tables, figures, PASS/FAIL reports) and the cases / DTS / coarse SIRT
files of ``--n-demo`` test patients for the demo. The big data (all cases, DICOM) stays behind.

    python scripts/make_kaggle_bundle.py --out /workspace/medlift_kaggle.zip --n-demo 15
"""
import argparse
import glob
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/workspace/medlift_kaggle.zip")
    ap.add_argument("--n-demo", type=int, default=15, help="test patients to include for the demo")
    ap.add_argument("--work", default="work")
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()
    os.chdir(ROOT)
    w = args.work
    files, missing = [], []
    for f in [f"{w}/splits.json", f"{w}/models/diffusion/best.pt", f"{w}/models/detector/best.pt"]:
        (files if os.path.exists(f) else missing).append(f)
    files += [f for f in glob.glob(f"{w}/results/*") if os.path.isfile(f)]
    files += [f for f in glob.glob(f"{args.outputs}/**/*", recursive=True) if os.path.isfile(f)
              and not f.startswith(f"{args.outputs}/12_demo/roi_")]          # skip the demo's click images
    test = []
    if os.path.exists(f"{w}/splits.json"):
        sp = json.load(open(f"{w}/splits.json"))["cases"]
        for c in sorted(c for c, s in sp.items() if s == "test"):
            need = [f"{w}/{d}/{c}.npz" for d in ("cases", "dts", "sirt")]
            if all(os.path.exists(f) for f in need):
                test.append(c)
                files += need
            if len(test) >= args.n_demo:
                break
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:       # .npz / .pt are already compressed
        for f in files:
            z.write(f)
    size = os.path.getsize(args.out) / 1e9
    print(f"wrote {args.out}: {len(files)} files, {size:.2f} GB, {len(test)} demo test patients")
    if missing:
        print("WARNING, missing (did that step run?):", ", ".join(missing))
    if not any(f.endswith("pipeline.jsonl") for f in files):
        print("WARNING: no work/results/pipeline.jsonl -> step 10 cannot run on Kaggle (run step 9 first)")


if __name__ == "__main__":
    main()
