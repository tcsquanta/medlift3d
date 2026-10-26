#!/usr/bin/env python
"""Step 15 - is the diffusion model / the detector under-trained, converged or overfitting?

Reads the training logs (no GPU, seconds) and writes outputs/15_fit/{fit_curves.png, fit_report.md}:

    python scripts/15_check_fit.py                       # logs/08_train.log + logs/14_detector.log
    python scripts/15_check_fit.py --diff-log my_kaggle_output.txt --det-log none

What it looks at:
  diffusion  training loss (falls almost forever: on its own it says little); the nodule-box gain of the
             validation samples over the DTS input (the number best.pt is chosen by; 4 columns, noisy by
             ~0.3 dB); and the denoising loss at fixed noise on 32 training vs 32 held-out pairs (the
             [fit] lines). Held-out rising while training-pair loss falls = overfitting.
  detector   training loss vs the CPM on the calibration patients (every 500 steps).

--gap also measures the final models directly on training vs held-out data (needs the prepared data
and preferably a GPU, a few minutes): a large gap means the model memorised its training patients.
"""
import importlib.util
import os

from _common import ROOT, base_parser, setup

import numpy as np

from medlift3d.fitcheck import detector_verdict, diffusion_verdict, parse_detector_log, parse_diffusion_log
from medlift3d.io import Paths, load_json


def _module(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), os.path.join(ROOT, "scripts", name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    if not path or path == "none" or not os.path.exists(path):
        return None
    with open(path, errors="replace") as f:
        return f.read()


def plot(path, dd, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for row, (d, name) in enumerate(((dd, "diffusion (step 8)"), (dt, "detector (step 14)"))):
        a, b = ax[row]
        if d is None or not d["steps"]:
            a.set_axis_off(), b.set_axis_off()
            a.set_title(f"{name}: no log")
            continue
        k = max(1, min(20, len(d["loss"]) // 20))
        a.plot(d["steps"][k - 1:], np.convolve(d["loss"], np.ones(k) / k, "valid"), color="tab:blue")
        a.set_yscale("log"), a.set_xlabel("step"), a.set_ylabel("training loss (moving average)")
        a.set_title(f"{name}: training loss"), a.grid(alpha=0.3)
        c = d["checks"]
        if not c:
            b.set_axis_off()
            continue
        s = [x["step"] for x in c]
        if "gain" in c[0]:
            b.plot(s, [x["gain"] for x in c], "o-", color="tab:green", label="nodule-box gain (dB, left)")
            b.axhline(0, color="gray", lw=0.8)
            b.set_ylabel("PSNR gain over the DTS input (dB)")
            ft, fv = [x["fit_train"] for x in c], [x["fit_heldout"] for x in c]
            if np.isfinite(ft).any():
                b2 = b.twinx()
                b2.plot(s, ft, "s--", color="tab:blue", label="denoising loss, train pairs (right)")
                if np.isfinite(fv).any():
                    b2.plot(s, fv, "s--", color="tab:red", label="denoising loss, held-out pairs (right)")
                b2.set_ylabel("fixed-noise denoising loss")
                b2.legend(loc="lower right", fontsize=8)
            i = int(np.argmax([x["gain"] for x in c]))
        else:
            b.plot(s, [x["cpm"] for x in c], "o-", color="tab:green", label="CPM on cal patients")
            b.set_ylabel("CPM (higher = better)")
            i = int(np.argmax([x["cpm"] for x in c]))
        b.axvline(s[i], color="tab:orange", ls=":", label=f"best.pt (step {s[i]})")
        b.set_xlabel("step"), b.set_title(f"{name}: validation"), b.grid(alpha=0.3)
        b.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def gap_diffusion(cfg, dev, n, n_samples):
    import torch
    from medlift3d.diffusion.checkpoint import load_for_sampling
    from medlift3d.diffusion.dataset import RoiPairDataset

    t08 = _module("08_train_diffusion.py")
    work = cfg["paths"]["work"]
    path = os.path.join(work, "models", "diffusion", "best.pt")
    tdir, vdir = os.path.join(work, "roi_pairs", "train"), os.path.join(work, "roi_pairs", "val")
    if not (os.path.exists(path) and os.path.isdir(tdir) and os.path.isdir(vdir)):
        return ["diffusion: needs work/models/diffusion/best.pt and work/roi_pairs/{train,val} (step 7)"]
    model, diffusion, dcfg = load_for_sampling(path, dev)
    dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else None
    files = {s: t08.evenly(RoiPairDataset(d, augment=False).files, n) for s, d in (("train", tdir), ("held-out", vdir))}
    rows = []
    vsp = np.asarray(cfg["roi"]["voxel_mm"], float)[::-1]
    box = np.asarray(cfg["roi"].get("box_mm", (40, 50, 40)), float)[::-1] / vsp / 2.0
    for s, fl in files.items():
        ds = RoiPairDataset(fl, augment=False)
        loss = t08.fixed_noise_loss(model, diffusion, [ds[i] for i in range(len(ds))], dev, dtype)
        meta = RoiPairDataset(t08.evenly(fl, n_samples), augment=False, return_meta=True)
        _, _, _, (bs, bc) = t08.evaluate(model, diffusion, [meta[i] for i in range(len(meta))],
                                         int(dcfg["sample_steps"]), dev, start_frac=float(dcfg.get("start_frac", 1.0)),
                                         box_half_vox=box)
        rows.append((s, len(fl), loss, len(meta), bs - bc))
        print(f"  diffusion {s}: denoising loss {loss:.4f} ({len(fl)} pairs), nodule-box gain {bs - bc:+.2f} dB "
              f"({len(meta)} sampled)", flush=True)
    (_, _, lt, _, gt), (_, _, lv, _, gv) = rows
    lines = ["| diffusion best.pt | pairs | denoising loss (fixed noise) | sampled | nodule-box gain |", "|---|---|---|---|---|"]
    lines += [f"| {s} | {a} | {b:.4f} | {c} | {g:+.2f} dB |" for s, a, b, c, g in rows]
    note = (f"Held-out / train denoising loss = {lv / lt:.2f}; gain train - held-out = {gt - gv:+.2f} dB. "
            + ("Close (< 1.15 and < 1 dB): the model generalises to new patients."
               if lv / lt < 1.15 and gt - gv < 1.0 else
               "Large: the model works clearly better on its training patients (memorising); more patients / "
               "augmentation or fewer hours."))
    return lines + ["", note]


def gap_detector(cfg, dev, n):
    from medlift3d.detector import froc, load_detector

    t14 = _module("14_train_detector.py")
    path = os.path.join(cfg["paths"]["work"], "models", "detector", "best.pt")
    paths = Paths(cfg)
    if not (os.path.exists(path) and os.path.exists(paths.p("splits.json"))):
        return ["detector: needs work/models/detector/best.pt and work/splits.json"]
    model, _ = load_detector(path, dev)
    dcfg = dict(cfg["detector"])
    sp = load_json(paths.p("splits.json"))["cases"]
    mu_w = float(cfg["physics"]["mu_water"])
    res = {}
    for s in ("train", "cal"):
        ids = sorted(c for c, v in sp.items() if v == s)
        ids = [ids[i] for i in sorted(set(np.linspace(0, len(ids) - 1, min(n, len(ids))).astype(int)))] if ids else []
        scans = t14.load_scans(paths, ids, dcfg, mu_w)
        if scans:
            res[s] = (len(scans), froc(t14.evaluate(model, scans, dev, int(dcfg["eval_top_k"])))["cpm"])
            print(f"  detector {s}: CPM {res[s][1]:.3f} on {res[s][0]} scans", flush=True)
    if len(res) < 2:
        return ["detector: train or cal scans missing (steps 4 and 6)"]
    gap = res["train"][1] - res["cal"][1]
    return ["| detector best.pt | scans | CPM |", "|---|---|---|",
            f"| train patients | {res['train'][0]} | {res['train'][1]:.3f} |",
            f"| cal patients (held out) | {res['cal'][0]} | {res['cal'][1]:.3f} |", "",
            f"Train - cal CPM = {gap:+.3f}. " + ("Small (< 0.15): generalises." if gap < 0.15 else
                                                 "Large: memorised its training patients; more patients would help.")]


def section(name, v):
    keys = {k: v[k] for k in v if k not in ("verdict", "reason", "action")}
    nums = ", ".join(f"{k} {val:.4g}" if isinstance(val, float) else f"{k} {val}" for k, val in keys.items())
    return [f"## {name}: **{v['verdict']}**", "", f"* why: {v['reason']}", f"* what to do: {v['action']}",
            f"* numbers: {nums}", ""]


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--diff-log", default=os.path.join(ROOT, "logs", "08_train.log"))
    ap.add_argument("--det-log", default=os.path.join(ROOT, "logs", "14_detector.log"))
    ap.add_argument("--gap", action="store_true", help="also measure best.pt on training vs held-out data")
    ap.add_argument("--n", type=int, default=32, help="--gap: pairs / scans per side")
    ap.add_argument("--n-samples", type=int, default=8, help="--gap: diffusion columns sampled per side")
    args = ap.parse_args()
    cfg, dev = setup(args)
    out = Paths(cfg).out("15_fit")
    td, tt = _read(args.diff_log), _read(args.det_log)
    dd = parse_diffusion_log(td) if td else None
    dt = parse_detector_log(tt) if tt else None
    lines = ["# Training fit check (step 15)", ""]
    for name, d, judge, src in (("Diffusion (step 8)", dd, diffusion_verdict, args.diff_log),
                                ("Detector (step 14)", dt, detector_verdict, args.det_log)):
        if d is None or not d["steps"]:
            lines += [f"## {name}: no training lines in {src}", ""]
            continue
        lines += section(name, judge(d))
        if name.startswith("Diffusion") and d["checks"] and not np.isfinite(d["checks"][-1]["fit_heldout"]):
            lines += ["(no [fit] lines: log from an older version; the verdict uses the nodule-box gain only, "
                      "which is noisy. `--gap` measures the held-out loss directly.)", ""]
    if args.gap:
        print("measuring best.pt on training vs held-out data ...", flush=True)
        lines += ["## Train vs held-out (best.pt, `--gap`)", ""] + gap_diffusion(cfg, dev, args.n, args.n_samples) + [""]
        lines += gap_detector(cfg, dev, args.n) + [""]
    plot(os.path.join(out, "fit_curves.png"), dd, dt)
    lines += ["How to read it: docs/RUNPOD_FULL_RUN.md, \"Is the model under- or overfitting?\".",
              "Curves: outputs/15_fit/fit_curves.png"]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "fit_report.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
