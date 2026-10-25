#!/usr/bin/env python
"""Step 14 - train the nodule detector on the standard DTS reconstruction (coarse SIRT).

A small 3D U-Net learns to put a peak at every radiologist-annotated nodule centre
(LIDC consensus) in the DTS volume that the doctor scrolls through:
  * trained on the *train*-split patients (the same patients as the diffusion prior),
  * checkpoint chosen on the *cal* patients (CPM: mean sensitivity at 1/8 ... 8 false
    positives per scan, as in LUNA16), operating threshold set there at 2 FP per scan,
  * evaluated once on the *test* patients, next to the training-free LoG detector.
The demo (step 12) then suggests these detections on unseen (test) patients.

Needs only steps 4 and 6 (work/cases, work/sirt). Time: ~15-30 min on a T4.

    python scripts/14_train_detector.py
    python scripts/14_train_detector.py --steps 10000 --base-ch 24        # bigger / longer
"""
import math
import os
import time

from _common import base_parser, setup

import numpy as np
import torch

from medlift3d.data.cases import load_case
from medlift3d.detect import detect_candidates
from medlift3d.detector import (DetNet, detect_learned, focal_loss, froc, heatmap_target, load_detector,
                                normalize_hu, save_detector, score_threshold)
from medlift3d.io import Paths, load_json, load_sirt, save_json
from medlift3d.physics import mu_to_hu
from medlift3d.utils import Checker
from medlift3d import viz


def load_scans(paths, ids, dcfg, mu_water):
    """Per case: coarse DTS volume (HU), its grid, nodules, heatmap target."""
    scans = []
    for cid in ids:
        if not (os.path.exists(paths.sirt(cid)) and os.path.exists(paths.case(cid))):
            continue
        mu, vg, _ = load_sirt(paths.sirt(cid))
        hu = mu_to_hu(mu.numpy(), mu_water).astype(np.float32)
        case = load_case(paths.case(cid), load_readers=False)
        nods = [{"xyz": [float(v) for v in n.center_xyz], "diameter_mm": float(n.diameter_mm)} for n in case.nodules]
        tgt = heatmap_target(vg, [n["xyz"] for n in nods], [n["diameter_mm"] for n in nods],
                             float(dcfg["min_sigma_mm"]), float(dcfg["depth_factor"]))
        scans.append({"case_id": cid, "hu": hu, "vg": vg, "nodules": nods, "target": tgt})
    return scans


class PatchSampler:
    """Random training patches: ``pos_frac`` of them contain a nodule (anywhere in the patch),
    the rest are random. Flips left-right and head-feet (not front-back: DTS is not symmetric there)."""

    def __init__(self, scans, patch, pos_frac, seed):
        self.patch = np.asarray(patch, int)                  # (z, y, x)
        self.pos_frac = pos_frac
        self.rng = np.random.default_rng(seed)
        self.items = []
        air = float(normalize_hu(np.float32(-1000.0)))
        for s in scans:
            pad = [(0, max(0, int(p) - n)) for p, n in zip(self.patch, s["hu"].shape)]
            x = np.pad(normalize_hu(s["hu"]), pad, constant_values=air).astype(np.float16)
            t = np.pad(s["target"], pad).astype(np.float16)
            centres = [np.round(s["vg"].world_to_index(np.asarray(n["xyz"]))[::-1]).astype(int) for n in s["nodules"]]
            self.items.append((x, t, centres))

    def sample(self, batch):
        xs, ts = [], []
        P = self.patch
        for _ in range(batch):
            x, t, centres = self.items[self.rng.integers(len(self.items))]
            hi = np.asarray(x.shape) - P
            if centres and self.rng.random() < self.pos_frac:
                c = centres[self.rng.integers(len(centres))]
                lo = c - P // 2 + self.rng.integers(-(P // 4), P // 4 + 1)
            else:
                lo = np.array([self.rng.integers(0, h + 1) for h in hi])
            lo = np.clip(lo, 0, hi)
            sl = tuple(slice(a, a + p) for a, p in zip(lo, P))
            xp, tp = x[sl].astype(np.float32), t[sl].astype(np.float32)
            for ax in (0, 2):                                   # z (head-feet), x (left-right)
                if self.rng.random() < 0.5:
                    xp, tp = np.flip(xp, ax), np.flip(tp, ax)
            xs.append(np.ascontiguousarray(xp))
            ts.append(np.ascontiguousarray(tp))
        return torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ts))[:, None]


def evaluate(model, scans, dev, top_k):
    out = []
    for s in scans:
        out.append({"case_id": s["case_id"], "nodules": s["nodules"],
                    "cands": detect_learned(model, s["hu"], s["vg"], dev, top_k=top_k)})
    return out


def baseline_log(scans, top_k):
    out = []
    for s in scans:
        rec = s["hu"]
        sel = (rec > -1000) & (rec < -600)
        lung = float(np.median(rec[sel])) if sel.any() else -850.0
        out.append({"case_id": s["case_id"], "nodules": s["nodules"],
                    "cands": detect_candidates(rec, s["vg"], baseline_hu=lung, top_k=top_k)})
    return out


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--base-ch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true", help="evaluate work/models/detector/best.pt, no training")
    args = ap.parse_args()
    cfg, dev = setup(args)
    dcfg = dict(cfg["detector"])
    for k, a in (("steps", args.steps), ("batch_size", args.batch_size), ("base_ch", args.base_ch),
                 ("lr", args.lr), ("val_every", args.val_every)):
        if a is not None:
            dcfg[k] = a
    paths = Paths(cfg)
    out = paths.out("14_detector")
    ck = Checker("Step 14: nodule detector on the DTS volume (train / cal / test patients)")
    mdir = os.path.join(cfg["paths"]["work"], "models", "detector")
    splits = load_json(paths.p("splits.json"))["cases"]
    by = {s: sorted(c for c, v in splits.items() if v == s) for s in ("train", "cal", "test")}
    mu_w = float(cfg["physics"]["mu_water"])
    t0 = time.time()
    data = {s: load_scans(paths, by[s], dcfg, mu_w) for s in by}
    pats = {s: {c.split("_")[0] for c in (d["case_id"] for d in data[s])} for s in data}
    ck.check("train / cal / test patients are disjoint", not (pats["train"] & pats["cal"]) and not
             (pats["train"] & pats["test"]) and not (pats["cal"] & pats["test"]),
             ", ".join(f"{s}: {len(data[s])} scans, {sum(len(d['nodules']) for d in data[s])} nodules" for s in data))
    if not data["train"] or not data["test"]:
        ck.check("train and test scans available", False, "run steps 4 and 6 first")
        ck.finish(out)
        return
    val = data["cal"] or data["test"][: max(1, len(data["test"]) // 2)]
    if not data["cal"]:
        ck.info("no cal scans: checkpoint chosen on half of the test scans (tiny run only)", len(val))
    top_k = int(dcfg["eval_top_k"])

    if not args.eval_only:
        torch.manual_seed(int(cfg["seed"]))
        model = DetNet(int(dcfg["base_ch"])).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        sampler = PatchSampler(data["train"], dcfg["patch_vox"], float(dcfg["pos_frac"]), int(cfg["seed"]))
        opt = torch.optim.AdamW(model.parameters(), lr=float(dcfg["lr"]), weight_decay=1e-4)
        steps, warm = int(dcfg["steps"]), min(200, int(dcfg["steps"]) // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
        amp = dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        losses, best, hist = [], -1.0, []
        print(f"detector: {n_par / 1e6:.2f} M parameters, {steps} steps x batch {dcfg['batch_size']}, "
              f"patch {dcfg['patch_vox']}, {len(data['train'])} train scans", flush=True)
        for step in range(1, steps + 1):
            model.train()
            x, t = sampler.sample(int(dcfg["batch_size"]))
            x, t = x.to(dev), t.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = focal_loss(logits, t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(f"step {step}/{steps}  loss {np.mean(losses[-100:]):.4f}  "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            if step % int(dcfg["val_every"]) == 0 or step == steps:
                r = froc(evaluate(model, val, dev, top_k))
                hist.append(r["cpm"])
                tag = ""
                if r["cpm"] > best:
                    best, tag = r["cpm"], "  <- best"
                    save_detector(os.path.join(mdir, "best.pt"), model, int(dcfg["base_ch"]), {"step": step, "val_cpm": best})
                print(f"[val] step {step}: CPM {r['cpm']:.3f}, sensitivity at 2 FP/scan "
                      f"{r['sensitivity'][2.0]:.2f}, recall top-10 {r['recall_top10']:.2f}{tag}", flush=True)
        save_detector(os.path.join(mdir, "last.pt"), model, int(dcfg["base_ch"]), {"step": steps})
        k = max(1, len(losses) // 10)
        ck.check("training loss decreased (last 10% vs first 10% of steps)", np.mean(losses[-k:]) < np.mean(losses[:k]),
                 f"{np.mean(losses[:k]):.3f} -> {np.mean(losses[-k:]):.3f}")
        viz.save_curve(os.path.join(out, "loss.png"), {"focal loss": np.convolve(losses, np.ones(k) / k, "valid")},
                       "detector training loss", logy=True)
        ck.info("validation CPM per check (cal patients)", [round(h, 3) for h in hist])

    best_path = os.path.join(mdir, "best.pt")
    ck.check("detector checkpoint saved", os.path.exists(best_path), best_path)
    if not os.path.exists(best_path):
        ck.finish(out)
        return
    model, extra = load_detector(best_path, dev)
    cal_res = evaluate(model, val, dev, top_k)
    thr = score_threshold(cal_res, float(dcfg["fp_per_scan"]))
    extra.update({"threshold": thr, "fp_per_scan": float(dcfg["fp_per_scan"]),
                  "val_cpm": froc(cal_res)["cpm"]})
    save_detector(best_path, model, int(model.e1[0].out_channels), extra)          # demo reads the threshold
    test_res = evaluate(model, data["test"], dev, top_k)
    rl, rb = froc(test_res), froc(baseline_log(data["test"], top_k))
    r2d = froc(test_res, depth_tol_mm=1e9)                  # diagnostic: right place in-plane, any depth
    lines = ["| detector (test patients) | CPM | sensitivity at 1 / 2 / 4 / 8 FP per scan | recall top-10 | nodules / scans |",
             "|---|---|---|---|---|"]
    for name, r in (("learned 3D U-Net (ours)", rl), ("training-free LoG + Hessian", rb),
                    ("learned, in-plane only (any depth; diagnostic)", r2d)):
        s = r["sensitivity"]
        lines.append(f"| {name} | {r['cpm']:.3f} | {100 * s[1.0]:.0f}% / {100 * s[2.0]:.0f}% / {100 * s[4.0]:.0f}% / "
                     f"{100 * s[8.0]:.0f}% | {100 * r['recall_top10']:.0f}% | {r['n_nodules']} / {r['n_scans']} |")
    note = ("A suggestion finds a nodule if it lies within max(6 mm, radius + 3 mm) in-plane and 12 mm + radius in depth "
            "(where the measurement searches). Only LIDC nodules marked by >= 3 radiologists and 4-30 mm are labels; "
            "suggestions on other real findings count as false positives. The diagnostic row ignores depth: the gap to "
            "the first row is suggestions in the right coronal position but at the wrong depth (the DTS depth smear).")
    text = "\n".join(lines) + "\n\n" + note + "\n"
    print("\n" + text)
    with open(os.path.join(out, "table_detection.md"), "w") as f:
        f.write(text)
    viz.save_froc(os.path.join(out, "froc_test.png"),
                  {"learned (ours)": (rl["curve_fp_per_scan"], rl["curve_sensitivity"]),
                   "training-free LoG": (rb["curve_fp_per_scan"], rb["curve_sensitivity"])},
                  f"FROC on {rl['n_scans']} test patients ({rl['n_nodules']} nodules)")
    save_json({"threshold": thr, "test": test_res, "cal": cal_res}, paths.p("results", "detections.json"))
    ck.info("operating threshold (2 FP per scan on the cal patients)", f"{thr:.3f}")
    ck.info("learned vs training-free detector on test patients: CPM", f"{rl['cpm']:.3f} vs {rb['cpm']:.3f}")
    ck.info("recall within the top 10 suggestions (test)", f"{100 * rl['recall_top10']:.0f}% vs {100 * rb['recall_top10']:.0f}%")
    ck.info("diagnostic: learned detector, in-plane position only (any depth): CPM / recall top-10",
            f"{r2d['cpm']:.3f} / {100 * r2d['recall_top10']:.0f}%")
    ck.info("table", os.path.join(out, "table_detection.md"))
    ck.finish(out)


if __name__ == "__main__":
    main()
