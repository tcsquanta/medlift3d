"""Read the training logs of step 8 (diffusion) and step 14 (detector) and judge the fit:
still improving (under-trained), converged, overfitting or not learning.

The logs are the console output written to ``logs/08_train.log`` / ``logs/14_detector.log`` by
``scripts/runpod_run.sh`` (any saved console output with the same lines works, e.g. from Kaggle).
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

_STEP = re.compile(r"^step (\d+)/\d+\s+loss ([\d.eE+-]+)")
_VAL = re.compile(r"\[val\] PSNR sample vs CT ([\d.-]+) dB \| condition vs CT ([\d.-]+) dB"
                  r"(?:\s+\|\| nodule box: sample ([\d.-]+) dB \| condition ([\d.-]+) dB)?")
_FIT = re.compile(r"\[fit\] denoising loss .*?train pairs ([\d.]+)(?: \| held-out pairs ([\d.]+))?")
_DVAL = re.compile(r"\[val\] step (\d+): CPM ([\d.]+), sensitivity at 2 FP/scan ([\d.]+)")


def parse_diffusion_log(text: str) -> Dict:
    """Training-loss points and one entry per validation check. A log holding several runs (a fresh
    start after an earlier one) keeps only the last run; a resumed run continues the same curve."""
    steps, losses, checks, last = [], [], [], 0
    for line in text.splitlines():
        m = _STEP.search(line)
        if m:
            s = int(m.group(1))
            if s < last:                                       # a new run from step 0: forget the old one
                steps, losses, checks = [], [], []
            last = s
            steps.append(s)
            losses.append(float(m.group(2)))
            continue
        m = _VAL.search(line)
        if m:
            ps, pc = float(m.group(1)), float(m.group(2))
            bs = float(m.group(3)) if m.group(3) else float("nan")
            bc = float(m.group(4)) if m.group(4) else float("nan")
            gain = bs - bc if np.isfinite(bs) else ps - pc
            checks.append({"step": last, "gain": gain, "fit_train": float("nan"), "fit_heldout": float("nan")})
            continue
        m = _FIT.search(line)
        if m and checks:
            checks[-1]["fit_train"] = float(m.group(1))
            checks[-1]["fit_heldout"] = float(m.group(2)) if m.group(2) else float("nan")
    return {"steps": steps, "loss": losses, "checks": checks}


def parse_detector_log(text: str) -> Dict:
    steps, losses, checks, last = [], [], [], 0
    for line in text.splitlines():
        m = _STEP.search(line)
        if m:
            s = int(m.group(1))
            if s < last:
                steps, losses, checks = [], [], []
            last = s
            steps.append(s)
            losses.append(float(m.group(2)))
            continue
        m = _DVAL.search(line)
        if m:
            checks.append({"step": int(m.group(1)), "cpm": float(m.group(2)), "sens2": float(m.group(3))})
    return {"steps": steps, "loss": losses, "checks": checks}


def _loss_trend(loss: List[float]) -> float:
    """Relative change of the training loss over the last 20% of the run (negative = still falling)."""
    if len(loss) < 10:
        return float("nan")
    k = max(2, len(loss) // 5)
    a, b = float(np.mean(loss[-2 * k:-k])), float(np.mean(loss[-k:]))
    return (b - a) / max(a, 1e-12)


def diffusion_verdict(d: Dict) -> Dict:
    """Verdict + reason + what to do, from the validation checks (every 2,000 steps by default)."""
    c = d["checks"]
    out = {"n_checks": len(c), "loss_trend": _loss_trend(d["loss"])}
    if len(c) < 3:
        return dict(out, verdict="TOO EARLY", reason=f"{len(c)} validation checks; at least 3 are needed.",
                    action="Let it train (one check per 2,000 steps).")
    gain = np.array([x["gain"] for x in c])
    ft = np.array([x["fit_train"] for x in c])
    fv = np.array([x["fit_heldout"] for x in c])
    ib = int(np.argmax(gain))
    out.update(best_step=c[ib]["step"], best_gain=float(gain[ib]), last_gain=float(gain[-1]))
    has_fit = np.isfinite(fv).sum() >= 3 and np.isfinite(ft).sum() >= 3
    if has_fit:
        iv = int(np.nanargmin(fv))
        out.update(heldout_min_step=c[iv]["step"], heldout_last=float(fv[-1]), heldout_min=float(fv[iv]),
                   gap=float(fv[-1] / ft[-1]))
    n = len(c)
    if n >= 5 and gain.max() <= 0.2:
        return dict(out, verdict="NOT LEARNING",
                    reason=f"after {n} checks the samples are still no better than the DTS input in the nodule box "
                           f"(best gain {gain.max():+.2f} dB).",
                    action="Check outputs/08_diffusion/val_step*.png and the training pairs (step 7); do not just "
                           "train longer. A loss that does not fall at all points at the data or the learning rate.")
    heldout_up = has_fit and iv <= n - 3 and fv[-1] > 1.03 * fv[iv] and ft[-1] < ft[iv]
    gain_down = ib <= n - 3 and gain[-1] < gain[ib] - 0.3
    if heldout_up or (gain_down and not has_fit):
        why = (f"the held-out denoising loss was lowest at step {c[iv]['step']} ({fv[iv]:.4f}) and has risen to "
               f"{fv[-1]:.4f} while the training-pair loss kept falling" if heldout_up else
               f"the nodule-box gain peaked at step {c[ib]['step']} ({gain[ib]:+.2f} dB) and is now {gain[-1]:+.2f} dB")
        return dict(out, verdict="OVERFITTING", reason=why + ".",
                    action=f"best.pt (chosen by the nodule-box gain) is the step-{c[ib]['step']} model, so the results use the "
                           "peak, not the last step. "
                           "Do not train longer. To generalise better: more patients, more augmentation "
                           "(step 7 --jitter), or retrain from scratch with fewer hours.")
    rising_gain = ib >= n - 2 and gain[-1] - gain[max(0, n - 4)] > 0.2
    rising_fit = has_fit and fv[-1] < 0.98 * fv[max(0, n - 3)]
    if rising_gain or rising_fit:
        why = []
        if rising_gain:
            why.append(f"the nodule-box gain is still rising ({gain[max(0, n - 4)]:+.2f} -> {gain[-1]:+.2f} dB over "
                       f"the last checks)")
        if rising_fit:
            why.append(f"the held-out denoising loss is still falling ({fv[max(0, n - 3)]:.4f} -> {fv[-1]:.4f})")
        return dict(out, verdict="STILL IMPROVING (under-trained)", reason="; ".join(why) + ".",
                    action="More training should help: MORE_TRAIN_HOURS=3 bash scripts/runpod_run.sh "
                           "(continues from last.pt; best.pt is kept unless beaten).")
    return dict(out, verdict="CONVERGED (plateau)",
                reason=f"gain and held-out loss are flat over the last checks (best gain {gain[ib]:+.2f} dB at step "
                       f"{c[ib]['step']}).",
                action="More hours will add little. Bigger gains need more data or a model / data change.")


def detector_verdict(d: Dict) -> Dict:
    c = d["checks"]
    out = {"n_checks": len(c), "loss_trend": _loss_trend(d["loss"])}
    if len(c) < 3:
        return dict(out, verdict="TOO EARLY", reason=f"{len(c)} validation checks; at least 3 are needed.",
                    action="Let it train (one check per 500 steps).")
    cpm = np.array([x["cpm"] for x in c])
    ib, n = int(np.argmax(cpm)), len(c)
    out.update(best_step=c[ib]["step"], best_cpm=float(cpm[ib]), last_cpm=float(cpm[-1]))
    if cpm.max() < 0.05:
        return dict(out, verdict="NOT LEARNING", reason=f"validation CPM never above 0.05 (best {cpm.max():.3f}).",
                    action="Check the labels / heatmap targets (tests/test_detector.py) and the scans before training longer.")
    if ib <= n - 3 and cpm[-1] < cpm[ib] - 0.03:
        return dict(out, verdict="OVERFITTING",
                    reason=f"validation CPM peaked at step {c[ib]['step']} ({cpm[ib]:.3f}) and fell to {cpm[-1]:.3f} "
                           f"while the training loss kept falling.",
                    action=f"best.pt is the step-{c[ib]['step']} model, so the test table uses the peak. To improve: "
                           "more patients, or fewer steps (--steps) so the learning rate decays around the peak.")
    if ib >= n - 2 and cpm[-1] - cpm[max(0, n - 4)] > 0.02:
        return dict(out, verdict="STILL IMPROVING (under-trained)",
                    reason=f"validation CPM still rising at the end ({cpm[max(0, n - 4)]:.3f} -> {cpm[-1]:.3f}).",
                    action="Retrain with more steps: rm work/models/detector/.done, then "
                           "python scripts/14_train_detector.py --steps 12000.")
    return dict(out, verdict="CONVERGED (plateau)",
                reason=f"validation CPM flat over the last checks (best {cpm[ib]:.3f} at step {c[ib]['step']}).",
                action="More steps will add little; more patients would.")
