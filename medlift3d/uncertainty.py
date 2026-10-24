"""Uncertainty: ensemble statistics, split-conformal intervals, error-ranking metrics.

Raw ensemble spread is *not* trusted as an error estimate. It is only used as
a per-nodule scale sigma, and split-conformal calibration on held-out nodules
turns it into intervals with a finite-sample coverage guarantee:

    P(true value in interval) >= 1 - alpha   (exchangeable nodules)

We work in log space (volume / mass errors are multiplicative). The
asymmetric variant uses separate lower/upper quantiles of the signed scores,
so a systematic bias (e.g. DTS over-estimating volume) gives a shifted
interval rather than an inflated one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def ensemble_stats(values: Sequence[float], min_value: float = 0.0) -> Dict[str, float]:
    """Ensemble summary. The median uses every member (a member that found nothing pulls it
    towards 0); the log-spread uses only members above ``min_value`` (those that found a nodule),
    so failed members do not blow up the interval; ``n_found`` counts them."""
    v = np.asarray(values, dtype=float)
    pos = v[v > min_value]
    lv = np.log(pos) if len(pos) else np.array([np.nan])
    return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "log_mean": float(lv.mean()), "log_std": float(lv.std(ddof=1)) if len(lv) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v)), "n_found": int(len(pos))}


def _order_stat(sorted_scores: np.ndarray, k: int) -> float:
    """k-th smallest (1-based); +inf / -inf outside the range."""
    n = len(sorted_scores)
    if k > n:
        return float("inf")
    if k < 1:
        return float("-inf")
    return float(sorted_scores[k - 1])


@dataclass
class ConformalCalibrator:
    alpha: float = 0.1
    mode: str = "asymmetric"                  # 'asymmetric' | 'symmetric'
    sigma_floor: float = 0.02
    q_lo: float = float("nan")
    q_hi: float = float("nan")
    n_cal: int = 0
    scores: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, pred: Sequence[float], sigma_log: Sequence[float], truth: Sequence[float]) -> "ConformalCalibrator":
        pred = np.asarray(pred, float)
        truth = np.asarray(truth, float)
        sig = np.asarray(sigma_log, float) + self.sigma_floor
        ok = (pred > 0) & (truth > 0) & np.isfinite(sig)
        e = (np.log(truth[ok]) - np.log(pred[ok])) / sig[ok]
        n = len(e)
        self.n_cal = n
        if n == 0:
            raise ValueError("no valid calibration points")
        if self.mode == "symmetric":
            s = np.sort(np.abs(e))
            q = _order_stat(s, int(math.ceil((n + 1) * (1 - self.alpha))))
            self.q_lo, self.q_hi = -q, q
        else:
            s = np.sort(e)
            self.q_hi = _order_stat(s, int(math.ceil((n + 1) * (1 - self.alpha / 2))))
            self.q_lo = _order_stat(s, int(math.floor((n + 1) * (self.alpha / 2))))
        self.scores = e
        return self

    def interval(self, pred, sigma_log) -> Tuple[np.ndarray, np.ndarray]:
        pred = np.asarray(pred, float)
        sig = np.asarray(sigma_log, float) + self.sigma_floor
        lp = np.log(np.clip(pred, 1e-9, None))
        with np.errstate(over="ignore"):
            lo = np.exp(lp + self.q_lo * sig)
            hi = np.exp(lp + self.q_hi * sig)
        return lo, hi

    def to_dict(self) -> Dict:
        return {"alpha": self.alpha, "mode": self.mode, "sigma_floor": self.sigma_floor,
                "q_lo": self.q_lo, "q_hi": self.q_hi, "n_cal": self.n_cal}


def coverage(lo, hi, truth) -> float:
    lo, hi, truth = (np.asarray(a, float) for a in (lo, hi, truth))
    return float(np.mean((truth >= lo) & (truth <= hi)))


def relative_width(lo, hi, pred) -> float:
    lo, hi, pred = (np.asarray(a, float) for a in (lo, hi, pred))
    return float(np.median((hi - lo) / np.clip(pred, 1e-9, None)))


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


def sparsification(errors, uncert, n_steps: int = 20) -> Dict[str, np.ndarray]:
    """Sparsification curves: RMSE of the remaining voxels after removing the
    most uncertain fraction (by predicted uncertainty) vs by true error (oracle).
    AUSE = area between the two curves (lower = uncertainty ranks errors better)."""
    e = np.asarray(errors, float).ravel()
    u = np.asarray(uncert, float).ravel()
    n = len(e)
    fr = np.linspace(0.0, 0.95, n_steps)
    by_u = np.argsort(-u)
    by_e = np.argsort(-np.abs(e))
    cu, co = [], []
    for f in fr:
        k = int(f * n)
        cu.append(np.sqrt(np.mean(e[by_u[k:]] ** 2)))
        co.append(np.sqrt(np.mean(e[by_e[k:]] ** 2)))
    cu = np.asarray(cu)
    co = np.asarray(co)
    norm = cu[0] if cu[0] > 0 else 1.0
    ause = float(np.trapezoid((cu - co) / norm, fr)) if hasattr(np, "trapezoid") else float(np.trapz((cu - co) / norm, fr))
    return {"fractions": fr, "by_uncertainty": cu, "oracle": co, "ause": ause}
