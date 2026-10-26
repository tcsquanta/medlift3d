"""Conformal predictive systems (``crepes``) on the per-nodule measurements of step 9.

Step 10 turns the ensemble spread into calibrated 90% intervals. This module goes one step further:
a *conformal predictive system* (CPS) gives every test nodule a whole calibrated distribution of its
true volume / mass, so we can state

* intervals at any level, with or without the ensemble spread as difficulty (normalised) and with
  separate calibration per nodule-size group (Mondrian: coverage holds *within* each group, not only
  on average);
* calibrated probabilities of clinical decisions: the volume categories of the European Position Statement
  on lung cancer screening (EUPS, Oudkerk et al. 2017: < 100 / 100-300 / >= 300 mm3, the categories the demo
  shows), compared with the uncalibrated ensemble vote.

Everything works in log space (errors are multiplicative): y = log(truth), y_hat = log(estimate),
calibration scores (y - y_hat) / sigma on the *cal* patients, evaluation on the *test* patients.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

VOLUME_THRESHOLDS_MM3 = (100.0, 300.0)          # EUPS (Oudkerk 2017), solid nodules: < 100 / 100-300 / >= 300 mm3
CATEGORIES = ("< 100 mm3", "100-300 mm3", ">= 300 mm3")
MIN_PRED = 1e-3                                 # a failed measurement (0) is kept as a tiny estimate, like step 10


def eq_diameter_mm(volume_mm3):
    return (6.0 * np.asarray(volume_mm3, float) / math.pi) ** (1.0 / 3.0)


def measurements(rows: List[Dict], method: str, qty: str) -> Dict[str, np.ndarray]:
    """Arrays for one method and quantity ('volume' | 'mass'): estimate, log-spread of the ensemble,
    truth and the member values (for the ensemble-vote baseline)."""
    key, skey = ("volume_mm3", "volume_log_std") if qty == "volume" else ("mass_mg", "mass_log_std")
    P, S, T, M, R = [], [], [], [], []
    for r in rows:
        e = r["methods"].get(method)
        t = r["gt"]["volume_mm3" if qty == "volume" else "mass_mg"]
        if e is None or not (np.isfinite(e[key]) and np.isfinite(t) and t > 0):
            continue
        P.append(max(float(e[key]), MIN_PRED))
        S.append(float(e[skey]) if np.isfinite(e[skey]) else 0.0)
        T.append(float(t))
        M.append(np.array([max(float(m[key]), 0.0) for m in e.get("members", [])] or [e[key]], float))
        R.append(r)
    return {"pred": np.array(P), "sigma": np.array(S), "truth": np.array(T), "members": M, "rows": R}


def size_groups(log_pred_cal, log_pred_test, n_groups: int):
    """Mondrian groups by the *estimated* size (known at prediction time), equal-count on cal.
    Returns (bins_cal, bins_test, inner group edges in the original units)."""
    from crepes.extras import binning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b_cal, thr = binning(np.asarray(log_pred_cal, float), bins=n_groups)
        b_test = binning(np.asarray(log_pred_test, float), bins=thr)
    return np.asarray(b_cal), np.asarray(b_test), [float(math.exp(t)) for t in thr if np.isfinite(t)]


class LogCPS:
    """A crepes ConformalPredictiveSystem in log space, optionally normalised by the ensemble spread
    (sigma = log-std + floor) and / or Mondrian by size group."""

    def __init__(self, normalised: bool = False, sigma_floor: float = 0.02):
        self.normalised = normalised
        self.floor = float(sigma_floor)
        self.cps = None
        self.n_cal: Dict[int, int] = {}

    def _sig(self, sigma):
        return (np.asarray(sigma, float) + self.floor) if self.normalised else None

    def fit(self, pred, sigma, truth, bins=None) -> "LogCPS":
        from crepes import ConformalPredictiveSystem

        res = np.log(np.asarray(truth, float)) - np.log(np.asarray(pred, float))
        self.cps = ConformalPredictiveSystem().fit(res, sigmas=self._sig(sigma), bins=bins)
        b = np.zeros(len(res), int) if bins is None else np.asarray(bins)
        self.n_cal = {int(g): int(np.sum(b == g)) for g in np.unique(b)}
        return self

    def interval(self, pred, sigma, alpha: float = 0.1, bins=None):
        """Central (1 - alpha) interval in original units; (0, inf) where calibration is too small."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iv = self.cps.predict(np.log(np.asarray(pred, float)), sigmas=self._sig(sigma), bins=bins,
                                  lower_percentiles=[100 * alpha / 2], higher_percentiles=[100 * (1 - alpha / 2)])
        iv = np.atleast_2d(iv)
        with np.errstate(over="ignore"):
            return np.exp(iv[:, 0]), np.exp(iv[:, 1])

    def cdf(self, pred, sigma, value, bins=None) -> np.ndarray:
        """Calibrated P(true value <= value) per nodule. ``value``: one threshold for all nodules or one per
        nodule. crepes' unsmoothed value is the upper conformal p-value (k + 1) / (n + 1) (k calibration
        scores below); its smoothed value adds a random tau in [0, 1) instead of the 1. We use the expected
        smoothed value, tau = 1/2: deterministic, valid on average, and reaching 1/(2(n+1)) and
        1 - 1/(2(n+1)) at both ends instead of stopping at 1/(n+1) on one side."""
        lp = np.log(np.asarray(pred, float))
        y = np.log(np.broadcast_to(np.asarray(value, float), lp.shape))
        upper = np.asarray(self.cps.predict(lp, sigmas=self._sig(sigma), bins=bins, y=y, smoothing=False), float)
        g = np.zeros(len(lp), int) if bins is None else np.asarray(bins)
        n = np.array([self.n_cal.get(int(b), 0) for b in g], float)
        return np.clip(upper - 0.5 / (n + 1.0), 0.0, 1.0)

    def cdf_curve(self, pred: float, sigma: float, grid, bin_=None) -> np.ndarray:
        """The calibrated distribution function of one nodule on a grid of values."""
        n = len(grid)
        return self.cdf(np.full(n, pred), np.full(n, sigma), grid, None if bin_ is None else np.full(n, bin_))

    def crps(self, pred, sigma, truth, bins=None) -> float:
        """Continuous ranked probability score of the calibrated distributions, log space (lower = better)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = self.cps.evaluate(np.log(np.asarray(pred, float)), np.log(np.asarray(truth, float)),
                                  sigmas=self._sig(sigma), bins=bins, metrics=["CRPS"])
        return float(m["CRPS"])


def category_probs(cdf_at_thresholds: Sequence[np.ndarray]) -> np.ndarray:
    """[n, 3] probabilities of the three volume categories from P(V <= 100), P(V <= 300)."""
    lo, hi = (np.clip(np.asarray(c, float), 0, 1) for c in cdf_at_thresholds)
    hi = np.maximum(hi, lo)
    return np.stack([lo, hi - lo, 1.0 - hi], 1)


def category_of(volume_mm3) -> np.ndarray:
    return np.digitize(np.asarray(volume_mm3, float), VOLUME_THRESHOLDS_MM3)


def category_sets(lo, hi) -> List[List[int]]:
    """Categories that overlap each interval [lo, hi]: the set the true category lies in (at the
    interval's confidence)."""
    a, b = category_of(lo), category_of(hi)
    return [list(range(int(x), int(y) + 1)) for x, y in zip(a, b)]


def vote_probability(members: List[np.ndarray], threshold: float) -> np.ndarray:
    """Uncalibrated baseline: share of ensemble members above the threshold."""
    return np.array([float(np.mean(m > threshold)) for m in members])


def lognormal_probability(pred, sigma, threshold: float, floor: float = 0.02) -> np.ndarray:
    """Uncalibrated baseline: the ensemble spread read as a log-normal error, P(V > threshold)."""
    from scipy.stats import norm

    z = (math.log(threshold) - np.log(np.asarray(pred, float))) / (np.asarray(sigma, float) + floor)
    return 1.0 - norm.cdf(z)


def brier(prob, event) -> float:
    return float(np.mean((np.asarray(prob, float) - np.asarray(event, float)) ** 2))


def ece(prob, event, n_bins: int = 5) -> float:
    """Expected calibration error with equal-width bins (few test nodules: 5 bins)."""
    p, e = np.asarray(prob, float), np.asarray(event, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(p[idx == b].mean() - e[idx == b].mean()) * np.mean(idx == b)
                     for b in range(n_bins) if np.any(idx == b)))


def reliability_curve(prob, event, n_bins: int = 5):
    """(mean predicted probability, observed frequency, count) per equal-width bin."""
    p, e = np.asarray(prob, float), np.asarray(event, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = [(float(p[idx == b].mean()), float(e[idx == b].mean()), int(np.sum(idx == b)))
           for b in range(n_bins) if np.any(idx == b)]
    return [list(v) for v in zip(*out)] if out else [[], [], []]


def coverage_by_group(lo, hi, truth, groups) -> Dict[int, Optional[float]]:
    lo, hi, truth, groups = (np.asarray(a) for a in (lo, hi, truth, groups))
    return {int(g): float(np.mean((truth[groups == g] >= lo[groups == g]) & (truth[groups == g] <= hi[groups == g])))
            for g in np.unique(groups)}


def category_probs_from_calibration(cal: Dict, pred: float, sigma: float) -> np.ndarray:
    """Calibrated probabilities of the three volume categories for one new nodule, from the calibration
    step 16 saves (``cps_calibration.json``; pooled normalised CPS). Plain numpy -- the demo needs no
    ``crepes`` -- and identical to :meth:`LogCPS.cdf` (expected smoothed p-value)."""
    floor = float(cal["sigma_floor"])
    scores = np.asarray(cal["residuals"], float) / (np.asarray(cal["sigma"], float) + floor)
    n = len(scores)
    s = [(math.log(t) - math.log(max(float(pred), MIN_PRED))) / (float(sigma) + floor) for t in VOLUME_THRESHOLDS_MM3]
    cdf = [np.array([(np.sum(scores <= v) + 0.5) / (n + 1.0)]) for v in s]
    return category_probs(cdf)[0]
