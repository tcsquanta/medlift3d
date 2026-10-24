"""Conformal coverage, sparsification, doubling times and virtual growth."""
import math

import numpy as np

from medlift3d.growth import (FAST, INDET, NOT_MEASURABLE, SLOW, doubling_time, growth_category, scale_for_doubling,
                              virtual_growth)
from medlift3d.phantom import sphere_volume
from medlift3d.uncertainty import ConformalCalibrator, coverage, relative_width, sparsification


def _synthetic(n, rng, bias=0.0):
    truth = np.exp(rng.uniform(3, 7, n))
    sigma = rng.uniform(0.05, 0.4, n)                      # per-item predicted log-std
    err = bias + sigma * rng.standard_normal(n) * 1.5      # true errors scale with sigma (miscalibrated x1.5)
    pred = truth * np.exp(-err)
    return pred, sigma, truth


def test_conformal_coverage_both_modes():
    rng = np.random.default_rng(0)
    for mode in ("symmetric", "asymmetric"):
        covs = []
        for rep in range(20):
            pc, sc, tc = _synthetic(150, rng, bias=0.3)
            pt, st, tt = _synthetic(1000, rng, bias=0.3)
            cal = ConformalCalibrator(alpha=0.1, mode=mode).fit(pc, sc, tc)
            lo, hi = cal.interval(pt, st)
            covs.append(coverage(lo, hi, tt))
        assert 0.87 <= np.mean(covs) <= 0.95, (mode, np.mean(covs))


def test_asymmetric_is_narrower_under_bias():
    """With a systematic bias, asymmetric intervals shift instead of widening (log-width)."""
    rng = np.random.default_rng(1)

    def data(n):
        truth = np.exp(rng.uniform(3, 7, n))
        sigma = np.full(n, 0.2)
        pred = truth * np.exp(-(0.5 + 0.2 * rng.standard_normal(n)))     # biased by exp(-0.5)
        return pred, sigma, truth
    pc, sc, tc = data(400)
    pt, st, tt = data(2000)
    w, cov = {}, {}
    for mode in ("symmetric", "asymmetric"):
        cal = ConformalCalibrator(alpha=0.1, mode=mode).fit(pc, sc, tc)
        lo, hi = cal.interval(pt, st)
        w[mode] = float(np.median(np.log(hi / lo)))
        cov[mode] = coverage(lo, hi, tt)
    assert w["asymmetric"] < 0.75 * w["symmetric"], w
    assert min(cov.values()) > 0.85, cov


def test_sparsification_oracle_vs_random():
    rng = np.random.default_rng(2)
    err = rng.standard_normal(5000)
    good = sparsification(err, np.abs(err))
    bad = sparsification(err, rng.random(5000))
    assert good["ause"] < 0.02 and bad["ause"] > good["ause"] + 0.1


def test_doubling_time_and_categories():
    assert abs(doubling_time(100.0, 200.0, 365.0) - 365.0) < 1e-9
    assert doubling_time(100.0, 90.0, 365.0) == float("inf")
    assert growth_category(200) == FAST and growth_category(500) == INDET and growth_category(900) == SLOW
    # a failed measurement (stored as ~1e-6) must not look like explosive growth
    assert math.isnan(doubling_time(1e-6, 500.0, 180.0)) and math.isnan(doubling_time(100.0, 0.0, 180.0))
    assert growth_category(float("nan")) == NOT_MEASURABLE and growth_category(float("inf")) == SLOW
    s = scale_for_doubling(180.0, 300.0)
    assert abs(s ** 3 - 2 ** (180.0 / 300.0)) < 1e-9


def test_virtual_growth_scales_volume_and_mass():
    r, hu, lung = 5.0, 20.0, -850.0
    vol, vg = sphere_volume(hu, r, spacing=0.5, size_mm=48.0, background_hu=lung)
    x, y, z = vg.axes()
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    mask = X ** 2 + Y ** 2 + Z ** 2 <= r * r
    s = 1.3
    new, gmask = virtual_growth(vol, vg, mask, s, lung)
    v_ratio = gmask.sum() / mask.sum()
    m_ratio = (new - lung).clip(0).sum() / (vol - lung).clip(0).sum()
    assert abs(v_ratio - s ** 3) / s ** 3 < 0.08, v_ratio
    assert abs(m_ratio - s ** 3) / s ** 3 < 0.08, m_ratio


def test_virtual_growth_exact_on_thick_slices():
    """Coarse (0.7 x 0.7 x 2.5 mm) CT grid: added mass is still (s^3 - 1) x nodule mass."""
    from medlift3d.geometry import VolumeGeometry

    vg = VolumeGeometry.centered((20, 40, 40), (0.7, 0.7, 2.5))
    x, y, z = vg.axes()
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    hu = (-850 + 870 / (1 + np.exp((r - 4.0) / 0.4))).astype(np.float32)
    mask = r <= 4.0
    s = 1.149
    new, gmask, info = virtual_growth(hu, vg, mask, s, -850.0, return_info=True)
    assert abs(info["added_mass_mg"] / ((s ** 3 - 1) * info["excess_mass_mg"]) - 1) < 0.03
    assert abs(gmask.sum() / mask.sum() - s ** 3) / s ** 3 < 0.2


def test_ensemble_spread_ignores_members_that_found_nothing():
    from medlift3d.uncertainty import ensemble_stats

    st = ensemble_stats([100.0, 110.0, 90.0, 1e-6, 1e-6], min_value=1e-2)
    assert st["n_found"] == 3 and st["log_std"] < 0.2          # not inflated by the two failed members
    assert ensemble_stats([1e-6, 1e-6, 50.0], min_value=1e-2)["median"] < 1e-3   # mostly failed -> not found
