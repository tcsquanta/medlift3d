"""Conformal predictive systems on per-nodule results (medlift3d.uq, step 16)."""
import numpy as np
import pytest

pytest.importorskip("crepes")

from medlift3d.uq import (LogCPS, brier, category_of, category_probs, category_sets, coverage_by_group, ece,
                          measurements, size_groups, vote_probability)


def fake_rows(n, split, seed, method="diffusion_gs", small_err=0.25):
    """Nodules whose log error has spread sigma (known to the ensemble) plus extra error for small nodules
    that the ensemble does not see (``small_err``)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        d = rng.uniform(4, 25)
        v = np.pi / 6 * d ** 3
        sig = rng.uniform(0.05, 0.4)
        small = small_err if d < 8 else 0.0
        err = 0.1 + sig * rng.normal() + small * rng.normal()
        est = v * np.exp(-err)
        mem = est * np.exp(sig * rng.normal(size=8))
        rows.append({"split": split, "patient_id": f"{split}{i}", "case_id": f"{split}{i}_c", "nid": f"{split}{i}_n",
                     "gt": {"volume_mm3": v, "mass_mg": v * 0.9},
                     "methods": {method: {"volume_mm3": est, "volume_log_std": sig, "mass_mg": est * 0.9,
                                          "mass_log_std": sig,
                                          "members": [{"volume_mm3": m, "mass_mg": 0.9 * m} for m in mem]}}})
    return rows


def test_measurements_and_cps_coverage():
    C = measurements(fake_rows(400, "cal", 0, small_err=0.0), "diffusion_gs", "volume")
    T = measurements(fake_rows(2000, "test", 1, small_err=0.0), "diffusion_gs", "volume")
    assert len(C["pred"]) == 400 and len(T["members"][0]) == 8
    covs, widths = {}, {}
    for norm in (False, True):
        m = LogCPS(norm).fit(C["pred"], C["sigma"], C["truth"])
        lo, hi = m.interval(T["pred"], T["sigma"], 0.1)
        covs[norm] = np.mean((T["truth"] >= lo) & (T["truth"] <= hi))
        widths[norm] = np.median((hi - lo) / T["pred"])
    assert all(abs(c - 0.9) < 0.03 for c in covs.values())            # marginal guarantee, both variants
    assert widths[True] < widths[False]                                # informative spread -> narrower


def test_mondrian_fixes_small_nodule_undercoverage():
    C = measurements(fake_rows(600, "cal", 2), "diffusion_gs", "volume")
    T = measurements(fake_rows(3000, "test", 3), "diffusion_gs", "volume")
    gc, gt, edges = size_groups(np.log(C["pred"]), np.log(T["pred"]), 3)
    assert len(edges) == 2 and set(np.unique(gc)) == {0, 1, 2}
    plain = LogCPS(True).fit(C["pred"], C["sigma"], C["truth"])
    mond = LogCPS(True).fit(C["pred"], C["sigma"], C["truth"], bins=gc)
    cp = coverage_by_group(*plain.interval(T["pred"], T["sigma"], 0.1), T["truth"], gt)
    cm = coverage_by_group(*mond.interval(T["pred"], T["sigma"], 0.1, bins=gt), T["truth"], gt)
    assert cp[0] < 0.86                                                # small group under-covered on its own
    assert min(cm.values()) > 0.86                                     # Mondrian: every group near 90%


def test_calibrated_probabilities_and_categories():
    C = measurements(fake_rows(500, "cal", 4), "diffusion_gs", "volume")
    T = measurements(fake_rows(3000, "test", 5), "diffusion_gs", "volume")
    m = LogCPS(True).fit(C["pred"], C["sigma"], C["truth"])
    p300 = 1 - m.cdf(T["pred"], T["sigma"], 300.0)
    ev = T["truth"] >= 300.0
    assert ece(p300, ev) < 0.05
    pv = vote_probability(T["members"], 300.0)                         # biased estimates: the vote is not calibrated
    assert brier(p300, ev) < brier(pv, ev)
    pr = category_probs([m.cdf(T["pred"], T["sigma"], t) for t in (100.0, 300.0)])
    assert np.allclose(pr.sum(1), 1) and (pr >= 0).all()
    curve = m.cdf_curve(T["pred"][0], T["sigma"][0], np.geomspace(1, 1e5, 50))
    assert np.all(np.diff(curve) >= -1e-9) and curve[0] < 0.05 and curve[-1] > 0.95


def test_category_helpers():
    assert list(category_of([50, 100, 150, 299, 300])) == [0, 1, 1, 1, 2]
    assert category_sets([50, 120, 50], [90, 250, 400]) == [[0], [1], [0, 1, 2]]


def test_numpy_category_probs_match_crepes():
    from medlift3d.uq import category_probs_from_calibration

    C = measurements(fake_rows(60, "cal", 6), "diffusion_gs", "volume")
    m = LogCPS(True, 0.02).fit(C["pred"], C["sigma"], C["truth"])
    cal = {"sigma_floor": 0.02, "residuals": (np.log(C["truth"]) - np.log(C["pred"])).tolist(), "sigma": C["sigma"].tolist()}
    for pred, sig in ((80.0, 0.1), (150.0, 0.3), (400.0, 0.05), (5000.0, 0.2)):
        ref = category_probs([m.cdf([pred], [sig], t) for t in (100.0, 300.0)])[0]
        assert np.allclose(category_probs_from_calibration(cal, pred, sig), ref, atol=1e-9)
