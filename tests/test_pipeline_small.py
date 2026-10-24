"""Small end-to-end run on the synthetic phantom (CPU, < ~2 min)."""
import numpy as np
import pytest
import torch

from medlift3d.config import load_config
from medlift3d.data.cases import case_from_phantom
from medlift3d.phantom import NoduleSpec, make_chest_phantom
from medlift3d.pipeline import ground_truth_for_roi, prepare_case_context, process_nodule
from medlift3d.roi import ROI


def small_cfg():
    return load_config(overrides={
        "geometry": {"det_shape": [150, 150], "det_pitch": [2.4, 2.4]},
        "simulation": {"step_mm": 1.5, "i0": 1e5},
        "sirt": {"voxel_mm": [4.0, 6.0, 4.0], "det_binning": 2, "n_iter": 15, "step_mm": 2.0},
        "roi": {"size_vox": [32, 88, 32], "voxel_mm": [2.0, 3.0, 2.0], "box_mm": [30.0, 45.0, 30.0],
                "sirt_iter": 30, "step_mm": 1.5},
        "gaussians": {"fit_iter": 25, "refine_iter": 25, "max_gaussians": 3000},
        "pipeline": {"n_members": 2},
    })


@pytest.fixture(scope="module")
def small_case():
    ph = make_chest_phantom(spacing=(3.0, 3.0, 3.0), fov_mm=(360, 260, 280), n_vessels=10, seed=3,
                            nodules=[NoduleSpec((-70.0, -10.0, 20.0), (8.0, 8.0, 8.0), 20.0, "big")])
    return case_from_phantom(ph, "ph_small")


def test_small_pipeline_runs_and_mass_is_close(small_case):
    cfg = small_cfg()
    torch.manual_seed(0)
    ctx = prepare_case_context(small_case, cfg, device="cpu", seed=0)
    assert ctx.sirt_history[-1] < ctx.sirt_history[0]               # SIRT reduces the residual
    nod = small_case.nodules[0]
    roi = ROI.from_config(nod.center_xyz, cfg["roi"], ctx.geom)
    gt = ground_truth_for_roi(small_case, 0, roi, cfg["measure"])
    res = process_nodule(ctx, nod.center_xyz, cfg, methods=["sirt_coarse", "roi_sirt", "roi_gs"], gt=gt)
    for name in ("sirt_coarse", "roi_sirt", "roi_gs"):
        e = res["methods"][name]
        assert np.isfinite(e["mass_mg"]) and np.isfinite(e["volume_mm3"])
    m_gt = res["gt"]["mass_mg"]
    v_gt = res["gt"]["volume_mm3"]
    m_roi = res["methods"]["roi_sirt"]["mass_mg"]
    v_roi = res["methods"]["roi_sirt"]["volume_mm3"]
    # physics-only reconstruction: volume is inflated by the depth smear (missing wedge),
    # mass is much less affected (it is measured by every view) -- hypothesis H1 on one nodule
    assert v_roi > 1.3 * v_gt, (v_roi, v_gt)
    assert abs(m_roi / m_gt - 1) < abs(v_roi / v_gt - 1), (m_roi, m_gt, v_roi, v_gt)
    assert abs(m_roi / m_gt - 1) < 0.5, (m_roi, m_gt)
    assert res["roi"]["n_rays"] > 1000


def test_roi_data_match_true_column_with_air_margins():
    """Regression (real LIDC chests): the CT grid reaches far beyond the patient in depth --
    in front of the detector and behind the back. Unconstrained DTS smear there was subtracted
    as background and the prism lost ~1/3 of its attenuation. With the patient depth support
    the prism data (measured - background) must equal the projection of the true prism."""
    from medlift3d.physics import hu_to_mu
    from medlift3d.pipeline import build_problem
    from medlift3d.roi import resample_to_grid

    cfg = small_cfg()
    ph = make_chest_phantom(spacing=(3.0, 3.0, 3.0), fov_mm=(360, 420, 280), n_vessels=10, seed=4,
                            nodules=[NoduleSpec((-70.0, -10.0, 20.0), (8.0, 8.0, 8.0), 20.0, "big")])
    case = case_from_phantom(ph, "ph_margins")
    ctx = prepare_case_context(case, cfg, device="cpu", seed=0)
    lo, hi = ctx.geom.depth_range()
    ct_y = case.vgeom.axes()[1]
    assert ((ct_y < lo) | (ct_y > hi)).sum() >= 10               # the CT grid really has air margins in depth
    cy, sp = ctx.coarse_vg.axes()[1], ctx.coarse_vg.spacing[1]
    assert cy.min() - 0.5 * sp >= lo - sp and cy.max() + 0.5 * sp <= hi + sp   # coarse grid cropped to the patient
    prob = build_problem(ctx, case.nodules[0].center_xyz, cfg)
    x_true = torch.as_tensor(hu_to_mu(resample_to_grid(case.hu, case.vgeom, prob.vg, order=1), 0.02),
                             dtype=torch.float32)
    with torch.no_grad():
        ratio = float(prob.residual.mean() / prob.forward(x_true).mean())
    assert 0.9 < ratio < 1.1, ratio


def test_nodule_spread_depth_smear_shrinks_with_sweep(small_case):
    """Artifact spread function (step 13): the nodule alone, reconstructed with the prism SIRT, is
    smeared in depth, less so with a wider sweep, and keeps its mass (measured by every view)."""
    from medlift3d.measure import depth_smear_metrics
    from medlift3d.physics import hu_to_mu
    from medlift3d.pipeline import CaseContext, build_problem, coarse_grid, nodule_spread
    from medlift3d.roi import resample_to_grid
    from medlift3d.simulate import geometry_for_volume

    cfg = small_cfg()
    nod = small_case.nodules[0]
    smear = {}
    for sweep in (30.0, 60.0):
        geom = geometry_for_volume(small_case.hu, small_case.vgeom, dict(cfg["geometry"], sweep_deg=sweep))
        cvg = coarse_grid(small_case.vgeom, cfg["sirt"]["voxel_mm"], geom.depth_range())
        cmu = torch.as_tensor(hu_to_mu(resample_to_grid(small_case.hu, small_case.vgeom, cvg), 0.02))
        ctx = CaseContext("ph", geom, torch.zeros((geom.n_views, *geom.det_shape)), cmu, cvg)  # rays only
        prob = build_problem(ctx, nod.center_xyz, cfg)
        rec, truth = nodule_spread(small_case, 0, prob, cfg["measure"], n_iter=30)
        assert abs(rec.sum() / truth.sum() - 1) < 0.2, (rec.sum(), truth.sum())
        m, t = (depth_smear_metrics(v, prob.vg, nod.center_xyz, 0.0) for v in (rec, truth))
        assert t["depth_fwhm_mm"] / t["inplane_fwhm_mm"] < 1.5                 # the sphere itself
        smear[sweep] = m["depth_fwhm_mm"] / m["inplane_fwhm_mm"]
    assert smear[30.0] > 1.5 and smear[60.0] < smear[30.0], smear
