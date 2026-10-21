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
