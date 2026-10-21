"""Configuration: one nested dict with defaults, optionally overridden by YAML.

Usage::

    from medlift3d.config import load_config
    cfg = load_config("configs/default.yaml")      # or load_config() for defaults
    cfg["roi"]["size_vox"]
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "paths": {
        "lidc_root": "data/LIDC-IDRI",      # DICOM root (NBIA download)
        "work": "work",                     # cases, projections, recons, results
        "outputs": "outputs",               # verification figures / tables
    },
    "physics": {
        "mu_water": 0.02,                   # mm^-1 at ~60-70 keV effective energy
    },
    "geometry": {                           # linear-sweep chest DTS
        "n_views": 15,
        "sweep_deg": 30.0,
        "sid": 1800.0,                      # source-to-detector distance (mm)
        "detector_to_iso": 150.0,           # used if not estimated from the body
        "det_shape": [536, 536],            # (rows along z, cols along x)
        "det_pitch": [0.8, 0.8],            # mm
    },
    "simulation": {
        "i0": 2.0e4,                        # incident photons / pixel / view (low dose)
        "electronic_sigma": 0.0,            # detector read noise (counts)
        "step_mm": 0.5,                     # ray-marching step for simulation
    },
    "sirt": {                               # whole-chest coarse reconstruction
        "voxel_mm": [2.5, 2.5, 2.5],        # (x, y, z)
        "det_binning": 3,
        "n_iter": 40,
        "relax": 1.0,
        "step_mm": 1.25,
    },
    "roi": {                                # prism = column through the whole chest depth
        "size_vox": [64, 128, 64],          # (x, y=depth, z)
        "voxel_mm": [1.0, 2.0, 1.0],        # 64 x 256 x 64 mm
        "depth_start_mm": 5.0,              # prism starts this far behind the detector plane
        "box_mm": [40.0, 50.0, 40.0],       # nodule box (Gaussians) inside the prism
        "min_ray_fraction": 0.5,            # keep rays with >= 50% of attenuation in the prism
        "margin_px": 3,
        "sirt_iter": 40,
        "step_mm": 1.0,
        "lung_hu": -850.0,                  # initial guess for the lung baseline
    },
    "gaussians": {
        "stride": 2,
        "sigma_factor": 0.6,
        "threshold_hu": 60.0,               # excess HU above baseline to place a Gaussian
        "max_gaussians": 30000,
        "radius_vox": 4,
        "fit_iter": 150,
        "refine_iter": 150,
        "lr_means": 0.05,                   # mm per step (Adam)
        "lr_scales": 0.1,                   # Adam lr on the logit of the scale
        "lr_amps": 0.05,                    # Adam lr on inverse-softplus(amplitude)
        "tv_weight": 1e-4,
        "prior_weight": 0.05,
    },
    "measure": {
        "mode": "halfmax",                  # 'halfmax' or 'fixed'
        "fixed_hu": -400.0,
        "floor_hu": -1000.0,                # absolute floor (off); use min_contrast_hu instead
        "min_contrast_hu": 50.0,            # threshold >= local lung + 50 HU (keeps faint GGO nodules)
        "max_extent_mm": [25.0, 50.0, 25.0],    # segmentation confined to this ellipsoid (x, y, z radii)
        "search_radius_mm": [6.0, 12.0, 6.0],   # peak search ellipsoid (x, y=depth, z)
        "open_radius_vox": 1,
        "mass_dilate_mm": [2.0, 6.0, 2.0],  # (x, y, z) minimum dilation of the mask for mass
        "mass_depth_factor": 1.5,           # depth dilation >= 1.5 x in-plane diameter (DTS smear)
    },
    "diffusion": {
        "base_ch": 32,
        "ch_mults": [1, 2, 4, 4],
        "num_res": 2,
        "attn_levels": [3],
        "dropout": 0.1,
        "timesteps": 1000,
        "prediction": "v",                  # 'v' (default) or 'eps'
        "start_frac": 1.0,                  # <1: start sampling from the noised ROI-SIRT (CCDF)
        "eta": 0.5,
        "lr": 2.0e-4,
        "batch_size": 4,
        "train_steps": 40000,
        "ema_decay": 0.999,
        "sample_steps": 50,
        "dc_iters": 2,                      # SIRT data-consistency iterations per DDIM step
        "dc_start_frac": 0.5,               # apply DC in the last 50% of DDIM steps
    },
    "pipeline": {
        "n_members": 8,                     # ensemble size K (diffusion methods)
        "n_members_physics": 4,             # ensemble size for the physics-only Gaussian baseline
        "methods": ["sirt_coarse", "roi_sirt", "roi_gs", "diffusion", "diffusion_gs"],
    },
    "uncertainty": {
        "alpha": 0.1,                       # 90% intervals
        "mode": "asymmetric",
        "sigma_floor": 0.02,                # in log units
    },
    "growth": {
        "interval_days": [90, 180, 365],
        "doubling_days": [150, 300, 450, 600, 900, 1e9],
    },
}


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None and os.path.exists(path):
        with open(path, "r") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
