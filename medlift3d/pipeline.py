"""End-to-end processing used by the scripts, the smoke test and the demo.

``prepare_case_context``  CT -> DTS projections (+noise) -> coarse SIRT volume
``process_nodule``        click -> prism ROI -> every method -> measurements (+ GT metrics)

Methods (``cfg['pipeline']['methods']``):
  sirt_coarse   coarse whole-chest SIRT, resampled into the prism (standard DTS)
  roi_sirt      physics-only fine prism reconstruction (background subtracted)
  roi_gs        physics-only 3D Gaussians in the nodule box (ensemble of perturbed inits)
  diffusion     conditional diffusion samples of the prism (+ data consistency)
  diffusion_gs  OURS: diffusion samples -> Gaussians in the nodule box -> refined on the X-rays
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .data.cases import Case
from .geometry import DTSGeometry, VolumeGeometry, bin_projections, depth_support
from .growth import MIN_MEASURABLE
from .measure import (mass_region, measure_ground_truth, measure_reconstruction, segment_nodule,
                      volume_mm3, gaussian_mass_mg)
from .metrics import dice, hd95, psnr, ssim3d
from .physics import hu_to_mu, mu_to_hu
from .projector import Projector
from .roi import ROI, ROIProblem, SubProblem, estimate_baseline_hu, resample_mask, resample_to_grid
from .simulate import add_noise, geometry_for_volume, simulate_clean
from .sirt import SIRT
from .uncertainty import ensemble_stats


@dataclass
class CaseContext:
    case_id: str
    geom: DTSGeometry
    proj: torch.Tensor                       # noisy line integrals [V, rows, cols]
    coarse_mu: torch.Tensor                  # coarse SIRT attenuation [D,H,W]
    coarse_vg: VolumeGeometry
    mu_water: float = 0.02
    device: str = "cpu"
    sirt_history: List[float] = field(default_factory=list)
    clean: Optional[torch.Tensor] = None

    @property
    def coarse_hu(self) -> np.ndarray:
        return mu_to_hu(self.coarse_mu.detach().cpu().numpy(), self.mu_water)


def coarse_grid(vg: VolumeGeometry, voxel_mm: Sequence[float], depth_range=None) -> VolumeGeometry:
    """Grid covering the CT field of view with coarse voxels -- in depth (y) only the part
    that can hold attenuation (``DTSGeometry.depth_range``: in front of the detector, inside
    the patient) when given; the rest of the field of view is air by assumption."""
    lo, hi = vg.box_min.astype(float), vg.box_max.astype(float)
    if depth_range is not None:
        lo[1], hi[1] = max(lo[1], float(depth_range[0])), min(hi[1], float(depth_range[1]))
    n = np.maximum(4, np.ceil((hi - lo) / np.asarray(voxel_mm, float))).astype(int)   # (nx, ny, nz)
    return VolumeGeometry.centered((n[2], n[1], n[0]), voxel_mm, 0.5 * (lo + hi))


def coarse_sirt(proj: torch.Tensor, geom: DTSGeometry, vg_ct: VolumeGeometry, scfg: dict, device="cpu",
                x0: Optional[torch.Tensor] = None, n_iter: Optional[int] = None):
    b = int(scfg["det_binning"])
    pb = bin_projections(proj, b)
    gb = geom.binned(b)
    cvg = coarse_grid(vg_ct, scfg["voxel_mm"], geom.depth_range())
    if x0 is not None and tuple(x0.shape) != tuple(cvg.shape):
        raise ValueError(f"coarse SIRT warm start has shape {tuple(x0.shape)}, grid is {cvg.shape}: "
                         "the saved coarse volume is from an older version -- re-run step 6")
    o, d = gb.rays(device=device)
    solver = SIRT(Projector(cvg, step_mm=scfg["step_mm"]), o, d, device=device,
                  support=torch.as_tensor(depth_support(cvg, geom), device=device))
    x, hist = solver.run(pb.reshape(-1).to(device), n_iter=int(n_iter or scfg["n_iter"]), x0=x0,
                         relax=float(scfg["relax"]))
    return x, cvg, hist


def prepare_case_context(case: Case, cfg: dict, device="cpu", seed: int = 0, clean: Optional[torch.Tensor] = None,
                         geom: Optional[DTSGeometry] = None, sirt_x0: Optional[torch.Tensor] = None,
                         sirt_iter: Optional[int] = None) -> CaseContext:
    mu_w = float(cfg["physics"]["mu_water"])
    geom = geom or geometry_for_volume(case.hu, case.vgeom, cfg["geometry"])
    if clean is None:
        clean = simulate_clean(case.hu, case.vgeom, geom, mu_w, cfg["simulation"]["step_mm"], device=device)
    clean = clean.to(device)
    noisy = add_noise(clean, cfg["simulation"]["i0"], seed=seed,
                      electronic_sigma=cfg["simulation"]["electronic_sigma"])
    cmu, cvg, hist = coarse_sirt(noisy, geom, case.vgeom, cfg["sirt"], device, x0=sirt_x0, n_iter=sirt_iter)
    return CaseContext(case.case_id, geom, noisy, cmu, cvg, mu_w, str(device), hist, clean)


def build_problem(ctx: CaseContext, click, cfg: dict) -> ROIProblem:
    rcfg = cfg["roi"]
    roi = ROI.from_config(click, rcfg, ctx.geom)
    return ROIProblem(roi, ctx.geom, ctx.proj, ctx.coarse_mu, ctx.coarse_vg, step_mm=float(rcfg["step_mm"]),
                      coarse_step_mm=float(cfg["sirt"]["step_mm"]), margin_px=int(rcfg["margin_px"]),
                      device=ctx.device, mu_water=ctx.mu_water, min_ray_fraction=float(rcfg["min_ray_fraction"]))


# ---------------------------------------------------------------------------
# per-member reconstructions
# ---------------------------------------------------------------------------
def gaussian_member(prism_hu: np.ndarray, prob: ROIProblem, base_mu: float, gcfg: dict, mu_water: float,
                    seed: int = 0, jitter_mm: float = 0.0, amp_noise: float = 0.0, use_prior: bool = True,
                    fine_voxel_mm: Optional[float] = 0.5):
    """Prism volume -> Gaussians in the nodule box (rest of the prism fixed) -> refined on the X-rays.

    Returns (composite prism HU, GaussianField, info).
    """
    from .gaussians import fit_to_volume, gaussians_from_volume, refine_with_projections

    roi = prob.roi
    sl = roi.box_slices()
    dev = prob.device
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    prism_mu = torch.as_tensor(hu_to_mu(prism_hu, mu_water), device=dev).float()
    sub = SubProblem(prob, prism_mu)
    bvg = sub.vg
    excess = (prism_mu[sl] - base_mu).clamp(min=0.0)
    gf = gaussians_from_volume(excess, bvg, stride=int(gcfg["stride"]), sigma_factor=float(gcfg["sigma_factor"]),
                               threshold_mu=float(gcfg["threshold_hu"]) * mu_water / 1000.0,
                               max_gaussians=int(gcfg["max_gaussians"]), radius_vox=int(gcfg["radius_vox"]),
                               jitter_mm=jitter_mm, generator=g).to(dev)
    if amp_noise > 0:
        with torch.no_grad():
            gf.amp_raw.add_(amp_noise * torch.randn(gf.amp_raw.shape, generator=g, device=dev))
    t0 = time.perf_counter()
    fit_hist = fit_to_volume(gf, excess, bvg, n_iter=int(gcfg["fit_iter"]), lr_means=float(gcfg["lr_means"]),
                             lr_scales=float(gcfg["lr_scales"]), lr_amps=float(gcfg["lr_amps"]))
    ref_hist = refine_with_projections(gf, sub, base_mu, n_iter=int(gcfg["refine_iter"]),
                                       lr_means=float(gcfg["lr_means"]), lr_scales=float(gcfg["lr_scales"]),
                                       lr_amps=float(gcfg["lr_amps"]), tv_weight=float(gcfg["tv_weight"]),
                                       prior=excess if use_prior else None,
                                       prior_weight=float(gcfg["prior_weight"]) if use_prior else 0.0)
    with torch.no_grad():
        box_mu = base_mu + gf.voxelize(bvg)
    comp = prism_mu.clone()
    comp[sl] = box_mu
    info = {"n_gaussians": gf.n, "fit_final": fit_hist[-1] if fit_hist else float("nan"),
            "data_first": ref_hist["data"][0] if ref_hist["data"] else float("nan"),
            "data_final": ref_hist["data"][-1] if ref_hist["data"] else float("nan"),
            "seconds": time.perf_counter() - t0}
    return mu_to_hu(comp.cpu().numpy(), mu_water), gf, info


def fine_volume_mm3(gf, bvg: VolumeGeometry, base_mu: float, mu_water: float, seed_xyz, baseline_hu: float,
                    mcfg: dict, fine_mm: float = 0.5) -> float:
    """Resolution-free volume: evaluate the continuous Gaussian field on a fine isotropic grid."""
    n = np.maximum(4, np.round(bvg.size_mm / fine_mm)).astype(int)
    fvg = VolumeGeometry.centered((n[2], n[1], n[0]), (fine_mm,) * 3, bvg.center)
    with torch.no_grad():
        f = mu_to_hu((base_mu + gf.voxelize(fvg, chunk=4096)).cpu().numpy(), mu_water)
    mask, _ = segment_nodule(f, fvg, seed_xyz, baseline_hu, mode=mcfg["mode"], fixed_hu=mcfg["fixed_hu"],
                             floor_hu=mcfg["floor_hu"], search_radius_mm=mcfg["search_radius_mm"],
                             open_radius_vox=max(1, int(round(mcfg["open_radius_vox"] * min(bvg.spacing) / fine_mm))),
                             min_contrast_hu=float(mcfg.get("min_contrast_hu", 50.0)),
                             max_extent_mm=mcfg.get("max_extent_mm", (25.0, 50.0, 25.0)))
    return volume_mm3(mask, fvg)


def diffusion_samples(model, diffusion, cond_hu: np.ndarray, prob: ROIProblem, dcfg: dict, n: int,
                      mu_water: float, seed: int = 0, batch: int = 4) -> List[np.ndarray]:
    """K conditional diffusion samples of the prism with SIRT data-consistency steps (HU)."""
    from .diffusion.ddpm import hu_to_unit, unit_to_hu

    dev = prob.device
    cond = torch.as_tensor(hu_to_unit(cond_hu), dtype=torch.float32, device=dev).clamp(-1.5, 1.5)[None, None]
    dc_iters = int(dcfg.get("dc_iters", 0))

    def dc_fn(x0):
        mu = hu_to_mu(unit_to_hu(x0[:, 0]), mu_water)
        mu = prob.sirt_steps(mu, n_iter=dc_iters)
        return hu_to_unit(mu_to_hu(mu, mu_water))[:, None]

    out: List[np.ndarray] = []
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    model.eval()
    for s in range(0, n, batch):
        b = min(batch, n - s)
        x = diffusion.ddim_sample(model, cond.expand(b, -1, -1, -1, -1), steps=int(dcfg["sample_steps"]),
                                  eta=float(dcfg.get("eta", 0.5)), generator=g,
                                  dc_fn=dc_fn if dc_iters > 0 else None,
                                  dc_start_frac=float(dcfg.get("dc_start_frac", 0.5)),
                                  start_frac=float(dcfg.get("start_frac", 1.0)))
        out.extend(list(unit_to_hu(x[:, 0]).float().cpu().numpy()))
    return out


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------
def ground_truth_for_roi(case: Case, nodule_index: int, roi: ROI, mcfg: dict) -> Dict:
    """GT prism (HU, resampled), GT mask in the prism, GT measurements at native resolution."""
    nod = case.nodules[nodule_index]
    full = nod.full_mask(case.hu.shape)
    z0, z1, y0, y1, x0, x1 = nod.bbox_zyx
    pad_mm = np.array([20.0, 20.0 + 1.5 * max(nod.diameter_mm, 4.0), 20.0])         # x, y(depth), z
    pad = np.ceil(pad_mm[::-1] / np.asarray(case.vgeom.spacing)[::-1]).astype(int)  # z, y, x
    lo = np.maximum(np.array([z0, y0, x0]) - pad, 0)
    hi = np.minimum(np.array([z1, y1, x1]) + pad, case.hu.shape)
    crop_hu = case.hu[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    crop_mask = full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    crop_vg = VolumeGeometry(crop_hu.shape, case.vgeom.spacing, tuple(case.vgeom.index_to_world(lo[::-1].astype(float))))
    gt_meas = measure_ground_truth(crop_hu, crop_vg, crop_mask, mcfg)
    gt_roi = resample_to_grid(case.hu, case.vgeom, roi.vgeom, order=1)
    gt_mask_roi = resample_mask(full, case.vgeom, roi.vgeom)
    out = {"meas": gt_meas, "roi_hu": gt_roi, "roi_mask": gt_mask_roi, "roi_vg": roi.vgeom}
    if nod.reader_masks is not None and len(nod.reader_masks) > 1:
        vols = [float(m.sum()) * case.vgeom.voxel_volume for m in nod.reader_masks]
        out["meas"]["reader_volume_cv"] = float(np.std(vols) / max(np.mean(vols), 1e-9))
        out["meas"]["reader_volumes"] = vols
    return out


# ---------------------------------------------------------------------------
# one nodule
# ---------------------------------------------------------------------------
def process_nodule(ctx: CaseContext, click_xyz, cfg: dict, model=None, diffusion=None,
                   methods: Optional[Sequence[str]] = None, n_members: Optional[int] = None, seed: int = 0,
                   gt: Optional[Dict] = None, keep_volumes: bool = False, n_members_physics: Optional[int] = None) -> Dict:
    methods = list(methods or cfg["pipeline"]["methods"])
    K = int(n_members or cfg["pipeline"]["n_members"])
    K_phys = int(n_members_physics or cfg["pipeline"].get("n_members_physics", max(2, K // 2)))
    mu_w = ctx.mu_water
    rcfg, gcfg, mcfg = cfg["roi"], cfg["gaussians"], cfg["measure"]
    t_all = time.perf_counter()
    prob = build_problem(ctx, click_xyz, cfg)
    roi = prob.roi
    vg = roi.vgeom
    sl = roi.box_slices()
    if gt is not None and "roi_vg" in gt:
        gvg = gt["roi_vg"]
        if gvg.shape != vg.shape or not np.allclose(gvg.origin, vg.origin, atol=1e-3):
            raise ValueError("ground truth ROI grid does not match the reconstruction grid: build it with "
                             "ROI.from_config(click, cfg['roi'], geom) using the same click and geometry")
    t0 = time.perf_counter()
    sirt_mu, sirt_hist = prob.sirt(n_iter=int(rcfg["sirt_iter"]), init_hu=float(rcfg["lung_hu"]))
    roi_sirt_hu = mu_to_hu(sirt_mu.cpu().numpy(), mu_w)
    t_sirt = time.perf_counter() - t0
    baseline = estimate_baseline_hu(roi_sirt_hu, vg, click_xyz, default=float(rcfg["lung_hu"]))
    base_mu = float(hu_to_mu(np.float32(baseline), mu_w))

    members: Dict[str, List[np.ndarray]] = {}
    fields: Dict[str, list] = {}
    timing: Dict[str, float] = {"roi_sirt": t_sirt}
    extra: Dict[str, list] = {}
    if "sirt_coarse" in methods:
        members["sirt_coarse"] = [resample_to_grid(ctx.coarse_hu, ctx.coarse_vg, vg, order=1)]
    if "roi_sirt" in methods:
        members["roi_sirt"] = [roi_sirt_hu]
    if "roi_gs" in methods:
        t0 = time.perf_counter()
        vols, gfs, infos = [], [], []
        for k in range(K_phys):
            v, gf, info = gaussian_member(roi_sirt_hu, prob, base_mu, gcfg, mu_w, seed=seed * 1000 + k,
                                          jitter_mm=0.5 if k > 0 else 0.0, amp_noise=0.2 if k > 0 else 0.0,
                                          use_prior=False)
            vols.append(v), gfs.append(gf), infos.append(info)
        members["roi_gs"], fields["roi_gs"], extra["roi_gs"] = vols, gfs, infos
        timing["roi_gs"] = time.perf_counter() - t0
    if model is not None and diffusion is not None and ("diffusion" in methods or "diffusion_gs" in methods):
        t0 = time.perf_counter()
        samples = diffusion_samples(model, diffusion, roi_sirt_hu, prob, cfg["diffusion"], K, mu_w, seed=seed)
        timing["diffusion"] = time.perf_counter() - t0
        if "diffusion" in methods:
            members["diffusion"] = samples
        if "diffusion_gs" in methods:
            t0 = time.perf_counter()
            vols, gfs, infos = [], [], []
            for k, smp in enumerate(samples):
                v, gf, info = gaussian_member(smp, prob, base_mu, gcfg, mu_w, seed=seed * 1000 + k, use_prior=True)
                vols.append(v), gfs.append(gf), infos.append(info)
            members["diffusion_gs"], fields["diffusion_gs"], extra["diffusion_gs"] = vols, gfs, infos
            timing["diffusion_gs"] = time.perf_counter() - t0 + timing["diffusion"]

    result: Dict = {"click_xyz": [float(c) for c in click_xyz], "baseline_hu": baseline,
                    "roi": {"size_vox": list(roi.size_vox), "voxel_mm": list(roi.voxel_mm), "box_mm": list(roi.box_mm),
                            "n_rays": int(prob.o.shape[0]), "n_window_rays": int(prob.keep.numel())},
                    "roi_sirt_residual": sirt_hist[-1] if sirt_hist else float("nan"),
                    "timing_s": timing, "methods": {}}
    box_vg = roi.box_vgeom
    for name, vols in members.items():
        per, masks = [], []
        for k, v in enumerate(vols):
            m, mask = measure_reconstruction(v, vg, click_xyz, baseline, mcfg)
            if name in fields:
                gf = fields[name][k]
                region = mass_region(mask, vg, mcfg["mass_dilate_mm"], float(mcfg.get("mass_depth_factor", 0.0)))
                m["mass_gauss_mg"] = gaussian_mass_mg(gf, region[sl], box_vg, mu_w)
                m["volume_fine_mm3"] = fine_volume_mm3(gf, box_vg, base_mu, mu_w, click_xyz, baseline, mcfg)
                m.update({f"gs_{kk}": vv for kk, vv in extra[name][k].items()})
            per.append(m)
            masks.append(mask)
        vol_stats = ensemble_stats([max(p["volume_mm3"], 1e-6) for p in per], min_value=MIN_MEASURABLE)
        mass_stats = ensemble_stats([max(p["mass_mg"], 1e-6) for p in per], min_value=MIN_MEASURABLE)
        vote = np.mean(masks, axis=0) >= 0.5
        mean_vol = np.mean(vols, axis=0)
        entry = {"volume_mm3": vol_stats["median"], "volume_log_std": vol_stats["log_std"],
                 "mass_mg": mass_stats["median"], "mass_log_std": mass_stats["log_std"],
                 "volume2d_mm3": float(np.median([p["volume2d_mm3"] for p in per])),
                 "d2d_long_mm": float(np.median([p["d2d_long_mm"] for p in per])),
                 "n_members": len(vols), "n_found": vol_stats["n_found"], "members": per}
        if name in fields:
            entry["mass_gauss_mg"] = float(np.median([p["mass_gauss_mg"] for p in per]))
            entry["volume_fine_mm3"] = float(np.median([p["volume_fine_mm3"] for p in per]))
        if gt is not None:
            rng = 1400.0
            gt_box = np.clip(gt["roi_hu"][sl], -1000, 400)
            mv_box = np.clip(mean_vol[sl], -1000, 400)
            entry["psnr"] = psnr(mv_box, gt_box, rng)
            entry["ssim"] = ssim3d(mv_box, gt_box, rng)
            entry["dice"] = dice(vote, gt["roi_mask"])
            entry["hd95_mm"] = hd95(vote, gt["roi_mask"], vg.spacing[::-1])
            if len(vols) > 1:
                from .uncertainty import sparsification, spearman

                std_map = np.std(vols, axis=0)[sl]
                err = np.abs(mean_vol[sl] - gt["roi_hu"][sl])
                entry["unc_spearman"] = spearman(std_map, err)
                entry["unc_ause"] = sparsification(err, std_map)["ause"]
        if keep_volumes:
            entry["mean_volume"] = mean_vol.astype(np.float32)
            entry["std_volume"] = np.std(vols, axis=0).astype(np.float32) if len(vols) > 1 else None
            entry["vote_mask"] = vote
        result["methods"][name] = entry
    if keep_volumes:
        result["roi_sirt_hu"] = roi_sirt_hu
        result["roi_vg"] = vg
        result["box_slices"] = sl
        result["fields"] = {k: [g.clone_detached().cpu() for g in v] for k, v in fields.items()}
    if gt is not None:
        result["gt"] = gt["meas"]
    result["timing_s"]["total"] = time.perf_counter() - t_all
    return result
