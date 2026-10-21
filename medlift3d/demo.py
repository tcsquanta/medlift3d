"""Backend of the click-to-3D demo (UI-free so it can be tested headless).

    sess = DemoSession(cfg, device)
    sess.load_case(case_id)                  # or sess.load_phantom(seed)
    img = sess.coronal_image(iy)             # DTS slice + candidate markers
    report = sess.analyze(x_mm, iy, z_mm)    # 3D nodule, mass/volume with intervals
    growth = sess.followup(days, doubling_days)   # virtual follow-up -> VDT / MDT
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from .data.cases import Case, case_from_phantom, load_case
from .detect import detect_candidates
from .growth import doubling_time, growth_category, scale_for_doubling
from .io import Paths, load_dts, load_sirt, stable_seed
from .phantom import make_chest_phantom
from .physics import mu_to_hu
from .pipeline import CaseContext, ground_truth_for_roi, prepare_case_context, process_nodule
from .roi import ROI
from .simulate import add_noise
from .uncertainty import ConformalCalibrator


def eups_volume_category(v_mm3: float) -> str:
    """European Position Statement volume categories for a solid baseline nodule."""
    if v_mm3 < 100:
        return "< 100 mm3: low risk (routine follow-up)"
    if v_mm3 < 300:
        return "100-300 mm3: indeterminate (3-month follow-up)"
    return ">= 300 mm3: high risk (refer)"


class DemoSession:
    def __init__(self, cfg: dict, device="cpu", ckpt: Optional[str] = None, calibration: Optional[str] = None):
        self.cfg = cfg
        self.device = str(device)
        self.paths = Paths(cfg)
        self.model = self.diffusion = None
        ckpt = ckpt or os.path.join(cfg["paths"]["work"], "models", "diffusion", "best.pt")
        if os.path.exists(ckpt):
            from .diffusion.checkpoint import load_for_sampling

            self.model, self.diffusion, _ = load_for_sampling(ckpt, device=self.device)
        self.method = "diffusion_gs" if self.model is not None else "roi_gs"
        cal = calibration or os.path.join(cfg["paths"]["outputs"], "10_evaluation", "conformal.json")
        self.calib = json.load(open(cal)) if os.path.exists(cal) else {}
        self.case: Optional[Case] = None
        self.ctx: Optional[CaseContext] = None
        self.candidates: List[Dict] = []
        self.last: Optional[Dict] = None

    # ------------------------------------------------------------------ cases
    def available_cases(self) -> List[str]:
        return [c for c in self.paths.case_ids() if os.path.exists(self.paths.sirt(c))]

    def load_case(self, case_id: str) -> None:
        case = load_case(self.paths.case(case_id), load_readers=False)
        clean, geom, _ = load_dts(self.paths.dts(case_id), device=self.device)
        cmu, cvg, meta = load_sirt(self.paths.sirt(case_id), device=self.device)
        noisy = add_noise(clean, self.cfg["simulation"]["i0"], seed=int(meta.get("noise_seed", stable_seed(case_id))))
        self.case = case
        self.ctx = CaseContext(case_id, geom, noisy, cmu, cvg, self.cfg["physics"]["mu_water"], self.device, clean=clean)
        self._after_load()

    def load_phantom(self, seed: int = 0, spacing: float = 2.5) -> None:
        ph = make_chest_phantom(spacing=(spacing,) * 3, seed=seed, n_random_nodules=4)
        self.case = case_from_phantom(ph, f"phantom_demo{seed}")
        self.ctx = prepare_case_context(self.case, self.cfg, device=self.device, seed=seed)
        self._after_load()

    def _after_load(self) -> None:
        rec = self.ctx.coarse_hu
        sel = (rec > -1000) & (rec < -600)
        lung = float(np.median(rec[sel])) if sel.any() else -850.0
        self.candidates = detect_candidates(rec, self.ctx.coarse_vg, baseline_hu=lung, top_k=10)
        self.last = None

    # ----------------------------------------------------------------- slices
    @property
    def n_depth(self) -> int:
        return int(self.ctx.coarse_vg.shape[1])

    def depth_mm(self, iy: int) -> float:
        return float(self.ctx.coarse_vg.index_to_world(np.array([0, iy, 0], float))[1])

    def coronal_image(self, iy: int, with_candidates: bool = True) -> np.ndarray:
        """RGB uint8 coronal DTS slice (superior at the top); candidates near this depth in yellow."""
        vol = self.ctx.coarse_hu
        iy = int(np.clip(iy, 0, vol.shape[1] - 1))
        sl = np.clip((vol[:, iy, :] + 1000.0) / 1400.0, 0, 1)[::-1]
        rgb = np.stack([sl] * 3, -1)
        if with_candidates:
            vg = self.ctx.coarse_vg
            for k, c in enumerate(self.candidates):
                iz, cy, ix = c["index_zyx"]
                if abs(cy - iy) * vg.spacing[1] > 12.0:
                    continue
                r = int(round((vol.shape[0] - 1) - iz))
                rad = max(3, int(round(2.5 * c["sigma_mm"] / vg.spacing[0])))
                yy, xx = np.ogrid[: rgb.shape[0], : rgb.shape[1]]
                ring = np.abs(np.hypot(yy - r, xx - ix) - rad) < 0.8
                rgb[ring] = [1.0, 0.85, 0.0]
        return (rgb * 255).astype(np.uint8)

    def pixel_to_world(self, row: int, col: int, iy: int) -> List[float]:
        vg = self.ctx.coarse_vg
        iz = (vg.shape[0] - 1) - row
        return [float(v) for v in vg.index_to_world(np.array([col, iy, iz], float))]

    # ---------------------------------------------------------------- analysis
    def _interval(self, value: float, sigma: float, qty: str):
        key = f"{self.method}:{qty}"
        if key in self.calib:
            c = self.calib[key]
            cal = ConformalCalibrator(alpha=c["alpha"], mode=c["mode"], sigma_floor=c["sigma_floor"],
                                      q_lo=c["q_lo"], q_hi=c["q_hi"], n_cal=c["n_cal"])
            lo, hi = cal.interval([value], [sigma])
            return float(lo[0]), float(hi[0]), f"calibrated {int(100 * (1 - c['alpha']))}%"
        return float(value * np.exp(-2 * sigma)), float(value * np.exp(2 * sigma)), "uncalibrated +-2 sd"

    def analyze(self, click_xyz, n_members: Optional[int] = None) -> Dict:
        methods = [self.method]
        res = process_nodule(self.ctx, click_xyz, self.cfg, self.model, self.diffusion, methods, n_members,
                             seed=stable_seed(self.ctx.case_id, *[round(c) for c in click_xyz]), keep_volumes=True)
        e = res["methods"][self.method]
        v_lo, v_hi, how = self._interval(e["volume_mm3"], e["volume_log_std"], "volume")
        m_lo, m_hi, _ = self._interval(e["mass_mg"], e["mass_log_std"], "mass")
        report = {"method": self.method, "click_xyz": click_xyz, "volume_mm3": e["volume_mm3"],
                  "volume_interval": [v_lo, v_hi], "mass_mg": e["mass_mg"], "mass_interval": [m_lo, m_hi],
                  "interval_type": how, "d2d_long_mm": e["d2d_long_mm"],
                  "volume_fine_mm3": e.get("volume_fine_mm3"), "mass_gauss_mg": e.get("mass_gauss_mg"),
                  "eups_category": eups_volume_category(e["volume_mm3"]), "n_members": e["n_members"],
                  "seconds": res["timing_s"]["total"]}
        self.last = {"res": res, "entry": e, "report": report, "click": list(click_xyz)}
        return report

    def nearest_nodule(self, click_xyz, max_mm: float = 15.0) -> Optional[int]:
        if self.case is None or not self.case.nodules:
            return None
        d = [np.linalg.norm(np.asarray(n.center_xyz) - np.asarray(click_xyz)) for n in self.case.nodules]
        j = int(np.argmin(d))
        return j if d[j] <= max_mm else None

    def followup(self, interval_days: float, doubling_days: float) -> Dict:
        """Virtual follow-up of the last analysed nodule (needs the CT of a simulated case)."""
        if self.last is None:
            raise RuntimeError("analyse a nodule first")
        j = self.nearest_nodule(self.last["click"], max_mm=20.0)
        if j is None:
            raise RuntimeError("no ground-truth nodule near the click to grow (virtual growth needs the CT)")
        import importlib.util

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location("growth_script", os.path.join(here, "scripts", "11_growth_experiment.py"))
        mod = importlib.util.module_from_spec(spec)
        import sys

        sys.path.insert(0, os.path.join(here, "scripts"))
        spec.loader.exec_module(mod)
        from .pipeline import coarse_sirt
        from .simulate import project_local_change

        s = scale_for_doubling(interval_days, doubling_days)
        case_f, gvg, dmu = mod.grow_case(self.case, j, s, self.cfg["measure"])
        clean = self.ctx.clean
        if clean is None:
            raise RuntimeError("clean projections unavailable")
        clean_f = clean + project_local_change(dmu, gvg, self.ctx.geom, device=self.device)
        noisy_f = add_noise(clean_f, self.cfg["simulation"]["i0"], seed=stable_seed(self.ctx.case_id, interval_days, doubling_days))
        cmu_f, _, _ = coarse_sirt(noisy_f, self.ctx.geom, self.case.vgeom, self.cfg["sirt"], device=self.device,
                                  x0=self.ctx.coarse_mu, n_iter=10)
        ctx_f = CaseContext(self.ctx.case_id, self.ctx.geom, noisy_f, cmu_f, self.ctx.coarse_vg, self.ctx.mu_water, self.device)
        res_f = process_nodule(ctx_f, self.last["click"], self.cfg, self.model, self.diffusion, [self.method],
                               seed=stable_seed(self.ctx.case_id, "fu"))
        eb, ef = self.last["entry"], res_f["methods"][self.method]
        vdt = doubling_time(eb["volume_mm3"], ef["volume_mm3"], interval_days)
        mdt = doubling_time(eb["mass_mg"], ef["mass_mg"], interval_days)
        true_dt = doubling_days if doubling_days < 1e8 else float("inf")
        return {"interval_days": interval_days, "true_doubling_days": true_dt, "true_category": growth_category(true_dt),
                "volume_followup_mm3": ef["volume_mm3"], "mass_followup_mg": ef["mass_mg"],
                "vdt_days": vdt, "vdt_category": growth_category(vdt), "mdt_days": mdt, "mdt_category": growth_category(mdt)}

    # -------------------------------------------------------------------- 3D
    def mesh(self, which: str = "vote"):
        """Vertices, faces and per-vertex uncertainty (HU std) of the last nodule surface."""
        from skimage.measure import marching_cubes

        if self.last is None:
            return None
        res = self.last["res"]
        e = res["methods"][self.method]
        vg = res["roi_vg"]
        mask = e["vote_mask"].astype(np.float32)
        if mask.sum() < 4:
            return None
        verts, faces, _, _ = marching_cubes(np.pad(mask, 1), 0.5, spacing=tuple(vg.spacing[::-1]))
        verts = verts - np.array(vg.spacing[::-1])                    # undo padding (z, y, x in mm)
        std = e.get("std_volume")
        if std is not None:
            idx = np.clip(np.round(verts / np.array(vg.spacing[::-1])).astype(int), 0, np.array(std.shape) - 1)
            col = std[idx[:, 0], idx[:, 1], idx[:, 2]]
        else:
            col = np.zeros(len(verts))
        xyz = verts[:, ::-1] + vg.origin_arr                          # world (x, y, z)
        return xyz, faces, col
