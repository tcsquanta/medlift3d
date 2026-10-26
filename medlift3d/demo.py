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
from .growth import MIN_MEASURABLE, doubling_time, growth_category, scale_for_doubling
from .io import Paths, load_dts, load_sirt, stable_seed
from .phantom import make_chest_phantom
from .physics import mu_to_hu
from .pipeline import CaseContext, ground_truth_for_roi, prepare_case_context, process_nodule
from .roi import ROI
from .simulate import add_noise
from .uncertainty import ConformalCalibrator


def member_reason(m: Dict, min_contrast_hu: float = 50.0) -> str:
    """Why a hypothesis' segmentation is empty (or 'found')."""
    if m.get("volume_mm3", 0.0) > MIN_MEASURABLE:
        return "found"
    if m.get("touches_limit"):
        return "merged with an adjacent structure (chest wall / mediastinum / vessel)"
    pk, base = m.get("peak_hu", float("nan")), m.get("baseline_hu", float("nan"))
    if np.isfinite(pk) and np.isfinite(base):
        return f"too faint (peak {pk - base:.0f} HU above lung, needs {2 * min_contrast_hu:.0f})"
    return "nothing at the click"


def eups_volume_category(v_mm3: float) -> str:
    """European Position Statement volume categories for a solid baseline nodule."""
    if v_mm3 < 100:
        return "< 100 mm3: low risk (routine follow-up)"
    if v_mm3 < 300:
        return "100-300 mm3: indeterminate (3-month follow-up)"
    return ">= 300 mm3: high risk (refer)"


class DemoSession:
    def __init__(self, cfg: dict, device="cpu", ckpt: Optional[str] = None, calibration: Optional[str] = None,
                 n_members: Optional[int] = None, display_scale: int = 3):
        self.cfg = cfg
        self.n_members = n_members                 # hypotheses per click (None: config value)
        self.display_scale = int(display_scale)    # coronal image is enlarged for clicking
        self.device = str(device)
        self.paths = Paths(cfg)
        self.model = self.diffusion = None
        ckpt = ckpt or os.path.join(cfg["paths"]["work"], "models", "diffusion", "best.pt")
        if os.path.exists(ckpt):
            from .diffusion.checkpoint import load_for_sampling

            self.model, self.diffusion, _ = load_for_sampling(ckpt, device=self.device)
        self.detector, self.det_threshold = None, 0.0
        det_ckpt = os.path.join(cfg["paths"]["work"], "models", "detector", "best.pt")
        if os.path.exists(det_ckpt):                       # learned detector (step 14); else training-free LoG
            from .detector import load_detector

            self.detector, extra = load_detector(det_ckpt, device=self.device)
            self.det_threshold = float(extra.get("threshold", 0.0))
        self.detector_name = "learned detector (step 14)" if self.detector is not None else "training-free LoG detector"
        self.method = "diffusion_gs" if self.model is not None else "roi_gs"
        cal = calibration or os.path.join(cfg["paths"]["outputs"], "10_evaluation", "conformal.json")
        self.calib = json.load(open(cal)) if os.path.exists(cal) else {}
        cps = os.path.join(cfg["paths"]["outputs"], "16_uncertainty", "cps_calibration.json")
        self.cps = json.load(open(cps)) if os.path.exists(cps) else None      # step 16: category probabilities
        self.case: Optional[Case] = None
        self.ctx: Optional[CaseContext] = None
        self.candidates: List[Dict] = []
        self.last: Optional[Dict] = None

    # ------------------------------------------------------------------ cases
    SPLIT_ORDER = {"test": 0, "cal": 1, "train": 2}

    def case_split(self, case_id: str) -> str:
        """'train' / 'cal' / 'test' from work/splits.json ('?' if unknown)."""
        if not hasattr(self, "_splits"):
            p = self.paths.p("splits.json")
            self._splits = json.load(open(p)).get("cases", {}) if os.path.exists(p) else {}
        return self._splits.get(case_id, "?")

    def available_cases(self, splits: Optional[List[str]] = None) -> List[str]:
        """Cases with a DTS reconstruction (only those ``splits`` if given), held-out (test, then
        calibration) patients first."""
        ids = [c for c in self.paths.case_ids() if os.path.exists(self.paths.sirt(c))
               and (splits is None or self.case_split(c) in splits)]
        return sorted(ids, key=lambda c: (self.SPLIT_ORDER.get(self.case_split(c), 3), c))

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
        top_k = int(self.cfg.get("detector", {}).get("top_k", 10))
        if self.detector is not None:
            from .detector import detect_learned

            self.candidates = detect_learned(self.detector, rec, self.ctx.coarse_vg, self.device, top_k=top_k)
        else:
            self.candidates = detect_candidates(rec, self.ctx.coarse_vg, baseline_hu=lung, top_k=top_k)
        from .detector import match_candidates

        ann = self.annotations()
        for c, j in zip(self.candidates, match_candidates(self.candidates, ann)):
            c["match"] = None if j is None else ann[j]["index"]          # radiologists' nodule it finds, if any
            c["suggested"] = self.detector is None or c["score"] >= self.det_threshold
        self.last = None

    def detection_summary(self) -> Dict:
        """How the suggestions compare with the radiologists' nodules of the loaded case."""
        ann = self.annotations()
        found = sorted({c["match"] for c in self.candidates if c.get("match") is not None})
        first = {}
        for k, c in enumerate(self.candidates, start=1):
            if c.get("match") is not None:
                first.setdefault(c["match"], k)
        return {"n_nodules": len(ann), "found": found, "missed": [a["index"] for a in ann if a["index"] not in found],
                "n_false": sum(c.get("match") is None for c in self.candidates), "n_suggestions": len(self.candidates),
                "rank_of_nodule": first}

    def ground_truth_at(self, click_xyz, max_mm: float = 15.0) -> Optional[Dict]:
        """Radiologists' ground truth (volume, mass, diameter) of the nodule nearest the click."""
        j = self.nearest_nodule(click_xyz, max_mm=max_mm)
        if j is None:
            return None
        roi = ROI.from_config(self.case.nodules[j].center_xyz, self.cfg["roi"], self.ctx.geom)
        g = ground_truth_for_roi(self.case, j, roi, self.cfg["measure"])["meas"]
        return {"index": j, "diameter_mm": float(self.case.nodules[j].diameter_mm), **g}

    # ----------------------------------------------------------------- slices
    @property
    def n_depth(self) -> int:
        return int(self.ctx.coarse_vg.shape[1])

    def depth_mm(self, iy: int) -> float:
        return float(self.ctx.coarse_vg.index_to_world(np.array([0, iy, 0], float))[1])

    def coronal_image(self, iy: int, with_candidates: bool = True, with_annotations: bool = True,
                      selected: Optional[int] = None) -> np.ndarray:
        """RGB uint8 coronal DTS slice (superior at the top); detector candidates near this depth
        in yellow (the ``selected`` one in cyan), radiologist-annotated nodules (LIDC ground truth)
        near this depth in green."""
        vol = self.ctx.coarse_hu
        vg = self.ctx.coarse_vg
        iy = int(np.clip(iy, 0, vol.shape[1] - 1))
        sl = np.clip((vol[:, iy, :] + 1000.0) / 1400.0, 0, 1)[::-1]
        rgb = np.stack([sl] * 3, -1)
        yy, xx = np.ogrid[: rgb.shape[0], : rgb.shape[1]]

        def ring(iz, ix, rad, colour):
            r = (vol.shape[0] - 1) - iz
            rgb[np.abs(np.hypot(yy - r, xx - ix) - rad) < 0.8] = colour

        if with_candidates:
            for k, c in enumerate(self.candidates):
                iz, cy, ix = c["index_zyx"]
                if abs(cy - iy) * vg.spacing[1] <= 12.0:
                    ring(iz, ix, max(3, int(round(2.5 * c["sigma_mm"] / vg.spacing[0]))),
                         [0.0, 0.9, 1.0] if k == selected else [1.0, 0.85, 0.0])
        if with_annotations:
            for a in self.annotations():
                if abs(a["iy"] - iy) * vg.spacing[1] <= max(12.0, a["diameter_mm"]):
                    ring(a["iz"], a["ix"], max(4, int(round((0.5 * a["diameter_mm"] + 4.0) / vg.spacing[0]))),
                         [0.1, 0.9, 0.2])
        s = self.display_scale
        return (np.repeat(np.repeat(rgb, s, axis=0), s, axis=1) * 255).astype(np.uint8)

    def annotations(self) -> List[Dict]:
        """Radiologist-annotated nodules of the loaded case (LIDC consensus; phantoms: true nodules)."""
        if self.case is None or self.ctx is None:
            return []
        vg = self.ctx.coarse_vg
        out = []
        for j, n in enumerate(self.case.nodules):
            ix, iy, iz = np.round(vg.world_to_index(np.asarray(n.center_xyz, float))).astype(int)
            out.append({"index": j, "xyz": [float(v) for v in n.center_xyz], "diameter_mm": float(n.diameter_mm),
                        "ix": int(ix), "iy": int(iy), "iz": int(iz)})
        return out

    def pixel_to_world(self, row: int, col: int, iy: int) -> List[float]:
        """Clicked pixel of the (enlarged) coronal image at depth slice ``iy`` -> world (x, y, z) mm."""
        vg = self.ctx.coarse_vg
        row, col = int(row) // self.display_scale, int(col) // self.display_scale
        iz = (vg.shape[0] - 1) - row
        return [float(v) for v in vg.index_to_world(np.array([col, iy, iz], float))]

    # ---------------------------------------------------------------- analysis
    def _interval(self, value: float, sigma: float, qty: str, method: str):
        key = f"{method}:{qty}"
        c = self.calib.get(key)
        if c is not None and all(c.get(q) is not None and np.isfinite(c[q]) for q in ("q_lo", "q_hi")):
            cal = ConformalCalibrator(alpha=c["alpha"], mode=c["mode"], sigma_floor=c["sigma_floor"],
                                      q_lo=c["q_lo"], q_hi=c["q_hi"], n_cal=c["n_cal"])
            lo, hi = cal.interval([value], [sigma])
            return float(lo[0]), float(hi[0]), f"calibrated {int(100 * (1 - c['alpha']))}%"
        how = "uncalibrated +-2 sd" + (" (too few calibration nodules)" if c is not None else "")
        return float(value * np.exp(-2 * sigma)), float(value * np.exp(2 * sigma)), how

    def analyze(self, click_xyz, n_members: Optional[int] = None) -> Dict:
        n_members = n_members or self.n_members
        """Reconstruct + measure at the click. If the main method finds no nodule there, the
        physics-only method (roi_gs) is tried and the report says so in ``note``."""
        seed = stable_seed(self.ctx.case_id, *[round(c) for c in click_xyz])
        method, note = self.method, ""
        res = process_nodule(self.ctx, click_xyz, self.cfg, self.model, self.diffusion, [method], n_members,
                             seed=seed, keep_volumes=True)
        secs = res["timing_s"]["total"]
        mcfg = self.cfg["measure"]
        if res["methods"][method]["volume_mm3"] <= MIN_MEASURABLE and method != "roi_gs":
            why = sorted({member_reason(m, float(mcfg.get("min_contrast_hu", 50.0)))
                          for m in res["methods"][method]["members"]} - {"found"})
            note = (f"{method} found no nodule at the click ({'; '.join(why)}); showing the physics-only "
                    "result (roi_gs) -- treat it with caution")
            method = "roi_gs"
            res = process_nodule(self.ctx, click_xyz, self.cfg, self.model, self.diffusion, [method], n_members,
                                 seed=seed, keep_volumes=True)
            secs += res["timing_s"]["total"]
        e = res["methods"][method]
        members = [{"volume_mm3": float(m["volume_mm3"]), "mass_mg": float(m["mass_mg"]),
                    "reason": member_reason(m, float(mcfg.get("min_contrast_hu", 50.0)))} for m in e["members"]]
        if e["volume_mm3"] <= MIN_MEASURABLE:
            why = sorted({m["reason"] for m in members} - {"found"})
            note = f"no nodule measured at the click: {'; '.join(why) or 'nothing found'}"
        v_lo, v_hi, how = self._interval(e["volume_mm3"], e["volume_log_std"], "volume", method)
        m_lo, m_hi, _ = self._interval(e["mass_mg"], e["mass_log_std"], "mass", method)
        report = {"method": method, "note": note, "click_xyz": click_xyz, "volume_mm3": e["volume_mm3"],
                  "volume_interval": [v_lo, v_hi], "mass_mg": e["mass_mg"], "mass_interval": [m_lo, m_hi],
                  "interval_type": how, "d2d_long_mm": e["d2d_long_mm"],
                  "volume_fine_mm3": e.get("volume_fine_mm3"), "mass_gauss_mg": e.get("mass_gauss_mg"),
                  "eups_category": eups_volume_category(e["volume_mm3"]), "n_members": e["n_members"], "n_found": e.get("n_found", e["n_members"]),
                  "seconds": secs, "members": members, "spread_factor": float(np.exp(2 * e["mass_log_std"])),
                  "low_confidence": bool(e.get("n_found", e["n_members"]) < 0.5 * e["n_members"] + 0.01
                                         or np.exp(2 * max(e["mass_log_std"], e["volume_log_std"])) > 10.0)}
        cal = (self.cps or {}).get("methods", {}).get(method)      # step 16's calibration of this method
        if cal is not None and e["volume_mm3"] > MIN_MEASURABLE:
            from .uq import CATEGORIES, category_probs_from_calibration

            p = category_probs_from_calibration(cal, e["volume_mm3"], e["volume_log_std"])
            report["category_probs"] = {c: float(v) for c, v in zip(CATEGORIES, p)}
        self.last = {"res": res, "entry": e, "report": report, "click": list(click_xyz), "method": method}
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
        method = self.last["method"]                                  # same method as the baseline scan
        res_f = process_nodule(ctx_f, self.last["click"], self.cfg, self.model, self.diffusion, [method],
                               seed=stable_seed(self.ctx.case_id, "fu"))
        eb, ef = self.last["entry"], res_f["methods"][method]
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
        e = res["methods"][self.last["method"]]
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

    def truth_mesh(self, max_mm: float = 20.0):
        """Vertices (world x, y, z) and faces of the radiologists' consensus nodule nearest the
        last click (LIDC ground truth, for comparison in the demo); None if there is none."""
        from skimage.measure import marching_cubes

        if self.last is None or self.case is None:
            return None
        j = self.nearest_nodule(self.last["click"], max_mm=max_mm)
        if j is None:
            return None
        n = self.case.nodules[j]
        if n.mask.sum() < 4:
            return None
        sp = np.asarray(self.case.vgeom.spacing, float)[::-1]          # (z, y, x) mm
        verts, faces, _, _ = marching_cubes(np.pad(n.mask, 1).astype(np.float32), 0.5, spacing=tuple(sp))
        z0, _, y0, _, x0, _ = n.bbox_zyx
        corner = self.case.vgeom.index_to_world(np.array([x0, y0, z0], float))   # world of crop voxel (0,0,0)
        xyz = (verts - sp)[:, ::-1] + corner                          # undo padding, to world (x, y, z)
        return xyz, faces
