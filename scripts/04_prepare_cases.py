#!/usr/bin/env python
"""Step 4 - build "cases": CT volume (HU) + radiologist consensus masks.

Sources:
  --source lidc     DICOM under paths.lidc_root + pylidc annotations (default)
  --source luna16   .mhd files (--mhd-dir) + pylidc annotations
  --source phantom  synthetic chest phantoms (no download; for testing)

Verification per case: DICOM slice positions == pylidc slice positions,
nodule masks contain nodule-like HU, mask volume vs radiologists' volume.
Also writes the patient-level train / cal / test split (work/splits.json).

    python scripts/04_prepare_cases.py --source lidc
    python scripts/04_prepare_cases.py --source phantom --n-phantoms 6
"""
import glob
import os

from _common import base_parser, setup

import numpy as np

from medlift3d.data.cases import case_from_phantom, load_case, save_case
from medlift3d.io import Paths, load_json, make_splits, save_json
from medlift3d.phantom import make_chest_phantom
from medlift3d.utils import Checker
from medlift3d import viz


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--source", choices=["lidc", "luna16", "phantom"], default="lidc")
    ap.add_argument("--mhd-dir", default=None)
    ap.add_argument("--n-phantoms", type=int, default=6)
    ap.add_argument("--phantom-spacing", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--figures", type=int, default=6, help="overlay figures for the first N cases")
    args = ap.parse_args()
    cfg, _ = setup(args)
    paths = Paths(cfg)
    out = paths.out("04_cases")
    ck = Checker(f"Step 4: prepare cases ({args.source})")
    built, n_nod, failures = [], 0, []
    hu_in, vol_ratio = [], []

    if args.source == "phantom":
        for i in range(args.n_phantoms):
            ph = make_chest_phantom(spacing=(args.phantom_spacing,) * 3, seed=100 + i, n_random_nodules=5)
            case = case_from_phantom(ph, f"phantom{i:03d}")
            save_case(case, paths.case(case.case_id))
            built.append(case.case_id)
            n_nod += len(case.nodules)
            for n in case.nodules:
                hu_in.append(float(case.hu[n.full_mask(case.hu.shape)].mean()))
    else:
        if args.source == "lidc":
            from medlift3d.data.lidc import build_case

            with open(paths.p("lidc_series.txt")) as f:
                items = [l.strip() for l in f if l.strip()]
        else:
            from medlift3d.data.luna16 import build_case_luna16

            items = sorted(glob.glob(os.path.join(args.mhd_dir, "*.mhd")))
        if args.limit:
            items = items[: args.limit]
        for k, it in enumerate(items, 1):
            try:
                case, info = build_case(it, paths.lidc_root) if args.source == "lidc" else build_case_luna16(it)
            except Exception as e:
                failures.append((it, str(e)))
                print(f"[{k}/{len(items)}] FAILED {os.path.basename(str(it))[-16:]}: {e}")
                continue
            if not case.nodules:
                print(f"[{k}/{len(items)}] {case.case_id}: no qualifying nodules, skipped")
                continue
            save_case(case, paths.case(case.case_id))
            built.append(case.case_id)
            n_nod += len(case.nodules)
            for n in case.nodules:
                m = n.full_mask(case.hu.shape)
                hu_in.append(float(case.hu[m].mean()))
                vp = n.meta.get("volume_mm3_pylidc")
                if vp:
                    vol_ratio.append(float(m.sum() * case.vgeom.voxel_volume / vp))
            print(f"[{k}/{len(items)}] {case.case_id}: {len(case.nodules)} nodules, shape {case.hu.shape}, "
                  f"spacing {tuple(round(s, 2) for s in case.vgeom.spacing)}, slice dz err {info['max_slice_dz_mm']:.3f} mm")

    ck.check("cases built", len(built) > 0, f"{len(built)} cases, {n_nod} nodules")
    if failures:
        ck.check("cases without errors", False, f"{len(failures)} failed (see log above)")
    hu_in = np.array(hu_in)
    ck.check("nodule masks contain nodule-like tissue (median mean HU > -600)",
             len(hu_in) > 0 and np.median(hu_in) > -600, f"median {np.median(hu_in):.0f} HU" if len(hu_in) else "n/a")
    if vol_ratio:
        vr = np.array(vol_ratio)
        ck.check("consensus-mask volume / pylidc mean reader volume (median)", 0.5 < np.median(vr) < 1.5,
                 f"{np.median(vr):.2f}", "0.5 - 1.5")
    # overlays
    for cid in built[: args.figures]:
        case = load_case(paths.case(cid), load_readers=False)
        for j, n in enumerate(case.nodules[:2]):
            viz.save_orthoslices(os.path.join(out, f"{cid}_nod{j}.png"), case.hu, case.vgeom, n.center_xyz,
                                 n.full_mask(case.hu.shape), f"{cid} d={n.diameter_mm:.1f}mm",
                                 zoom_mm=max(30.0, 4 * n.diameter_mm))
    # splits (merge with an existing file so re-runs keep assignments)
    sp_path = paths.p("splits.json")
    old = load_json(sp_path) if os.path.exists(sp_path) else {}
    pats = {}
    for cid in paths.case_ids():
        c = load_case(paths.case(cid), load_readers=False)
        pats[cid] = c.patient_id
    new_pats = [p for p in set(pats.values()) if p not in old.get("patients", {})]
    assign = dict(old.get("patients", {}))
    assign.update(make_splits(new_pats, seed=cfg["seed"]))
    save_json({"patients": assign, "cases": {cid: assign[p] for cid, p in pats.items()}}, sp_path)
    counts = {s: sum(1 for v in assign.values() if v == s) for s in ("train", "cal", "test")}
    ck.info("patient split train/cal/test", counts)
    ck.finish(out)


if __name__ == "__main__":
    main()
