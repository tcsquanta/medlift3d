#!/usr/bin/env python
"""End-to-end smoke test: every step on synthetic chest phantoms, CPU only.

Proves the whole pipeline runs (data -> DTS -> SIRT -> diffusion training ->
reconstruction -> evaluation -> growth -> demo) before you spend GPU money.
Uses configs/smoke.yaml (tiny sizes); results are NOT representative.

    python scripts/smoke_test.py --fresh        # ~30-60 min on 4 CPU cores

At the end it prints one table of all verification checks. "science" checks
(the H1-H4 hypotheses, diffusion beating the physics baseline) need real data
and full training, so at smoke scale they are reported but do not fail the run.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "smoke.yaml")

STEPS = [
    ("00", "00_check_env.py", []),
    ("01", "01_verify_physics.py", ["--fast"]),
    ("04", "04_prepare_cases.py", ["--source", "phantom", "--n-phantoms", "6", "--phantom-spacing", "2.5"]),
    ("05", "05_simulate_dts.py", []),
    ("06", "06_reconstruct_sirt.py", []),
    ("07", "07_build_roi_dataset.py", ["--jitter", "3", "--negatives", "2"]),
    ("08", "08_train_diffusion.py", ["--steps", "600", "--lr", "1e-3", "--sample-every", "300", "--n-val", "2", "--workers", "0"]),
    ("09", "09_run_pipeline.py", ["--members", "3"]),
    ("10", "10_evaluate.py", []),
    ("11", "11_growth_experiment.py", ["--limit", "2", "--pairs-per-nodule", "2"]),
    ("12", "12_demo_app.py", ["--selftest"]),
]
SCIENCE = ("H1", "H2", "H3", "H4", "beat the physics-only", "candidate detector")


def is_science(name: str) -> bool:
    return any(name.startswith(k) or k in name for k in SCIENCE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="delete work_smoke/ and outputs_smoke/ first")
    ap.add_argument("--only", nargs="+", default=None, help="run only these step numbers, e.g. --only 09 10")
    ap.add_argument("--config", default=CFG)
    args = ap.parse_args()
    import yaml

    cfg = yaml.safe_load(open(args.config))
    work = os.path.join(ROOT, cfg["paths"]["work"])
    outs = os.path.join(ROOT, cfg["paths"]["outputs"])
    if args.fresh:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(outs, ignore_errors=True)
    t_all = time.time()
    status = {}
    for num, script, extra in STEPS:
        if args.only and num not in args.only:
            continue
        cmd = [sys.executable, os.path.join(ROOT, "scripts", script), "--config", args.config] + extra
        print(f"\n{'#' * 78}\n# STEP {num}: {' '.join(cmd[1:])}\n{'#' * 78}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=ROOT)
        status[num] = (rc, time.time() - t0)
        print(f"# step {num} finished with exit code {rc} in {time.time() - t0:.0f}s", flush=True)
    # summary of every verification.json -------------------------------------------------
    print("\n" + "=" * 78 + "\nSMOKE TEST SUMMARY\n" + "=" * 78)
    code_fail, sci_fail = 0, 0
    for f in sorted(glob.glob(os.path.join(outs, "*", "verification.json"))):
        v = json.load(open(f))
        for r in v["rows"]:
            if r["ok"] is None:
                continue
            sci = is_science(r["check"])
            tag = "PASS" if r["ok"] else ("SCIENCE-FAIL" if sci else "FAIL")
            if not r["ok"]:
                if sci:
                    sci_fail += 1
                else:
                    code_fail += 1
            print(f"[{tag:12s}] {v['title'][:34]:34s} | {r['check'][:70]} = {r['value'][:60]}")
    crashed = [n for n, (rc, _) in status.items() if rc not in (0, 1)]
    print("-" * 78)
    print(f"steps run: {len(status)}, crashed: {crashed or 'none'}, code checks failed: {code_fail}, "
          f"science checks not met at smoke scale: {sci_fail}, total time {(time.time() - t_all) / 60:.1f} min")
    ok = not crashed and code_fail == 0
    print("SMOKE TEST:", "OK - pipeline runs end to end" if ok else "PROBLEMS FOUND (see above)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
