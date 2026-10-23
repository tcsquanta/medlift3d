#!/usr/bin/env python
"""One time-budgeted run for a Kaggle session: prepare data, then train until the budget is used.

    python scripts/kaggle_run.py --total-hours 9 --n-patients 40

Runs steps 2-7 and then step 8 (diffusion training) with whatever time is left of
``--total-hours`` minus ``--reserve-hours``; step 8 stops by itself at its budget, after
evaluating and saving ``work/models/diffusion/{best,last}.pt``. Finished work is skipped when the
script is started again (steps 3-6 skip finished cases; use --resume-training to continue step 8).
Afterwards: ``python scripts/12_demo_app.py --share`` (see docs/GPU_VASTAI.md).

Use "Save Version -> Save & Run All (Commit)" so the run survives a closed browser tab.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def training_budget_hours(total_hours: float, elapsed_hours: float, reserve_hours: float) -> float:
    """Hours left for step 8 after preparation, keeping ``reserve_hours`` for saving the outputs."""
    return max(0.0, total_hours - elapsed_hours - reserve_hours)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-hours", type=float, default=9.0, help="session length to plan for")
    ap.add_argument("--reserve-hours", type=float, default=0.4, help="kept free at the end (saving outputs)")
    ap.add_argument("--n-patients", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4, help="parallel TCIA downloads")
    ap.add_argument("--batch-size", type=int, default=4, help="step 8 (use 2 if the GPU runs out of memory)")
    ap.add_argument("--amp", default="fp16", help="fp16 for T4 / P100 (no native bf16)")
    ap.add_argument("--sample-every", type=int, default=1000, help="step 8 validation / checkpoint interval")
    ap.add_argument("--min-train-hours", type=float, default=2.0, help="refuse to train if less than this is left")
    ap.add_argument("--keep-dicom", action="store_true", help="keep the DICOM files (default: delete after step 4)")
    ap.add_argument("--gpus", type=int, default=None,
                    help="GPUs for step 8 (default: all visible, e.g. 2 on Kaggle 'GPU T4 x2')")
    ap.add_argument("--resume-training", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    t0 = time.time()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation
    py = sys.executable

    def run(*cmd, launcher=()) -> None:
        line = " ".join([*launcher, os.path.relpath(cmd[0], ROOT)] + [str(c) for c in cmd[1:]])
        print(f"\n##### [{(time.time() - t0) / 3600:.2f} h] python {line}", flush=True)
        if args.dry_run:
            return
        code = subprocess.call([py, "-u", *launcher, *[str(c) for c in cmd]], cwd=ROOT)
        if code != 0:
            sys.exit(f"step failed with exit code {code}: {line}")

    S = lambda n: os.path.join(ROOT, "scripts", n)  # noqa: E731
    run(S("02_select_lidc.py"), "--n-patients", args.n_patients)
    run(S("03_download_lidc.py"), "--workers", args.workers)
    run(S("04_prepare_cases.py"), "--source", "lidc")
    if not args.keep_dicom and not args.dry_run:
        from medlift3d.config import load_config

        root = os.path.join(ROOT, load_config(os.path.join(ROOT, "configs", "default.yaml"))["paths"]["lidc_root"])
        if os.path.isdir(root):
            shutil.rmtree(root)
            print(f"deleted {root} (the cases are built; saves ~4.5 GB of the 20 GB output limit)")
    run(S("05_simulate_dts.py"))
    run(S("06_reconstruct_sirt.py"))
    run(S("07_build_roi_dataset.py"), "--jitter", 3, "--negatives", 2)

    elapsed = (time.time() - t0) / 3600
    budget = training_budget_hours(args.total_hours, elapsed if not args.dry_run else 1.5, args.reserve_hours)
    print(f"\npreparation took {elapsed:.2f} h; training budget {budget:.2f} h "
          f"(= {args.total_hours:g} total - preparation - {args.reserve_hours:g} reserve)", flush=True)
    if budget < args.min_train_hours and not args.dry_run:
        sys.exit(f"only {budget:.2f} h left for training (< {args.min_train_hours:g} h): lower --n-patients, "
                 "or start a new session and run step 8 on its own")
    cmd = [S("08_train_diffusion.py"), "--steps", 1_000_000, "--max-hours", f"{budget:.3f}", "--batch-size", args.batch_size,
           "--amp", args.amp, "--sample-every", args.sample_every]
    if args.resume_training:
        cmd.append("--resume")
    gpus = args.gpus
    if gpus is None:
        import torch

        gpus = torch.cuda.device_count()
    # several GPUs: one training process per GPU (DistributedDataParallel), --batch-size samples each
    launcher = ("-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(gpus)) if gpus > 1 else ()
    print(f"step 8 on {max(gpus, 1)} {'GPU' if gpus else 'CPU'} process(es)", flush=True)
    run(*cmd, launcher=launcher)
    print("\nDone. Checkpoints: work/models/diffusion/best.pt (best nodule-box gain) and last.pt.\n"
          "Demo: python scripts/12_demo_app.py --share", flush=True)


if __name__ == "__main__":
    main()
