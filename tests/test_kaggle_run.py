"""The time-budgeted Kaggle driver: budget arithmetic and the printed plan (nothing is run)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_training_budget_arithmetic():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from kaggle_run import training_budget_hours

    assert abs(training_budget_hours(9.0, 1.5, 0.4) - 7.1) < 1e-9
    assert training_budget_hours(9.0, 9.0, 0.4) == 0.0                 # never negative


def test_dry_run_lists_every_step_in_order():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "kaggle_run.py"), "--dry-run", "--n-patients", "7"],
                         capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    order = [out.stdout.index(f"scripts/{s}") for s in ("02_select", "03_download", "04_prepare", "05_simulate",
                                                          "06_reconstruct", "07_build", "08_train")]
    assert order == sorted(order)
    assert "--n-patients 7" in out.stdout and "--max-hours 7.100" in out.stdout and "--amp fp16" in out.stdout
