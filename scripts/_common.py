"""Shared CLI plumbing for the step scripts (adds the repo to sys.path)."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from medlift3d.config import load_config  # noqa: E402
from medlift3d.utils import get_device, seed_everything  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--work", default=None, help="override paths.work")
    p.add_argument("--outputs", default=None, help="override paths.outputs")
    p.add_argument("--cases", nargs="+", default=None, help="only these case ids (default: all)")
    return p


def pick_cases(ids, args):
    """Keep only the ``--cases`` given on the command line (all if none), in the given order."""
    if not getattr(args, "cases", None):
        return list(ids)
    missing = set(args.cases) - set(ids)
    if missing:
        print(f"note: --cases not in this selection (wrong split or id?): {sorted(missing)}")
    return [c for c in ids if c in set(args.cases)]


def setup(args):
    cfg = load_config(args.config)
    if args.work:
        cfg["paths"]["work"] = args.work
    if args.outputs:
        cfg["paths"]["outputs"] = args.outputs
    seed_everything(int(cfg["seed"]))
    dev = get_device(args.device)
    return cfg, dev
