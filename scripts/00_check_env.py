#!/usr/bin/env python
"""Step 0 - environment check: versions, GPU, and a projector speed benchmark.

    python scripts/00_check_env.py
"""
import importlib
import platform
import time

from _common import base_parser, setup

import numpy as np
import torch

from medlift3d.geometry import DTSGeometry, VolumeGeometry
from medlift3d.projector import Projector
from medlift3d.utils import Checker


def main():
    args = base_parser(__doc__).parse_args()
    cfg, dev = setup(args)
    ck = Checker("Step 0: environment")
    ck.info("python", platform.python_version())
    ck.check("torch >= 2.0", tuple(int(v) for v in torch.__version__.split(".")[:2]) >= (2, 0), torch.__version__)
    for mod in ("numpy", "scipy", "skimage", "matplotlib", "yaml", "pydicom"):
        try:
            m = importlib.import_module(mod)
            ck.check(f"import {mod}", True, getattr(m, "__version__", "ok"))
        except Exception as e:
            ck.check(f"import {mod}", False, str(e))
    try:
        from medlift3d.data.lidc import import_pylidc

        pl = import_pylidc()
        n = pl.query(pl.Scan).count()
        ck.check("pylidc annotation database", n > 1000, f"{n} scans")
    except Exception as e:
        ck.check("pylidc annotation database", False, f"{e} (pip install pylidc)")
    ck.info("device", str(dev))
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        ck.info("GPU", f"{p.name}, {p.total_memory / 2**30:.1f} GB")
    # benchmark: 64^3 ROI, 15 views x 80 x 80 rays (one ROI forward + backward)
    vg = VolumeGeometry.centered((64, 64, 64), (1.0, 1.0, 1.0))
    geom = DTSGeometry(det_shape=(80, 80), det_pitch=(1.0, 1.0))
    o, d = geom.rays(device=dev)
    P = Projector(vg, step_mm=0.5)
    x = torch.rand(vg.shape, device=dev, requires_grad=True)
    P(x, o, d).sum().backward()                                  # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        x.grad = None
        P(x, o, d).sum().backward()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 3
    ck.info("ROI projector fwd+bwd (64^3, 96k rays)", f"{dt * 1000:.0f} ms")
    ck.check("projector produced finite gradients", bool(torch.isfinite(x.grad).all()), "")
    ck.finish(None)


if __name__ == "__main__":
    main()
