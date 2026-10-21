"""SIRT reconstruction (the standard iterative DTS reconstruction).

    x <- x + lambda * C * A^T R (b - A x),   R = 1/(A 1),  C = 1/(A^T 1)

with non-negativity. Used for (1) the coarse whole-chest DTS volume that the
doctor scrolls through and that provides the out-of-ROI background, and
(2) the physics-only ROI baseline / diffusion condition.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from .projector import Projector


class SIRT:
    def __init__(self, projector: Projector, origins: torch.Tensor, dirs: torch.Tensor, device="cpu"):
        self.P = projector
        self.o = origins.to(device)
        self.d = dirs.to(device)
        self.device = device
        D, H, W = projector.vg.shape
        with torch.no_grad():
            ones = torch.ones((D, H, W), device=device)
            row = self.P(ones, self.o, self.d)
        col = self.P.backproject(torch.ones_like(row), self.o, self.d)
        self.Wr = torch.where(row > 1e-6, 1.0 / row.clamp(min=1e-6), torch.zeros_like(row))
        self.Wc = torch.where(col > 1e-6, 1.0 / col.clamp(min=1e-6), torch.zeros_like(col))
        self.row_sum = row
        self.seen = col > 1e-6

    def run(self, b: torch.Tensor, n_iter: int = 40, x0: Optional[torch.Tensor] = None, relax: float = 1.0,
            nonneg: bool = True, callback: Optional[Callable[[int, torch.Tensor, float], None]] = None
            ) -> Tuple[torch.Tensor, List[float]]:
        b = b.reshape(-1).to(self.device).float()
        D, H, W = self.P.vg.shape
        x = torch.zeros((D, H, W), device=self.device) if x0 is None else x0.detach().clone().to(self.device).float()
        bn = float(b.norm()) + 1e-12
        hist: List[float] = []
        for it in range(int(n_iter)):
            y, g = self.P.forward_and_adjoint(x, self.o, self.d, lambda yy: self.Wr * (b - yy))
            hist.append(float((b - y).norm()) / bn)
            x = x + relax * self.Wc * g
            if nonneg:
                x = x.clamp(min=0.0)
            if callback is not None:
                callback(it, x, hist[-1])
        with torch.no_grad():
            hist.append(float((b - self.P(x, self.o, self.d)).norm()) / bn)
        return x, hist
