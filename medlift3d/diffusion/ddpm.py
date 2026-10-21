"""DDPM training objective and DDIM sampling with physics data-consistency.

Training (v-prediction by default, cosine schedule):
    x_t = sqrt(a_t) x_0 + sqrt(1 - a_t) eps,   v = sqrt(a_t) eps - sqrt(1 - a_t) x_0
    loss = || v - v_theta([x_t, c], t) ||^2

Sampling (DDIM, optional data consistency "DDS-style"): at each step the
denoised estimate x0_hat is mapped to attenuation, pushed through a few SIRT
iterations against the measured ROI projections, and mapped back. This keeps
every sample consistent with the X-rays in the directions they measure while
the prior fills in the directions they do not (depth).
"""
from __future__ import annotations

import copy
import math
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn

HU_LO, HU_HI = -1000.0, 400.0


def hu_to_unit(hu):
    """HU in [-1000, 400] -> [-1, 1] (values outside are not clipped here)."""
    return (hu - (HU_LO + HU_HI) / 2.0) / ((HU_HI - HU_LO) / 2.0)


def unit_to_hu(u):
    return u * ((HU_HI - HU_LO) / 2.0) + (HU_LO + HU_HI) / 2.0


def cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class GaussianDiffusion:
    """prediction='v' (default): the network predicts v = sqrt(a) eps - sqrt(1-a) x0.
    At high noise this forces it to predict the clean image from the condition,
    which trains much faster than epsilon prediction for conditional tasks."""

    def __init__(self, timesteps: int = 1000, prediction: str = "v"):
        self.T = int(timesteps)
        self.prediction = prediction
        self.ac = cosine_alphas_cumprod(self.T)                     # alpha-bar_t, t = 0..T-1

    def _a(self, t: torch.Tensor, device) -> torch.Tensor:
        return self.ac.to(device)[t].view(-1, 1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self._a(t, x0.device)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model: nn.Module, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        out = model(torch.cat([xt, cond], dim=1), t)
        if self.prediction == "v":
            a = self._a(t, x0.device)
            target = a.sqrt() * noise - (1 - a).sqrt() * x0
        else:
            target = noise
        return ((out.float() - target.float()) ** 2).mean()

    def split(self, out: torch.Tensor, x: torch.Tensor, a_t: torch.Tensor):
        """Network output -> (eps, x0) at noise level a_t."""
        if self.prediction == "v":
            x0 = a_t.sqrt() * x - (1 - a_t).sqrt() * out
            eps = (1 - a_t).sqrt() * x + a_t.sqrt() * out
        else:
            eps = out
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp(min=1e-4)
        return eps, x0

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, cond: torch.Tensor, steps: int = 50, eta: float = 0.0,
                    generator: Optional[torch.Generator] = None, dc_fn: Optional[Callable] = None,
                    dc_start_frac: float = 0.5, clip: float = 1.5, x_init: Optional[torch.Tensor] = None,
                    return_traj: bool = False, start_frac: float = 1.0):
        """Sample x_0 given condition ``cond`` [B,1,D,H,W] (unit scale).

        dc_fn(x0_hat) -> x0_hat' applies data consistency (unit scale in/out).
        start_frac < 1: "come-closer-diffuse-faster" -- start at t = start_frac * T
        from the condition diffused forward to that noise level instead of pure
        noise (fewer steps, closer to the measured data).
        """
        dev = cond.device
        B = cond.shape[0]
        shape = (B, 1) + tuple(cond.shape[2:])
        t_start = int(round(min(1.0, max(0.05, start_frac)) * (self.T - 1)))
        if x_init is None:
            noise = torch.randn(shape, generator=generator, device=dev)
            if t_start < self.T - 1:
                a0 = self.ac.to(dev)[t_start]
                x = a0.sqrt() * cond + (1 - a0).sqrt() * noise
            else:
                x = noise
        else:
            x = x_init
        ts = np.linspace(t_start, 0, steps).round().astype(int)
        ac = self.ac.to(dev)
        traj = []
        for i, t in enumerate(ts):
            tt = torch.full((B,), int(t), device=dev, dtype=torch.long)
            a_t = ac[t]
            a_prev = ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            eps, x0 = self.split(model(torch.cat([x, cond], dim=1), tt).float(), x, a_t)
            x0 = x0.clamp(-clip, clip)
            if dc_fn is not None and i >= int(dc_start_frac * len(ts)):
                x0 = dc_fn(x0)
                eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp(min=1e-6)
            if return_traj:
                traj.append(x0.detach().cpu())
            if i + 1 == len(ts):
                x = x0
                break
            sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp(min=0).sqrt()
            noise = torch.randn(shape, generator=generator, device=dev) if eta > 0 else 0.0
            x = a_prev.sqrt() * x0 + (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * noise
        return (x, traj) if return_traj else x


class EMA:
    """Exponential moving average of model weights (used for sampling).

    Warm-up: the effective decay is min(decay, (1 + n) / (10 + n)) after n
    updates, so short runs still produce a usable EMA model.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.n_updates = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n_updates += 1
        d = min(self.decay, (1.0 + self.n_updates) / (10.0 + self.n_updates))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)
