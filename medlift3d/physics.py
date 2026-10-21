"""X-ray physics helpers: HU <-> linear attenuation, Poisson (low-dose) noise."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

MU_WATER = 0.02  # mm^-1, water at ~60-70 keV effective energy (120 kVp beam)


def hu_to_mu(hu, mu_water: float = MU_WATER):
    """Linear attenuation (mm^-1) from Hounsfield units; air (-1000 HU) -> 0."""
    if isinstance(hu, torch.Tensor):
        return (mu_water * (1.0 + hu / 1000.0)).clamp(min=0.0)
    return np.clip(mu_water * (1.0 + np.asarray(hu, dtype=np.float32) / 1000.0), 0.0, None)


def mu_to_hu(mu, mu_water: float = MU_WATER):
    return 1000.0 * (mu / mu_water - 1.0)


def add_poisson_noise(p_clean: torch.Tensor, i0: float, generator: Optional[torch.Generator] = None,
                      electronic_sigma: float = 0.0, min_counts: float = 1.0) -> torch.Tensor:
    """Low-dose measurement model.

    counts ~ Poisson(i0 * exp(-p)) (+ Gaussian read noise); returns -log(counts / i0).
    """
    expected = float(i0) * torch.exp(-p_clean.float())
    counts = torch.poisson(expected, generator=generator)
    if electronic_sigma > 0:
        counts = counts + electronic_sigma * torch.randn(counts.shape, generator=generator,
                                                         device=counts.device)
    counts = counts.clamp(min=min_counts)
    return -torch.log(counts / float(i0))


def transmission_snr(p: float, i0: float) -> float:
    """SNR of a single detector pixel behind line integral p (Poisson): sqrt(i0 e^-p)."""
    return float(np.sqrt(i0 * np.exp(-p)))
