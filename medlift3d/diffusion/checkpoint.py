"""Save / load diffusion checkpoints (model + EMA + optimiser + config)."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from .ddpm import GaussianDiffusion
from .unet3d import UNet3D


def build_model(dcfg: dict) -> UNet3D:
    return UNet3D(in_ch=2, out_ch=1, base_ch=int(dcfg["base_ch"]), ch_mults=tuple(dcfg["ch_mults"]),
                  num_res=int(dcfg["num_res"]), attn_levels=tuple(dcfg["attn_levels"]),
                  dropout=float(dcfg["dropout"]))


def save_checkpoint(path: str, model, ema, opt, step: int, dcfg: dict, extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict() if ema is not None else None,
                "opt": opt.state_dict() if opt is not None else None, "step": int(step), "dcfg": dcfg,
                "extra": extra or {}}, path)


def load_for_sampling(path: str, device="cpu", use_ema: bool = True) -> Tuple[UNet3D, GaussianDiffusion, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    dcfg = ck["dcfg"]
    model = build_model(dcfg)
    sd = ck["ema"] if (use_ema and ck.get("ema") is not None) else ck["model"]
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, GaussianDiffusion(int(dcfg["timesteps"]), dcfg.get("prediction", "eps")), dcfg
