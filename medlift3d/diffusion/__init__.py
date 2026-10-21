"""Conditional 3D diffusion prior for nodule ROIs.

The model learns p(CT ROI | DTS ROI reconstruction): given the physics-only
ROI-SIRT volume (depth-smeared), it proposes plausible CT-like nodule volumes.
Sampling K times gives K hypotheses for what the X-rays cannot see (mostly
depth); data-consistency steps keep every hypothesis faithful to the X-rays.
"""
from .unet3d import UNet3D
from .ddpm import GaussianDiffusion, EMA, hu_to_unit, unit_to_hu

__all__ = ["UNet3D", "GaussianDiffusion", "EMA", "hu_to_unit", "unit_to_hu"]
