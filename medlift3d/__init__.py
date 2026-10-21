"""MedLift-3D: click-to-3D lung nodule mass and growth assessment from
low-dose chest digital tomosynthesis (DTS).

Pipeline (each stage has a verification script under ``scripts/``):

1. ``simulate``   CT (LIDC-IDRI) -> 15-view low-dose DTS projections
2. ``sirt``       standard whole-chest DTS reconstruction (coarse)
3. ``detect``     candidate nodules on the coarse DTS volume; doctor clicks one
4. ``roi``        ROI box around the click + out-of-box background subtraction
5. ``diffusion``  conditional 3D diffusion prior proposes K plausible nodule ROIs
6. ``gaussians``  each proposal -> 3D Gaussians, refined to match the X-rays
7. ``measure``    nodule mass (closed form from Gaussians) and volume
8. ``uncertainty``/``growth``  calibrated intervals, VDT / MDT
"""

__version__ = "0.2.0"
