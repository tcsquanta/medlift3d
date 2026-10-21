# MedLift-3D

**Click-to-3D lung nodule mass and growth assessment from low-dose chest digital tomosynthesis (DTS), with calibrated uncertainty.**

Final Year Design Project, Dept. of CSE, United International University -
Jubair Ahmed, Foyez Ahammed Naeem, Md Motasim Billah.

## The idea in four lines

1. DTS (15 low-dose X-rays over 30 degrees) finds lung nodules well but **cannot see depth** (the missing wedge).
2. Every view still measures a nodule's **mass** exactly; only its **depth extent**, which volume needs, is unseen
   (projection moments: mass 1x, depth position ~4x, depth extent ~15x worse conditioned).
3. So we reconstruct only a **column through the chest at the clicked nodule** with a
   **conditional 3D diffusion prior + 3D Gaussians kept consistent with the X-rays**, and report
   **mass and volume with calibrated 90% intervals**.
4. Growth is tracked with **mass doubling time (MDT)** alongside volume doubling time (VDT).

Details, theory, hypotheses (H1-H4) and novelty: [`docs/PROPOSAL.md`](docs/PROPOSAL.md).

```
CT (LIDC-IDRI) --simulate--> 15-view low-dose DTS --SIRT--> coronal DTS slices --(candidates + click)-->
prism ROI (64 x 256 x 64 mm column) --> diffusion prior x K (+ X-ray consistency) --> 3D Gaussians in the
nodule box --> mass (closed form) / volume (continuous field) --> conformal intervals --> VDT / MDT
```

## Quick start

```bash
pip install -r requirements.txt          # on Kaggle / Vast.ai keep the preinstalled torch
python -m pytest -q                      # 45 unit tests: physics, Gaussians, measurement, diffusion, LIDC masks, DICOM, download, pipeline
python scripts/smoke_test.py --fresh     # every step end-to-end on synthetic phantoms (CPU, ~70 min on 4 cores)
```

## Real data: run the steps in order

Each script prints **PASS / FAIL** checks and saves figures in `outputs/<step>/`
(what to look for: [`docs/VERIFY.md`](docs/VERIFY.md)). GPU needs and times per step:
[`docs/GPU_VASTAI.md`](docs/GPU_VASTAI.md). Steps 5, 6, 7, 9 and 11 accept `--cases <case_id> ...` to (re)run
only some cases, e.g. `python scripts/06_reconstruct_sirt.py --cases LIDC-IDRI-0012_457842 --overwrite`.

| step | command | what it does |
|---|---|---|
| 0 | `python scripts/00_check_env.py` | versions, GPU, projector speed |
| 1 | `python scripts/01_verify_physics.py` | X-ray physics checks (mass measured by every view, missing wedge, ...) |
| 2 | `python scripts/02_select_lidc.py --n-patients 200` | choose LIDC-IDRI nodules from the radiologist annotations (offline) |
| 3 | `python scripts/03_download_lidc.py --workers 8` | download only those CT series from TCIA (~110 MB each) |
| 4 | `python scripts/04_prepare_cases.py --source lidc` | CT + 4-radiologist consensus masks, alignment checks, patient split |
| 5 | `python scripts/05_simulate_dts.py` | 15-view low-dose DTS per CT |
| 6 | `python scripts/06_reconstruct_sirt.py` | standard DTS reconstruction + candidate nodules |
| 7 | `python scripts/07_build_roi_dataset.py` | (DTS column, CT column) training pairs |
| 8 | `python scripts/08_train_diffusion.py --batch-size 8` | **train the diffusion prior (rented GPU, ~3-4 h on RTX 4090)** |
| 9 | `python scripts/09_run_pipeline.py --members 8` | all methods on held-out nodules |
| 10 | `python scripts/10_evaluate.py` | accuracy tables, calibrated intervals, H1-H3 |
| 11 | `python scripts/11_growth_experiment.py` | virtual follow-up scans: VDT vs MDT, H4 |
| 12 | `python scripts/12_demo_app.py --share` | interactive click-to-3D demo (Gradio) |

### Why LIDC-IDRI

LIDC-IDRI has up to four radiologists' 3D outlines and ratings for every nodule. The
"Radiologist Annotations" download on the TCIA page is the database bundled with `pylidc`, so
masks come for free. Only the CT images need downloading, and step 2 picks the ~200 most useful
patients (~22 GB instead of 133 GB). LUNA16 (same scans as `.mhd`) works too: `--source luna16`.

## Repository layout

```
medlift3d/            library
  geometry.py         DTS geometry (15 views, +-15 deg, SID 1800 mm), voxel grids
  projector.py        differentiable ray-marching X-ray projector (pure PyTorch)
  simulate.py         CT -> DTS projections, Poisson noise, local-change projection
  sirt.py             SIRT reconstruction
  roi.py              prism ROI, ray selection, background subtraction, nodule box
  diffusion/          3D U-Net, DDPM/DDIM with data consistency, dataset, checkpoints
  gaussians.py        3D Gaussian field, voxeliser, closed-form mass, X-ray refinement
  measure.py          segmentation, volume, mass, 2D diameter
  uncertainty.py      ensembles, split-conformal intervals, sparsification
  growth.py           virtual growth, VDT/MDT, NELSON categories
  detect.py           training-free candidate detector
  pipeline.py         end-to-end per-nodule processing (all methods)
  data/               LIDC-IDRI (pylidc + DICOM), LUNA16, cases
  demo.py             demo backend
scripts/              steps 00-12 + smoke_test.py
tests/                pytest unit tests
configs/              default.yaml (real runs), smoke.yaml (CPU smoke test)
docs/                 PROPOSAL, VERIFY, GPU_VASTAI
```

## Limitations

Simulation study: no public chest DTS + CT dataset exists. Monochromatic X-rays, no scatter.
Virtual growth is isotropic. The candidate detector is training-free and modest. Radiologist
"ground truth" varies between readers (reported as context).
