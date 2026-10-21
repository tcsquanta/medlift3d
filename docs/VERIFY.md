# How to verify that every step works

Every script ends with a block like

```
[PASS] 5a every view measures the mass (max |detector integral*cos(g)/M^2 - mass|/mass): 0.0002826  (expected < 0.02)
...
Step 1: X-ray physics: 8 passed, 0 failed -> ALL PASS
```

and writes `outputs/<step>/verification.md` + `verification.json` and figures. A **FAIL** in a
code check means something is wrong: stop and look before continuing. Lines starting with
**H1-H4** are *scientific results*, not code checks; they may honestly come out either way.

Run the unit tests any time: `python -m pytest -q` (about 5 min on CPU; 30+ tests).

## Step 0 - environment (`00_check_env.py`)
* PASS: torch, numpy, scipy, scikit-image, matplotlib, yaml, pydicom import; pylidc database has 1018 scans.
* INFO: GPU name, projector speed (CPU ~2 s, GPU ~10-50 ms per 64^3 forward+backward).

## Step 1 - physics (`01_verify_physics.py`) -- the most important check
| check | meaning | typical value |
|---|---|---|
| 1 sphere chord | line integrals match the exact path length through a sphere | < 1% error |
| 2 adjoint | back-projection is the exact transpose of projection | ~0 |
| 3 noise | Poisson noise has the theoretical std | ratio 1.00 |
| 4 parallax | a blob 40 mm above the isocentre moves across the views exactly as geometry predicts | < 0.01 px |
| 5a mass | **every view measures the nodule mass** (detector integral x cos(gamma)/M^2) | < 0.1% |
| 5b depth | a 3x deeper blob with the same mass gives the same detector integral | < 0.1% |
| 6a SIRT | DTS reconstruction residual falls | 1.0 -> 0.02 |
| 6b missing wedge | nodule appears ~3x longer in depth than in-plane | ratio > 1.5 |

Figures: `phantom_dts_views.png` (15 X-ray views; the red circle follows the nodule as the tube
moves), `phantom_dts_sirt.png` (coronal = sharp nodule, sagittal = depth smear), `phantom_ct.png`.

## Step 2 - selection (`02_select_lidc.py`)
* PASS: qualifying nodules found; every chosen nodule has >= 3 readers.
* INFO: patients / series / nodules chosen, diameter distribution, expected download size.
* Figure: `selection_histograms.png` (diameter, malignancy, texture).

## Step 3 - download (`03_download_lidc.py`)
* PASS: all series downloaded. Failed series are listed in `work/lidc_failed.txt`. Simply re-run.

## Step 4 - cases (`04_prepare_cases.py`)
* Per case: `slice dz err 0.000 mm` = DICOM slice positions equal pylidc's (mask alignment).
* PASS: masks contain nodule-like HU (median > -600); consensus-mask volume / readers' volume in 0.5-1.5.
* Figures: `<case>_nod0.png` -- the red contour must sit exactly on the nodule in all 3 views.

## Step 5 - DTS simulation (`05_simulate_dts.py`)
* PASS: line integrals physical (max < 15); median nodule contrast-to-noise in one view > 1.
* Figures: `<case>_views.png` -- radiograph-like views, red circle on the nodule in every view.

## Step 6 - coarse SIRT (`06_reconstruct_sirt.py`)
* PASS: residual drops; depth/in-plane FWHM ratio > 1.3 (missing wedge visible).
* INFO: candidate detector recall@10 / @3 (training-free; modest on purpose -- the doctor can click anywhere).
* Figures: `<case>_sirt.png` vs `<case>_ct.png` (coronal similar; sagittal smeared).

## Step 7 - diffusion training pairs (`07_build_roi_dataset.py`)
* PASS: pairs written (train / val split by patient).
* INFO: PSNR of the physics-only condition vs CT -- **the bar the diffusion model must beat**.
* Figures: `pair_*.png` (condition = blurry / smeared, target = crisp CT).

## Step 8 - diffusion training (`08_train_diffusion.py`)
* First: `--tiny` sanity run (CPU, ~7 min) must PASS both checks (loss falls; samples beat the blurred condition).
* Real run: PASS loss decreased; **PASS "diffusion samples beat the physics-only condition"** is the
  key check of the trained model. Figures `val_step*.png` show condition / sample / CT side by side.

## Step 9 - all methods (`09_run_pipeline.py`)
* PASS: all nodules processed. INFO lines give median volume / mass error per method.
* Figures: `<case>_n<j>.png` -- rows = CT truth, input, each method; columns = coronal / sagittal (+ uncertainty std).

## Step 10 - evaluation (`10_evaluate.py`)
* `table_accuracy.md`, `table_intervals.md`, `summary.csv`, `conformal.json`, box plots, Bland-Altman, scatter.
* H1-H3 result lines (scientific). Intervals need >= 19 calibration nodules for 90% asymmetric intervals.

## Step 11 - growth (`11_growth_experiment.py`)
* PASS: virtual growth exact on CT (mass ratio within 10% of s^3).
* `table_growth.md`: category accuracy (NELSON) and doubling-time error for VDT vs MDT; H4 line.

## Step 12 - demo (`12_demo_app.py`)
* `--selftest`: loads a case, analyses a nodule, builds the 3D mesh, simulates a follow-up, builds the UI.
* `python scripts/12_demo_app.py --share` -> public link for the defense.

## Smoke test (`smoke_test.py`)
Runs steps 0-12 on synthetic phantoms with tiny settings (CPU, ~45-90 min) and prints one table.
"SCIENCE-FAIL" lines are expected at this scale (tiny model trained for minutes); any plain
"FAIL" or a crashed step is a real problem.
