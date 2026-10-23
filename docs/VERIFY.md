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

Run the unit tests any time: `python -m pytest -q` (45 tests, about 5-10 min on CPU).

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
* PASS: all series downloaded. Failed series are listed in `work/lidc_failed.txt`. Simply re-run
  (finished series are skipped; an interrupted series is never mistaken for a finished one).
* PASS: every series has at least as many DICOM files as the pylidc database lists slices.
* Real example (4 patients): `4/4 complete`, 0.45 GB in 30 s.

## Step 4 - cases (`04_prepare_cases.py`)
* Per case: `slice dz err 0.000 mm` = DICOM slice positions equal pylidc's (mask alignment).
* PASS: masks contain nodule-like HU (median > -600); consensus-mask volume / readers' volume in 0.5-1.5.
* Real example (4 patients, 39 nodules): dz err 0.000 mm in every case, median -202 HU, volume ratio 0.81.
* Figures: `<case>_nod0.png` -- the red contour must sit exactly on the nodule in all 3 views
  (top row: whole slice; bottom row: zoomed on the nodule).

## Step 5 - DTS simulation (`05_simulate_dts.py`)
* PASS: line integrals physical (max < 15); median nodule contrast-to-noise in one view > 1.
* Real example (LIDC-IDRI-0012): max line integral 6.83, median nodule CNR 3.5.
* Figures: `<case>_views.png` -- radiograph-like views, red circle on the nodule in every view.
  The 15 views look alike (only +-15 deg apart); `<case>_parallax.png` shows how they differ: the
  nodule moves against the ribs / diaphragm by an amount set by its depth (what DTS reconstructs from).

## Step 6 - coarse SIRT (`06_reconstruct_sirt.py`)
* The volume is confined in depth to the patient (detector to back; `support_y` recorded by step 5).
* PASS: residual drops; depth/in-plane FWHM ratio > 1.3 (missing wedge visible).
* INFO: candidate detector recall@10 / @3 (training-free; modest on purpose -- the doctor can click anywhere).
* Figures: `<case>_sirt.png` vs `<case>_ct.png` (coronal similar; sagittal smeared).

## Step 7 - diffusion training pairs (`07_build_roi_dataset.py`)
* PASS: pairs written (train / val split by patient).
* PASS: the condition keeps the prism's total attenuation (sum of mu, condition / CT, median 0.85-1.15).
  DTS measures each column's attenuation well, so a low ratio means the background subtraction is
  wrong (this check caught the depth-smear bug: 0.67-0.81 on real LIDC before the fix).
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
* PASS: virtual growth construction exact (added mass = (s^3 - 1) x nodule mass, < 3%; typical 0.2%).
* `table_growth.md`: category accuracy (NELSON), doubling-time error, false growth alarms and
  **not measurable** (a scan where the method found no nodule; counted as a wrong category,
  never as fast growth) for VDT vs MDT.
* H4 line: MDT category accuracy >= VDT, and most pairs must be measurable.
* `--limit` is the total number of nodules: re-running with the same limit only rebuilds the tables.

## Step 12 - demo (`12_demo_app.py`)
* `--selftest`: loads a case, analyses a nodule, builds the 3D mesh, simulates a follow-up, builds the
  UI and clicks through it (9 checks). `INFO method used` says which method produced the result: if the
  diffusion model finds no nodule at a click, the demo shows the physics-only result and says so.
* Intervals are "calibrated 90%" after step 10 with >= 19 calibration nodules, otherwise labelled
  "uncalibrated +-2 sd".
* `python scripts/12_demo_app.py --share` -> public link for the defense.

## Smoke test (`smoke_test.py`)
Runs steps 0-12 on synthetic phantoms with tiny settings (CPU, ~45 min on 4 cores) and prints one table.
"SCIENCE-FAIL" lines are expected at this scale (tiny model trained for minutes on 2-3 mm voxels;
H1, H2, H3b and H4 typically fail there); any plain "FAIL" or a crashed step is a real problem.
Expected last line: `steps run: 11, crashed: none, code checks failed: 0, ...`.
Re-run single steps with `--only 09 10`.
