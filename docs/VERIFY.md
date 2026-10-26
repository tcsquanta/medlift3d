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

Run the unit tests any time: `python -m pytest -q` (56 tests, about 5-10 min on CPU).

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
* Every check also prints `[fit] denoising loss ... train pairs A | held-out pairs B`: the same 32 training
  and 32 held-out pairs with the same noise every time. Both falling = still learning; B rising while A
  falls = overfitting (step 15 judges this).

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
* `--selftest`: loads a test patient, analyses a nodule, builds the 3D mesh, simulates a follow-up, builds
  the UI and clicks through it: AI suggestions, ground truth on request, "Reconstruct in 3D" on suggestion #1
  (report with the radiologists' values next to ours), a click on the nodule, the follow-up button.
  The 3D check tests the coordinates (the radiologists' surface sits on the annotated nodule); how close our
  reconstruction is to it is reported as INFO (an accuracy result, e.g. a juxtapleural nodule the diffusion
  model cannot separate from the chest wall, not a code error).
* The report lists every hypothesis (mass / volume, or "nothing"), says why a method found nothing ("too faint"
  or "merged with an adjacent structure"), and flags **low confidence** when most hypotheses found nothing or
  they disagree by more than 10x. `INFO method used` says which method produced the result: if the
  diffusion model finds no nodule at a click, the demo shows the physics-only result and says so.
* Intervals are "calibrated 90%" after step 10 with >= 19 calibration nodules, otherwise labelled
  "uncalibrated +-2 sd".
* `python scripts/12_demo_app.py --share` -> public link for the defense.

## Step 14 - nodule detector (`14_train_detector.py`)
A small 3D U-Net (0.34 M parameters) learns a nodule-centre heatmap on the coarse DTS volume from the LIDC
radiologists' nodules: trained on the *train* patients, checkpoint and operating threshold (2 false
positives per scan) chosen on the *cal* patients, evaluated once on the *test* patients.
* PASS: the three patient groups are disjoint, the training loss decreases, a checkpoint is saved.
* `[val] step ...: CPM ...` lines during training: CPM = mean sensitivity at 1/8 ... 8 false positives per
  scan (LUNA16); it should rise above the first check.
* `table_detection.md` + `froc_test.png` (scientific, report as it comes out): learned detector vs the
  training-free LoG detector on the same test patients and the same hit rule (within max(6 mm, r + 3 mm)
  in-plane and 12 mm + r in depth: a click there lets the measurement find the nodule).
* LIDC labels only nodules >= 4 mm marked by >= 3 of 4 radiologists: a suggestion on another real finding
  counts as a false positive, so the false-positive numbers are pessimistic.
* Diagnostic row "in-plane only (any depth)": the gap to the first row = suggestions at the right coronal
  position but the wrong depth. `--eval-only` re-evaluates the saved detector without training (~1 min).
* Measured (40-patient Kaggle run, Oct 25): test CPM 0.155 vs 0.024 for the training-free detector; recall in
  the top 10: 26% vs 11%; sensitivity 17% at 2 FP per scan. The cal CPM peaked at step 1000 (0.176) while the
  training loss kept falling: with 24 training patients the detector overfits; best.pt is the step-1000 model.
* The demo (step 12) uses `work/models/detector/best.pt` automatically when it exists.

## Step 13 - acquisition geometry (`13_geometry_sensitivity.py`)
Our DTS uses 15 views over a 30 deg linear sweep: the angular range of the GE VolumeRAD chest unit, but
fewer views than real units (60-74) to save compute, and Shimadzu units sweep up to 40 deg. This step
re-simulates the same patients with other geometries (default 15x30, 60x30, 15x40, 74x40; equal total
dose) and runs the physics-only methods on the same nodules.
* PASS: no errors, every geometry has results, the nodule is on the detector in every view.
* `table_geometry.md` (scientific, report it as it comes out):
  - *depth smear* and *nodule peak kept* come from the nodule's **artifact spread function**: the
    nodule alone (CT excess over lung) projected along the prism's rays and reconstructed with the
    same SIRT, so overlapping anatomy cannot hide its depth profile (the standard DTS depth-resolution
    measure). CT truth ~1 (a round nodule).
  - *views* (15x30 vs 60x30): the depth smear should barely change (it is set by the sweep angle);
    contrast / clutter may change a little (fewer streaks).
  - *angle* (15x30 vs 15x40): the depth smear should shrink, roughly with tan(half sweep), i.e.
    ~1.3x less at +-20 deg than at +-15 deg.
    Measured on a 16 mm phantom nodule (unit test setting): depth smear 3.85 (15x30), 3.99 (60x30),
    2.98 (15x40), 2.79 (15x60); nodule peak kept 27%, 26%, 34%, 36%.
  - mass error should depend much less on the geometry than volume error (H1 again).
* Figures `<case>_n<j>_roi_sirt.png` (the same nodule's ROI-SIRT per geometry next to the CT) and
  `<case>_n<j>_spread.png` (the nodule alone, reconstructed per geometry).
* Diffusion is not run here (trained on the 15x30 geometry only).

## Step 15 - under/overfitting check (`15_check_fit.py`)
* Reads `logs/08_train.log` and `logs/14_detector.log`; writes `outputs/15_fit/fit_report.md` (verdict per
  model: STILL IMPROVING / CONVERGED / OVERFITTING / NOT LEARNING, the reason and what to do) and
  `fit_curves.png` (training loss and validation curves, where `best.pt` was taken).
* `--gap`: the final `best.pt` models on training vs held-out data. Healthy: diffusion held-out / train
  denoising loss < 1.15; detector train CPM - cal CPM < 0.15.
* How to act on it: [`RUNPOD_FULL_RUN.md`, sections 7 and 9](RUNPOD_FULL_RUN.md#7-is-the-model-under--or-overfitting).

## Step 16 - uncertainty with conformal predictive systems (`16_uncertainty.py`)
* PASS: CPS 90% intervals cover ~90% of test nodules (volume, mass). The row "cross-check: step 10's own
  implementation" should match "normalised by ensemble spread" closely (same method, two implementations).
* Read the widths (normalised narrower than standard = the ensemble spread is informative), the per-size-group
  coverage (Mondrian should bring every group near 90%), and the Brier / ECE of P(V >= 100 / 300 mm3) (EUPS categories)
  (conformal vs ensemble vote). These are results (H5), not code checks. Details: [`XAI_UQ.md`](XAI_UQ.md).

## Step 17 - explainability (`17_explain.py`)
* PASS: the seen / unseen split is accurate (median |A unseen| / |A x| < 0.05).
* PASS: the true nodule's mass is seen (unseen share of the box mass < 10%) -- the mechanism of H1.
* Results (H6): unseen share of the true nodules (how much DTS cannot see), unseen skill of ours (> 0 = the
  prior recovers the invisible part, < 0 = it invents), error drivers, detector saliency true vs false
  positives. Figures `<nodule>_seen_unseen.png`: truth row = compact blob / depth-smeared seen part / sharp
  core with negative side lobes as unseen part; ROI-SIRT row has an (almost) empty unseen column.

## Smoke test (`smoke_test.py`)
Runs steps 0-12 on synthetic phantoms with tiny settings (CPU, ~45 min on 4 cores) and prints one table.
"SCIENCE-FAIL" lines are expected at this scale (tiny model trained for minutes on 2-3 mm voxels;
H1, H2, H3b and H4 typically fail there); any plain "FAIL" or a crashed step is a real problem.
Expected last line: `steps run: 11, crashed: none, code checks failed: 0, ...`.
Re-run single steps with `--only 09 10`.
