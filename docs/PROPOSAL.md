# MedLift-3D: Click-to-3D Lung Nodule Mass and Growth Assessment from Low-Dose Chest Tomosynthesis with Calibrated Uncertainty

**Team:** Jubair Ahmed, Foyez Ahammed Naeem, Md Motasim Billah (UIU, CSE, FYDP)

## 1. Problem in one paragraph

Lung nodules are followed up by measuring how fast they grow (volume doubling time, VDT).
CT measures 3D size well but costs radiation and money for every follow-up. Digital chest
tomosynthesis (DTS) takes ~15 low-dose X-rays over a 30 degree sweep (~0.1 mSv, close to a
chest X-ray pair) and is good at *finding* nodules, but it cannot see depth: a 30 degree sweep
leaves a "missing wedge" of information, so any 3D volume measured from DTS is partly a guess.
Generative AI can fill the gap, but a model that invents ("hallucinates") anatomy is dangerous
for cancer follow-up.

## 2. Key insight: measure what DTS can actually see

*Every* projection measures the nodule's **mass** (its total extra attenuation over the
surrounding lung), but no projection measures how far it extends in depth.

Projection moments make this exact. For a projection direction tilted by angle theta from the
depth axis, with detector coordinate v along e_theta = (0, -sin theta, cos theta):

| quantity | what the projection gives | conditioning (sweep +-15 deg) |
|---|---|---|
| mass M (0th moment) | integral of p over the detector = M (x M^2 / cos gamma for cone beam) | 1x, every view |
| depth position (1st moment) | -sin(theta) M y_c + cos(theta) M z_c | 1/sin 15 deg ~ 3.9x worse than in-plane |
| depth extent (2nd moment) | sin^2(theta) S_yy - 2 sin cos S_yz + cos^2 S_zz | 1/sin^2 15 deg ~ 15x worse |

So mass is perfectly conditioned, while the depth extent that volume needs is ~15x
ill-conditioned. `scripts/01_verify_physics.py` checks this numerically: the detector
integral matches the mass to 0.03% in all 15 views, and a 3x deeper object of equal mass is
invisible to it (0.03%).

**Consequences we test:**
* **H1** For every DTS reconstruction, mass error < volume error.
* **H2** A learned prior (diffusion) + physics refinement reduces the volume error vs physics-only.
* **H3** Split-conformal intervals reach their nominal 90% coverage, and mass intervals are narrower than volume intervals.
* **H4** Mass doubling time (MDT, an established CT biomarker) classifies growth from DTS more accurately than VDT.

Any of these can come out false. The report must state whichever results we actually get.

## 3. Method (what the code does)

1. **Data:** LIDC-IDRI CT (TCIA) + radiologist annotations via `pylidc` (consensus mask
   = >= 50% of the 4 readers). Nodules 4-30 mm marked by >= 3 readers, slice thickness <= 2.5 mm.
2. **DTS simulation (virtual clinical trial):** the CT table is removed (largest body component),
   then 15 views, +-15 deg linear sweep, SID 1800 mm,
   43 cm flat panel (0.8 mm pixels), monochromatic attenuation, Poisson noise (I0 = 2e4/pixel).
   Reference system: chest DTS units move the tube on a *linear* track with a stationary (GE VolumeRAD)
   or counter-moving (Shimadzu SonialVision) detector; the angular range matches the GE VolumeRAD
   (+-15 deg, as in published VolumeRAD simulation models; Shimadzu allows up to 40 deg). Real units take
   60-74 views; we take 15 to save compute (every simulation / reconstruction step scales with the views).
   Step 13 re-runs the physics-only methods with 60/74 views and a 40 deg sweep at equal total dose to
   measure how much this choice matters. Measured on 12 LIDC nodules of 2 test patients (Oct 25; depth
   smear = depth / in-plane FWHM of the nodule's artifact spread function, CT 0.95): 5.13 at 15 views /
   30 deg, 5.34 at 60 views / 30 deg (4x the compute), 4.69 at 15 views / 40 deg, 4.61 at 74 views / 40 deg;
   nodule peak kept 18 / 17 / 20 / 20%. The number of views does not matter at equal dose; a 40 deg sweep
   reduces the smear by ~9% (median; ~23% on one patient, close to tan 20 / tan 15), so +-15 deg is the
   conservative case. Physics-only methods stay poor in every geometry (roi_gs finds 75-83% of these
   ~7 mm nodules with ~80-90% median error): their limit is the missing wedge + anatomical clutter
   (contrast / clutter 1.2-1.5 in all geometries), not our simplified acquisition. (Breast tomosynthesis uses 15-50 deg and arcs; not our setting.)
   Simulated at CT resolution with fine ray sampling; reconstructed on different grids (no inverse crime).
3. **Standard DTS reconstruction:** coarse whole-chest SIRT (2.5 mm), confined in depth to the patient
   (between the detector and the back, from the patient thickness that a DTS unit measures at acquisition;
   here the CT body outline + 5 mm). Without this support DTS smears attenuation into the air in front of and
   behind the patient, and on real LIDC chests the prism then lost ~1/3 of its attenuation as false
   "background". The doctor scrolls coronal slices;
   a nodule detector (a small 3D U-Net heatmap trained on the train patients' LIDC nodules, step 14;
   baseline: a training-free LoG + Hessian blob detector) suggests candidates; the click gives (x, z) and the slice gives depth y.
4. **Prism ROI ("zoom in-plane, not in depth"):** a 64 x 64 mm column through the whole chest
   (64 x 128 x 64 voxels, 1 x 2 x 1 mm; shifted back for deep nodules in thick chests). Rays are kept only
   if >= 50% of their attenuation lies in the prism, so the out-of-prism background (from the coarse SIRT)
   is a small correction. Check (step 7): the prism keeps the column's total attenuation (ratio ~1).
   (A small 3D box fails: each ray's attenuation is mostly outside the box and the background error
   swamps the nodule. We measured a 73% relative error for the box vs 5-7% for the prism.)
5. **Conditional 3D diffusion prior:** a 3D U-Net (~17M parameters, v-prediction, cosine schedule)
   learns p(CT prism | DTS-SIRT prism) from training patients (nodule ROIs + nodule-free lung ROIs).
   DDIM sampling with **data consistency** (a few SIRT steps on every denoised estimate) gives
   K = 8 hypotheses that all agree with the X-rays but differ where the X-rays are blind (depth).
6. **3D Gaussians:** for each hypothesis, the nodule box is represented by anisotropic 3D
   Gaussians (initialised from the sample, refined against the measured projections with the rest of the
   prism fixed). Gaussians give **closed-form mass** (a (2 pi)^1.5 s1 s2 s3 per Gaussian) and a continuous
   field that can be evaluated on a fine grid for resolution-free volume.
7. **Measurement:** seeded half-maximum segmentation (confined to a plausible nodule extent; leaks
   into the chest wall are detached or reported as not measurable), volume, excess mass over local lung,
   2D diameter baseline. Ground-truth mass = inside the radiologists' consensus mask + 1 mm rim
   (mass / volume 0.7-0.85 mg/mm^3 on real LIDC nodules). A reconstruction's mass window extends in
   depth by half the segmentation's own depth extent, so it covers the DTS smear without summing the
   neighbouring vessels / chest wall of a sharp reconstruction.
8. **Uncertainty:** ensemble spread -> split-conformal intervals calibrated on held-out patients
   (asymmetric, log scale). These give a finite-sample coverage guarantee, unlike raw ensemble variance.
9. **Growth:** virtual follow-up scans (nodule grown by a known factor in the CT, projections updated
   linearly, new noise) -> VDT and MDT vs the true doubling time, NELSON categories (<400 / 400-600 / >600 d).

## 4. Baselines and metrics

Baselines: 2D diameter -> sphere volume (clinical practice); standard DTS SIRT; prism ROI-SIRT
(physics only); physics-only Gaussians; diffusion without Gaussians.
Metrics: volume / mass error (median signed, MdAPE, Bland-Altman), failure rate, Dice / HD95 vs
consensus mask, PSNR / SSIM in the nodule box, interval coverage and width, Spearman(sigma, |error|),
doubling-time error and category accuracy, runtime. Inter-reader volume variability in LIDC is
reported as context.

## 5. Novelty (to the best of our literature search, Oct 2026)

| contribution | closest existing work | what is new |
|---|---|---|
| Mass-first DTS biomarker (MDT) justified by a projection-moment identifiability argument | Mass / MDT on CT (de Hoop et al. 2010; JTD studies); DTS growth studies use diameters / reader judgement | first DTS-specific argument + test |
| Click-to-3D prism ROI reconstruction for DTS with 3D Gaussians + closed-form mass | R2-Gaussian, 3DGR-CT (whole-volume sparse-view CT), no DTS | DTS geometry, prism ROI, Gaussian mass |
| Conditional diffusion prior with SIRT data consistency inside the prism | DDS / DPS for CT | applied to DTS nodule columns, combined with Gaussians |
| Calibrated intervals on clinical numbers (mass, volume, doubling time) | conformal CT metrics (2024); 3DGS-CT uncertainty preprint (Jul 2026) shows raw variance is a poor error map | biomarker-level calibration for DTS growth |

Do **not** claim: "first use of 3DGS in medical imaging", "first uncertainty for 3DGS", "first
conformal prediction in CT" -- all exist.

## 5b. First real-data observations (4 LIDC-IDRI patients, 39 nodules, CPU, Oct 21)

Measured with the pipeline before the GPU run (useful for Ch.4 / the defense):

* Data path: DICOM slice positions = pylidc's in every scan (0.000 mm); consensus masks sit on
  nodule tissue (median -202 HU); consensus volume / mean reader volume 0.81.
* Simulated low-dose DTS (I0 = 2e4 photons / 0.8 mm pixel / view, ~1/20-1/40 of clinical DTS dose):
  median nodule CNR in a single view 3.3; SIRT residual 1 -> 0.02; nodules ~3.5x longer in depth
  than in-plane (missing wedge).
* Background bug found and fixed: without a depth support, DTS smear behind the detector / patient
  was subtracted as background and the prism kept only 67-81% of its attenuation; now 98.7%.
* Physics-only limit: a 6-7 mm solid nodule (700-850 HU above lung in CT) is only 57-68 HU above the
  lung in the prism SIRT, against 35-41 HU of anatomical texture -- with or without noise, and with
  40 or 120 iterations or relaxation 1.9. The depth smear and clutter, not convergence or dose, limit
  physics-only DTS; this is the gap the learned prior has to close (H2).
* The training-free candidate detector found mostly end-on vessels (recall@10 6%).
* Learned detector (step 14, 40-patient Kaggle run, Oct 25; 24 train / 8 cal / 8 test patients, 47 test
  nodules): test CPM 0.155 vs 0.024 for the training-free detector, recall within the top 10 suggestions 26% vs
  11%, sensitivity 17% at 2 false positives per scan. Validation CPM peaked after 1000 of 6000 steps while
  the training loss kept falling (overfitting on 148 training nodules): more patients is the main lever.

## 6. Limitations (state them in the report)

* Simulation only: no public chest DTS dataset with CT ground truth exists. Monochromatic beam, no scatter.
* Acquisition: 15 views instead of the 60-74 of clinical units (same angular range as the GE VolumeRAD,
  equal total dose); the narrow 30 deg sweep is the conservative end (Shimadzu: up to 40 deg = less depth
  smear). Step 13 quantifies both for the physics-only methods; the diffusion prior is trained and
  tested on the 15-view, 30 deg geometry only.
* Virtual growth is isotropic scaling; real nodules grow irregularly.
* LIDC "ground truth" is itself variable between radiologists (we report reader CV).
* The nodule detector is trained on few patients (24 train patients, ~150 nodules in the 40-patient run)
  on simulated DTS; the training-free baseline mostly found end-on vessels (recall@10 0-11%). LIDC labels
  only nodules >= 4 mm marked by >= 3 radiologists, so suggestions on other real findings count as false
  positives. The clinician can always click anywhere.
* The patient's antero-posterior thickness is assumed known (it bounds the reconstruction in depth);
  here it comes from the CT body outline. CT field of view limits the simulated chest (rays outside
  the CT see air).

## 7. What to change in the FYDP report

* Chapter 1.3 objectives: drop Patch-Volume autoencoder / BiFlowNet / ControlNet / MRI / 512^3
  (copied from 3D MedDiffusion). Use objectives O1-O4 = H1-H4 above.
* Chapter 3: replace the architecture with this pipeline (prism ROI, conditional diffusion + DC,
  Gaussians, conformal). Rename "The Hallucinator" -> "anatomical prior".
* Chapter 4: results tables from `outputs/10_evaluation/*.md` and `outputs/11_growth/*.md`,
  figures from `outputs/*/`.
* Chapter 5 P4: remove "3DGS has not been applied to medical imaging" (false).
* Chapter 6: limitations above.

## 8. Plan to the Oct 30 deadline

| date | work | who |
|---|---|---|
| Oct 21-22 | `pytest` + `smoke_test.py` on a laptop / Kaggle (CPU is fine); read `docs/VERIFY.md` | all |
| Oct 22 | rent Vast.ai RTX 4090; steps 0-4 (select 200 patients, download ~22 GB, build cases) | Foyez |
| Oct 23 | steps 5-7 (DTS simulation, SIRT, training pairs); check every PASS/FAIL and figure | Foyez + Motasim |
| Oct 23-24 | step 8: diffusion training (~3-4 h; watch the nodule-box PSNR line) | Jubair |
| Oct 24-25 | steps 9-11: all methods, evaluation, growth experiment (~4-5 h GPU) | Jubair + Foyez |
| Oct 25 | download `outputs/` + results; destroy the instance | Foyez |
| Oct 26-27 | demo (`12_demo_app.py --share`), figures for slides | Motasim |
| Oct 27-29 | report: Ch.1.3 objectives, Ch.3 method, Ch.4 results (tables from `outputs/10_evaluation`, `outputs/11_growth`), Ch.6 limitations; slides | all |
| Oct 30 | rehearsal / buffer | all |

If step 8 is late, `09_run_pipeline.py --methods sirt_coarse roi_sirt roi_gs` still produces the
physics-only results (H1, H3, H4 with physics methods), and the diffusion rows are added when
the model is ready.
