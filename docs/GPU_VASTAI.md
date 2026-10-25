# Running MedLift-3D on a rented GPU (Vast.ai) -- what needs a big GPU and how long it takes

## Which steps need which machine

| step | what | measured on Kaggle 2x T4, 40 patients | RTX 4090, 200 patients (estimate) |
|---|---|---|---|
| setup | clone, pip install, 00 check, unit tests | - | 15-20 min |
| 2 | select patients (annotations only) | 6 min | ~6 min |
| 3 | download CT (~22 GB for 200) | 2.4 min for 40 series (4.5 GB) | 15-60 min (TCIA speed) |
| 4 | DICOM -> cases (CPU) | 10 min (15 s / case) | ~50 min |
| 5 | DTS simulation | 3.6 min | 5-10 min |
| 6 | coarse SIRT + candidates | 9 min | 15-20 min |
| 7 | diffusion training pairs | 24 min | 40-60 min |
| **8** | **diffusion training** | **0.96 s/step at batch 2 x 2 GPUs: 28,186 steps in 8.3 h** | **~9-14 samples/s (batch 2-4 fits in 24 GB): 6 h cap = ~190-290k samples** |
| 14 | nodule detector | 22.5 min (one T4) | 10-15 min |
| **9** | **all methods, K = 8, `--limit 150` nodules** | demo click with K = 4: 110-230 s | **3-5 h** (K = 4: about half) |
| 10 | evaluation tables | ~1 min | ~1 min |
| **11** | **growth, 20 nodules x 2 follow-ups** | - | **1.5-2.5 h** |
| 13 | geometry sensitivity (optional) | 25 min (12 nodules x 4 geometries) | ~10-15 min |
| 12 | demo self-test | - | ~10 min |
| | **total** | | **~14-19 h, about $5-10 at $0.35-0.50/h** |

The RTX 4090 column scales our Kaggle measurements: a 4090 is assumed to train this 3D U-Net ~4-6x
faster than one T4 (bf16; step 8 picks the largest batch that fits, 2-4 on 24 GB), and steps 4 and 7 grow with the number of patients (5x). Step 8
prints its `s/step` after 50 steps: check it and set `--max-hours` to fit your budget. An H100 / A100
(80 GB) is ~1.5-3x faster than a 4090 (total ~8-11 h, but ~$1-2.5/h).

**Recommendation:** do *everything* from step 2 onwards on one Vast.ai RTX 4090 instance and cap step 8
with `--max-hours` (e.g. 6). Kaggle's 20 GB `/kaggle/working` limit is too small for 200 patients'
DICOM, and its 12 h session limit is short for step 8. Use Kaggle or your laptop for the smoke test and code checks.

## Kaggle (free): environment check + 10-patient pilot, before paying for Vast.ai

Notebook settings: Accelerator **GPU** (T4 or P100), **Internet on** (needs a verified phone number).
Ten patients (~1.1 GB of DICOM) fit easily in `/kaggle/working`. Paste into cells:

```bash
!git clone -b claude/funny-hopper-m4j8do https://github.com/tcsquanta/medlift3d.git
%cd medlift3d
!pip install -q pydicom pylidc sqlalchemy scikit-image pyyaml gradio plotly
!python scripts/00_check_env.py                       # GPU listed? projector speed?
!python -m pytest -q                                  # 56 tests, all must pass
!python scripts/02_select_lidc.py --n-patients 10
!python scripts/03_download_lidc.py --workers 4
!python scripts/04_prepare_cases.py --source lidc
!python scripts/05_simulate_dts.py
!python scripts/06_reconstruct_sirt.py
!python scripts/07_build_roi_dataset.py --jitter 1 --negatives 1
!python scripts/08_train_diffusion.py --tiny          # GPU sanity run; note its s/step
!python scripts/09_run_pipeline.py --methods sirt_coarse roi_sirt roi_gs --limit 10
```

Check: every step ends `ALL PASS` (step 7: "condition keeps the prism's attenuation" ~0.99),
the figures in `outputs/04_cases` (contours on the nodules) and `outputs/07_roi_dataset`.
Physics-only methods miss many small nodules on real data (the depth smear leaves them only
~60 HU above the lung): that is a result, not a bug -- it is what the diffusion prior is for.
Then do the full run on Vast.ai below (keep all data on one machine: moving ~30 GB of DICOM and
intermediate files between Kaggle and Vast.ai costs more time than it saves money).

## Kaggle: one 9-hour run (prepare + train), then the interactive demo

`scripts/kaggle_run.py` runs steps 2-7 and then trains the diffusion model with all the time that is
left (step 8 stops itself at its budget, after validating and saving `best.pt` / `last.pt`).
Training is *not* the full 40,000-step recipe: a T4 reaches maybe 10-20k steps in ~7 h (measure it:
the log prints `s/step`), which is enough to see whether the model beats the physics-only input.

**1. Train (background, survives a closed tab).** New notebook, Accelerator **GPU T4 x2**, Internet **On**,
then two cells (with two GPUs step 8 trains on both automatically -- one process per GPU,
`--batch-size` samples each; expected ~1.7-1.9x the speed of one T4, not measured; the log says `x 2 GPU(s)`):

```bash
!git clone -b claude/funny-hopper-m4j8do https://github.com/tcsquanta/medlift3d.git
%cd medlift3d
!pip install -q pydicom pylidc sqlalchemy scikit-image pyyaml gradio plotly
!python scripts/kaggle_run.py --total-hours 9 --n-patients 40     # add --batch-size 2 if out of memory
```

Press **Save Version -> Save & Run All (Commit)**. Everything under `/kaggle/working` becomes the
version's output (the DICOM files are deleted after step 4 to stay far below the 20 GB limit).
Watch the log for `[val] ... nodule box: sample X dB | condition Y dB` every 1000 steps: the sample
should rise above the condition. If the session dies early, add the output as input data, copy
`work/models/diffusion/last.pt` back and run `kaggle_run.py --resume-training`.

**2. Demo (interactive session, ~1 h).** New notebook with GPU + Internet, **Add Data -> Notebook
Output Files** -> your committed notebook, then:

```bash
!git clone -b claude/funny-hopper-m4j8do https://github.com/tcsquanta/medlift3d.git
%cd medlift3d
!pip install -q pydicom pylidc sqlalchemy scikit-image pyyaml gradio plotly
!rm -rf work && cp -r /kaggle/input/<your-notebook-name>/medlift3d/work work   # path: look in the input panel
!python scripts/14_train_detector.py                                # nodule detector, ~15-30 min; ALL PASS
!python scripts/12_demo_app.py --selftest                           # must end ALL PASS
!python scripts/12_demo_app.py --share                              # open the printed gradio.live link
```

The demo offers only *test* patients (never seen in training). Pick one: the detector suggests possible
nodules (yellow). Pick a suggestion and press **Reconstruct in 3D** (or click anywhere on the slice).
Then tick **show radiologists' ground truth** to see the green LIDC nodules, which suggestions were real,
and the radiologists' mass / volume next to MedLift-3D's. If the diffusion model finds nothing at a point it
says so and shows the physics-only result. Without step 14 the demo falls back to the training-free
detector. Stop the session when done: it counts against your ~30 GPU hours per week. Keep
`work/models/detector/` (Save Version) so the next demo session does not need to retrain it.

This does not include steps 9-11 (evaluation tables, calibrated intervals, growth): they need a
trained model and take hours on a T4, so run them on the rented GPU, or in a second session.

**3. Geometry sensitivity (optional, ~1-2 h, needs only `work/cases`).** Same setup as the demo
session (copy `work` from the input), then:

```bash
!python scripts/13_geometry_sensitivity.py --limit 12      # 15 vs 60 views, 30 vs 40 deg; prints ALL PASS
!cat outputs/13_geometry/table_geometry.md
```

It re-simulates each patient 4 times (15, 60, 15 and 74 views; the log prints the simulation + SIRT
time per case) and keeps the results in `work/results/geometry.jsonl`, so a stopped run continues
where it left off.

## Before you rent: the decisions you make

Everything below is a choice with a trade-off. The recommended value is what `scripts/runpod_run.sh`
uses by default; each decision has its own section further down.

| # | decision | recommended | main trade-off | section |
|---|---|---|---|---|
| D1 | platform | RunPod Secure Cloud | reliability vs price (Vast.ai is cheaper, hosts vary) | [Platform](#d1-platform-runpod-or-vastai) |
| D2 | GPU | L40S 48 GB ($1.09/h) | money vs hours; memory decides the batch | [Which GPU](#d2-which-gpu) |
| D3 | patients | 583 (all usable) | more data vs download + preparation time | [How many patients](#d3-how-many-patients) |
| D4 | training time | 6 h (`TRAIN_HOURS`) | better prior vs cost and time left for steps 9-11 | [Step 8](#step-08-diffusion-training-the-expensive-one) |
| D5 | evaluation size | 150 nodules, 8 hypotheses; growth 20 nodules | reliable tables vs hours | [Steps 09 / 11](#steps-09-11-and-10-measurement-and-evaluation) |
| D6 | storage | 300 GB container disk + small Global volume | speed vs safety vs monthly cost | [Pod settings](#d6-pod-settings-and-storage-runpod) |
| D7 | results transfer | zip -> Kaggle dataset (and Google Drive) | effort vs safety | [Getting results out](#d7-getting-the-results-out) |

Two rules that are not trade-offs:

* **Keep the project settings (`configs/default.yaml`) fixed** for the rented run unless you have a reason
  you can defend, and decide before you see any test result. Changing them after looking at the test
  numbers would bias the results. The [project settings](#project-settings-configsdefaultyaml) section says
  which ones may change without retraining.
* **Run steps 9-10 on Kaggle with the current 40-patient model first**, so the report has results even
  if the rented run goes wrong. The 40 vs 583-patient comparison is itself a result.

## D1. Platform: RunPod or Vast.ai

| | Vast.ai | RunPod |
|---|---|---|
| what it is | marketplace of independent hosts | own data centres ("Secure Cloud") + community hosts |
| A100 80 GB price (typical) | ~$0.8-1.5/h | ~$1.2-1.9/h (Secure Cloud) |
| reliability | varies by host | consistent in Secure Cloud |
| disk / network | depends on the host: filter carefully | predictable |

For a ~16-21 h unattended run before a deadline, reliability is worth the ~$5-15 difference: **RunPod
Secure Cloud**. On Vast.ai filter for reliability > 99%, verified / data-centre hosts, >= 500 Mbps,
>= 250 GB disk, on-demand. The commands are the same on both.

## D2. Which GPU?

Memory decides the batch size: one 64 x 128 x 64 training sample needs ~5 GB (a 16 GB T4 fitted batch 2,
batch 4 ran out of memory); step 8 halves the batch automatically until it fits. Speed-ups are relative to
an RTX 4090 for this 3D U-Net (estimates from public benchmarks; measure with step 8's `s/step`). Prices
change daily.

| GPU (memory) | batch that fits | speed vs 4090 | RunPod $/h (seen Oct 25) | 583 patients: total (estimate) |
|---|---|---|---|---|
| A40 (48 GB) | 8 | ~0.6-0.7x | 0.59 | ~24-30 h, ~$15-18 |
| RTX 4090 (24 GB) | 2-4 | 1x | 0.89 (Vast.ai ~0.30-0.50) | ~18-24 h |
| **L40S (48 GB)** | 8 | ~1.1-1.3x | **1.09** | **~16-21 h, ~$18-24** |
| A100 80 GB SXM | 8-16 | ~1.5-2x | 1.79 | ~13-17 h, ~$25-32 |
| H100 80 GB SXM | 8-16 | ~2-3x | 3.99 | ~10-13 h, ~$40-55 |

* **L40S** = best value: fits the full batch of 8, slightly faster than a 4090, ~60% of the A100 price.
* **A100 SXM** if finishing ~4 h earlier is worth ~$8. **A40** only if money is very tight and time is not.
* Skip: H100 (our small 3D model cannot use its speed), 24 GB cards (half the batch), anything smaller.
* About 4-6 h of a full run does not depend on the GPU at all: the 64 GB download and step 4
  (DICOM -> cases) run on the CPU and network.
* **One GPU.** Only step 8 can use several (`torchrun --standalone --nproc_per_node N
  scripts/08_train_diffusion.py ...`); steps 5-7, 9 and 11 use one, so extra GPUs sit idle.
* **Test before committing:** start step 8 and read `s/step` after 50 steps. An L40S at batch 8 should show
  roughly 0.6-0.9 s/step (estimate); much slower -> switch machine (billing is per millisecond).

## D3. How many patients?

1,010 LIDC patients exist, but only **583** have a qualifying nodule (4-30 mm, >= 3 readers, slices <= 2.5 mm);
the others add no nodule labels. Step 2 takes the most nodule-rich patients first (counted from the
annotation database):

| patients | qualifying nodules | share | L40S total time (estimate) | DICOM download |
|---|---|---|---|---|
| 40 (Kaggle run) | 246 | 22% | - | 4.5 GB |
| 200 | 681 | 60% | ~13-17 h | ~22 GB |
| 300 | 861 | 75% | ~14-18 h | ~33 GB |
| 583 (all) | 1,144 | 100% | ~16-21 h | ~64 GB (disk >= 250 GB) |

More patients mainly lengthen steps 3, 4, 5-7 (they scale with patients); step 8 is capped by `--max-hours`
and steps 9 / 11 by `--limit`. Beyond ~300 patients each extra patient adds only one nodule. More patients
help the detector most (it overfitted on 148 training nodules) and give a larger test set (~230 test
nodules with 583 patients vs ~136 with 200).

## D6. Pod settings and storage (RunPod)

| setting | value | why |
|---|---|---|
| template | PyTorch 2.8 (CUDA 12.8) | PyTorch preinstalled; we only `pip install` small packages |
| GPU count | 1 | see D2 |
| pricing | On-Demand | cannot be taken away mid-run (spot / interruptible can) |
| region | any with the GPU available | a Global volume does not tie you to a region |
| container disk | **300 GB** | fast local disk for all data and training; peak use ~110 GB (64 GB DICOM + ~45 GB data); **erased when the pod stops** |
| persistent storage | **Global volume** at `/workspace` | backups only (models, results, outputs, logs: a few GB, ~$0.09/GB per month); survives a crash |

Persistent storage options: **Global volume** (pay for what is stored, any region: recommended),
Network volume (fixed size, ties the pod to one data centre; fine at ~20 GB if Global misbehaves),
Volume disk (legacy, deleted with the pod: no protection). Do not train from a network / global volume:
it is slower and Global charges per disk operation.

Stop vs Terminate: a stopped pod still charges for its disk and the container disk is erased anyway.
Copy the results off, then **Terminate**; delete the Global volume once the zip is safe.

## Rented GPU in one command, the rest on Kaggle

`scripts/runpod_run.sh` runs everything that needs the big GPU (steps 2-8, 14, 9, 11, plus step 10)
with a log per step in `logs/`, a backup of models / results / outputs to `/workspace/medlift_backup`
after every step, and at the end two zips: `/workspace/medlift_kaggle.zip` (models, splits, results, all
outputs, 15 test patients for the demo; ~1-2 GB; upload it as a Kaggle dataset for step 10 and the demo)
and `/workspace/medlift_full_results.zip` (everything produced, see D7). The README has the full
RunPod + Kaggle commands.

```bash
N_PATIENTS=583 TRAIN_HOURS=6 bash scripts/runpod_run.sh        # inside tmux; RESUME=1 after a crash
```

### Settings at the top of the script

| variable | default | what it does | why this value |
|---|---|---|---|
| `N_PATIENTS` | 583 | how many LIDC patients | all patients with a usable nodule (1,144 nodules) |
| `TRAIN_HOURS` | 6 | time cap for diffusion training | ends at a known time; ~2x the training of the Kaggle run |
| `LIMIT_09` | 150 | nodules measured in step 9 | enough for reliable tables and >= 19 calibration nodules; without it ~460 nodules (10+ h) |
| `LIMIT_11` | 20 | nodules in the growth experiment | 40 follow-up scans: enough for the H4 comparison |
| `BACKUP` | `/workspace/medlift_backup` | copy after every step | the persistent volume survives a crash |
| `BUNDLE` | `/workspace/medlift_kaggle.zip` | the zip for Kaggle (step 10 + demo) | |
| `FULL` | `/workspace/medlift_full_results.zip` | archive of everything produced (`scripts/archive_results.py`) | keep it: all outputs, logs, results, both models incl. `last.pt` |
| `WITH_DATA` | not set | `WITH_DATA=1` adds the prepared data (~45 GB) to the full archive | only to re-run steps 9-14 elsewhere without re-downloading LIDC |
| `KEEP_DICOM` | not set | `KEEP_DICOM=1` keeps the downloaded LIDC DICOM (~64 GB) instead of deleting it after step 4 | to save the raw CT to Google Drive (never download from TCIA again) |
| `RESUME` | not set | `RESUME=1` continues training from `last.pt` | restart after a crash |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | always | PyTorch reuses GPU memory more flexibly | fewer out-of-memory crashes |
| `set -euo pipefail` | always | stops the script if a step crashes | later steps never run on broken data |

Change any of them in front of the command, e.g. `N_PATIENTS=300 TRAIN_HOURS=8 bash scripts/runpod_run.sh`.

### Data steps (02-07)

| command | option | meaning |
|---|---|---|
| `02_select_lidc.py` | `--n-patients 583` | the 583 patients with the most qualifying nodules first |
| `03_download_lidc.py` | `--workers 8` | 8 CT series downloaded at once; finished downloads are skipped |
| `04_prepare_cases.py` | `--source lidc` | cases from LIDC DICOM + radiologist outlines; afterwards the script deletes `data/LIDC-IDRI` (64 GB) and writes `work/.cases_done` so a re-run does not download again |
| `05_simulate_dts.py` | (config) | 15 views, +-15 deg, SID 1800 mm, 20,000 photons / pixel / view |
| `06_reconstruct_sirt.py` | (config) | 40 SIRT iterations, 2.5 mm voxels |
| `07_build_roi_dataset.py` | `--jitter 3` | 3 pairs per nodule: 1 centred + 2 shifted up to +-6 mm (the model must not expect a centred nodule) |
| | `--negatives 2` | 2 nodule-free lung columns per patient (teach the model not to invent nodules) |
| | (`--val-frac 0.1`) | 10% of train patients kept for training previews |

### Step 08: diffusion training (the expensive one)

`08_train_diffusion.py --batch-size 8 --steps 40000 --max-hours 6 [--resume]`

| option | value | meaning and reason |
|---|---|---|
| `--batch-size 8` | 8 samples per step | the recipe's batch; ~5 GB per sample fits in 48 GB; halved automatically (8 -> 4 -> 2) if it does not fit |
| `--steps 40000` | target length | the full recipe; in practice the time cap usually stops training first |
| `--max-hours 6` | time cap | final validation, save `best.pt` / `last.pt`, stop; the learning rate is constant, so stopping early cuts nothing short |
| `--resume` | if `RESUME=1` | continue from `last.pt` (model, averaged model, optimiser, step count) |
| `--amp` (default `auto`) | bf16 on L40S / A100 | half-precision maths: ~2x faster, half the memory |
| learning rate (config) | 2e-4 | warm-up over the first 500 steps, then constant |
| EMA (config) | 0.999 | a smoothed average of the weights; the averaged model is used for sampling (more stable) |
| `--sample-every` (default) | 2000 | every 2,000 steps: 4 validation columns, nodule-region PSNR, save `last.pt`, and `best.pt` when the gain over the DTS input is the best so far |
| `--workers` (default) | 4 | CPU processes loading training pairs |

Model (config): 3D U-Net, 17.4 M parameters, v-prediction, cosine schedule, 1,000 noise levels.
Expected on an L40S (estimate): ~0.6-0.9 s/step -> ~24-36k steps in 6 h (~190-290k samples); with
~2,800 training pairs the model sees each about 70-100 times.

**Deciding D4 (training hours):** watch the `[val] ... nodule box: sample X dB | condition Y dB` lines.
Still rising clearly at the cap and time / budget left -> `--resume` for a few more hours (or start with
`TRAIN_HOURS=8`). Flat for the last few checks -> more hours will not help much.

### Step 14: detector training (settings in `configs/default.yaml` -> `detector`)

| setting | value | meaning |
|---|---|---|
| network | 3D U-Net, base 16 (0.34 M parameters) | small on purpose: few labels, a larger one overfits faster |
| patch | 64 x 64 x 64 voxels (160 mm) | one training sample = a piece of the DTS volume |
| batch / steps / lr | 8 / 6,000 / 1e-3 (cosine decay) | ~10-15 min on an L40S (estimate) |
| `pos_frac` | 0.6 | 60% of patches contain a nodule |
| heatmap target | blob of max(2.5 mm, d/4), 2x longer in depth | peak = nodule centre; longer in depth because DTS cannot localise depth |
| validation | every 500 steps on the **cal** patients | `best.pt` = highest CPM, so late overfitting does not matter |
| threshold | 2 false positives per scan, set on cal | used by the demo |
| final test | once, on the **test** patients | next to the training-free detector |

### Steps 09, 11 and 10: measurement and evaluation

| command | option | meaning |
|---|---|---|
| `09_run_pipeline.py` | `--members 8` | 8 diffusion hypotheses per nodule (physics Gaussians: 4); more = better uncertainty, slower |
| | `--limit 150` | 150 nodules from cal + test patients; re-runs skip finished ones |
| `11_growth_experiment.py` | `--limit 20` | 20 test nodules |
| | `--pairs-per-nodule 2` | 2 virtual follow-ups each (random from 90 / 180 / 365-day intervals x 150 / 300 / 450 / 600 / 900-day or no doubling) |
| | (`--warm-sirt-iter 10`) | follow-up DTS image: 10 SIRT iterations from the baseline image |
| `10_evaluate.py` | - | accuracy tables + calibrated intervals (1 min; repeated on Kaggle) |
| `make_kaggle_bundle.py` | `--n-demo 15` | models, results, outputs and 15 test patients for the demo |

**Deciding D5:** `LIMIT_09=100` or `--members 4` roughly halves step 9; keep at least ~100 nodules so the
calibration set has >= 19 nodules and the tables are not noise. Larger limits give tighter numbers.

## Project settings (`configs/default.yaml`)

These define the experiment. **Keep them identical between the Kaggle and rented runs** (otherwise the
two runs cannot be compared) and do not tune them on test results. Column "change?" says what a change
costs.

| group | setting | value | meaning | change? |
|---|---|---|---|---|
| general | `seed` | 42 | fixes the patient split and random choices | no: a new split changes every result |
| geometry | `n_views`, `sweep_deg` | 15, 30 | DTS views over +-15 deg (GE VolumeRAD range) | no: needs new data + retraining (step 13 measured the effect) |
| | `sid`, `det_shape`, `det_pitch` | 1800 mm, 536 x 536, 0.8 mm | machine geometry, 43 cm panel | no (same) |
| simulation | `i0` | 20,000 | photons / pixel / view (low dose) | no: lower = harder, needs retraining |
| sirt | `voxel_mm`, `n_iter`, `det_binning` | 2.5 mm, 40, 3 | the standard DTS image + background for the prism | no: it is the detector's input and the prism background |
| roi | `size_vox`, `voxel_mm` | 64 x 128 x 64, 1 x 2 x 1 mm | the prism column = the diffusion model's input size | **never after training** (the model expects this shape) |
| | `box_mm` | 40 x 50 x 40 | nodule box for the Gaussians | evaluation only, but keep fixed for comparability |
| | `min_ray_fraction`, `sirt_iter` | 0.5, 40 | rays used, physics iterations (the condition) | no: changes the diffusion input |
| diffusion | `base_ch`, `ch_mults`, `num_res`, `attn_levels` | 32, [1,2,4,4], 2, [3] | network size (17.4 M) | before training only |
| | `timesteps`, `lr`, `ema_decay` | 1000, 2e-4, 0.999 | training schedule | before training only |
| | `sample_steps` | 50 | denoising steps per hypothesis | at evaluation: fewer = faster, rougher |
| | `dc_iters`, `dc_start_frac` | 2, 0.6 | 2 SIRT data-consistency iterations in the last 40% of steps | at evaluation; more = closer to the X-rays, slower |
| gaussians | `fit_iter`, `refine_iter`, `max_gaussians` | 100, 100, 30000 | blob fitting and X-ray refinement | at evaluation: fewer = faster, less exact |
| measure | `mode`, `min_contrast_hu`, `max_extent_mm`, `mass_dilate_mm`, `gt_mass_dilate_mm` | half-max, 50 HU, 25 x 50 x 25 mm, 2 x 6 x 2 mm, 1 mm | the definitions of volume and mass (same for truth and every method) | no: fixed before seeing test results |
| pipeline | `n_members`, `n_members_physics` | 8, 4 | hypotheses per nodule | at evaluation (`--members`) |
| uncertainty | `alpha`, `mode` | 0.1, asymmetric | 90% intervals, separate low / high sides (corrects bias) | no: 90% is the hypothesis H3 |
| growth | `interval_days`, `doubling_days` | 90/180/365; 150/300/450/600/900/none | follow-up scenarios | no: defines H4 |
| detector | (see step 14) | | | before training only |

What can change **without retraining**: `--members`, `sample_steps`, `dc_iters`, Gaussian iterations and the
`--limit` values (speed vs precision). What needs **retraining** (new data and step 8): geometry, dose,
SIRT settings, prism size, network size.

## D7. Getting the results out

* The run ends with two zips in `/workspace`:
  * `medlift_kaggle.zip` (~1-2 GB): what Kaggle needs (both `best.pt`, splits, results, all outputs,
    15 demo test patients);
  * `medlift_full_results.zip` (a few GB): **everything produced** -- all `outputs/` (tables, figures,
    PASS/FAIL reports), all step logs, `work/results/`, both models including `last.pt`, training logs and
    previews, the LIDC selection files and `configs/`. `WITH_DATA=1` adds the prepared data (~45 GB:
    cases, DTS, SIRT, training pairs); a Google Drive free account holds 15 GB, so only do that with
    enough space. Make it any time: `python scripts/archive_results.py --out /workspace/medlift_full_results.zip`.
* Download both in JupyterLab (right-click -> Download) or copy them to Google Drive with rclone (one-time
  `rclone config`: remote `gdrive`, type `drive`, "auto config" = n, run the printed `rclone authorize ...`
  on your laptop), then
  `rclone copy /workspace/medlift_kaggle.zip gdrive:medlift3d -P` and
  `rclone copy /workspace/medlift_full_results.zip gdrive:medlift3d -P`.
* Upload the zip as a Kaggle dataset and follow Part B of the README (step 10 + demo).
* The big data (all cases, ~45 GB) is not needed afterwards. To demo more than the 15 bundled patients,
  run `python scripts/make_kaggle_bundle.py --n-demo 30` before terminating.

## Saving everything to Google Drive (including the prepared data and the raw CT)

With a large Google Drive (e.g. 5 TB) keep **everything**: results, models, the prepared data (~45 GB) and
even the downloaded LIDC CT (~64 GB). Then you never have to download from TCIA or re-run steps 2-7
again: a later run (more training, a changed evaluation) can start directly at step 8, 9 or 14.

| what | size (583 patients, estimate) | gives you |
|---|---|---|
| full results (`archive_results.py`) | a few GB | all outputs, logs, results, both models incl. `last.pt` |
| + prepared data (`--with-data`) | ~45 GB more | cases, DTS, SIRT volumes, training pairs: restart at step 8 / 9 / 14 |
| + raw LIDC DICOM (`KEEP_DICOM=1`) | ~64 GB more | the CT scans themselves: re-run step 4 onwards with other settings |

Write the big files to the **container disk** (`/root/...`), not to the `/workspace` volume: the container
disk is fast and free, the Global volume charges per GB and per disk operation. Disk check: data ~45 GB +
DICOM ~64 GB + their archives ~110 GB = ~220 GB, so 300 GB is enough if the DICOM folder is deleted after
it is archived (below); otherwise take a 400 GB container disk.

**1. rclone, once per pod** (Google login runs in your laptop's browser):

```bash
curl https://rclone.org/install.sh | bash
rclone config
#   n                     new remote
#   name> gdrive
#   Storage> drive        (Google Drive)
#   client_id / client_secret: press Enter (empty)
#   scope> 1              (full access)
#   service_account_file: Enter;  Edit advanced config? n
#   Use auto config? n    -> it prints:  rclone authorize "drive" "..."
#   on your laptop: install rclone (rclone.org/downloads), run that exact command, log in to Google,
#   copy the token it prints back into the pod
#   Configure as Shared Drive? n;  then y (keep) and q (quit)
rclone lsd gdrive:            # should list your Drive folders
```

**2. Start the run so nothing is deleted and the full archive includes the data:**

```bash
mkdir -p /root/to_drive
KEEP_DICOM=1 WITH_DATA=1 FULL=/root/to_drive/medlift_everything.zip \
  N_PATIENTS=583 TRAIN_HOURS=6 bash scripts/runpod_run.sh
```

(Already running without these options? Then the DICOM is gone, but you can still archive the data at
the end: `python scripts/archive_results.py --out /root/to_drive/medlift_everything.zip --with-data`.)

**3. When the run says `ALL DONE`, archive the raw CT and upload everything:**

```bash
cd /root/medlift3d
tar cf /root/to_drive/lidc_dicom.tar data/LIDC-IDRI && rm -rf data/LIDC-IDRI   # one big file uploads much faster than ~120k DICOM files
cp /workspace/medlift_kaggle.zip /root/to_drive/                               # the Kaggle bundle too
ls -lh /root/to_drive
rclone copy /root/to_drive gdrive:medlift3d -P --transfers 4 --drive-chunk-size 256M
rclone check /root/to_drive gdrive:medlift3d --one-way                         # verifies every file (checksums)
```

`rclone check` must report `0 differences found` before you terminate the pod. Upload time depends on the
pod's network and on Google: roughly 30 min to 2 h for ~160 GB (estimate). Google allows 750 GB of
uploads per day per account, far above this.

In Drive you then have `medlift3d/`: `medlift_kaggle.zip` (Kaggle: step 10 + demo),
`medlift_everything.zip` (results, logs, models, prepared data) and `lidc_dicom.tar` (raw CT).

**4. Restoring on a new machine later** (another rented GPU):

```bash
git clone -b claude/funny-hopper-m4j8do https://github.com/tcsquanta/medlift3d.git && cd medlift3d
pip install pydicom pylidc sqlalchemy scikit-image pyyaml gradio plotly pytest
# rclone config as in 1., then:
rclone copy gdrive:medlift3d/medlift_everything.zip . -P
unzip -q medlift_everything.zip && rm medlift_everything.zip      # work/, outputs/, logs/ are back
# optional, only to redo step 4 onwards with other settings:
rclone copy gdrive:medlift3d/lidc_dicom.tar . -P && tar xf lidc_dicom.tar && rm lidc_dicom.tar
```

The restored `work/.cases_done` makes `runpod_run.sh` skip steps 2-4. To re-run only some steps, call
them directly, e.g. more training then a new evaluation:

```bash
python scripts/08_train_diffusion.py --batch-size 8 --steps 60000 --max-hours 4 --resume
rm -f work/results/pipeline.jsonl work/results/growth.jsonl        # otherwise steps 9 / 11 keep the old results
python scripts/09_run_pipeline.py --members 8 --limit 150
python scripts/11_growth_experiment.py --limit 20 --pairs-per-nodule 2
python scripts/10_evaluate.py
```

The 45 GB data is too large for Kaggle's 20 GB working disk; use `medlift_kaggle.zip` there.

## Manual step-by-step (any rented GPU)

The same as `runpod_run.sh`, one command at a time:

1. **Instance:** L40S (or A100 80 GB), >= 64 GB RAM, **>= 300 GB disk**, on-demand; PyTorch (CUDA 12) image.
2. **Setup:**
   ```bash
   apt-get update && apt-get install -y tmux && tmux new -s ml
   git clone -b claude/funny-hopper-m4j8do https://github.com/tcsquanta/medlift3d.git
   cd medlift3d
   pip install pydicom pylidc sqlalchemy scikit-image pyyaml gradio plotly pytest   # torch is preinstalled
   python scripts/00_check_env.py && python -m pytest -q
   ```
3. **Steps** (each prints PASS/FAIL; stop and check if anything FAILs):
   ```bash
   python scripts/02_select_lidc.py --n-patients 583
   python scripts/03_download_lidc.py --workers 8
   python scripts/04_prepare_cases.py --source lidc && rm -rf data/LIDC-IDRI
   python scripts/05_simulate_dts.py
   python scripts/06_reconstruct_sirt.py
   python scripts/07_build_roi_dataset.py --jitter 3 --negatives 2
   python scripts/08_train_diffusion.py --batch-size 8 --steps 40000 --max-hours 6
   python scripts/14_train_detector.py
   python scripts/09_run_pipeline.py --members 8 --limit 150
   python scripts/11_growth_experiment.py --limit 20 --pairs-per-nodule 2
   python scripts/10_evaluate.py
   python scripts/make_kaggle_bundle.py --out /workspace/medlift_kaggle.zip --n-demo 15
   python scripts/archive_results.py --out /workspace/medlift_full_results.zip        # everything produced
   ```
4. Download both zips, then **terminate** the instance.

## Tips for step 8

* Watch the `[val]` lines every 2,000 steps: the nodule-box `sample` PSNR should rise above `condition`
  within the first ~10k steps. If it does not after ~15k steps, stop and check
  `outputs/08_diffusion/val_step*.png`.
* Out of memory: the batch is halved automatically; `--batch-size 4` forces it.
* `--resume` continues from `work/models/diffusion/last.pt` (saved at every validation).

## If TCIA downloads are slow or blocked

* Use NBIA Data Retriever with `work/lidc_subset.tcia` (written by step 2) on any computer,
  then upload the DICOM folder to the instance (`rsync`/`scp`) and set `paths.lidc_root`.
* Fewer patients still work: `--n-patients 300` (~33 GB) gives 861 nodules.
* LUNA16 (same scans, `.mhd`) also works: `python scripts/04_prepare_cases.py --source luna16 --mhd-dir /path/to/subset0`.
