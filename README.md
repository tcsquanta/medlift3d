# MedLift-3D

**Beyond the Missing Wedge: Calibrated 3D Lung Nodule Mass and Growth Assessment from Low-Dose Chest Tomosynthesis.**

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
python -m pytest -q                      # 56 unit tests: physics, Gaussians, measurement, diffusion, LIDC masks, DICOM, download, pipeline
python scripts/smoke_test.py --fresh     # every step end-to-end on synthetic phantoms (CPU, ~45 min on 4 cores)
```

## Full run: rented GPU (RunPod) for the heavy steps, Kaggle for the rest

**Complete step-by-step guide (renting, settings, the run, under/overfitting, saving, retraining, Kaggle):
[`docs/RUNPOD_FULL_RUN.md`](docs/RUNPOD_FULL_RUN.md).** Below is the short version.

| where | steps | why |
|---|---|---|
| **RunPod** (L40S 48 GB, ~$1.09/h) | 02-08 data + diffusion training, 14 detector, 09 all methods, 11 growth (+ 10) | need a big GPU and the full 583-patient data (~45 GB) |
| **Kaggle** (free T4) | 10 evaluation tables, 16 uncertainty, 17 explainability, 12 demo | need only a 1-2 GB bundle |

Expected time on one L40S: ~16-21 h for 583 patients (estimate scaled from our Kaggle run; details and
other GPUs in [`docs/GPU_VASTAI.md`](docs/GPU_VASTAI.md)). On your own **NVIDIA DGX Spark** the same script runs
in ~1-2 days at no hourly cost; setup (NGC container or Arm64 wheels) in
[`docs/GPU_VASTAI.md`](docs/GPU_VASTAI.md#nvidia-dgx-spark-your-own-machine-instead-of-renting).

### Part A - RunPod

Pod settings: **L40S, 1 GPU, On-Demand**, container disk **300 GB** (fast; erased when the pod stops),
persistent storage: a **Global volume** mounted at `/workspace` (backups only), template **PyTorch 2.8**.

**A1. Setup** (Web Terminal or JupyterLab -> Terminal):

```bash
apt-get update && apt-get install -y tmux
tmux new -s ml                         # detach: Ctrl-b d   re-attach: tmux attach -t ml
cd /root
git clone -b claude/funny-hopper-m4j8do https://github.com/tcsquanta/medlift3d.git
cd medlift3d
pip install pydicom pylidc sqlalchemy scikit-image pyyaml gradio plotly pytest crepes scikit-learn
python scripts/00_check_env.py         # GPU listed?
python -m pytest -q                    # all tests must pass
```

**A2. Run everything that needs the GPU** (one command, inside tmux):

```bash
N_PATIENTS=583 TRAIN_HOURS=6 bash scripts/runpod_run.sh
```

[`scripts/runpod_run.sh`](scripts/runpod_run.sh) runs steps 02-04 (and deletes the 64 GB of DICOM once the
cases are built), 05-07, 08 (diffusion training, stops itself after `TRAIN_HOURS`), 14, 09 (150 nodules),
11 (20 nodules x 2 follow-ups), 10, 15 (fit check), 16 (uncertainty) and 17 (explainability, 30 nodules); one log per step in `logs/`; after every step a copy of models,
results, outputs and logs in `/workspace/medlift_backup`; at the end two zips:

| file | contents | size | for |
|---|---|---|---|
| `/workspace/medlift_kaggle.zip` | both `best.pt` models, splits, results, all outputs, 15 test patients | ~1-2 GB | Kaggle: step 10 + demo |
| `/workspace/medlift_full_results.zip` | **everything produced**: all `outputs/` (tables, figures, PASS/FAIL reports), all step logs, `work/results/`, both models incl. `last.pt` + training logs and previews, LIDC selection files, configs | a few GB | keeping: report, appendix, re-analysis |

`WITH_DATA=1` also puts the prepared data (cases, DTS, SIRT, training pairs, ~45 GB) into the full zip;
only needed to re-run steps 9-14 elsewhere without downloading LIDC again. Make the full archive at any
time with `python scripts/archive_results.py --out /workspace/medlift_full_results.zip`.

**A3. While it runs**

- `tail -f logs/08_train.log`: after ~50 steps check `s/step` (L40S at batch 8: roughly 0.6-0.9 s/step,
  estimated); much slower -> switch pod (billing is per millisecond).
- `grep -h "passed,.*failed" logs/*.log`: every step must end `ALL PASS`.
- The script stopped (crash, `Ctrl-c`) but the pod still runs: run the same command again. Finished steps
  are skipped, an interrupted training continues from `last.pt`, finished nodules are kept.
- The pod itself was lost: new pod with the same Global volume, restore with
  `mkdir -p work && cp -r /workspace/medlift_backup/work/. work/`, then the same command (the data steps
  02-07 run again, training / detector / measured nodules are taken from the backup).
- Under- or overfitting? `python scripts/15_check_fit.py` (also runs at the end; report in
  `outputs/15_fit/fit_report.md`). Retraining from scratch: `RETRAIN=run2 bash scripts/runpod_run.sh`
  (see the full guide, section 9).
- Validation PSNR still rising at the 6 h cap: `MORE_TRAIN_HOURS=3 bash scripts/runpod_run.sh` continues
  training for 3 h and re-runs steps 9, 11, 10 with the new model (details: `docs/GPU_VASTAI.md`,
  "Continuing training after the time cap").

**A4. Download** when the log says `ALL DONE`:

- JupyterLab: file browser -> `/workspace` -> right-click `medlift_kaggle.zip` and
  `medlift_full_results.zip` -> Download, or
- Google Drive (one-time `rclone config`, remote `gdrive`, type `drive`, "auto config" = n, then run the
  printed `rclone authorize ...` on your laptop):
  ```bash
  curl https://rclone.org/install.sh | bash
  rclone copy /workspace/medlift_kaggle.zip gdrive:medlift3d -P
  rclone copy /workspace/medlift_full_results.zip gdrive:medlift3d -P
  ```

**A5. Terminate the pod** (a stopped pod still charges for its disk); delete the Global volume once the zip is safe.

### Part B - Kaggle

**B1.** Kaggle -> Datasets -> New Dataset -> upload `medlift_kaggle.zip` (e.g. name it `medlift-583`).

**B2.** New notebook, **GPU T4**, Internet on, Add Input -> your dataset. Then:

Cell 1 - setup
```bash
!git clone -b claude/funny-hopper-m4j8do https://github.com/tcsquanta/medlift3d.git
%cd medlift3d
!pip install -q pydicom pylidc sqlalchemy scikit-image pyyaml gradio plotly crepes scikit-learn
```

Cell 2 - copy the bundle in (finds it wherever Kaggle put it)
```bash
%%bash
cd /kaggle/working/medlift3d
S=$(find /kaggle/input -name splits.json -path "*work*" | head -1)
if [ -z "$S" ]; then                       # Kaggle kept the zip packed
  Z=$(find /kaggle/input -name "*.zip" | head -1); unzip -q "$Z" -d .
else
  B=$(dirname "$(dirname "$S")"); rm -rf work outputs logs
  cp -r "$B/work" work; cp -r "$B/outputs" outputs; [ -d "$B/logs" ] && cp -r "$B/logs" logs
fi
ls work/models/diffusion work/models/detector; ls work/cases | wc -l
```

Cell 3 - evaluation tables (step 10, ~1 min)
```bash
!python scripts/10_evaluate.py
!cat outputs/10_evaluation/table_accuracy.md outputs/10_evaluation/table_intervals.md
!cat outputs/11_growth/table_growth.md outputs/14_detector/table_detection.md
```

Cell 4 - uncertainty (step 16, seconds) and explainability (step 17, ~30-50 min; `--skip-maps`: minutes)
```bash
!python scripts/16_uncertainty.py
!python scripts/17_explain.py --limit 12 --members 4 --outputs /kaggle/working/outputs_kaggle
!cat outputs/16_uncertainty/table_uq_*.md /kaggle/working/outputs_kaggle/17_explain/table_*.md
```
What they mean and what is new: [`docs/XAI_UQ.md`](docs/XAI_UQ.md).

Cell 5 - demo
```bash
!python scripts/12_demo_app.py --selftest      # must end ALL PASS
!python scripts/12_demo_app.py --share         # open the printed gradio.live link
```

The first line should read `detector: learned detector (step 14) | calibration: yes | cases offered (test): 15 | category probabilities (step 16): yes`.
In the demo: pick a test patient -> the detector suggests nodules -> "Reconstruct in 3D" -> tick
"show radiologists' ground truth" to compare.

## Real data: run the steps in order

Each script prints **PASS / FAIL** checks and saves figures in `outputs/<step>/`
(what to look for: [`docs/VERIFY.md`](docs/VERIFY.md)). GPU needs and times per step:
[`docs/GPU_VASTAI.md`](docs/GPU_VASTAI.md). Steps 5, 6, 7, 9, 11 and 13 accept `--cases <case_id> ...` to (re)run
only some cases, e.g. `python scripts/06_reconstruct_sirt.py --cases LIDC-IDRI-0012_457842 --overwrite`.

| step | command | what it does |
|---|---|---|
| 0 | `python scripts/00_check_env.py` | versions, GPU, projector speed |
| 1 | `python scripts/01_verify_physics.py` | X-ray physics checks (mass measured by every view, missing wedge, ...) |
| 2 | `python scripts/02_select_lidc.py --n-patients 583` | choose LIDC-IDRI nodules from the radiologist annotations (offline) |
| 3 | `python scripts/03_download_lidc.py --workers 8` | download only those CT series from TCIA (~110 MB each) |
| 4 | `python scripts/04_prepare_cases.py --source lidc` | CT + 4-radiologist consensus masks, alignment checks, patient split |
| 5 | `python scripts/05_simulate_dts.py` | 15-view low-dose DTS per CT |
| 6 | `python scripts/06_reconstruct_sirt.py` | standard DTS reconstruction + candidate nodules |
| 7 | `python scripts/07_build_roi_dataset.py` | (DTS column, CT column) training pairs |
| 8 | `python scripts/08_train_diffusion.py --batch-size 8 --steps 1000000 --max-hours 6` (add `--resume` to continue) | **train the diffusion prior (rented GPU; stops at the time cap, saves best.pt)** |
| 9 | `python scripts/09_run_pipeline.py --members 8 --limit 150` | all methods on held-out nodules |
| 10 | `python scripts/10_evaluate.py` | accuracy tables, calibrated intervals, H1-H3 |
| 11 | `python scripts/11_growth_experiment.py --limit 20 --pairs-per-nodule 2` | virtual follow-up scans: VDT vs MDT, H4 |
| 12 | `python scripts/12_demo_app.py --share` | interactive click-to-3D demo (Gradio) |
| 14 | `python scripts/14_train_detector.py` | train the nodule detector on the DTS volume (train patients), FROC on test patients; the demo suggests its detections |
| 13 | `python scripts/13_geometry_sensitivity.py --limit 12` | same nodules with 60 views / a 40 deg sweep: how much the 15-view, 30 deg choice matters |
| 15 | `python scripts/15_check_fit.py [--gap]` | under/overfitting check of both models from the training logs (`--gap`: final models on train vs held-out data) |
| 16 | `python scripts/16_uncertainty.py` | conformal predictive systems (`crepes`): intervals (standard / ensemble-normalised / Mondrian size groups), calibrated EUPS volume-category probabilities (< 100 / 100-300 / >= 300 mm3) |
| 17 | `python scripts/17_explain.py` | explainability: seen / unseen (null-space) split of truth, physics and ours; detector saliency; error drivers ([`docs/XAI_UQ.md`](docs/XAI_UQ.md)) |
| 18 | `python scripts/18_report.py [--outputs <folder>]` | **one HTML page with every table, figure and check** (`outputs/report.html`; plain Python, also on an unzipped results folder) |

### Why LIDC-IDRI

LIDC-IDRI has up to four radiologists' 3D outlines and ratings for every nodule. The
"Radiologist Annotations" download on the TCIA page is the database bundled with `pylidc`, so
masks come for free. Only the CT images need downloading: 583 of the 1,010 patients have a qualifying
nodule (1,144 nodules, ~64 GB); step 2 takes the most nodule-rich patients first (200 patients already hold
60% of the nodules). LUNA16 (same scans as `.mhd`) works too: `--source luna16`.

## Repository layout

```
medlift3d/            library
  geometry.py         DTS geometry (15 views, +-15 deg linear sweep, SID 1800 mm), voxel grids
  projector.py        differentiable ray-marching X-ray projector (pure PyTorch)
  simulate.py         CT -> DTS projections, Poisson noise, local-change projection
  sirt.py             SIRT reconstruction
  roi.py              prism ROI, ray selection, background subtraction, nodule box
  diffusion/          3D U-Net, DDPM/DDIM with data consistency, dataset, checkpoints
  gaussians.py        3D Gaussian field, voxeliser, closed-form mass, X-ray refinement
  measure.py          segmentation, volume, mass, 2D diameter
  uncertainty.py      ensembles, split-conformal intervals, sparsification
  growth.py           virtual growth, VDT/MDT, NELSON categories
  detect.py           training-free candidate detector (LoG + Hessian; baseline)
  detector.py         learned nodule detector (3D U-Net heatmap, FROC / CPM evaluation)
  fitcheck.py         training-log parser: under-trained / converged / overfitting verdicts (step 15)
  uq.py               conformal predictive systems on per-nodule results (crepes; step 16)
  explain.py          seen / unseen split (CGLS null space), detector saliency, correlations (step 17)
  pipeline.py         end-to-end per-nodule processing (all methods)
  data/               LIDC-IDRI (pylidc + DICOM), LUNA16, cases
  demo.py             demo backend
scripts/              steps 00-17, runpod_run.sh (all rented-GPU steps), make_kaggle_bundle.py, archive_results.py,
                      kaggle_run.py (steps 2-8 in one Kaggle session), smoke_test.py
tests/                pytest unit tests
configs/              default.yaml (real runs), smoke.yaml (CPU smoke test)
docs/                 PROPOSAL (method), VERIFY (what each check means), GPU_VASTAI (GPUs, times, costs),
                      RUNPOD_FULL_RUN (the full rented-GPU run step by step), XAI_UQ (steps 16-17),
                      MedLift3D_Project_Guide.docx (the whole project in plain English)
```

## Limitations

Simulation study: no public chest DTS + CT dataset exists. Monochromatic X-rays, no scatter.
Virtual growth is isotropic. The nodule detector is trained on few patients (24 in the 40-patient run). Radiologist
"ground truth" varies between readers (reported as context).
