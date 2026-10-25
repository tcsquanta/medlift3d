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
| **8** | **diffusion training** | **0.96 s/step at batch 2 x 2 GPUs: 28,186 steps in 8.3 h** | **~0.6-0.9 s/step at batch 8: 30k steps 5-7.5 h, 40k steps 6.5-10 h** |
| 14 | nodule detector | 22.5 min (one T4) | 10-15 min |
| **9** | **all methods, K = 8, `--limit 150` nodules** | demo click with K = 4: 110-230 s | **3-5 h** (K = 4: about half) |
| 10 | evaluation tables | ~1 min | ~1 min |
| **11** | **growth, 20 nodules x 2 follow-ups** | - | **1.5-2.5 h** |
| 13 | geometry sensitivity (optional) | 25 min (12 nodules x 4 geometries) | ~10-15 min |
| 12 | demo self-test | - | ~10 min |
| | **total** | | **~14-19 h, about $5-10 at $0.35-0.50/h** |

The RTX 4090 column scales our Kaggle measurements: a 4090 is assumed to train this 3D U-Net ~4-6x
faster than one T4 (bf16, batch 8), and steps 4 and 7 grow with the number of patients (5x). Step 8
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

## Vast.ai checklist

1. **Instance:** RTX 4090 (or A100 40 GB), >= 32 GB RAM, **>= 150 GB disk**, reliability > 99%,
   download speed > 500 Mbps, **on-demand** (not interruptible). Template: official *PyTorch*
   (CUDA 12.x) image.
2. **Setup** (in the instance terminal):
   ```bash
   git clone -b claude/funny-hopper-m4j8do https://github.com/tcsquanta/medlift3d.git
   cd medlift3d
   pip install pydicom pylidc sqlalchemy scikit-image pyyaml gradio plotly pytest   # torch is preinstalled
   python scripts/00_check_env.py                 # GPU listed? projector speed?
   python -m pytest -q                            # ~2-5 min, all tests should pass
   ```
3. **Use `tmux`** so long jobs survive a closed browser tab: `tmux new -s ml`, detach with
   `Ctrl-b d`, re-attach with `tmux attach -t ml`.
4. **Data + training** (each step prints PASS/FAIL; stop and check if anything FAILs):
   ```bash
   python scripts/02_select_lidc.py --n-patients 200
   python scripts/03_download_lidc.py --workers 8
   python scripts/04_prepare_cases.py --source lidc
   python scripts/05_simulate_dts.py
   python scripts/06_reconstruct_sirt.py
   python scripts/07_build_roi_dataset.py --jitter 3 --negatives 2
   python scripts/08_train_diffusion.py --batch-size 8 --steps 40000 --max-hours 6   # stops at 6 h; resume with --resume
   python scripts/14_train_detector.py                                   # nodule detector, ~10-15 min
   python scripts/09_run_pipeline.py --members 8 --limit 150             # ~3-5 h
   python scripts/10_evaluate.py
   python scripts/11_growth_experiment.py --limit 20 --pairs-per-nodule 2   # ~1.5-2.5 h
   python scripts/12_demo_app.py --selftest
   python scripts/13_geometry_sensitivity.py --limit 20                 # optional
   ```
5. **Save results before destroying the instance** (storage keeps costing money):
   `outputs/` (all figures + tables), `work/results/*.jsonl`, `work/models/diffusion/best.pt`,
   `work/splits.json`. For example: `tar czf results.tgz outputs work/results work/models/diffusion/best.pt work/splits.json`
   then download it via the Vast.ai web file browser or `scp`.
6. **Destroy** the instance when done.

## Tips for step 8 (the expensive one)

* Watch the validation lines: `PSNR sample vs CT` should rise above `condition vs CT` within the
  first ~10k steps. If it does not after ~15k steps, stop and check `outputs/08_diffusion/val_step*.png`.
* Out of memory: `--batch-size 4` (or 2). bf16 is used automatically on RTX 30xx/40xx/A100.
* `--resume` continues from `work/models/diffusion/last.pt` (saved every `--sample-every` steps).
* Smaller budget: `--steps 20000` (~1.5-2 h) is a reasonable first model; train longer if the
  validation PSNR is still improving.

## If TCIA downloads are slow or blocked

* Use NBIA Data Retriever with `work/lidc_subset.tcia` (written by step 2) on any computer,
  then upload the DICOM folder to the instance (`rsync`/`scp`) and set `paths.lidc_root`.
* Fewer patients still work: `--n-patients 100` (~11 GB) gives ~250 nodules.
* LUNA16 (same scans, `.mhd`) also works: `python scripts/04_prepare_cases.py --source luna16 --mhd-dir /path/to/subset0`.
