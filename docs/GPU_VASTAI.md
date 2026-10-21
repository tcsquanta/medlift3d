# Running MedLift-3D on a rented GPU (Vast.ai) -- what needs a big GPU and how long it takes

## Which steps need which machine

| step | script | needs | RTX 4090 (24 GB) estimate | free Kaggle T4 (16 GB) | CPU only |
|---|---|---|---|---|---|
| 0-1 | check env, physics checks | anything | 1 min | 2 min | 2-3 min |
| smoke test | `smoke_test.py` | anything (CPU fine) | 10 min | 20 min | measured: 71 min (4 cores) |
| 2 | select LIDC cases (annotations only) | CPU | 2-4 min | 2-4 min | 2-4 min |
| 3 | download ~200 LIDC series (~22 GB) | **disk + bandwidth** | 30-90 min (TCIA speed) | disk limit is tight | measured: 4 series (0.45 GB) in 30 s |
| 4 | DICOM -> cases | CPU / RAM | 40-80 min | 40-80 min | measured: ~23 s per case |
| 5 | DTS simulation (200 cases) | GPU | ~5 min | ~15 min | measured: ~4 min per real CT (4 cores) |
| 6 | coarse SIRT (200 cases) | GPU | ~15 min | ~45 min | many hours |
| 7 | diffusion training pairs (~2000) | GPU | ~30-45 min | ~1.5-2.5 h | too slow |
| **8** | **diffusion training (40k steps)** | **big GPU** | **~3-4 h (batch 8, bf16)** | ~10-14 h (batch 4, fp16) | no |
| **9** | **all methods on ~150 held-out nodules (K = 8)** | **big GPU** | **~2-3 h** | ~6-8 h | no |
| 10 | evaluation tables / figures | CPU | 1 min | 1 min | 1 min |
| **11** | growth experiment (40 nodules x 2 follow-ups) | **big GPU** | ~1.5-2 h | ~5 h | no |
| 12 | Gradio demo | GPU nice, CPU OK | instant click (~30-60 s / nodule) | similar | 2-5 min / nodule |

These GPU times are **estimates**, not measurements. The CPU numbers in this table are
measured in this repo; the GPU numbers come from the FLOP count of the 3D U-Net (~0.5
TFLOP per 64x128x64 forward pass) and grid-sample throughput. Run `scripts/00_check_env.py` on the
rented machine first: it prints the projector speed, and step 8 prints its `s/step` and ETA after
50 steps. If the ETA is far off, change `--steps` / `--batch-size` accordingly.

**Recommendation:** do *everything* from step 3 onwards on one Vast.ai RTX 4090 instance
(~8-11 GPU hours in total, about $3-5 at $0.30-0.45/h). Kaggle's 20 GB `/kaggle/working`
limit is too small for the DICOM download, and its 12 h session limit is short for step 8.
Use Kaggle or your laptop only for the smoke test and code checks.

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
   python scripts/08_train_diffusion.py --batch-size 8 --steps 40000      # ~3-4 h; resume with --resume
   python scripts/09_run_pipeline.py --members 8
   python scripts/10_evaluate.py
   python scripts/11_growth_experiment.py --limit 40 --pairs-per-nodule 2
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
