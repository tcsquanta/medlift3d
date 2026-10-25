#!/usr/bin/env bash
# Everything that needs the rented GPU, in order, with logs and backups (RunPod / Vast.ai).
#
#   bash scripts/runpod_run.sh                            # 583 patients, 6 h of diffusion training
#   N_PATIENTS=300 TRAIN_HOURS=6 bash scripts/runpod_run.sh
#
# Steps 2-7 (data), 8 (diffusion training), 14 (detector), 9 (all methods), 11 (growth),
# 10 (tables, 1 min), then
#   scripts/make_kaggle_bundle.py -> /workspace/medlift_kaggle.zip        (what Kaggle needs: step 10 + demo)
#   scripts/archive_results.py    -> /workspace/medlift_full_results.zip  (everything produced: all outputs,
#                                    logs, results, both models incl. last.pt + training logs, configs;
#                                    WITH_DATA=1 adds the prepared data, ~45 GB)
# KEEP_DICOM=1 keeps the downloaded LIDC DICOM (~64 GB) instead of deleting it after step 4
# (to save it to Google Drive; see docs/GPU_VASTAI.md, "Saving everything to Google Drive").
# After every step: copy of models / results / outputs / logs to $BACKUP (the persistent volume).
# Safe to start again after a crash: finished cases and nodules are skipped; RESUME=1 continues
# step 8 from work/models/diffusion/last.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
N_PATIENTS=${N_PATIENTS:-583}
TRAIN_HOURS=${TRAIN_HOURS:-6}
LIMIT_09=${LIMIT_09:-150}
LIMIT_11=${LIMIT_11:-20}
BACKUP=${BACKUP:-/workspace/medlift_backup}
BUNDLE=${BUNDLE:-/workspace/medlift_kaggle.zip}
FULL=${FULL:-/workspace/medlift_full_results.zip}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

step() {   # step <log name> <script> [args...]
  local name=$1; shift
  echo "===== $(date '+%F %T')  $name: python $*" | tee -a "logs/$name.log"
  python -u "$@" 2>&1 | tee -a "logs/$name.log"
}
backup() {
  mkdir -p "$BACKUP"
  for p in work/models work/results work/splits.json outputs logs; do
    [ -e "$p" ] && mkdir -p "$BACKUP/$(dirname "$p")" && cp -r "$p" "$BACKUP/$(dirname "$p")/"
  done
  echo "----- backup -> $BACKUP"
}

# ---- data (CPU + network heavy); the DICOM files are deleted once the cases are built
if [ ! -f work/.cases_done ]; then
  step 02_select   scripts/02_select_lidc.py --n-patients "$N_PATIENTS"
  step 03_download scripts/03_download_lidc.py --workers 8
  step 04_cases    scripts/04_prepare_cases.py --source lidc
  if [ -z "${KEEP_DICOM:-}" ]; then
    rm -rf data/LIDC-IDRI && echo "deleted data/LIDC-IDRI (cases built)"
  else
    echo "KEEP_DICOM=1: data/LIDC-IDRI kept"
  fi
  touch work/.cases_done
fi
step 05_dts   scripts/05_simulate_dts.py
step 06_sirt  scripts/06_reconstruct_sirt.py
step 07_pairs scripts/07_build_roi_dataset.py --jitter 3 --negatives 2
backup

# ---- the expensive part: diffusion training (stops itself after TRAIN_HOURS, saves best.pt / last.pt)
step 08_train scripts/08_train_diffusion.py --batch-size 8 --steps 40000 --max-hours "$TRAIN_HOURS" ${RESUME:+--resume}
backup

# ---- detector (train patients), all methods on held-out nodules, growth experiment
step 14_detector scripts/14_train_detector.py
backup
step 09_pipeline scripts/09_run_pipeline.py --members 8 --limit "$LIMIT_09"
backup
step 11_growth   scripts/11_growth_experiment.py --limit "$LIMIT_11" --pairs-per-nodule 2
backup
step 10_evaluate scripts/10_evaluate.py              # 1 min; Kaggle can re-run it from the bundle
backup

python scripts/make_kaggle_bundle.py --out "$BUNDLE" --n-demo 15 | tee -a logs/bundle.log
python scripts/archive_results.py --out "$FULL" ${WITH_DATA:+--with-data} | tee -a logs/bundle.log
echo "===== $(date '+%F %T')  ALL DONE. Download $BUNDLE (for Kaggle) and $FULL (everything), then terminate the pod."
