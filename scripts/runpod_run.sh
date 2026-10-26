#!/usr/bin/env bash
# Everything that needs the rented GPU, in order, with logs and backups (RunPod / Vast.ai).
#
#   bash scripts/runpod_run.sh                            # 583 patients, 6 h of diffusion training
#   N_PATIENTS=300 TRAIN_HOURS=6 bash scripts/runpod_run.sh
#
# Steps 2-7 (data), 8 (diffusion training), 14 (detector), 9 (all methods), 11 (growth),
# 10 (tables, 1 min), 15 (under/overfitting check of both models, a few min), then
#   scripts/make_kaggle_bundle.py -> /workspace/medlift_kaggle.zip        (what Kaggle needs: step 10 + demo)
#   scripts/archive_results.py    -> /workspace/medlift_full_results.zip  (everything produced: all outputs,
#                                    logs, results, both models incl. last.pt + training logs, configs;
#                                    WITH_DATA=1 adds the prepared data, ~45 GB)
# KEEP_DICOM=1 keeps the downloaded LIDC DICOM (~64 GB) instead of deleting it after step 4
# (to save it to Google Drive; see docs/GPU_VASTAI.md, "Saving everything to Google Drive").
# After every step: copy of models / results / outputs / logs to $BACKUP (the persistent volume).
#
# Safe to start again (after a crash, or to change something): finished cases, training pairs,
# training, detector and measured nodules are skipped (marker files .done); an interrupted training
# continues from work/models/diffusion/last.pt automatically (for TRAIN_HOURS more hours).
# Train longer after a finished run:  MORE_TRAIN_HOURS=3 bash scripts/runpod_run.sh
#   -> continues from last.pt for 3 h (best.pt is kept unless beaten), moves the old step 9 / 11
#      results to work/results/old/ (they belong to the old model) and re-runs 9, 11, 10 and the zips.
#
# Retrain from scratch (e.g. the fit check says overfitting, or new training pairs):
#   RETRAIN=run2 bash scripts/runpod_run.sh   -> the finished model moves to work/models/diffusion_before_run2,
#   the old step 9 / 11 results to work/results/old/, a new model trains for TRAIN_HOURS; safe to re-run with the
#   same name. RETRAIN_DETECTOR=run2 [DET_STEPS=3000] does the same for the detector.
#   New training pairs first: rm -rf work/roi_pairs, then e.g. JITTER=6 NEGATIVES=4 RETRAIN=run2 ...
#
# Two-pod plan (steps 2-4 need no GPU):  on a cheap CPU pod   ONLY_CASES=1 bash scripts/runpod_run.sh
#   -> downloads + builds the cases, packs work/ + logs/ into $CASES_TAR on the persistent volume, stops.
#   On the GPU pod (same volume) the normal command unpacks it automatically and starts at step 5.
set -euo pipefail
cd "$(dirname "$0")/.."
N_PATIENTS=${N_PATIENTS:-583}
TRAIN_HOURS=${TRAIN_HOURS:-6}
LIMIT_09=${LIMIT_09:-150}
LIMIT_11=${LIMIT_11:-20}
BACKUP=${BACKUP:-/workspace/medlift_backup}
BUNDLE=${BUNDLE:-/workspace/medlift_kaggle.zip}
FULL=${FULL:-/workspace/medlift_full_results.zip}
CASES_TAR=${CASES_TAR:-/workspace/medlift_cases.tar}
JITTER=${JITTER:-3}
NEGATIVES=${NEGATIVES:-2}
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
old_results() {   # step 9 / 11 results belong to the old diffusion model: keep them aside, measure again
  mkdir -p work/results/old
  for f in pipeline.jsonl growth.jsonl; do
    [ -f "work/results/$f" ] && mv "work/results/$f" "work/results/old/${f%.jsonl}_$(date +%Y%m%d_%H%M%S).jsonl"
  done
  return 0
}

# ---- data (CPU + network heavy); the DICOM files are deleted once the cases are built
if [ ! -f work/.cases_done ] && [ -z "${ONLY_CASES:-}" ] && [ -f "$CASES_TAR" ]; then
  echo "unpacking the cases prepared on the CPU pod: $CASES_TAR"
  tar xf "$CASES_TAR"
fi
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
if [ -n "${ONLY_CASES:-}" ]; then
  mkdir -p "$(dirname "$CASES_TAR")"
  tar cf "$CASES_TAR" work logs
  echo "===== $(date '+%F %T')  ONLY_CASES done: $(du -h "$CASES_TAR" | cut -f1) packed to $CASES_TAR."
  echo "      Terminate this CPU pod; on the GPU pod (same volume) run the normal command."
  exit 0
fi
step 05_dts   scripts/05_simulate_dts.py                 # skips finished cases
step 06_sirt  scripts/06_reconstruct_sirt.py             # skips finished cases
if [ ! -f work/roi_pairs/.done ]; then
  step 07_pairs scripts/07_build_roi_dataset.py --jitter "$JITTER" --negatives "$NEGATIVES"
  touch work/roi_pairs/.done
fi
backup

# ---- the expensive part: diffusion training (stops itself after the time cap, saves best.pt / last.pt).
# --steps is only an upper bound: the time cap decides (the learning rate is constant after warm-up).
DIFF=work/models/diffusion
TRAIN=(scripts/08_train_diffusion.py --batch-size 8 --steps 1000000)
if [ -n "${RETRAIN:-}" ] && [ -f "$DIFF/.done" ] && [ ! -e "${DIFF}_before_$RETRAIN" ]; then
  mv "$DIFF" "${DIFF}_before_$RETRAIN"
  old_results
  echo "RETRAIN=$RETRAIN: finished model moved to ${DIFF}_before_$RETRAIN; new training for $TRAIN_HOURS h"
fi
if [ -n "${MORE_TRAIN_HOURS:-}" ]; then
  [ -f "$DIFF/last.pt" ] || { echo "MORE_TRAIN_HOURS: no $DIFF/last.pt to continue from"; exit 1; }
  step 08_train "${TRAIN[@]}" --max-hours "$MORE_TRAIN_HOURS" --resume
  old_results
  touch "$DIFF/.done"
elif [ ! -f "$DIFF/.done" ]; then
  RES=""; { [ -f "$DIFF/last.pt" ] || [ -n "${RESUME:-}" ]; } && RES="--resume"   # interrupted earlier: continue
  step 08_train "${TRAIN[@]}" --max-hours "$TRAIN_HOURS" $RES
  touch "$DIFF/.done"
else
  echo "step 08: training already finished ($DIFF/.done); MORE_TRAIN_HOURS=N continues it"
fi
backup

# ---- detector (train patients), all methods on held-out nodules, growth experiment
DET=work/models/detector
if [ -n "${RETRAIN_DETECTOR:-}" ] && [ -f "$DET/.done" ] && [ ! -e "${DET}_before_$RETRAIN_DETECTOR" ]; then
  mv "$DET" "${DET}_before_$RETRAIN_DETECTOR"
  echo "RETRAIN_DETECTOR=$RETRAIN_DETECTOR: finished detector moved to ${DET}_before_$RETRAIN_DETECTOR"
fi
if [ ! -f "$DET/.done" ]; then
  step 14_detector scripts/14_train_detector.py ${DET_STEPS:+--steps "$DET_STEPS"}
  touch "$DET/.done"
fi
backup
step 09_pipeline scripts/09_run_pipeline.py --members 8 --limit "$LIMIT_09"
backup
step 11_growth   scripts/11_growth_experiment.py --limit "$LIMIT_11" --pairs-per-nodule 2
backup
step 10_evaluate scripts/10_evaluate.py              # 1 min; Kaggle can re-run it from the bundle
step 15_fit scripts/15_check_fit.py --gap || echo "step 15 (fit check) failed: not needed for the results"
backup

python scripts/make_kaggle_bundle.py --out "$BUNDLE" --n-demo 15 | tee -a logs/bundle.log
python scripts/archive_results.py --out "$FULL" ${WITH_DATA:+--with-data} | tee -a logs/bundle.log
echo "===== $(date '+%F %T')  ALL DONE. Download $BUNDLE (for Kaggle) and $FULL (everything), then terminate the pod."
