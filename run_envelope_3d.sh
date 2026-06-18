#!/usr/bin/env bash
# Full 3D re-run with the new LRW support-function envelope + clearance relaxation + p_base=5
# (HANDOFF §"Action"). Caches already invalidated. Fresh outcomes (mean±std over 17 seeds) +
# corrected ECP timing (warmup-excluded, per-step pooled) + envelope-dependent figures.
set -uo pipefail
cd /home/sju5379/cp_scratch
LOG=/home/sju5379/cp_scratch/run_envelope_3d.log
: > "$LOG"
echo "=== START $(date -Is) ===" | tee -a "$LOG"

echo "--- [1/3] make_3d_results.py (dense: new envelope, 17 outcome seeds, 10 timing seeds) ---" | tee -a "$LOG"
conda run --no-capture-output -n cp python make_3d_results.py \
  --seeds 20 21 22 23 24 30 31 32 33 34 35 36 37 38 39 40 41 \
  --timing-seeds 20 21 22 23 24 30 31 32 33 34 >> "$LOG" 2>&1
RC1=$?
echo "--- make_3d_results exit=$RC1 $(date -Is) ---" | tee -a "$LOG"

echo "--- [2/3] run_sparse_3d.py (sparse N_obs=50, mean±std) ---" | tee -a "$LOG"
conda run --no-capture-output -n cp python run_sparse_3d.py >> "$LOG" 2>&1
RC2=$?
echo "--- run_sparse_3d exit=$RC2 $(date -Is) ---" | tee -a "$LOG"

echo "--- [3/3] make_fig_conformal_3d.py (Func_cp_3d_zoom: new envelope) ---" | tee -a "$LOG"
conda run --no-capture-output -n cp python make_fig_conformal_3d.py >> "$LOG" 2>&1
RC3=$?
echo "--- conformal exit=$RC3 $(date -Is) ---" | tee -a "$LOG"

echo "=== DONE $(date -Is) rc=$RC1/$RC2/$RC3 ===" | tee -a "$LOG"
