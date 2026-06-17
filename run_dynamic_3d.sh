#!/usr/bin/env bash
# Dynamic-env 3D re-run after the §1b fairness fixes (env RNG reset parity +
# unified goal_finish_dist). Traj-seeds auto-selected by make_3d_results.
set -uo pipefail
cd /home/sju5379/cp_scratch
LOG=/home/sju5379/cp_scratch/run_dynamic_3d.log
: > "$LOG"
echo "=== START $(date -Is) ===" | tee -a "$LOG"

echo "--- [1/2] make_3d_results.py (fair env; auto traj-seeds) ---" | tee -a "$LOG"
conda run --no-capture-output -n cp python make_3d_results.py \
  --seeds 20 21 22 23 24 30 31 32 33 34 35 36 37 38 39 40 41 >> "$LOG" 2>&1
RC1=$?
echo "--- make_3d_results.py exit=$RC1 $(date -Is) ---" | tee -a "$LOG"

echo "--- [2/2] run_sparse_3d.py (sparse table N_obs=50) ---" | tee -a "$LOG"
conda run --no-capture-output -n cp python run_sparse_3d.py >> "$LOG" 2>&1
RC2=$?
echo "--- run_sparse_3d.py exit=$RC2 $(date -Is) ---" | tee -a "$LOG"

echo "=== DONE $(date -Is) rc1=$RC1 rc2=$RC2 ===" | tee -a "$LOG"
exit $(( RC1 != 0 ? RC1 : RC2 ))
