#!/usr/bin/env bash
# Dynamic-env 3D run (per HANDOFF_DYNAMIC_3D.md). Held awake via systemd-inhibit.
set -uo pipefail
cd /home/sju5379/cp_scratch
LOG=/home/sju5379/cp_scratch/run_dynamic_3d.log
: > "$LOG"

echo "=== START $(date -Is) ===" | tee -a "$LOG"

echo "--- [1/2] make_3d_results.py (dense table + traj + scalability) ---" | tee -a "$LOG"
conda run --no-capture-output -n cp python make_3d_results.py \
  --seeds 20 21 22 23 24 30 31 32 33 34 35 36 37 38 39 40 41 \
  --traj-seeds 20 22 30 >> "$LOG" 2>&1
RC1=$?
echo "--- make_3d_results.py exit=$RC1 $(date -Is) ---" | tee -a "$LOG"

echo "--- [2/2] run_sparse_3d.py (sparse table N_obs=50) ---" | tee -a "$LOG"
conda run --no-capture-output -n cp python run_sparse_3d.py >> "$LOG" 2>&1
RC2=$?
echo "--- run_sparse_3d.py exit=$RC2 $(date -Is) ---" | tee -a "$LOG"

echo "=== DONE $(date -Is) rc1=$RC1 rc2=$RC2 ===" | tee -a "$LOG"
exit $(( RC1 != 0 ? RC1 : RC2 ))
