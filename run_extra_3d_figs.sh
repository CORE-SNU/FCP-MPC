#!/usr/bin/env bash
# Regenerate the remaining dynamic-env 3D figures (conformal zoom + traj overlay).
set -uo pipefail
cd /home/sju5379/cp_scratch
LOG=/home/sju5379/cp_scratch/run_extra_3d_figs.log
: > "$LOG"
echo "=== START $(date -Is) ===" | tee -a "$LOG"

echo "--- [1/2] make_fig_conformal_3d.py (Func_cp_3d_zoom.png; recompute) ---" | tee -a "$LOG"
conda run --no-capture-output -n cp python quadrotor/make_fig_conformal_3d.py >> "$LOG" 2>&1
RC1=$?
echo "--- conformal exit=$RC1 $(date -Is) ---" | tee -a "$LOG"

echo "--- [2/2] make_traj_3d_overlay.py (traj_3d_overlay.png) ---" | tee -a "$LOG"
conda run --no-capture-output -n cp python quadrotor/make_traj_3d_overlay.py >> "$LOG" 2>&1
RC2=$?
echo "--- overlay exit=$RC2 $(date -Is) ---" | tee -a "$LOG"

echo "=== DONE $(date -Is) rc1=$RC1 rc2=$RC2 ===" | tee -a "$LOG"
exit $(( RC1 != 0 ? RC1 : RC2 ))
