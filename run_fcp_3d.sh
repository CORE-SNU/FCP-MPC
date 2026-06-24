#!/usr/bin/env bash
set -uo pipefail
cd /home/sju5379/cp_scratch
LOG=/home/sju5379/cp_scratch/run_fcp_3d.log
: > "$LOG"
echo "=== FCP START $(date -Is) ===" | tee -a "$LOG"
conda run --no-capture-output -n cp python quadrotor/run_subset_3d.py --which fcp >> "$LOG" 2>&1
RC1=$?
echo "--- fcp subset exit=$RC1 $(date -Is) ---" | tee -a "$LOG"
echo "--- assemble dense table + figures ---" | tee -a "$LOG"
conda run --no-capture-output -n cp python quadrotor/assemble_3d.py >> "$LOG" 2>&1
RC2=$?
echo "=== FCP+ASSEMBLE DONE $(date -Is) rc=$RC1/$RC2 ===" | tee -a "$LOG"
