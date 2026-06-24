#!/usr/bin/env bash
set -uo pipefail
cd /home/sju5379/cp_scratch
LOG=/home/sju5379/cp_scratch/run_sparse_now.log
: > "$LOG"
echo "=== SPARSE START $(date -Is) ===" | tee -a "$LOG"
conda run --no-capture-output -n cp python quadrotor/run_sparse_3d.py >> "$LOG" 2>&1
echo "=== SPARSE DONE $(date -Is) rc=$? ===" | tee -a "$LOG"
