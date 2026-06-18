#!/usr/bin/env bash
set -uo pipefail
cd /home/sju5379/cp_scratch
LOG=/home/sju5379/cp_scratch/run_baseline_3d.log
: > "$LOG"
echo "=== BASELINE START $(date -Is) ===" | tee -a "$LOG"
conda run --no-capture-output -n cp python run_subset_3d.py --which baseline >> "$LOG" 2>&1
echo "=== BASELINE DONE $(date -Is) rc=$? ===" | tee -a "$LOG"
