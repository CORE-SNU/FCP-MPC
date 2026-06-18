#!/usr/bin/env bash
set -uo pipefail
cd /home/sju5379/cp_scratch
LOG=/home/sju5379/cp_scratch/run_retime_3d.log
: > "$LOG"
echo "=== RETIME START $(date -Is) ===" | tee -a "$LOG"
# reuse cached 17-seed outcomes (mean±std) + re-measure timing only (ECP warmup fix); 10 timing seeds for speed
conda run --no-capture-output -n cp python make_3d_results.py --fix-timing \
  --seeds 20 21 22 23 24 30 31 32 33 34 35 36 37 38 39 40 41 \
  --timing-seeds 20 21 22 23 24 30 31 32 33 34 >> "$LOG" 2>&1
echo "=== RETIME DONE $(date -Is) rc=$? ===" | tee -a "$LOG"
