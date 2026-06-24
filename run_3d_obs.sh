#!/bin/bash
set -euo pipefail

METHODS="cc,fcp,ecp"  
SEED_FROM=20
SEED_TO=22
OUT_DIR="metric_3d"
CSV_NAME="ablation_nobs.csv"
CSV_PATH="${OUT_DIR}/${CSV_NAME}"

OBS_LIST=(10 50 100 150 200 280)

mkdir -p "${OUT_DIR}"

echo "[Ablation] Starting experiment... CSV will be saved to ${CSV_PATH}"

for N_OBS in "${OBS_LIST[@]}"; do
  echo "========================================="
  echo " Running N_OBS = ${N_OBS}"
  echo "========================================="

  python3 quadrotor/runner_3d.py \
    --methods "${METHODS}" \
    --seed-from "${SEED_FROM}" \
    --seed-to "${SEED_TO}" \
    --n-obs "${N_OBS}" \
    --csv-path "${CSV_PATH}" \
    --dump-json
done

echo "All ablation tests completed!"