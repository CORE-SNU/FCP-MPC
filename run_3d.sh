#!/bin/bash
set -euo pipefail

# defaults
METHODS="nocp,cc,fcp,ecp"
SEED_FROM=25
SEED_TO=27
OUT_DIR="metric_3d"
CSV_NAME="quad_suite.csv"
OUT_DT_FAIL_FRAC="0.10"
FAIL_ON="loop"   # loop or ctrl
N_OBS=200
SAVE_TRAJ_IMG=0           # set via --save-traj-img to dump per-run trajectory PNGs
TRAJ_IMG_DIR="traj_3d"
TRAJ_IMG_MAX_SEEDS=3      # only dump trajectory PNGs for the first N seeds

PY="python3"
SCRIPT="quadrotor/runner_3d.py"

# parse args
while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --methods) METHODS="$2"; shift 2 ;;
    --seed-from) SEED_FROM="$2"; shift 2 ;;
    --seed-to) SEED_TO="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --csv-name) CSV_NAME="$2"; shift 2 ;;
    --out-dt-fail-frac) OUT_DT_FAIL_FRAC="$2"; shift 2 ;;
    --fail-on) FAIL_ON="$2"; shift 2 ;;
    --n-obs) N_OBS="$2"; shift 2 ;;
    --save-traj-img) SAVE_TRAJ_IMG=1; shift 1 ;;
    --traj-img-dir) TRAJ_IMG_DIR="$2"; shift 2 ;;
    --traj-img-max-seeds) TRAJ_IMG_MAX_SEEDS="$2"; shift 2 ;;

    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUT_DIR}"

CSV_PATH="${OUT_DIR}/${CSV_NAME}"

echo "[run] methods=${METHODS} seeds=${SEED_FROM}-${SEED_TO} csv=${CSV_PATH} fail_on=${FAIL_ON} out_dt_fail_frac=${OUT_DT_FAIL_FRAC} n_obs=${N_OBS} save_traj_img=${SAVE_TRAJ_IMG}"

EXTRA_ARGS=()
if [[ "${SAVE_TRAJ_IMG}" -eq 1 ]]; then
  EXTRA_ARGS+=(--save-traj-img --traj-img-dir "${TRAJ_IMG_DIR}" --traj-img-max-seeds "${TRAJ_IMG_MAX_SEEDS}")
fi

${PY} ${SCRIPT} \
  --methods "${METHODS}" \
  --seed-from "${SEED_FROM}" \
  --seed-to "${SEED_TO}" \
  --csv-path "${CSV_PATH}" \
  --out-dt-fail-frac "${OUT_DT_FAIL_FRAC}" \
  --fail-on "${FAIL_ON}" \
  --n-obs "${N_OBS}" \
  "${EXTRA_ARGS[@]}" \
  --dump-json