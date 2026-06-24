#!/usr/bin/env python3
"""Run a SUBSET of the 3D methods and cache its outcomes + timing, so the UNCHANGED
baselines (ACP/CC/ECP) are computed once and only FCP is re-run as its envelope evolves.

  python run_subset_3d.py --which baseline   # ACP/CC/ECP  -> baseline_3d_cache.pkl
  python run_subset_3d.py --which fcp        # FCP hard/soft -> fcp_3d_cache.pkl

Then `python assemble_3d.py` merges the two caches into the dense table + traj + scalability.
Outcomes run in parallel; timing runs sequentially (contention-free), with the ECP
warmup-step exclusion + per-step pooling already in make_3d_results / sim_ecp_3d.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import pickle
import make_3d_results as D

SEEDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
TIMING_SEEDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34]
N_OBS_MAIN = 280
N_OBS_SWEEP = [10, 50, 100, 150, 200, 280]
TIMING_STEPS = 40
GROUPS = {"baseline": ["ACP-MPC", "CC-MPC", "ECP-MPC"],
          "fcp": ["FCP-MPC (hard)", "FCP-MPC (soft)"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=list(GROUPS), required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    labels = GROUPS[args.which]

    eb = dict(D.EXP_BASE); eb["max_steps"] = 250; eb["n_jobs"] = 1

    # outcomes @ n_obs_main (parallel) -- want_traj=True so the figure has trajectories
    jobs = [(lab, s, N_OBS_MAIN, eb, True) for s in SEEDS for lab in labels]
    print(f"[{args.which}] outcomes: {len(jobs)} episodes (parallel)", flush=True)
    main_results = D.run_jobs(jobs, args.workers)

    # timing sweep (sequential). run_timing_sequential builds jobs from D.METHOD_LABELS.
    D.METHOD_LABELS = labels
    print(f"[{args.which}] timing: {len(TIMING_SEEDS)}x{len(N_OBS_SWEEP)}x{len(labels)}", flush=True)
    timing = D.run_timing_sequential(TIMING_SEEDS, N_OBS_SWEEP, TIMING_STEPS, eb)

    out = os.path.join(D.HERE, f"{args.which}_3d_cache.pkl")
    with open(out, "wb") as f:
        pickle.dump({"main": main_results, "seeds": SEEDS, "timing": timing,
                     "labels": labels}, f)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
