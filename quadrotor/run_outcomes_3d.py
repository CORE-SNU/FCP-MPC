#!/usr/bin/env python3
"""Outcomes-only 3D re-run (parallel) after the reset/kinematic-state fix in
quad_env.reset (first observation now reflects the true z=1.0 start instead of the
stale ~0.11 m floor-line altitude that spuriously crashed controllers at step 1).

Only closed-loop OUTCOMES are recomputed here (collision / infeasible / steps /
reached). Control-time numbers are NOT affected by the start-altitude fix, so the
existing contention-free sequential timing pass is reused as-is.

Runs both densities used in the paper (N_obs = 50 and 280), all five methods, the
same 17 seeds as the subset runner, with want_traj=True so the trajectory figures
can be regenerated.
"""
from __future__ import annotations
import os
import pickle

import make_3d_results as D

SEEDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
N_OBS = [50, 280]
LABELS = ["ACP-MPC", "CC-MPC", "ECP-MPC", "FCP-MPC (hard)", "FCP-MPC (soft)"]
OUT = os.path.join(D.HERE, "outcomes_3d_fixed.pkl")


def main():
    workers = max(1, (os.cpu_count() or 2) - 1)
    eb = dict(D.EXP_BASE)
    eb["max_steps"] = 250
    eb["n_jobs"] = 1
    jobs = [(lab, s, n, eb, True) for n in N_OBS for s in SEEDS for lab in LABELS]
    print(f"[outcomes] {len(jobs)} episodes (N_obs={N_OBS}) on {workers} workers", flush=True)
    res = D.run_jobs(jobs, workers)
    with open(OUT, "wb") as f:
        pickle.dump({"main": res, "seeds": SEEDS, "n_obs": N_OBS, "labels": LABELS}, f)
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
