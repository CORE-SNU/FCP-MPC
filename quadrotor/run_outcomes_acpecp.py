#!/usr/bin/env python3
"""Outcomes-only re-run for ACP and ECP after switching them to the SHARED
goal-anchored sampling-based planner (the one CC and FCP already use), so all four
3D controllers search the same candidate set and differ only in their conformal
safety logic. CC and FCP are unchanged -> reuse their existing outcomes.
"""
from __future__ import annotations
import os
import pickle
import make_3d_results as D

SEEDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
N_OBS = [50, 280]
LABELS = ["ACP-MPC", "ECP-MPC"]
OUT = os.path.join(D.HERE, "outcomes_acpecp.pkl")


def main():
    workers = max(1, (os.cpu_count() or 2) - 1)
    eb = dict(D.EXP_BASE); eb["max_steps"] = 250; eb["n_jobs"] = 1
    jobs = [(lab, s, n, eb, True) for n in N_OBS for s in SEEDS for lab in LABELS]
    print(f"[acp/ecp outcomes] {len(jobs)} episodes on {workers} workers", flush=True)
    res = D.run_jobs(jobs, workers)
    with open(OUT, "wb") as f:
        pickle.dump({"main": res, "seeds": SEEDS, "n_obs": N_OBS, "labels": LABELS}, f)
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
