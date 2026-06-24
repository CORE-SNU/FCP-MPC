"""Scan a pool of seeds to find good trajectory-figure panels: seeds where
FCP-MPC reaches the goal AND the baselines (especially ECP) survive long enough
to be visible. Reuses the driver's episode runner; the paper TABLE is unaffected
(it stays on seeds 30-34) -- this only picks representative panels for the
qualitative trajectory figure.

Outputs a ranked summary and caches all episode results so the figure can be
rendered from any chosen 3 seeds without recomputation.
"""
from __future__ import annotations
import os, pickle
import numpy as np

from make_3d_results import (run_jobs, METHOD_LABELS, EXP_BASE, CACHE,
                             render_traj)

POOL = [20, 21, 22, 23, 24, 35, 36, 37, 38, 39, 40, 41]
N_OBS = 280
OUT_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "traj_scan_results.pkl")


def main():
    exp_base = dict(EXP_BASE)
    exp_base["max_steps"] = 250
    exp_base["n_jobs"] = 1

    # reuse already-computed 30-34 episodes from the driver cache
    cached = []
    if os.path.isfile(CACHE):
        c = pickle.load(open(CACHE, "rb"))
        cached = c.get("main", [])
        print(f"[scan] loaded {len(cached)} cached episodes (seeds {sorted({r['seed'] for r in cached})})")

    jobs = [(lab, s, N_OBS, exp_base, True) for s in POOL for lab in METHOD_LABELS]
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"[scan] running {len(jobs)} episodes on {workers} workers over pool {POOL}", flush=True)
    fresh = run_jobs(jobs, workers)

    results = cached + fresh
    pickle.dump(results, open(OUT_PKL, "wb"))
    print(f"[scan] cached combined results -> {OUT_PKL}")

    seeds = sorted({r["seed"] for r in results})
    print("\nseed |  FCP reach/steps |  ECP steps |  ACP steps |  CC steps | FCP coll")
    print("-" * 78)
    for s in seeds:
        rs = {r["label"]: r for r in results if r["seed"] == s}
        def st(lab):
            r = rs.get(lab)
            return r["metrics"]["steps"] if r else -1
        def reach(lab):
            r = rs.get(lab)
            return int(r["metrics"]["reached_goal"]) if r else -1
        fcp = rs.get("FCP-MPC (ours)")
        coll = fcp["metrics"]["collision_rate"] if fcp else -1
        flag = "  <== FCP reaches" if reach("FCP-MPC (ours)") == 1 else ""
        print(f"{s:>4} |    {reach('FCP-MPC (ours)')} / {st('FCP-MPC (ours)'):>4}     |"
              f"   {st('ECP-MPC'):>4}    |   {st('ACP-MPC'):>4}    |  {st('CC-MPC'):>4}   |"
              f"  {coll:.3f}{flag}")


if __name__ == "__main__":
    main()
