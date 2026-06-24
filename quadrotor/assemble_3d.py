#!/usr/bin/env python3
"""Merge baseline_3d_cache.pkl + fcp_3d_cache.pkl (from run_subset_3d.py) into the dense
3D deliverables: table_3d_results.tex (mean±std), traj_3d_seeds.png, control_time_3d.png.

Lets the unchanged baselines be reused across FCP envelope iterations:
  python run_subset_3d.py --which baseline   # once
  python run_subset_3d.py --which fcp        # after each FCP change
  python assemble_3d.py                      # regenerate dense table + figures
"""
from __future__ import annotations
import os
import pickle
import make_3d_results as D


def _load(name):
    p = os.path.join(D.HERE, name)
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    b = _load("baseline_3d_cache.pkl")
    f = _load("fcp_3d_cache.pkl")
    main_results = b["main"] + f["main"]
    timing = b["timing"] + f["timing"]
    seeds = b["seeds"]

    D.METHOD_LABELS = list(D.TABLE_ORDER)  # all five for table/figures

    N_OBS_MAIN = 280
    clean = D.clean_ctrl_by_method(timing, N_OBS_MAIN)
    D.write_table(main_results, clean, seeds)   # -> table_3d_results.tex (mean±std)
    D.write_scalability(timing)

    # trajectory-panel seeds: FCP-soft success AND baselines travel before crashing
    def _metric(seed, label, key):
        for r in main_results:
            if r["seed"] == seed and r["label"] == label:
                return r["metrics"].get(key)
        return None

    BASE = ["ACP-MPC", "CC-MPC", "ECP-MPC"]
    succ = [s for s in seeds if bool(_metric(s, "FCP-MPC (soft)", "reached_goal"))]

    def _bmin(s):
        return min([(_metric(s, bl, "steps") or 0) for bl in BASE]) if BASE else 0

    def _fsteps(s):
        return _metric(s, "FCP-MPC (soft)", "steps") or 1e9

    ranked = sorted(succ, key=lambda s: (_bmin(s), -_fsteps(s)), reverse=True)
    traj_seeds = ranked[:3] or list(seeds)[:3]
    print(f"[traj] FCP-soft reaches on {succ}; figure uses {traj_seeds}", flush=True)
    D.render_traj(main_results, traj_seeds)
    D.render_scalability()
    print("[assemble] dense table + traj + scalability regenerated", flush=True)


if __name__ == "__main__":
    main()
