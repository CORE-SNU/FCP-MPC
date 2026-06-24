"""Density study: FCP-MPC (hard) vs (soft) at a *lower* obstacle count, to show that
the hard variant is overconservative only in the pathologically dense regime and
recovers feasibility / reliable goal-reaching as density drops, while soft is robust
throughout. Same 17 seeds and config as the main 3D table; in-process so the
controller settings apply. Outcomes are what matter here (deterministic).
"""
from __future__ import annotations
import os, pickle, sys, time
import numpy as np
from make_figs_3d import EXP_BASE, build_env
from sim_func_3d import run_one_episode_visual_3d as run

SEEDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
DENSITIES = [int(x) for x in sys.argv[1:]] or [100]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "density_3d_results.pkl")
METHODS = {"FCP-MPC (hard)": dict(CP=True, safety_mode="hard"),
           "FCP-MPC (soft)": dict(CP=True, safety_mode="soft")}

import inspect
ALLOWED = set(inspect.signature(run).parameters)


def main():
    rows = []
    print(f"{'n_obs':>5} {'method':16s} {'seed':>4} {'reach':>5} {'coll':>6} {'infeas':>6}", flush=True)
    for n_obs in DENSITIES:
        for lab, over in METHODS.items():
            for s in SEEDS:
                exp = dict(EXP_BASE); exp["max_steps"] = 250; exp["n_jobs"] = 1
                exp.update(break_on_collision=True, **over)
                exp = {k: v for k, v in exp.items() if k in ALLOWED}
                t0 = time.perf_counter()
                r = run(build_env(s, n_obs), **exp)
                st = max(1, r["steps"])
                row = dict(n_obs=n_obs, label=lab, seed=s,
                           reached=int(r["reached_goal"]), steps=r["steps"],
                           collision_rate=r["collisions"] / st,
                           infeas_rate=r["infeasible_steps"] / st,
                           ctrl_ms=float(np.mean(r["ctrl_times_ms"])) if r["ctrl_times_ms"] else float("nan"))
                rows.append(row)
                print(f"{n_obs:>5} {lab:16s} {s:>4} {row['reached']:>5} "
                      f"{row['collision_rate']:>6.3f} {row['infeas_rate']:>6.3f} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    pickle.dump(rows, open(OUT, "wb"))
    print("\n=== means ===", flush=True)
    for n_obs in DENSITIES:
        for lab in METHODS:
            rs = [r for r in rows if r["n_obs"] == n_obs and r["label"] == lab]
            reached = [r["steps"] for r in rs if r["reached"]]
            print(f"  n_obs={n_obs:>3} {lab:16s} reach={np.mean([r['reached'] for r in rs]):.2f} "
                  f"coll={np.mean([r['collision_rate'] for r in rs]):.3f} "
                  f"infeas={np.mean([r['infeas_rate'] for r in rs]):.3f} "
                  f"steps={np.mean(reached) if reached else float('nan'):.1f} "
                  f"ctrl={np.nanmean([r['ctrl_ms'] for r in rs]):.1f}ms", flush=True)
    print(f"[saved] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
