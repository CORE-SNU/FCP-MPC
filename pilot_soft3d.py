"""Pilot the new 3D soft variant: FCP-hard vs FCP-soft vs Nominal on the same
seeds/scenarios (n_calib_samples kept from EXP_BASE so the env RNG -- and thus the
obstacle trajectories -- are identical across methods). Expectation: soft removes
the hard variant's stalling (infeasible rate -> ~0), so it reaches more while the
conformal penalty keeps collisions low.
"""
from __future__ import annotations
import inspect
import numpy as np
from make_figs_3d import EXP_BASE, build_env
from sim_func_3d import run_one_episode_visual_3d as run

SEEDS = [20, 30, 31, 32, 36, 41]   # mix incl. seeds where hard struggled
N_OBS = 280
ALLOWED = set(inspect.signature(run).parameters)


def run_one(seed, **over):
    exp = dict(EXP_BASE); exp["max_steps"] = 250; exp["n_jobs"] = 1
    exp.update(break_on_collision=True, **over)
    exp = {k: v for k, v in exp.items() if k in ALLOWED}
    env = build_env(seed, N_OBS)
    r = run(env, **exp)
    s = max(1, r["steps"])
    return dict(reach=int(r["reached_goal"]), steps=r["steps"],
                cr=r["collisions"] / s, ir=r["infeasible_steps"] / s)


def main():
    methods = {
        "FCP-hard": dict(CP=True, safety_mode="hard"),
        "FCP-soft": dict(CP=True, safety_mode="soft"),
        "Nominal ": dict(CP=False, safety_mode="hard"),
    }
    agg = {m: [] for m in methods}
    print(f"{'method':9s} {'seed':>4} {'reach':>5} {'steps':>5} {'coll_r':>7} {'infeas':>7}", flush=True)
    for seed in SEEDS:
        for m, over in methods.items():
            r = run_one(seed, **over)
            agg[m].append(r)
            print(f"{m:9s} {seed:>4} {r['reach']:>5} {r['steps']:>5} {r['cr']:>7.3f} {r['ir']:>7.3f}",
                  flush=True)
    print("\n=== means ===", flush=True)
    for m, rs in agg.items():
        print(f"{m:9s} reach={np.mean([x['reach'] for x in rs]):.2f} "
              f"coll_r={np.mean([x['cr'] for x in rs]):.3f} "
              f"infeas={np.mean([x['ir'] for x in rs]):.3f} "
              f"steps(reached)={np.mean([x['steps'] for x in rs if x['reach']] or [float('nan')]):.1f}",
              flush=True)


if __name__ == "__main__":
    main()
