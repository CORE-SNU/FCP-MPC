"""Does a less-conservative envelope (larger alpha) fix FCP-hard's stalling-and-
getting-hit in 3D? Run FCP-MPC (hard) at alpha in {0.10, 0.15, 0.20} on the seeds
where it stalled badly, and see whether reach goes up / infeasible+collision go
down. (Smaller alpha = larger envelope = more conservative, so we go the OTHER way.)
"""
from __future__ import annotations
import inspect
import numpy as np
from make_figs_3d import EXP_BASE, build_env
from sim_func_3d import run_one_episode_visual_3d as run

SEEDS = [23, 31, 36, 37, 40]   # the high-stall / high-collision seeds
ALPHAS = [0.10, 0.15, 0.20]
N_OBS = 280
ALLOWED = set(inspect.signature(run).parameters)


def run_one(seed, alpha):
    exp = dict(EXP_BASE); exp["max_steps"] = 250; exp["n_jobs"] = 1
    exp.update(CP=True, safety_mode="hard", break_on_collision=True, alpha=alpha)
    exp = {k: v for k, v in exp.items() if k in ALLOWED}
    r = run(build_env(seed, N_OBS), **exp)
    s = max(1, r["steps"])
    return r["reached_goal"], r["collisions"] / s, r["infeasible_steps"] / s


def main():
    print(f"{'alpha':>5} {'seed':>4} {'reach':>5} {'coll':>6} {'infeas':>6}", flush=True)
    agg = {a: [] for a in ALPHAS}
    for a in ALPHAS:
        for s in SEEDS:
            reach, cr, ir = run_one(s, a)
            agg[a].append((reach, cr, ir))
            print(f"{a:>5.2f} {s:>4} {reach:>5} {cr:>6.3f} {ir:>6.3f}", flush=True)
    print("\n=== means over the 5 hard seeds ===", flush=True)
    for a in ALPHAS:
        v = agg[a]
        print(f"alpha={a:.2f} reach={np.mean([x[0] for x in v]):.2f} "
              f"coll={np.mean([x[1] for x in v]):.3f} infeas={np.mean([x[2] for x in v]):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
