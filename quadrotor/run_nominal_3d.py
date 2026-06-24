"""Run a Nominal-MPC baseline (the same sampling controller as FCP-MPC but with
the conformal bound switched OFF -> raw predicted distances) over the same 17
seeds and configuration as the 3D table. This is the safety ablation: it isolates
what the conformal lower bound buys, since Nominal traverses the field but without
calibrated inflation it collides where FCP does not.

Sequential (single-thread) so the control timing is contention-free and directly
comparable to the table's other methods. Outcomes are deterministic.
"""
from __future__ import annotations
import os, pickle, time
import numpy as np

from make_figs_3d import ENV_KWARGS, EXP_BASE, build_env
from sim_func_3d import run_one_episode_visual_3d as run_fcp

SEEDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
N_OBS = 280
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nominal_3d_results.pkl")


def main():
    exp = dict(EXP_BASE)
    exp["max_steps"] = 250
    exp["n_jobs"] = 1
    # Match FCP's config exactly (incl. n_calib_samples from EXP_BASE) so the env
    # RNG advances identically and Nominal sees the SAME obstacle trajectories;
    # only CP=False differs. The calibration is computed but unused (CP off).
    exp.update(CP=False, break_on_collision=True)
    # keep only kwargs the function accepts
    import inspect
    allowed = set(inspect.signature(run_fcp).parameters)
    exp = {k: v for k, v in exp.items() if k in allowed}

    rows = []
    print(f"{'seed':>4} {'reach':>5} {'steps':>5} {'coll':>5} {'coll_rate':>9} "
          f"{'infeas_rate':>11} {'ctrl_ms':>8}", flush=True)
    for s in SEEDS:
        env = build_env(s, N_OBS)
        t0 = time.perf_counter()
        r = run_fcp(env, **exp)
        steps = max(1, r["steps"])
        cr = r["collisions"] / steps
        ir = r["infeasible_steps"] / steps
        ctrl = float(np.mean(r["ctrl_times_ms"])) if r["ctrl_times_ms"] else float("nan")
        rows.append(dict(seed=s, reached=int(r["reached_goal"]), steps=r["steps"],
                         collisions=r["collisions"], collision_rate=cr,
                         infeas_rate=ir, ctrl_ms=ctrl))
        print(f"{s:>4} {int(r['reached_goal']):>5} {r['steps']:>5} {r['collisions']:>5} "
              f"{cr:>9.3f} {ir:>11.3f} {ctrl:>8.1f}  ({time.perf_counter()-t0:.0f}s)",
              flush=True)

    pickle.dump(rows, open(OUT, "wb"))
    reach = np.mean([x["reached"] for x in rows])
    coll = np.mean([x["collision_rate"] for x in rows])
    inf = np.mean([x["infeas_rate"] for x in rows])
    reached_steps = [x["steps"] for x in rows if x["reached"]]
    steps = np.mean(reached_steps) if reached_steps else float("nan")
    ctrl = np.nanmean([x["ctrl_ms"] for x in rows])
    print("\n=== Nominal MPC aggregate (17 seeds) ===", flush=True)
    print(f"reach={reach:.2f} coll_rate={coll:.3f} infeas_rate={inf:.3f} "
          f"steps(reached)={steps:.1f} ctrl_ms={ctrl:.1f}", flush=True)
    print(f"[saved] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
