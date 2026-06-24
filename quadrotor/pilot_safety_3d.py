"""Pilot: does the 3D setting actually exercise *collision safety* (not just
navigability)? We compare FCP-MPC (conformal bound ON) against a Nominal MPC
(same sampling controller, conformal bound OFF -> raw predicted distances) under
the current noise and a harder, more erratic-obstacle setting. If the conformal
bound meaningfully lowers collisions while both traverse, that is the safety
story; it also tells us which noise level to use for the final run.
"""
from __future__ import annotations
import itertools
import numpy as np
from quad_env import QuadWorldEnv3D
from sim_func_3d import run_one_episode_visual_3d

BASE = dict(
    dt=0.1, horizon=20, world_bounds_xyz=((-3, 7), (-3, 7), (0, 8)),
    mode_switch_p=0.95, mode_min_ttl=1, mode_max_ttl=6, stop_go_p=0.6, gui=False,
)
EXP = dict(nx=40, ny=40, nz=40, time_horizon=12, n_skip=4, n_paths=2000,
           max_steps=250, n_calib_samples=20, alpha=0.10, p_base=8, k_mix=10,
           visualize=False, break_on_collision=False)

# (name, noise overrides)
SETTINGS = {
    "current": dict(pred_model_noise=0.20, obs_process_noise=0.22,
                    gt_future_noise=0.20, turn_rate_std=3.0),
    "harder":  dict(pred_model_noise=0.20, obs_process_noise=0.40,
                    gt_future_noise=0.35, turn_rate_std=5.0),
}
SEEDS = [30, 31, 32]
N_OBS = 280


def main():
    print(f"{'setting':9s} {'seed':>4} {'CP':>5} {'reach':>5} {'steps':>5} "
          f"{'coll':>5} {'coll_rate':>9} {'infeas_rate':>11}", flush=True)
    rows = []
    for sname, seed, cp in itertools.product(SETTINGS, SEEDS, [True, False]):
        env = QuadWorldEnv3D(seed=seed, n_obs=N_OBS, **BASE, **SETTINGS[sname])
        r = run_one_episode_visual_3d(env, CP=cp, **EXP)
        steps = max(1, r["steps"])
        cr = r["collisions"] / steps
        ir = r["infeasible_steps"] / steps
        rows.append((sname, seed, cp, r["reached_goal"], r["steps"],
                     r["collisions"], cr, ir))
        print(f"{sname:9s} {seed:>4} {str(cp):>5} {str(r['reached_goal']):>5} "
              f"{r['steps']:>5} {r['collisions']:>5} {cr:>9.3f} {ir:>11.3f}",
              flush=True)

    print("\n=== summary (mean over seeds) ===", flush=True)
    for sname in SETTINGS:
        for cp in [True, False]:
            sub = [x for x in rows if x[0] == sname and x[2] == cp]
            reach = np.mean([x[3] for x in sub])
            cr = np.mean([x[6] for x in sub]); ir = np.mean([x[7] for x in sub])
            tag = "FCP " if cp else "Nom."
            print(f"{sname:9s} {tag} reach={reach:.2f} coll_rate={cr:.3f} infeas={ir:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
