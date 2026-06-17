#!/usr/bin/env python3
"""Generate a 3-D baseline-overlay trajectory figure (CC / ACP / FCP) for one seed.

Runs one episode each for CC-MPC, ACP-MPC, and FCP-MPC in a M=280-obstacle 3D
environment (same env_kwargs as runner_3d.py), then overlays the three robot_traj
arrays on a single 3-D axes and writes:

    T_RO2026/traj_3d_overlay.png

Usage:
    conda run -n cp python make_traj_3d_overlay.py [--seed 25] [--n-obs 280]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from quad_env import QuadWorldEnv3D
from sim_cp_3d   import run_one_episode_rerun_simple as run_cc
from sim_acp_3d  import run_one_episode_acp_3d       as run_acp
from sim_func_3d import run_one_episode_visual_3d    as run_fcp

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

ENV_KWARGS = dict(
    dt=0.1, horizon=20,
    world_bounds_xyz=((-3, 7), (-3, 7), (0, 8)),
    pred_model_noise=0.20, obs_process_noise=0.22, gt_future_noise=0.20,
    mode_switch_p=0.95, mode_min_ttl=1, mode_max_ttl=6,
    turn_rate_std=3.0, stop_go_p=0.6, gui=False,
    # Dynamic environment (paper's main setting): half the obstacles are crossing
    # pedestrians that traverse the workspace toward goals; the rest random-walk.
    goal_directed_frac=0.5,
)
EXP_BASE = dict(
    nx=40, ny=40, nz=40,
    time_horizon=12,
    n_skip=4,
    n_paths=2000,
    max_steps=250,
    n_calib_samples=20,
    backend="loky",
    visualize=False,
    save_rrd=False,
)

METHOD_COLORS = {
    "CC-MPC":  "#E64B35",   # red
    "ACP-MPC": "#4DBBD5",   # cyan
    "FCP-MPC": "#00A087",   # green
}
METHOD_LS = {
    "CC-MPC":  "-",
    "ACP-MPC": "--",
    "FCP-MPC": "-.",
}

OUT_PATH = os.path.join(os.path.dirname(__file__), "T_RO2026", "traj_3d_overlay.png")


def run_one(env_kwargs, exp_kwargs, run_fn, allowed_keys=None) -> dict:
    if allowed_keys is not None:
        exp_kwargs = {k: v for k, v in exp_kwargs.items() if k in allowed_keys}
    return run_fn(env_kwargs["_env"], **{k: v for k, v in exp_kwargs.items()})


def build_env(env_kwargs_base: dict, seed: int, n_obs: int) -> QuadWorldEnv3D:
    kw = dict(env_kwargs_base)
    kw["seed"] = seed
    kw["n_obs"] = n_obs
    return QuadWorldEnv3D(**kw)


def run_method(name: str, seed: int, n_obs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (robot_traj (T,3), goal (3,), obs_snapshot (M,3))."""
    import inspect

    env = build_env(ENV_KWARGS, seed=seed, n_obs=n_obs)
    exp = dict(EXP_BASE)

    if name == "CC-MPC":
        run_fn = run_cc
    elif name == "ACP-MPC":
        run_fn = run_acp
        exp["target_miscoverage_level"] = 0.10
        exp["step_size"] = 0.05
    elif name == "FCP-MPC":
        run_fn = run_fcp
        exp["CP"] = True
        exp["alpha"] = 0.10
    else:
        raise ValueError(f"Unknown method: {name}")

    allowed = set(inspect.signature(run_fn).parameters.keys())
    exp_clean = {k: v for k, v in exp.items() if k in allowed}

    t0 = time.perf_counter()
    result = run_fn(env, **exp_clean)
    elapsed = time.perf_counter() - t0

    traj = np.asarray(result["robot_traj"], dtype=np.float32).reshape(-1, 3)
    goal = np.asarray(env.goal_xyz, dtype=np.float32)
    # obstacle positions at end of episode (env.obstacles is always up-to-date)
    obs = np.zeros((0, 3), dtype=np.float32)
    try:
        if hasattr(env, "obstacles") and env.obstacles:
            obs = np.array([ob.pos for ob in env.obstacles], dtype=np.float32)
    except Exception:
        pass

    print(
        f"  [{name}] steps={result.get('steps', len(traj))} "
        f"coll={result.get('collisions', '?')} "
        f"reached={result.get('reached_goal', '?')} "
        f"elapsed={elapsed:.1f}s"
    )
    return traj, goal, obs


def make_overlay(
    trajs: dict,       # method_name -> (T,3) array
    goal: np.ndarray,
    obs: np.ndarray,
    bounds,
    out_path: str,
    seed: int,
    n_obs: int,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # obstacles
    if obs.size:
        rng = np.random.default_rng(0)
        max_obs = 350
        if obs.shape[0] > max_obs:
            idx = rng.choice(obs.shape[0], max_obs, replace=False)
            obs_plot = obs[idx]
        else:
            obs_plot = obs
        ax.scatter(
            obs_plot[:, 0], obs_plot[:, 1], obs_plot[:, 2],
            c="0.65", s=6, alpha=0.35, depthshade=True, zorder=1,
        )

    # trajectories
    for name, traj in trajs.items():
        color = METHOD_COLORS[name]
        ls    = METHOD_LS[name]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=color, lw=2.0, ls=ls, label=name, zorder=5)
        # start dot
        ax.scatter(*traj[0], c=color, s=60, marker="o", depthshade=False, zorder=6)

    # goal star
    ax.scatter(*goal, c="#111111", s=120, marker="*", label="goal", depthshade=False, zorder=7)

    (x0, x1), (y0, y1), (z0, z1) = bounds
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_xlabel("$x$ [m]", fontsize=12, labelpad=5)
    ax.set_ylabel("$y$ [m]", fontsize=12, labelpad=5)
    ax.set_zlabel("$z$ [m]", fontsize=12, labelpad=3)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.8)
    ax.view_init(elev=22, azim=-55)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",  type=int, default=25)
    ap.add_argument("--n-obs", type=int, default=280)
    ap.add_argument("--out",   default=OUT_PATH)
    args = ap.parse_args()

    methods = ["CC-MPC", "ACP-MPC", "FCP-MPC"]
    trajs   = {}
    goals   = []
    obs_snap = np.zeros((0, 3), dtype=np.float32)

    for name in methods:
        print(f"\n>>> Running {name}  (seed={args.seed}, n_obs={args.n_obs}) ...")
        try:
            traj, goal, obs = run_method(name, seed=args.seed, n_obs=args.n_obs)
            trajs[name] = traj
            goals.append(goal)
            if obs.shape[0] > obs_snap.shape[0]:
                obs_snap = obs
        except Exception as e:
            import traceback
            print(f"  [WARN] {name} failed: {e}")
            traceback.print_exc()

    if not trajs:
        sys.exit("All methods failed — nothing to plot.")

    goal = goals[0] if goals else np.array([2.0, 2.0, 2.0])
    bounds = ((-3, 7), (-3, 7), (0, 8))

    make_overlay(
        trajs=trajs,
        goal=goal,
        obs=obs_snap,
        bounds=bounds,
        out_path=args.out,
        seed=args.seed,
        n_obs=args.n_obs,
    )


if __name__ == "__main__":
    main()
