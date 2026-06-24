#!/usr/bin/env python3
"""Build the qualitative 3D trajectory figure for the paper (row of seeds).

Mirrors the 2D figure (``make_figs_2d.py``): each panel is one seed of the
M=280-obstacle 3D environment, and within each panel the robot trajectory of
every controller --- CC-MPC, ECP-MPC, ACP-MPC and FCP-MPC (ours) --- is overlaid
so the qualitative behaviour can be compared at a glance. Panels are labelled
(a), (b), (c); method colours and labels are kept consistent with the 2D figure.

Because each panel runs four full closed-loop episodes the simulation is slow, so
the trajectories are cached to ``--cache`` after the sweep; pass ``--replot`` to
re-render the figure (layout/sizing tweaks) without re-simulating.

    T_RO2026/traj_3d_seeds.png   (override dirs via FCP_PAPER_DIR / --out)

Usage:
    conda run -n cp python make_figs_3d.py                  # full sweep (seeds 25-28)
    conda run -n cp python make_figs_3d.py --replot         # re-render from cache
"""
from __future__ import annotations

import argparse
import inspect
import os
import time

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from quad_env import QuadWorldEnv3D
from sim_cp_3d   import run_one_episode_rerun_simple as run_cc
from sim_ecp_3d  import run_one_episode_ecp_3d_rerun  as run_ecp
from sim_acp_3d  import run_one_episode_acp_3d        as run_acp
from sim_func_3d import run_one_episode_visual_3d     as run_fcp

# ---- shared publication (paper) style, matched to make_figs_2d.py ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.environ.get("FCP_PAPER_DIR", os.path.join(HERE, "T_RO2026"))
OUT_PATH = os.path.join(PAPER_DIR, "traj_3d_seeds.png")
CACHE_PATH = os.path.join(PAPER_DIR, "traj_3d_seeds_data.npz")

WORLD_BOUNDS = ((-3, 7), (-3, 7), (0, 8))

ENV_KWARGS = dict(
    dt=0.1, horizon=20,
    world_bounds_xyz=WORLD_BOUNDS,
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
    # Pinned so EVERY method uses the identical goal tolerance (fairness): without
    # this, ACP/FCP fell back to their default 0.8 while CC/ECP used 0.3 (~2.7x
    # closer), spuriously biasing the goal-reach comparison toward FCP.
    goal_finish_dist=0.8,
    backend="loky",
    visualize=False,
    save_rrd=False,
)

# (label, run_fn, method-specific kwargs, colour, linewidth, zorder).
# Colours/labels mirror make_figs_2d.py; FCP (ours) is emphasised and drawn on top.
METHODS = [
    ("CC-MPC",      run_cc,  {"break_on_collision": True},                       "#7f7f7f", 1.6, 2),
    ("ECP-MPC",     run_ecp, {"miscoverage_level": 0.10, "step_size": 0.05, "break_on_collision": True}, "#ff7f0e", 1.6, 2),
    ("ACP-MPC",     run_acp, {"target_miscoverage_level": 0.10, "step_size": 0.05, "break_on_collision": True}, "#9467bd", 1.6, 2),
    ("FCP-MPC (ours)", run_fcp, {"CP": True, "alpha": 0.10, "break_on_collision": True}, "#1f77b4", 2.6, 4),
]


def build_env(seed: int, n_obs: int) -> QuadWorldEnv3D:
    kw = dict(ENV_KWARGS)
    kw["seed"] = seed
    kw["n_obs"] = n_obs
    return QuadWorldEnv3D(**kw)


def run_method(run_fn, extras: dict, env, exp_base: dict) -> np.ndarray:
    """Run one closed-loop episode and return the robot trajectory (T,3)."""
    exp = dict(exp_base, **extras)
    allowed = set(inspect.signature(run_fn).parameters.keys())
    exp_clean = {k: v for k, v in exp.items() if k in allowed}
    result = run_fn(env, **exp_clean)
    return np.asarray(result["robot_traj"], dtype=np.float32).reshape(-1, 3), result


def simulate(seeds, n_obs, exp_base, cache_path=None) -> dict:
    """Run every (seed, method) episode. Returns {seed: {start, goal, obs, trajs}}."""
    data = {}
    for seed in seeds:
        print(f"\n===== seed {seed} (n_obs={n_obs}) =====")
        start = goal = None
        obs = np.zeros((0, 3), dtype=np.float32)
        trajs = {}
        for name, run_fn, extras, *_ in METHODS:
            env = build_env(seed, n_obs)
            if start is None:  # geometry is identical across methods for a seed
                start = np.asarray(env.start_xyz_yaw[:3], dtype=np.float32)
                goal = np.asarray(env.goal_xyz, dtype=np.float32)
                try:
                    if getattr(env, "obstacles", None):
                        obs = np.array([ob.pos for ob in env.obstacles], dtype=np.float32)
                except Exception:
                    pass
            try:
                t0 = time.perf_counter()
                traj, res = run_method(run_fn, extras, env, exp_base)
                trajs[name] = traj
                print(f"  [{name:16s}] steps={res.get('steps', len(traj)):>4} "
                      f"reached={res.get('reached_goal', '?')!s:>5} "
                      f"coll={res.get('collisions', '?')} "
                      f"elapsed={time.perf_counter() - t0:5.1f}s")
            except Exception as e:
                import traceback
                print(f"  [{name:16s}] FAILED: {e}")
                traceback.print_exc()
        data[seed] = dict(start=start, goal=goal, obs=obs, trajs=trajs)
        if cache_path:
            save_cache(data, cache_path)  # persist after each seed
    return data


def save_cache(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, data=np.array(data, dtype=object))
    print(f"[cached] trajectories -> {path}")


def load_cache(path: str) -> dict:
    blob = np.load(path, allow_pickle=True)
    return blob["data"].item()


def _smooth_path(traj, n_out: int = 240, frac: float = 0.25):
    """Render a control trajectory as a smooth curve: a Savitzky-Golay low-pass over the
    raw closed-loop waypoints (removes the per-step zig-zag without overshooting into fake
    altitude drops) followed by arclength cubic up-sampling. True endpoints are preserved so
    the start/goal/crash markers still land on the real path."""
    traj = np.asarray(traj, dtype=float)
    m = traj.shape[0]
    if m < 5:
        return traj
    win = max(5, int(m * frac))
    if win % 2 == 0:
        win += 1
    win = min(win, m if m % 2 == 1 else m - 1)
    if win < 5:
        return traj
    sm = np.column_stack([savgol_filter(traj[:, k], win, 3) for k in range(3)])
    sm[0], sm[-1] = traj[0], traj[-1]          # keep the real endpoints
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(sm, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return sm
    u = d / d[-1]
    keep = np.r_[True, np.diff(u) > 1e-9]
    u, sm = u[keep], sm[keep]
    if sm.shape[0] < 4:
        return sm
    uu = np.linspace(0.0, 1.0, n_out)
    return np.column_stack([interp1d(u, sm[:, k], kind="cubic")(uu) for k in range(3)])


def make_figure(data: dict, out_path: str) -> None:
    seeds = sorted(data.keys())[:3]
    n = len(seeds)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    fig = plt.figure(figsize=(2.6 * n, 3.3))

    (x0, x1), (y0, y1), (z0, z1) = WORLD_BOUNDS
    for i, seed in enumerate(seeds):
        d = data[seed]
        ax = fig.add_subplot(1, n, i + 1, projection="3d")

        obs = d["obs"]
        if isinstance(obs, np.ndarray) and obs.size:
            ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2],
                       c="0.45", s=7, alpha=0.30, depthshade=False, zorder=0)

        crashed = d.get("crashed", {})
        for name, _fn, _ex, color, lw, z in METHODS:
            traj = d["trajs"].get(name)
            if traj is None or len(traj) == 0:
                continue
            if len(traj) >= 2:
                sm = _smooth_path(traj)
                ax.plot(sm[:, 0], sm[:, 1], sm[:, 2],
                        color=color, lw=lw, alpha=0.95, zorder=z)
            else:
                # crashed almost immediately (one logged pose): still show the
                # method with a colored dot so it does not vanish from the panel.
                ax.scatter(*traj[0], color=color, s=28, depthshade=False, zorder=z)
            if crashed.get(name):  # lost control / fell to the floor -> mark the crash point
                ax.scatter(*traj[-1], c="#d62728", marker="x", s=55,
                           linewidths=2.0, depthshade=False, zorder=9)

        # start marker at where the trajectories actually begin (matches the former
        # overlay figure), not the env's nominal start pose which can differ slightly
        start_pt = None
        for _name, *_rest in METHODS:
            tr = d["trajs"].get(_name)
            if tr is not None and len(tr) >= 1:
                start_pt = tr[0]
                break
        if start_pt is None:
            start_pt = d.get("start")
        if start_pt is not None:
            ax.scatter(*start_pt, c="#2ca02c", s=45, marker="s",
                       depthshade=False, zorder=6)
        if d.get("goal") is not None:
            ax.scatter(*d["goal"], c="#111111", s=90, marker="*",
                       depthshade=False, zorder=7)

        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
        ax.set_box_aspect((1, 1, 0.7))
        ax.view_init(elev=22, azim=-55)
        ax.text2D(0.04, 0.90, panel_labels[i], transform=ax.transAxes, fontsize=12)
        ax.set_xlabel("$x$ [m]", fontsize=8, labelpad=-6)
        ax.set_ylabel("$y$ [m]", fontsize=8, labelpad=-6)
        ax.set_zlabel("$z$ [m]", fontsize=8, labelpad=-6)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.zaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=6, pad=-2)

    # shared legend at the bottom, matching the 2D figure
    handles = [Line2D([0], [0], color=c, lw=2.2) for _n, _f, _e, c, _lw, _z in METHODS]
    labels = [m[0] for m in METHODS]
    handles += [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#2ca02c",
               markersize=8, label="start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111111",
               markersize=12, label="goal"),
        Line2D([0], [0], marker="x", color="#d62728", linestyle="none",
               markersize=8, markeredgewidth=2.0, label="crash"),
    ]
    labels += ["start", "goal", "crash"]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.3, handletextpad=0.5)

    fig.subplots_adjust(left=0.0, right=1.0, top=1.02, bottom=0.16, wspace=0.0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] qualitative 3D trajectory figure ({len(seeds)} seeds) -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[25, 26, 27])
    ap.add_argument("--n-obs", type=int, default=280)
    ap.add_argument("--max-steps", type=int, default=None, help="override EXP_BASE max_steps")
    ap.add_argument("--n-paths", type=int, default=None, help="override EXP_BASE n_paths")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--cache", nargs="+", default=[CACHE_PATH],
                    help="cache file(s); first is written when simulating, all merged on --replot")
    ap.add_argument("--replot", action="store_true", help="re-render from cache(s), skip simulation")
    ap.add_argument("--no-plot", action="store_true", help="simulate and cache only, skip the figure")
    args = ap.parse_args()

    if args.replot:
        data = {}
        for c in args.cache:
            if not os.path.isfile(c):
                raise SystemExit(f"--replot given but no cache at {c}")
            data.update(load_cache(c))
        print(f"[loaded] {len(data)} seeds from {len(args.cache)} cache file(s)")
    else:
        exp_base = dict(EXP_BASE)
        if args.max_steps is not None:
            exp_base["max_steps"] = args.max_steps
        if args.n_paths is not None:
            exp_base["n_paths"] = args.n_paths
        data = simulate(args.seeds, args.n_obs, exp_base, cache_path=args.cache[0])

    if not args.no_plot:
        make_figure(data, args.out)


if __name__ == "__main__":
    main()
