"""
Static trajectory-image helpers (asset-free, headless-safe).

These are used to dump a quick-look PNG of the closed-loop robot trajectory
whenever an experiment is run, so every run leaves behind a figure that can be
dropped straight into the paper / slides without launching Rerun.

Both helpers force the non-interactive Agg backend and never raise: a failed
plot must not crash a simulation sweep.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_traj_image_3d(
    *,
    robot_traj: np.ndarray,
    goal: Optional[np.ndarray] = None,
    start: Optional[np.ndarray] = None,
    obstacles: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
    title: str = "",
    out_path: str = "traj_3d/traj.png",
    max_obstacles: int = 400,
    dpi: int = 150,
) -> Optional[str]:
    """Save a static 3-D plot of a quadrotor's closed-loop trajectory.

    robot_traj : (T, 3) world-frame path.
    obstacles  : (M, 3) snapshot of obstacle positions (optional, sub-sampled).
    bounds     : ((xmin,xmax),(ymin,ymax),(zmin,zmax)) world box (optional).
    Returns the written path, or None on failure / empty trajectory.
    """
    try:
        tr = np.asarray(robot_traj, dtype=np.float32).reshape(-1, 3)
        if tr.shape[0] < 1:
            return None

        _ensure_dir(out_path)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        if obstacles is not None:
            ob = np.asarray(obstacles, dtype=np.float32).reshape(-1, 3)
            if ob.shape[0] > max_obstacles:
                idx = np.random.default_rng(0).choice(ob.shape[0], max_obstacles, replace=False)
                ob = ob[idx]
            if ob.size:
                ax.scatter(ob[:, 0], ob[:, 1], ob[:, 2],
                           c="0.55", s=8, alpha=0.5, label="obstacles", depthshade=True)

        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="#FFB300", lw=2.0, label="robot path")

        s = tr[0] if start is None else np.asarray(start, dtype=np.float32).reshape(3)
        ax.scatter(*s, c="#1E90FF", s=70, marker="o", label="start", depthshade=False)
        if goal is not None:
            g = np.asarray(goal, dtype=np.float32).reshape(3)
            ax.scatter(*g, c="#111111", s=90, marker="*", label="goal", depthshade=False)

        if bounds is not None:
            (x0, x1), (y0, y1), (z0, z1) = bounds
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_zlim(z0, z1)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        if title:
            ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as e:  # never crash a sweep on a plotting error
        print(f"[viz_traj] 3D plot failed for {out_path}: {e}")
        plt.close("all")
        return None


def save_traj_image_2d(
    *,
    trajectories: Sequence[np.ndarray],
    goal: Optional[np.ndarray] = None,
    start: Optional[np.ndarray] = None,
    title: str = "",
    out_path: str = "traj/traj.png",
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    dpi: int = 150,
) -> Optional[str]:
    """Save a static top-down plot of one or more 2-D robot trajectories.

    trajectories : iterable of (T, 2) world-frame paths (one per scenario).
    Asset-free fallback for the 2-D pedestrian experiments (no background frame
    or homography required). Returns the written path or None.
    """
    try:
        trajs = [np.asarray(t, dtype=np.float32).reshape(-1, 2) for t in trajectories]
        trajs = [t for t in trajs if t.shape[0] >= 1]
        if not trajs:
            return None

        _ensure_dir(out_path)
        fig, ax = plt.subplots(figsize=(7, 6))

        marked_start = False
        for t in trajs:
            ax.plot(t[:, 0], t[:, 1], color="#1f77b4", lw=2.0, alpha=0.8)
            ax.scatter(t[-1, 0], t[-1, 1], color="#1f77b4", s=25, zorder=5)
            if not marked_start:
                s = t[0] if start is None else np.asarray(start, dtype=np.float32).reshape(2)
                ax.scatter(*s, c="#2ca02c", s=80, marker="s", zorder=6, label="start")
                marked_start = True

        if goal is not None:
            g = np.asarray(goal, dtype=np.float32).reshape(2)
            ax.scatter(*g, c="#111111", s=100, marker="*", zorder=6, label="goal")

        if bounds is not None:
            (x0, x1), (y0, y1) = bounds
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if title:
            ax.set_title(title, fontsize=11)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"[viz_traj] 2D plot failed for {out_path}: {e}")
        plt.close("all")
        return None
