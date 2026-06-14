#!/usr/bin/env python3
"""Build the curated qualitative 2D trajectory figure for the paper.

Reads the per-(dataset, controller) trajectories dumped by runner_2d.py
(``traj/{dataset}_{controller}.npy``) and the matching metric JSON (used only to
trim the padding back to each scene's executed length), and renders a 2x2 panel
(one ETH-UCY scene per dataset) overlaying FCP-MPC (soft, ours) against the
baselines so the qualitative behaviour can be compared at a glance.

Output goes to the paper folder so it can be \\includegraphics'd directly:
    T_RO2026/traj_2d.png   (override with FCP_PAPER_DIR)
"""
from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- shared publication (paper) style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
})

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.environ.get("FCP_PAPER_DIR", os.path.join(HERE, "T_RO2026"))
OUT_PATH = os.path.join(PAPER_DIR, "traj_2d.png")

DATASETS = ["zara1", "zara2", "eth", "univ"]
SCENE_IDX = 0  # representative scene per dataset (keeps the overlay legible)

# Start / goal per dataset (mirror of runner_2d.py eval_task_configs).
TASK = {
    "zara1": {"start": (12.0, 5.0), "goal": (3.0, 6.0)},
    "zara2": {"start": (1.0, 6.0), "goal": (14.0, 5.0)},
    "eth":   {"start": (5.0, 1.0), "goal": (3.0, 10.0)},
    "univ":  {"start": (3.5, 2.0), "goal": (11.5, 8.5)},
}

# (controller key, label, color, linewidth, linestyle, zorder). Ours is emphasized.
METHODS = [
    ("cc",                "CC-MPC",            "#7f7f7f", 1.6, "-",  2),
    ("ecp-mpc",           "ECP-MPC",           "#ff7f0e", 1.6, "-",  2),
    ("acp-mpc",           "ACP-MPC",           "#9467bd", 1.6, "-",  2),
    ("fcp-hard-adaptive", "FCP-MPC (hard, ours)", "#08519c", 2.4, "--", 4),
    ("fcp-soft-adaptive", "FCP-MPC (soft, ours)", "#1f77b4", 2.8, "-",  5),
]


def _scene_lengths(dataset, key):
    """Executed length (#points) per scene = steps + 1, from the metric JSON."""
    path = os.path.join(HERE, "metric", f"{dataset}_{key}.json")
    if not os.path.isfile(path):
        return None
    d = json.load(open(path))
    return [int(t) + 1 for t in d.get("time", [])]


def _load_scene(dataset, key, scene_idx):
    """Return the (L,2) trajectory for one scene, trimmed to its executed length."""
    npy = os.path.join(HERE, "traj", f"{dataset}_{key}.npy")
    if not os.path.isfile(npy):
        return None
    arr = np.load(npy)  # (S, maxL, 2)
    if scene_idx >= arr.shape[0]:
        return None
    traj = arr[scene_idx]
    lengths = _scene_lengths(dataset, key)
    L = traj.shape[0]
    if lengths is not None and scene_idx < len(lengths):
        L = min(lengths[scene_idx], L)
    traj = traj[:L]
    # Drop any trailing exact-zero padding (FCP pads with zeros; baselines do not).
    nz = np.where(np.any(traj != 0.0, axis=1))[0]
    if nz.size:
        traj = traj[: int(nz[-1]) + 1]
    return traj if traj.shape[0] >= 2 else None


def main():
    os.makedirs(PAPER_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    axes = axes.ravel()

    for ax, dataset in zip(axes, DATASETS):
        for key, label, color, lw, ls, z in METHODS:
            traj = _load_scene(dataset, key, SCENE_IDX)
            if traj is None:
                continue
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=lw, ls=ls,
                    alpha=0.9, zorder=z, label=label)

        start = TASK[dataset]["start"]
        goal = TASK[dataset]["goal"]
        ax.scatter(*start, c="#2ca02c", s=110, marker="s", zorder=6, label="start")
        ax.scatter(*goal, c="#111111", s=170, marker="*", zorder=6, label="goal")

        ax.set_title(dataset, fontsize=13)
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="in", length=3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] qualitative 2D trajectory figure -> {OUT_PATH}")


if __name__ == "__main__":
    main()
