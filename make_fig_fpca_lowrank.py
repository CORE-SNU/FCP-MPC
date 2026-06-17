#!/usr/bin/env python3
"""L3 evidence — the residual (prediction-error) field is LOW-RANK / compressible.

This is the most direct justification for calibrating the functional spatial bound
OFFLINE once and evaluating it cheaply ONLINE: if the ensemble of per-episode residual
fields {f_i(x)} lives in a low-dimensional subspace, a few components capture it, so it
is learnable offline and need not be re-estimated online (the FCP compute advantage, L4).

Method (functional PCA on the discretised field):
  1. For each dataset, reuse the SAME residual-field builder the controller calibrates on
     (`sims.sim_func_cp.build_training_residuals_from_file`) on the SAME grid
     (`utils.build_grid` + `_infer_world`), giving per-episode fields (N, H, Hg, Wg).
  2. Collapse the prediction horizon (mean over h) -> one spatial field per episode (N, D).
  3. Centre and SVD -> explained-variance scree (how many components for X% of variance)
     + leading eigenfunction phi_1(x) reshaped to the grid.

Outputs:
  T_RO2026/fcp_lowrank_fpca.png   scree (all datasets) + phi_1 overlay (representative)
  prints cumulative variance explained by the first k components per dataset.

Usage:
  python make_fig_fpca_lowrank.py                         # all ETH-UCY, univ for phi_1
  python make_fig_fpca_lowrank.py --datasets univ zara1   # subset
  python make_fig_fpca_lowrank.py --phi-dataset zara1     # which scene draws phi_1
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sims"))

from utils import build_grid
from sims.sim_func_cp import _infer_world, build_training_residuals_from_file

PRED_DIR = os.path.join(HERE, "predictions")
PAPER_DIR = os.path.join(HERE, "T_RO2026")
DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]
TIME_HORIZON = 12
GRID = 40  # Hg = Wg


def _collect_all_points(all_data):
    """All (x,y) across prediction+future, to infer the scene's world box."""
    pts = []
    for key in ("prediction", "future"):
        d = all_data.get(key, {})
        for scene in d.values():
            for traj in scene.values():
                arr = np.asarray(traj, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                    pts.append(arr)
    return np.vstack(pts) if pts else np.zeros((0, 2), dtype=np.float32)


def residual_fields(dataset):
    """(N, Hg, Wg) per-episode residual fields (mean over horizon) for one dataset,
    plus the grid (xs, ys) for plotting."""
    path = os.path.join(PRED_DIR, f"{dataset}.pkl")
    with open(path, "rb") as f:
        all_data = pickle.load(f)

    pts = _collect_all_points(all_data)
    world_center, box, _ = _infer_world(pts, margin=2.0)
    xs, ys, Xg, Yg = build_grid(float(box), GRID, GRID)

    scene_ids = sorted(list(all_data["prediction"].keys()))
    res = build_training_residuals_from_file(
        all_data_dict=all_data, scene_ids=scene_ids,
        Xg=Xg, Yg=Yg, world_center=world_center, time_horizon=TIME_HORIZON,
    )  # (N, H, Hg, Wg)
    if res.size == 0:
        return np.zeros((0, GRID, GRID), np.float32), xs, ys
    fields = res.mean(axis=1)  # collapse horizon -> (N, Hg, Wg)
    # drop degenerate (all-zero) episodes
    keep = np.array([np.any(np.abs(f) > 1e-9) for f in fields], dtype=bool)
    return fields[keep], xs, ys


def fpca(fields):
    """Return (cumvar, phi1_grid, n) for a stack of (N, Hg, Wg) fields."""
    N = fields.shape[0]
    if N < 3:
        return None, None, N
    X = fields.reshape(N, -1).astype(np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # economy SVD: singular values^2 give component variances
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    cumvar = np.cumsum(var) / np.sum(var)
    phi1 = Vt[0].reshape(fields.shape[1], fields.shape[2])
    # sign convention: make the dominant lobe positive
    if np.abs(phi1.min()) > np.abs(phi1.max()):
        phi1 = -phi1
    return cumvar, phi1, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--phi-dataset", default="univ")
    ap.add_argument("--out", default=os.path.join(PAPER_DIR, "fcp_lowrank_fpca.png"))
    args = ap.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    fig, (ax_s, ax_p) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    summary = {}
    phi_payload = None
    for ds in args.datasets:
        try:
            fields, xs, ys = residual_fields(ds)
        except Exception as e:
            print(f"[{ds}] SKIP ({e})")
            continue
        cumvar, phi1, N = fpca(fields)
        if cumvar is None:
            print(f"[{ds}] too few episodes (N={N})")
            continue
        k_for = {pct: int(np.searchsorted(cumvar, pct) + 1) for pct in (0.8, 0.9, 0.95)}
        summary[ds] = dict(N=N, cumvar=cumvar, k=k_for)
        print(f"[{ds}] N={N:4d}  cumvar: c1={cumvar[0]:.3f} c2={cumvar[1]:.3f} "
              f"c3={cumvar[2]:.3f} c5={cumvar[4] if len(cumvar) > 4 else float('nan'):.3f} "
              f"| #comp for 80/90/95% = {k_for[0.8]}/{k_for[0.9]}/{k_for[0.95]}")
        ax_s.plot(np.arange(1, min(11, len(cumvar) + 1)), cumvar[:10],
                  marker="o", ms=3.5, lw=1.6, label=ds)
        if ds == args.phi_dataset:
            phi_payload = (phi1, xs, ys, N, cumvar)

    ax_s.axhline(0.9, ls="--", c="0.5", lw=1.0)
    ax_s.set_xlabel("# principal components")
    ax_s.set_ylabel("cumulative variance explained")
    ax_s.set_title("Residual field is low-rank", fontsize=10)
    ax_s.set_ylim(0, 1.02)
    ax_s.legend(fontsize=7, loc="lower right")

    if phi_payload is not None:
        phi1, xs, ys, N, cumvar = phi_payload
        ext = [xs[0], xs[-1], ys[0], ys[-1]]
        vmax = np.abs(phi1).max()
        im = ax_p.imshow(phi1, origin="lower", extent=ext, cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax_p, fraction=0.046, label=r"$\phi_1(x)$")
        ax_p.set_title(rf"Leading eigenfunction $\phi_1$ ({args.phi_dataset}, "
                       rf"{cumvar[0]*100:.0f}\% var)", fontsize=10)
        ax_p.set_xlabel("x (m, ego-centred)")
        ax_p.set_ylabel("y (m, ego-centred)")

    fig.tight_layout()
    os.makedirs(PAPER_DIR, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.out}")

    # machine-readable dump for the paper text
    np.savez(os.path.join(HERE, "fcp_lowrank_fpca.npz"),
             **{f"cumvar_{ds}": summary[ds]["cumvar"] for ds in summary})


if __name__ == "__main__":
    main()
