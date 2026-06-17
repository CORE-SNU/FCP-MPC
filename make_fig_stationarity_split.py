#!/usr/bin/env python3
"""L2 evidence — the residual (prediction-error) field is TIME-INVARIANT / stationary.

If the per-cell error field built from the EARLY half of an episode stream agrees with
the field built from the LATE half, the field is a property of the (fixed) scene rather
than of the moment -- which is what licenses calibrating it ONCE, offline (L4), and is
the linchpin the paper's "S admits low-dimensional structure" argument needs.

Method:
  - scene ids in the ETH-UCY prediction files are frame timestamps, so sorting them and
    cutting in two gives a genuine EARLY-vs-LATE temporal split (disjoint, no leakage).
  - on the SAME grid the controller calibrates on, build the per-episode residual fields
    for each half (`build_training_residuals_from_file`), average over episodes+horizon
    to a per-cell mean field, and compare the two halves:
      * Pearson r over cells (high r => stationary),
      * scatter of early vs late cell means,
      * the two heatmaps side by side.

Outputs:
  T_RO2026/fcp_stationarity_split.png   (per dataset; --datasets to choose)
  prints the split-halves Pearson r per dataset.
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
from make_fig_fpca_lowrank import _collect_all_points, PRED_DIR, PAPER_DIR, TIME_HORIZON, GRID


def half_field(all_data, scene_ids, Xg, Yg, world_center):
    res = build_training_residuals_from_file(
        all_data_dict=all_data, scene_ids=scene_ids,
        Xg=Xg, Yg=Yg, world_center=world_center, time_horizon=TIME_HORIZON,
    )  # (N, H, Hg, Wg)
    if res.size == 0:
        return None
    return res.mean(axis=(0, 1))  # mean over episodes + horizon -> (Hg, Wg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["univ", "zara1", "zara2", "eth", "hotel"])
    ap.add_argument("--phi-dataset", default="univ")
    ap.add_argument("--out", default=os.path.join(PAPER_DIR, "fcp_stationarity_split.png"))
    args = ap.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})

    results = {}
    panels = {}
    for ds in args.datasets:
        try:
            with open(os.path.join(PRED_DIR, f"{ds}.pkl"), "rb") as f:
                all_data = pickle.load(f)
        except Exception as e:
            print(f"[{ds}] SKIP ({e})")
            continue
        pts = _collect_all_points(all_data)
        world_center, box, _ = _infer_world(pts, margin=2.0)
        xs, ys, Xg, Yg = build_grid(float(box), GRID, GRID)

        sids = sorted(list(all_data["prediction"].keys()))
        mid = len(sids) // 2
        early, late = sids[:mid], sids[mid:]
        fe = half_field(all_data, early, Xg, Yg, world_center)
        fl = half_field(all_data, late, Xg, Yg, world_center)
        if fe is None or fl is None:
            print(f"[{ds}] no valid residuals in a half")
            continue

        # compare only cells where BOTH halves saw structure
        m = (np.abs(fe) > 1e-9) & (np.abs(fl) > 1e-9)
        a, b = fe[m].ravel(), fl[m].ravel()
        r = float(np.corrcoef(a, b)[0, 1]) if a.size >= 5 else float("nan")
        results[ds] = dict(r=r, n_cells=int(m.sum()))
        print(f"[{ds}] split-halves corr(early,late) = {r:+.3f} over {int(m.sum())} cells "
              f"(early={len(early)} scenes, late={len(late)})")
        if ds == args.phi_dataset:
            panels[ds] = (fe, fl, a, b, r, xs, ys)

    if args.phi_dataset in panels:
        fe, fl, a, b, r, xs, ys = panels[args.phi_dataset]
        ext = [xs[0], xs[-1], ys[0], ys[-1]]
        vmax = max(np.abs(fe).max(), np.abs(fl).max())
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
        for k, (fld, tag) in enumerate([(fe, "early half"), (fl, "late half")]):
            im = ax[k].imshow(fld, origin="lower", extent=ext, cmap="viridis",
                              vmin=-vmax, vmax=vmax, aspect="auto")
            ax[k].set_title(f"{args.phi_dataset}: {tag}", fontsize=10)
            ax[k].set_xlabel("x (m)"); ax[k].set_ylabel("y (m)")
            fig.colorbar(im, ax=ax[k], fraction=0.046)
        ax[2].scatter(a, b, s=5, alpha=0.4, color="#1f77b4")
        lim = [min(a.min(), b.min()), max(a.max(), b.max())]
        ax[2].plot(lim, lim, "--", c="0.5", lw=1)
        ax[2].set_title(rf"cell means agree ($r={r:+.2f}$)", fontsize=10)
        ax[2].set_xlabel("early-half cell mean"); ax[2].set_ylabel("late-half cell mean")
        fig.tight_layout()
        os.makedirs(PAPER_DIR, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
