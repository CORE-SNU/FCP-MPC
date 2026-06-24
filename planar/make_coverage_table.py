#!/usr/bin/env python3
"""Calibration-validity experiment for the FCP envelope (reuses the early
validation harness in multi_pedestrians_cp.py).

For each scene we (i) fit the FPCA+GMM functional envelope offline on the train
split at level alpha, and (ii) measure, on the held-out test split, the empirical
*field* coverage --- the fraction of timesteps (and episodes) on which the
conformal lower-bound unsafe region contains the true unsafe region. If the
construction is calibrated, timestep coverage should sit near 1-alpha.

We report:
  * in-distribution coverage for all five ETH-UCY scenes at alpha=0.10  (table)
  * a reliability sweep over alpha in {0.05,0.1,0.2,0.3}                 (figure)
  * a cross-scene block (train zara1 -> test zara2 and vice versa) that
    demonstrates the distribution shift across ETH-UCY scenes.

Outputs:
  coverage_2d.csv
  T_RO2026/table_coverage.tex
  T_RO2026/reliability_2d.png
"""
from __future__ import annotations
import os
import numpy as np

import multi_pedestrians_cp as M
from multi_pedestrians_cp import (
    build_grid, load_eth_pickle_preprocessed,
    build_training_residuals_valid_only, compute_g_upper_unified,
    evaluate_coverage_model_multi, align_mask_dims,
)

BOX, H, W = M.BOX, M.H, M.W
TSTEPS = M.TSTEPS
# Validate the coverage the theorem actually certifies: the i=1 (applied / 1-step-ahead)
# envelope. The closed-loop guarantee invokes only the applied step, so report 1-step coverage.
TIME_HORIZON = 1
SAFE_THRESHOLD = M.SAFE_THRESHOLD
P_BASE, K_MIX, TEST_SIZE = M.P_BASE, M.K_MIX, M.TEST_SIZE
RANDOM_STATE, N_JOBS, BACKEND = M.RANDOM_STATE, M.N_JOBS, M.BACKEND
N_TRAIN, NUM_PEDS, SEED = M.N_TRAIN, M.NUM_PEDS, M.SEED
BASE_DIR = M.BASE_DIR

DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]


def _load(ds, seed=SEED):
    """Return train/test (true,pred,mask) in (N,T,M,2)/(N,T,M) layout, using the
    dataset's own (Trajectron++) predictions from the pkl."""
    tr_t, tr_p, tr_m, te_t, te_p, te_m = load_eth_pickle_preprocessed(
        dataset=ds, T=TSTEPS, split_ratio=0.8, seed=seed,
        base_dir=BASE_DIR, num_peds=NUM_PEDS,
    )
    tr_t = np.transpose(tr_t, (0, 2, 1, 3)); tr_p = np.transpose(tr_p, (0, 2, 1, 3))
    te_t = np.transpose(te_t, (0, 2, 1, 3)); te_p = np.transpose(te_p, (0, 2, 1, 3))
    _, T_dim, M_dim, _ = tr_t.shape
    tr_m = align_mask_dims(tr_m, T_dim, M_dim)
    _, T_dim2, M_dim2, _ = te_t.shape
    te_m = align_mask_dims(te_m, T_dim2, M_dim2)
    return (tr_t, tr_p, tr_m), (te_t, te_p, te_m)


def fit_envelope(train, alpha, rng):
    tr_t, tr_p, tr_m = train
    _, _, Xg, Yg = build_grid(BOX, H, W)
    n = tr_t.shape[0]
    sel = rng.choice(n, size=min(N_TRAIN, n), replace=False)
    res, t_off = build_training_residuals_valid_only(
        obst_true=tr_t[sel], obst_pred=tr_p[sel], masks=tr_m[sel],
        Xg=Xg, Yg=Yg, horizon=TIME_HORIZON,
    )
    g_upper = compute_g_upper_unified(
        residuals=res, p_base=P_BASE, K_mix=K_MIX, alpha=alpha,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, n_jobs=N_JOBS, backend=BACKEND,
    )
    return g_upper, t_off, Xg, Yg


def eval_cov(test, g_upper, t_off, Xg, Yg):
    te_t, te_p, te_m = test
    tt = te_t[:, t_off:, :, :]; tp = te_p[:, t_off:, :, :]; tm = te_m[:, t_off:, :]
    T_eff = min(tt.shape[1], g_upper.shape[0])
    return evaluate_coverage_model_multi(
        test_true=tt[:, :T_eff], test_pred=tp[:, :T_eff], test_mask=tm[:, :T_eff],
        Xg=Xg, Yg=Yg, g_upper=g_upper[:T_eff], safe_threshold=SAFE_THRESHOLD,
    )


def main():
    rng = np.random.default_rng(SEED)
    rows = []          # (label, alpha, ep_mean, ep_std, ts_mean, ts_std)
    cache = {}         # ds -> (train, test)

    def data(ds):
        if ds not in cache:
            cache[ds] = _load(ds)
        return cache[ds]

    # ---- in-distribution, all scenes, sweep alpha ----
    for ds in DATASETS:
        train, test = data(ds)
        for alpha in (0.05, 0.10, 0.20, 0.30):
            g, t_off, Xg, Yg = fit_envelope(train, alpha, np.random.default_rng(SEED))
            ep_m, ep_s, ts_m, ts_s = eval_cov(test, g, t_off, Xg, Yg)
            rows.append((ds, alpha, ep_m, ep_s, ts_m, ts_s))
            print(f"[in-dist] {ds:6} a={alpha:.2f}  ep={ep_m*100:5.1f}+-{ep_s*100:4.1f}  ts={ts_m*100:5.1f}+-{ts_s*100:4.1f}", flush=True)

    # ---- cross-scene (distribution shift) at alpha=0.10 ----
    cross = []
    for tr_ds, te_ds in [("zara1", "zara1"), ("zara1", "zara2"),
                         ("zara2", "zara2"), ("zara2", "zara1")]:
        train, _ = data(tr_ds)
        _, test = data(te_ds)
        g, t_off, Xg, Yg = fit_envelope(train, 0.10, np.random.default_rng(SEED))
        ep_m, ep_s, ts_m, ts_s = eval_cov(test, g, t_off, Xg, Yg)
        cross.append((tr_ds, te_ds, ep_m, ep_s, ts_m, ts_s))
        print(f"[cross]  {tr_ds}->{te_ds}  ep={ep_m*100:5.1f}+-{ep_s*100:4.1f}  ts={ts_m*100:5.1f}+-{ts_s*100:4.1f}", flush=True)

    np.savez("coverage_2d.npz",
             rows=np.array(rows, dtype=object), cross=np.array(cross, dtype=object))
    _write_csv(rows, cross)
    _write_tex(rows, cross)
    _write_reliability_table(rows)   # reliability is reported as a TABLE, not a figure


def _write_csv(rows, cross):
    import csv
    with open("coverage_2d.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "train", "test", "alpha", "ep_mean", "ep_std", "ts_mean", "ts_std"])
        for ds, a, em, es, tm, ts in rows:
            w.writerow(["in-dist", ds, ds, a, em, es, tm, ts])
        for tr, te, em, es, tm, ts in cross:
            w.writerow(["cross", tr, te, 0.10, em, es, tm, ts])
    print("[saved] coverage_2d.csv")


def _write_tex(rows, cross):
    # in-distribution table at alpha=0.10
    a0 = 0.10
    lines = [r"\begin{tabular}{lcc}", r"\hline",
             r"Scene & Episode coverage (\%) & Timestep coverage (\%) \\", r"\hline"]
    for ds in DATASETS:
        r = [x for x in rows if x[0] == ds and abs(x[1] - a0) < 1e-9][0]
        lines.append(rf"\texttt{{{ds}}} & ${r[2]*100:.1f}\pm{r[3]*100:.1f}$ & ${r[4]*100:.1f}\pm{r[5]*100:.1f}$ \\")
    lines += [r"\hline", r"\end{tabular}"]
    with open(os.path.join("T_RO2026", "table_coverage.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[saved] T_RO2026/table_coverage.tex")

    # cross-scene table
    cl = [r"\begin{tabular}{llcc}", r"\hline",
          r"Calib.\ & Test & Episode coverage (\%) & Timestep coverage (\%) \\", r"\hline"]
    for tr, te, em, es, tm, ts in cross:
        tag = "" if tr == te else r"\,$^{\dagger}$"
        cl.append(rf"\texttt{{{tr}}} & \texttt{{{te}}}{tag} & ${em*100:.1f}\pm{es*100:.1f}$ & ${tm*100:.1f}\pm{ts*100:.1f}$ \\")
    cl += [r"\hline", r"\end{tabular}"]
    with open(os.path.join("T_RO2026", "table_coverage_cross.tex"), "w") as f:
        f.write("\n".join(cl) + "\n")
    print("[saved] T_RO2026/table_coverage_cross.tex")


def _write_reliability_table(rows):
    """Reliability as a LaTeX table: target coverage 1-alpha (columns) vs empirical
    timestep coverage per scene (rows)."""
    alphas = [0.05, 0.10, 0.20, 0.30]
    head = " & ".join([rf"$1-\alpha={1-a:.2f}$" for a in alphas])
    lines = [r"\begin{tabular}{l" + "c" * len(alphas) + "}", r"\hline",
             rf"Scene & {head} \\", r"\hline"]
    for ds in DATASETS:
        cells = []
        for a in alphas:
            r = [x for x in rows if x[0] == ds and abs(x[1] - a) < 1e-9]
            cells.append(f"{r[0][4]*100:.1f}" if r else "--")
        lines.append(rf"\texttt{{{ds}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    with open(os.path.join("T_RO2026", "table_reliability.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[saved] T_RO2026/table_reliability.tex")


def _write_reliability(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix"})
    alphas = [0.05, 0.10, 0.20, 0.30]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1.0, label="ideal")
    for ds in DATASETS:
        xs = [1 - a for a in alphas]
        ys = []
        for a in alphas:
            r = [x for x in rows if x[0] == ds and abs(x[1] - a) < 1e-9][0]
            ys.append(r[4])  # timestep coverage mean
        ax.plot(xs, ys, marker="o", ms=4, lw=1.6, label=ds)
    ax.set_xlabel(r"target coverage $1-\alpha$")
    ax.set_ylabel("empirical timestep coverage")
    ax.set_xlim(0.6, 1.0); ax.set_ylim(0.6, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join("T_RO2026", "reliability_2d.png"), dpi=200, bbox_inches="tight")
    print("[saved] T_RO2026/reliability_2d.png")


if __name__ == "__main__":
    main()
