"""Exploratory: is ETH-UCY prediction uncertainty spatially structured, and is it
driven by geometry (turning) or by social density?

Per scene, bin pedestrian-frames into a spatial grid (world coords, metres) and
compute per cell: mean prediction error (FDE/ADE of pred vs GT future), mean total
heading change over the future (a turning/geometry proxy), and pedestrian count
(density). Then correlate cell-level error against turning and against density.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = ["zara1", "zara2", "univ"]   # scenes with usable (>=2-step) GT futures
CELL = 0.8                            # grid cell size (m)
MIN_COUNT = 5                         # min ped-frames per cell for stable stats
TABLE_TEX = os.path.join(HERE, "T_RO2026", "table_spatial_uncertainty.tex")  # paper \input dir


def total_heading_change(traj):
    d = np.diff(traj, axis=0)
    n = np.linalg.norm(d, axis=1)
    keep = n > 1e-3
    d, n = d[keep], n[keep]
    if len(d) < 2:
        return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    dang = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    return float(np.sum(dang))


def collect(ds):
    r = pickle.load(open(os.path.join(HERE, f"predictions/{ds}.pkl"), "rb"))
    pred, fut = r["prediction"], r["future"]
    pos, err, turn = [], [], []
    for fr in pred:
        if fr not in fut:
            continue
        for pid, p in pred[fr].items():
            if pid not in fut[fr]:
                continue
            p = np.asarray(p, np.float32); f = np.asarray(fut[fr][pid], np.float32)
            k = min(len(p), len(f))
            if k < 2:
                continue
            pos.append(p[0])
            err.append(float(np.mean(np.linalg.norm(p[:k] - f[:k], axis=1))))  # ADE
            turn.append(total_heading_change(f[:k]))
    return np.array(pos), np.array(err), np.array(turn)


def grid_stats(pos, val, x0, y0, nx, ny):
    ix = np.clip(((pos[:, 0] - x0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((pos[:, 1] - y0) / CELL).astype(int), 0, ny - 1)
    s = np.zeros((ny, nx)); c = np.zeros((ny, nx))
    for a, b, v in zip(iy, ix, val):
        s[a, b] += v; c[a, b] += 1
    mean = np.full((ny, nx), np.nan)
    m = c >= MIN_COUNT
    mean[m] = s[m] / c[m]
    return mean, c


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 5:
        return float("nan"), 0
    return float(np.corrcoef(a, b)[0, 1]), len(a)


fig, axes = plt.subplots(len(SCENES), 3, figsize=(13, 4.2 * len(SCENES)))
if len(SCENES) == 1:
    axes = axes[None, :]
table_rows = []  # (scene, r_turn, r_dens, n_cells, n_windows)

for row, ds in enumerate(SCENES):
    pos, err, turn = collect(ds)
    x0, y0 = pos[:, 0].min(), pos[:, 1].min()
    nx = int((pos[:, 0].max() - x0) / CELL) + 1
    ny = int((pos[:, 1].max() - y0) / CELL) + 1
    err_g, cnt = grid_stats(pos, err, x0, y0, nx, ny)
    turn_g, _ = grid_stats(pos, turn, x0, y0, nx, ny)
    dens_g = np.where(cnt > 0, cnt, np.nan)

    ext = [x0, x0 + nx * CELL, y0, y0 + ny * CELL]
    for ax, g, title, cm in [
        (axes[row, 0], err_g, f"{ds}: mean prediction error (ADE, m)", "magma"),
        (axes[row, 1], turn_g, f"{ds}: mean total heading change (rad)", "viridis"),
        (axes[row, 2], np.log10(dens_g), f"{ds}: log10 density (count)", "cividis"),
    ]:
        im = ax.imshow(g, origin="lower", extent=ext, aspect="equal", cmap=cm)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # correlations over cells
    ef = err_g.ravel(); tf = turn_g.ravel(); df = np.log10(dens_g).ravel()
    r_turn, n1 = pearson(ef, tf)
    r_dens, n2 = pearson(ef, df)
    # top error cells
    flat = err_g.copy()
    order = np.argsort(np.where(np.isfinite(flat), flat, -1).ravel())[::-1][:5]
    tops = []
    for idx in order:
        a, b = divmod(idx, nx)
        if np.isfinite(err_g[a, b]):
            tops.append((round(x0 + (b + .5) * CELL, 1), round(y0 + (a + .5) * CELL, 1),
                         round(float(err_g[a, b]), 2)))
    print(f"\n=== {ds} ===  cells>=MIN_COUNT={int(np.sum(cnt>=MIN_COUNT))}  ped-frames={len(err)}")
    print(f"  corr(error, turning)  = {r_turn:+.3f}   (n_cells={n1})")
    print(f"  corr(error, density)  = {r_dens:+.3f}   (n_cells={n2})")
    print(f"  top-5 error cells (x,y,ADE): {tops}")
    table_rows.append((ds, r_turn, r_dens, n1, len(err)))

fig.suptitle("ETH-UCY: spatial structure of prediction uncertainty", y=1.0)
fig.tight_layout()
out = os.path.join(HERE, "spatial_uncertainty_preview.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\n[saved] {out}")

# ---- LaTeX table: spatial correlation of prediction error ----
os.makedirs(os.path.dirname(TABLE_TEX), exist_ok=True)
lines = [
    r"\begin{tabular}{lccc}",
    r"\hline",
    r"Scene & Corr(error, turning) $\uparrow$ & Corr(error, density) & \# cells \\",
    r"\hline",
]
for ds, rt, rd, nc, _ in table_rows:
    lines.append(f"{ds} & \\textbf{{{rt:+.2f}}} & {rd:+.2f} & {nc} \\\\")
lines += [r"\hline", r"\end{tabular}"]
with open(TABLE_TEX, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"[saved] {TABLE_TEX}")
