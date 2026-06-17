"""Overlay the spatial prediction-uncertainty field on the real ETH-UCY scene frame.

For each scene: bin pedestrian-frames into a world-coord grid, compute mean
prediction error (ADE) per cell, then back-project onto the video frame via the
scene homography (image px -> world: world = H @ [u, v, 1]). High-error cells are
the turning / decision regions (entrances, bends, crossings).
"""
import os, pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = ["zara1", "zara2"]
CELL = 0.8
MIN_COUNT = 5
ALPHA = 0.55


def total_heading_change(traj):
    d = np.diff(traj, axis=0); n = np.linalg.norm(d, axis=1)
    d = d[n > 1e-3]
    if len(d) < 2:
        return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.sum(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)))


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
            err.append(float(np.mean(np.linalg.norm(p[:k] - f[:k], axis=1))))
            turn.append(total_heading_change(f[:k]))
    return np.array(pos), np.array(err), np.array(turn)


def grid_mean(pos, val, x0, y0, nx, ny):
    ix = np.clip(((pos[:, 0] - x0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((pos[:, 1] - y0) / CELL).astype(int), 0, ny - 1)
    s = np.zeros((ny, nx)); c = np.zeros((ny, nx))
    np.add.at(s, (iy, ix), val); np.add.at(c, (iy, ix), 1)
    g = np.full((ny, nx), np.nan); m = c >= MIN_COUNT; g[m] = s[m] / c[m]
    return g, c


fig, axes = plt.subplots(1, len(SCENES), figsize=(7.0 * len(SCENES), 5.2))
if len(SCENES) == 1:
    axes = [axes]

for ax, ds in zip(axes, SCENES):
    pos, err, turn = collect(ds)
    x0, y0 = pos[:, 0].min(), pos[:, 1].min()
    nx = int((pos[:, 0].max() - x0) / CELL) + 1
    ny = int((pos[:, 1].max() - y0) / CELL) + 1
    err_g, cnt = grid_mean(pos, err, x0, y0, nx, ny)
    turn_g, _ = grid_mean(pos, turn, x0, y0, nx, ny)

    cap = cv2.VideoCapture(os.path.join(HERE, f"assets/videos/{ds}.avi"))
    ok, frame = cap.read(); cap.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    H = np.loadtxt(os.path.join(HERE, f"assets/homographies/{ds}.txt"))

    # back-project every pixel to world, sample the error grid
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones_like(uu)
    P = np.stack([uu, vv, ones], -1).reshape(-1, 3).T          # (3, N)
    Wd = H @ P; Wd /= Wd[2]
    wx, wy = Wd[0], Wd[1]
    gx = ((wx - x0) / CELL).astype(int); gy = ((wy - y0) / CELL).astype(int)
    inb = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    vals = np.full(gx.shape, np.nan)
    vals[inb] = err_g[gy[inb], gx[inb]]
    vals = vals.reshape(h, w)

    vmin, vmax = np.nanpercentile(err, 10), np.nanpercentile(err, 95)
    norm = plt.Normalize(vmin, vmax)
    rgba = cm.turbo(norm(vals))
    rgba[..., 3] = np.where(np.isfinite(vals), ALPHA, 0.0)

    ax.imshow(frame)
    ax.imshow((rgba * 255).astype(np.uint8))
    ax.set_title(f"{ds}: prediction uncertainty (ADE, m) on scene", fontsize=11)
    ax.axis("off")

    # annotate the top-3 highest-uncertainty cells (project world->pixel via inv(H))
    Hinv = np.linalg.inv(H)
    order = np.argsort(np.where(np.isfinite(err_g), err_g, -1).ravel())[::-1][:3]
    for idx in order:
        a, b = divmod(idx, nx)
        wxc, wyc = x0 + (b + .5) * CELL, y0 + (a + .5) * CELL
        pc = Hinv @ np.array([wxc, wyc, 1.0]); pc /= pc[2]
        if 0 <= pc[0] < w and 0 <= pc[1] < h:
            ax.plot(pc[0], pc[1], "o", mfc="none", mec="white", mew=2.0, ms=16)
            ax.text(pc[0] + 8, pc[1], f"{err_g[a, b]:.2f}", color="white",
                    fontsize=9, va="center")

    sm = cm.ScalarMappable(norm=norm, cmap="turbo")
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label="ADE (m)")

fig.tight_layout()
# Exploratory preview only (gitignored). The ETH-UCY overlay is NOT a paper figure:
# under count-mask + full-future + interior controls the turning signal halves and the
# hot cells are boundary/low-sample artifacts (see diagnose_spatial_uncertainty.py).
# The spatial-structure claim is to be shown on rounD / SDD-deathCircle instead.
out = os.path.join(HERE, "spatial_uncertainty_overlay.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"[saved] {out}")
