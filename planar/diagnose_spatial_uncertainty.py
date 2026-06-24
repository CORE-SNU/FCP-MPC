"""Decisive diagnostic for the spatial-uncertainty claim: does the error<->turning
correlation survive once we remove the boundary/low-sample confounds?

Controls:
  (1) FULL-length futures only (k == PRED_LEN) -> no ADE truncation at frame edges.
  (2) keep only well-sampled cells (count >= NMIN).
  (3) keep only interior cells (trim TRIM cells from the data bbox edge).
Reports corr(error, turning) and corr(error, density) raw vs. controlled, and the
corridor-vs-periphery error split. Also dumps per-cell stats + raw trajectories
(npz) for external (FPCA / signed-score) re-analysis.
"""
import os, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = ["zara1", "zara2", "univ"]
CELL = 0.8
OBS_LEN, PRED_LEN = 8, 12
NMIN = 20      # min full-future windows per cell
TRIM = 2       # interior trim (cells) from bbox edges


def total_heading_change(traj):
    d = np.diff(traj, axis=0); n = np.linalg.norm(d, axis=1); d = d[n > 1e-3]
    if len(d) < 2:
        return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.sum(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)))


def pearson(a, b, m=None):
    ok = np.isfinite(a) & np.isfinite(b)
    if m is not None:
        ok &= m
    a, b = a[ok], b[ok]
    return (float(np.corrcoef(a, b)[0, 1]) if len(a) >= 5 else float("nan")), int(len(a))


for ds in SCENES:
    r = pickle.load(open(os.path.join(HERE, f"predictions/{ds}.pkl"), "rb"))
    pred, fut = r["prediction"], r["future"]
    pos, err, turn, full = [], [], [], []
    raw_tracks = []
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
            pos.append(p[0]); err.append(float(np.mean(np.linalg.norm(p[:k] - f[:k], axis=1))))
            turn.append(total_heading_change(f[:k])); full.append(k >= PRED_LEN)
            raw_tracks.append(f[:k])
    pos = np.array(pos); err = np.array(err); turn = np.array(turn); full = np.array(full)

    x0, y0 = pos[:, 0].min(), pos[:, 1].min()
    nx = int((pos[:, 0].max() - x0) / CELL) + 1
    ny = int((pos[:, 1].max() - y0) / CELL) + 1
    ix = np.clip(((pos[:, 0] - x0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((pos[:, 1] - y0) / CELL).astype(int), 0, ny - 1)

    def cellgrid(mask):
        s = np.zeros((ny, nx)); st = np.zeros((ny, nx)); c = np.zeros((ny, nx))
        np.add.at(s, (iy[mask], ix[mask]), err[mask])
        np.add.at(st, (iy[mask], ix[mask]), turn[mask])
        np.add.at(c, (iy[mask], ix[mask]), 1)
        return s, st, c

    # RAW (all windows, like the original figure)
    s, st, c = cellgrid(np.ones(len(pos), bool))
    eg = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    tg = np.where(c > 0, st / np.maximum(c, 1), np.nan)
    r_raw, n_raw = pearson(eg.ravel(), tg.ravel())

    # CONTROLLED: full futures only, count>=NMIN, interior trim
    s, st, c = cellgrid(full)
    eg2 = np.where(c >= NMIN, s / np.maximum(c, 1), np.nan)
    tg2 = np.where(c >= NMIN, st / np.maximum(c, 1), np.nan)
    interior = np.zeros((ny, nx), bool)
    interior[TRIM:ny - TRIM, TRIM:nx - TRIM] = True
    msk = np.isfinite(eg2) & interior
    r_ctrl, n_ctrl = pearson(eg2.ravel(), tg2.ravel(), msk.ravel())
    r_dens, _ = pearson(eg2.ravel(), np.log10(np.where(c > 0, c, np.nan)).ravel(), msk.ravel())

    # corridor (top-quartile density) vs periphery error
    cc = c[msk]; ee = eg2[msk]
    hi = cc >= np.quantile(cc, 0.75)
    print(f"\n=== {ds} ===  windows={len(pos)} (full={int(full.sum())})")
    print(f"  RAW         corr(err,turn) = {r_raw:+.3f}  (cells={n_raw})")
    print(f"  CONTROLLED  corr(err,turn) = {r_ctrl:+.3f}  (cells={n_ctrl}, full-future + count>={NMIN} + interior)")
    print(f"  CONTROLLED  corr(err,dens) = {r_dens:+.3f}")
    print(f"  err: high-density cells={np.mean(ee[hi]):.3f}  low-density cells={np.mean(ee[~hi]):.3f}")

    out = os.path.join(HERE, f"spatial_cells_{ds}.npz")
    np.savez(out, cell=CELL, x0=x0, y0=y0, nx=nx, ny=ny,
             err_grid=eg2, turn_grid=tg2, count_grid=c,
             pos=pos, err=err, turn=turn, full=full)
    print(f"  [saved] per-cell + raw -> {out}")
