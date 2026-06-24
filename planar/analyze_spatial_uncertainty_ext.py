"""Artifact-robust spatial-uncertainty analysis on SDD (and rounD), to show that
prediction uncertainty is spatially structured at turning / decision regions
(roundabouts, intersections) -- the justification for a functional, space-dependent
conformal bound.

This deliberately applies the controls that ETH-UCY/zara failed, so the result is
trustworthy rather than a boundary/low-sample artifact:
  (1) FULL-length futures only (k == PRED_LEN)  -> no ADE truncation at the frame edge
  (2) per-cell count >= NMIN                     -> drop low-sample (noisy, up-biased) cells
  (3) interior trim                              -> drop boundary cells (entry/exit/occlusion)
  (4) density control                            -> report corr(error, density); show the
                                                    signal is not just visitation frequency
  (5) per-cell error variance                    -> across-window spread = envelope-width proxy
It reports RAW vs CONTROLLED corr(error, turning), saves a 3-panel diagnostic
(error / count / turning) + an overlay on the scene reference image, and dumps the
per-cell stats (npz) for deeper FPCA / signed-score re-analysis.

No learned predictor needed: a constant-velocity (CV) predictor residual is the
uncertainty proxy.

SDD (free)  : <data_dir>/annotations/<scene>/<video>/annotations.txt (+ reference.jpg)
              `deathCircle` is a roundabout; `hyang`/`gates` are intersections.
rounD (gated): <data_dir>/<rec>_tracks.csv, <rec>_recordingMeta.csv, <rec>_background.png

Usage:
  conda run -n cp python analyze_spatial_uncertainty_ext.py --dataset sdd --data-dir <SDD>
  conda run -n cp python analyze_spatial_uncertainty_ext.py --dataset sdd --data-dir <SDD> \
        --scene deathCircle --video video0
"""
import os, glob, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

OBS_LEN, PRED_LEN = 8, 12


# ------------------------------------------------------------------ core analysis
def total_heading_change(traj):
    d = np.diff(traj, axis=0); n = np.linalg.norm(d, axis=1); d = d[n > 1e-6]
    if len(d) < 2:
        return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.sum(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)))


def cv_windows(traj):
    """Yield (pos, ade, turn, full) for each 8/12 sliding window; CV from obs velocity."""
    T = len(traj)
    for t in range(OBS_LEN - 1, T - 2):
        obs = traj[t - OBS_LEN + 1:t + 1]; fut = traj[t + 1:t + 1 + PRED_LEN]
        k = len(fut)
        if k < 2:
            continue
        vel = (obs[-1] - obs[0]) / (OBS_LEN - 1)
        pred = obs[-1] + vel * np.arange(1, k + 1)[:, None]
        ade = float(np.mean(np.linalg.norm(pred - fut[:k], axis=1)))
        yield obs[-1], ade, total_heading_change(fut[:k]), (k >= PRED_LEN)


def pearson(a, b, m=None):
    ok = np.isfinite(a) & np.isfinite(b)
    if m is not None:
        ok = ok & m
    a, b = a[ok], b[ok]
    return (float(np.corrcoef(a, b)[0, 1]) if len(a) >= 5 else float("nan")), int(len(a))


def analyze(trajs, cell, nmin, trim, tag, bg, to_px, out_prefix):
    pos, err, turn, full = [], [], [], []
    for tr in trajs:
        for p, e, t, fl in cv_windows(tr):
            pos.append(p); err.append(e); turn.append(t); full.append(fl)
    pos = np.array(pos); err = np.array(err); turn = np.array(turn); full = np.array(full, bool)
    if len(pos) < 50:
        raise RuntimeError(f"too few windows ({len(pos)}) -- check data/downsampling")

    x0, y0 = pos[:, 0].min(), pos[:, 1].min()
    nx = int((pos[:, 0].max() - x0) / cell) + 1
    ny = int((pos[:, 1].max() - y0) / cell) + 1
    ix = np.clip(((pos[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((pos[:, 1] - y0) / cell).astype(int), 0, ny - 1)

    def cells(mask, vals):
        s = np.zeros((ny, nx)); ss = np.zeros((ny, nx)); c = np.zeros((ny, nx))
        np.add.at(s, (iy[mask], ix[mask]), vals[mask])
        np.add.at(ss, (iy[mask], ix[mask]), vals[mask] ** 2)
        np.add.at(c, (iy[mask], ix[mask]), 1)
        return s, ss, c

    allm = np.ones(len(pos), bool)
    se, _, ce = cells(allm, err); st, _, _ = cells(allm, turn)
    err_raw = np.where(ce > 0, se / np.maximum(ce, 1), np.nan)
    turn_raw = np.where(ce > 0, st / np.maximum(ce, 1), np.nan)
    r_raw, n_raw = pearson(err_raw.ravel(), turn_raw.ravel())

    # controlled: full futures only, count>=nmin, interior trim
    se, sse, c = cells(full, err); st2, _, _ = cells(full, turn)
    mean = np.where(c >= nmin, se / np.maximum(c, 1), np.nan)
    var = np.where(c >= nmin, sse / np.maximum(c, 1) - (se / np.maximum(c, 1)) ** 2, np.nan)
    turn_c = np.where(c >= nmin, st2 / np.maximum(c, 1), np.nan)
    interior = np.zeros((ny, nx), bool); interior[trim:ny - trim, trim:nx - trim] = True
    msk = np.isfinite(mean) & interior
    r_ctrl, n_ctrl = pearson(mean.ravel(), turn_c.ravel(), msk.ravel())
    r_dens, _ = pearson(mean.ravel(), np.log10(np.where(c > 0, c, np.nan)).ravel(), msk.ravel())
    cc, ee = c[msk], mean[msk]
    hi = cc >= np.quantile(cc, 0.75) if cc.size else np.array([], bool)

    print(f"\n=== {tag} ===  windows={len(pos)} (full={int(full.sum())})")
    print(f"  RAW         corr(err,turn) = {r_raw:+.3f}  (cells={n_raw})")
    print(f"  CONTROLLED  corr(err,turn) = {r_ctrl:+.3f}  (cells={n_ctrl}; full + count>={nmin} + interior)")
    print(f"  CONTROLLED  corr(err,dens) = {r_dens:+.3f}")
    if cc.size:
        print(f"  err hi-density={np.mean(ee[hi]):.3f}  lo-density={np.mean(ee[~hi]):.3f}")
    verdict = ("SUPPORTED" if (np.isfinite(r_ctrl) and r_ctrl >= 0.35 and n_ctrl >= 30)
               else "WEAK/UNSUPPORTED")
    print(f"  >>> controlled spatial-turning structure: {verdict}")

    # diagnostic panels
    ext = [x0, x0 + nx * cell, y0, y0 + ny * cell]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, g, ttl, cmp in [(ax[0], np.where(msk, mean, np.nan), "controlled error (CV ADE)", "turbo"),
                           (ax[1], np.log10(np.where(c > 0, c, np.nan)), "log10 count", "cividis"),
                           (ax[2], np.where(msk, turn_c, np.nan), "turning (heading change)", "viridis")]:
        im = a.imshow(g, origin="lower", extent=ext, cmap=cmp, aspect="equal")
        a.set_title(f"{tag}: {ttl}", fontsize=9); plt.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_prefix + "_diag.png", dpi=130, bbox_inches="tight")

    # overlay on reference image (controlled cells only)
    if bg is not None:
        h, w = bg.shape[:2]
        fig2, a2 = plt.subplots(figsize=(8, 7)); a2.imshow(bg)
        ys, xs = np.mgrid[0:ny, 0:nx]
        P = np.stack([(x0 + (xs + .5) * cell).ravel(), (y0 + (ys + .5) * cell).ravel()], 1)
        Ppx = to_px(P); ev = np.where(msk, mean, np.nan).ravel(); good = np.isfinite(ev)
        vmin, vmax = np.nanpercentile(ee, 10) if ee.size else 0, np.nanpercentile(ee, 95) if ee.size else 1
        norm = plt.Normalize(vmin, vmax)
        a2.scatter(Ppx[good, 0], Ppx[good, 1], c=ev[good], cmap="turbo", norm=norm,
                   s=130, marker="s", alpha=0.55, edgecolors="none")
        a2.set_xlim(0, w); a2.set_ylim(h, 0); a2.axis("off")
        a2.set_title(f"{tag}: controlled uncertainty (corr w/ turning={r_ctrl:+.2f}, {verdict})", fontsize=10)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap="turbo"), ax=a2, fraction=0.046, label="CV ADE")
        fig2.tight_layout(); fig2.savefig(out_prefix + "_overlay.png", dpi=140, bbox_inches="tight")

    np.savez(out_prefix + "_cells.npz", cell=cell, x0=x0, y0=y0, nx=nx, ny=ny,
             err_mean=mean, err_var=var, turn=turn_c, count=c, mask=msk,
             pos=pos, err=err, turn_w=turn, full=full)
    print(f"  [saved] {out_prefix}_diag.png, _overlay.png, _cells.npz")
    return dict(tag=tag, r_raw=r_raw, r_ctrl=r_ctrl, r_dens=r_dens, n_ctrl=n_ctrl, verdict=verdict)


# ------------------------------------------------------------------ loaders
def load_sdd(ann_path):
    """Load one SDD annotation file (pixel bbox -> centre tracks). Robust to the
    standard layout (annotations/<scene>/<video>/annotations.txt) and to mirrors like
    flclain (<scene>/<video>/annotation.txt). Finds a nearby reference image."""
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"SDD annotation file not found: {ann_path}")
    rows = {}
    for r in np.genfromtxt(ann_path, dtype=None, encoding="utf-8"):
        tid, xmin, ymin, xmax, ymax, frame, lost = (int(r[0]), float(r[1]), float(r[2]),
                                                    float(r[3]), float(r[4]), int(r[5]), int(r[6]))
        if lost:
            continue
        rows.setdefault(tid, []).append((frame, 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)))
    trajs = []
    for pts in rows.values():
        pts.sort()
        a = np.array([[p[1], p[2]] for p in pts], np.float32)[::12]  # ~30fps -> ~2.5fps
        if len(a) >= OBS_LEN + 2:
            trajs.append(a)
    # reference image: try the annotation dir and its parent (scene) dir
    d = os.path.dirname(ann_path)
    bg = None
    for cand in (os.path.join(d, "reference.jpg"), os.path.join(d, "reference.png"),
                 os.path.join(os.path.dirname(d), "reference.jpg"),
                 os.path.join(os.path.dirname(d), "reference.png")):
        if os.path.isfile(cand):
            bg = cv2.cvtColor(cv2.imread(cand), cv2.COLOR_BGR2RGB); break
    return trajs, bg, (lambda P: P)


def sdd_autofind(data_dir):
    """Return annotation-file paths discovered anywhere under data_dir (handles both
    `annotations.txt` and `annotation.txt`, any nesting)."""
    found = sorted(set(glob.glob(os.path.join(data_dir, "**", "annotation*.txt"),
                                 recursive=True)))
    return found


def sdd_tag(ann_path, data_dir):
    rel = os.path.relpath(os.path.dirname(ann_path), data_dir)
    return "SDD/" + rel.replace(os.sep, "/")


def load_round(data_dir, rec):
    import csv
    tracks = os.path.join(data_dir, f"{rec}_tracks.csv")
    if not os.path.isfile(tracks):
        raise FileNotFoundError(f"rounD tracks not found: {tracks}")
    rows = {}
    with open(tracks) as f:
        for r in csv.DictReader(f):
            rows.setdefault(int(r["trackId"]), []).append((int(r["frame"]), float(r["xCenter"]), float(r["yCenter"])))
    trajs = []
    for pts in rows.values():
        pts.sort()
        a = np.array([[p[1], p[2]] for p in pts], np.float32)[::25]
        if len(a) >= OBS_LEN + 2:
            trajs.append(a)
    px2m = 1.0
    meta = os.path.join(data_dir, f"{rec}_recordingMeta.csv")
    if os.path.isfile(meta):
        with open(meta) as f:
            px2m = float(next(csv.DictReader(f)).get("orthoPxToMeter", "1") or 1.0)
    bgp = os.path.join(data_dir, f"{rec}_background.png")
    bg = cv2.cvtColor(cv2.imread(bgp), cv2.COLOR_BGR2RGB) if os.path.isfile(bgp) else None
    return trajs, bg, (lambda P: np.stack([P[:, 0] / px2m, -P[:, 1] / px2m], 1))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sdd", "round"], default="sdd")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scene", default=None, help="SDD scene; omit to auto-find all")
    ap.add_argument("--video", default=None)
    ap.add_argument("--recording", default="00")
    ap.add_argument("--cell", type=float, default=None)
    ap.add_argument("--nmin", type=int, default=20)
    ap.add_argument("--trim", type=int, default=2)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    if args.dataset == "sdd":
        ann_paths = sdd_autofind(args.data_dir)
        if args.scene:                       # optional substring filter (e.g. deathCircle)
            ann_paths = [a for a in ann_paths if args.scene in a]
        if not ann_paths:
            raise FileNotFoundError(
                f"no SDD annotation*.txt found under {args.data_dir} "
                f"(expected e.g. <dir>/<scene>/<video>/annotation.txt)")
        print(f"[auto-find] {len(ann_paths)} SDD annotation file(s)")
        summary = []
        for ann in ann_paths:
            trajs, bg, to_px = load_sdd(ann)
            tag = sdd_tag(ann, args.data_dir)
            pref = os.path.join(here, "sdd_" + tag[len("SDD/"):].replace("/", "_"))
            try:
                summary.append(analyze(trajs, args.cell or 40.0, args.nmin, args.trim,
                                       tag, bg, to_px, pref))
            except RuntimeError as e:
                print(f"  [skip] {tag}: {e}")
        print("\n==== SUMMARY (controlled corr error<->turning) ====")
        for s in summary:
            print(f"  {s['tag']:30s} raw={s['r_raw']:+.2f} ctrl={s['r_ctrl']:+.2f} "
                  f"dens={s['r_dens']:+.2f} cells={s['n_ctrl']} -> {s['verdict']}")
    else:
        trajs, bg, to_px = load_round(args.data_dir, args.recording)
        cell = args.cell or 2.0
        analyze(trajs, cell, args.nmin, args.trim, f"rounD/{args.recording}",
                bg, to_px, os.path.join(here, f"round_{args.recording}"))


if __name__ == "__main__":
    main()
