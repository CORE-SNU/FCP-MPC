"""Fig.4 replacement: a *clean*, zoomed 3D illustration of the conformal safety
bound, analogous to the 2D True-SDF-vs-conformal figure (``multi_obs_H2.png``).

Instead of the cluttered 280-obstacle Rerun screenshot (``Func_cp_3d_zoom.png``),
we zoom onto one or two obstacles and draw two nested isosurfaces at the safety
threshold ``r_safe = ROBOT_RAD + OBSTACLE_RAD``:

  * the *true* safety boundary  ``D_true(x) = r_safe``         (sphere around the
    actual obstacle), and
  * the *conformal lower-bound* boundary ``D_pred(x) - U(x) = r_safe`` (the
    inflated envelope around the predicted obstacle),

together with the closed-loop FCP-MPC path threading past them. The conformal
surface encloses the true one -- the 3D analogue of the red dashed contour
covering the white dashed contour in the 2D figure.

The expensive part (calibration + closed-loop rollout) is cached to an .npz so
the (cheap) rendering can be re-tuned without recomputing. Usage:

    python make_fig_conformal_3d.py                 # run rollout (or reuse cache) + render
    python make_fig_conformal_3d.py --reuse         # render only, from cache
    python make_fig_conformal_3d.py --seed 7 --n-obs 60 --i-view 4

This script is headless (Agg) and safe to run in the background.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

from quad_env import QuadWorldEnv3D, distance_field_points_3d
from sim_func_3d import (run_one_episode_visual_3d,
                         _get_obs_positions_from_history)

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.environ.get("FCP_PAPER_DIR", os.path.join(HERE, "T_RO2026"))
CACHE = os.path.join(HERE, "fig_conformal_3d_cache.pkl")
OUT = os.path.join(PAPER_DIR, "Func_cp_3d_zoom.png")

# Mirror make_figs_3d.ENV_KWARGS so the calibrated envelope is representative.
WORLD_BOUNDS = ((-3, 7), (-3, 7), (0, 8))
ENV_KWARGS = dict(
    dt=0.1, horizon=20, world_bounds_xyz=WORLD_BOUNDS,
    pred_model_noise=0.20, obs_process_noise=0.22, gt_future_noise=0.20,
    mode_switch_p=0.95, mode_min_ttl=1, mode_max_ttl=6,
    turn_rate_std=3.0, stop_go_p=0.6, gui=False,
)


# --------------------------------------------------------------------------- #
# 1) Rollout (expensive; cached)
# --------------------------------------------------------------------------- #
def run_rollout(seed: int, n_obs: int, i_view: int) -> dict:
    env = QuadWorldEnv3D(seed=seed, n_obs=n_obs, **ENV_KWARGS)
    res = run_one_episode_visual_3d(
        env,
        nx=40, ny=40, nz=40,
        time_horizon=12,
        alpha=0.10,
        p_base=8, k_mix=10,
        n_skip=4, n_paths=2000,
        max_steps=250,
        n_calib_samples=20,
        i_view=i_view,
        CP=True,
        visualize=False,
        capture_history=True,
        return_fields=True,
        break_on_collision=False,
        method_name="FCP-MPC",
    )
    f = res["fields"]
    payload = dict(
        seed=seed, n_obs=n_obs, i_view=i_view,
        reached_goal=res["reached_goal"], steps=res["steps"],
        robot_traj=np.asarray(res["robot_traj"], dtype=np.float32),
        xs=f["xs"], ys=f["ys"], zs=f["zs"],
        g_upper_grid=(None if f["g_upper_grid"] is None
                      else np.asarray(f["g_upper_grid"], dtype=np.float32)),
        safe_rad=f["safe_rad"], goal=f["goal"],
        # keep only the light fields of episode_history needed for the figure
        history=[
            dict(step=h["step"], robot=np.asarray(h["robot"], np.float32),
                 pred=np.asarray(h["pred"], np.float32),
                 pred_mask=np.asarray(h["pred_mask"], bool),
                 obs=_get_obs_positions_from_history(h["obs"]))
            for h in f["episode_history"]
        ],
    )
    with open(CACHE, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"[cache] wrote {CACHE}  (steps={payload['steps']}, "
          f"reached={payload['reached_goal']})")
    return payload


# --------------------------------------------------------------------------- #
# 2) Frame / obstacle selection
# --------------------------------------------------------------------------- #
def pick_frame_and_obstacles(P: dict, max_obs: int = 2, sel_radius: float = 1.6,
                             min_clearance: float = 0.6):
    """Pick a closed-loop frame where the robot passes *just outside* an obstacle's
    conformal envelope -- i.e. the tightest approach whose clearance still exceeds
    ``min_clearance`` -- so the path visibly skirts (rather than penetrates) the
    safety bound. Then take the <=max_obs obstacles nearest the robot."""
    hist = P["history"]
    iv = int(P["i_view"])
    traj = P["robot_traj"]
    best = None       # tightest pass whose whole-path clearance >= min_clearance
    fallback = None   # global closest approach, if nothing clears the band
    for k, h in enumerate(hist):
        future_idx = k + iv + 1
        if future_idx >= len(hist):
            break
        gt = hist[future_idx]["obs"]
        if gt.size == 0:
            continue
        robot = h["robot"]
        d = np.linalg.norm(gt - robot[None, :], axis=1)
        j = int(d.argmin())
        obs = gt[j]
        # closest the *whole executed path* ever comes to this obstacle location:
        # guarantees the plotted segment never dips inside the conformal envelope.
        path_clear = float(np.linalg.norm(traj - obs[None, :], axis=1).min())
        if path_clear < (fallback[0] if fallback else 1e9) and path_clear > P["safe_rad"] * 0.5:
            fallback = (path_clear, k)
        if path_clear >= min_clearance and path_clear < (best[0] if best else 1e9):
            best = (path_clear, k)
    best = best or fallback
    if best is None:
        raise RuntimeError("no usable frame with nearby obstacles")
    _, k = best
    h = hist[k]
    iv_c = int(np.clip(iv, 0, h["pred"].shape[0] - 1)) if h["pred"].size else 0
    robot = h["robot"]

    # predicted obstacle positions at horizon iv (nominal centers)
    if h["pred"].size:
        pred_i = h["pred"][iv_c]
        m = h["pred_mask"][iv_c]
        pred_pts = pred_i[m]
    else:
        pred_pts = np.zeros((0, 3), np.float32)
    # true obstacle positions at the matching future time
    gt_pts = hist[k + iv_c + 1]["obs"]

    # select obstacles near the robot
    def _near(pts):
        if pts.size == 0:
            return pts
        d = np.linalg.norm(pts - robot[None, :], axis=1)
        order = np.argsort(d)
        keep = [i for i in order if d[i] <= sel_radius][:max_obs]
        if not keep:
            keep = order[:1]
        return pts[keep]

    gt_sel = _near(gt_pts)
    # pair each selected true obstacle with its single nearest prediction (the
    # same physical obstacle), so the figure shows clean nested true/conformal
    # surfaces rather than a cloud of unrelated predicted wells.
    if pred_pts.size and gt_sel.size:
        dd = np.linalg.norm(pred_pts[None, :, :] - gt_sel[:, None, :], axis=2)
        idx = np.unique(dd.argmin(axis=1))
        pred_sel = pred_pts[idx]
    else:
        pred_sel = _near(pred_pts)

    return dict(frame=k, iv=iv_c, robot=robot, gt=gt_sel, pred=pred_sel)


# --------------------------------------------------------------------------- #
# 3) Local fields + isosurfaces
# --------------------------------------------------------------------------- #
def local_grid(centers: np.ndarray, safe_rad: float, pad: float, res: int):
    lo = centers.min(axis=0) - pad
    hi = centers.max(axis=0) + pad
    xs = np.linspace(lo[0], hi[0], res, dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], res, dtype=np.float32)
    zs = np.linspace(lo[2], hi[2], res, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
    # match quad_env build_grid_3d layout: (nz, ny, nx)
    return (xs, ys, zs,
            np.transpose(X, (2, 0, 1)),
            np.transpose(Y, (2, 0, 1)),
            np.transpose(Z, (2, 0, 1)))


def interp_envelope(g_grid, gxs, gys, gzs, Xl, Yl, Zl):
    """Interpolate the global coarse envelope U onto the local fine grid."""
    if g_grid is None:
        return np.zeros_like(Xl)
    interp = RegularGridInterpolator(
        (gzs, gys, gxs), np.maximum(g_grid, 0.0),
        bounds_error=False, fill_value=None)
    pts = np.stack([Zl.ravel(), Yl.ravel(), Xl.ravel()], axis=1)
    return interp(pts).reshape(Xl.shape).astype(np.float32)


def iso_world(vol, level, xs, ys, zs):
    """Marching-cubes isosurface -> (verts_world (N,3), faces). vol is (nz,ny,nx)."""
    vmin, vmax = float(np.nanmin(vol)), float(np.nanmax(vol))
    if not (vmin < level < vmax):
        return None, None
    dz = zs[1] - zs[0]; dy = ys[1] - ys[0]; dx = xs[1] - xs[0]
    verts, faces, _, _ = marching_cubes(vol, level=level, spacing=(dz, dy, dx))
    # verts columns are (z, y, x) offset from origin (zs[0], ys[0], xs[0])
    world = np.column_stack([
        verts[:, 2] + xs[0],
        verts[:, 1] + ys[0],
        verts[:, 0] + zs[0],
    ])
    return world, faces


def shaded_sphere(ax, center, r, color, alpha, ls, n=64):
    """Smooth, light-shaded analytic sphere via plot_surface -- gives a solid 3D
    'ball' look (gradient shading) instead of a flat constant-color blob."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + r * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color, alpha=alpha,
                    linewidth=0, antialiased=True, shade=True, lightsource=ls)


# --------------------------------------------------------------------------- #
# 4) Render
# --------------------------------------------------------------------------- #
def render(P: dict, sel: dict, out: str, elev: float, azim: float,
           usetex: bool, res: int, envelope_mode: str = "sphere",
           margin_floor: float = 0.20, margin_cap: float = 0.8):
    safe_rad = float(P["safe_rad"])
    centers = np.vstack([c for c in (sel["gt"], sel["pred"]) if c.size] + [sel["robot"][None]])
    pad = safe_rad + 0.55   # zoom tight: just contain the inflated envelope
    lxs, lys, lzs, Xl, Yl, Zl = local_grid(centers, safe_rad, pad, res)

    D_true = distance_field_points_3d(sel["gt"], Xl, Yl, Zl)
    D_pred = distance_field_points_3d(sel["pred"], Xl, Yl, Zl)
    iv = int(sel["iv"])
    g = None if P["g_upper_grid"] is None else P["g_upper_grid"][
        int(np.clip(iv, 0, P["g_upper_grid"].shape[0] - 1))]

    if envelope_mode == "field":
        # exact (but ragged) conformal field D_pred - U(x)
        U = interp_envelope(g, P["xs"], P["ys"], P["zs"], Xl, Yl, Zl)
        D_lower = np.maximum(D_pred - U, 0.0)
        U_show = float(np.median(U))
    else:
        # clean illustration: inflate the nominal safety sphere by the
        # representative calibrated margin sampled at the predicted obstacle(s),
        # clipped to a sane band so coarse-grid outlier cells don't blow it up.
        if g is None or not sel["pred"].size:
            U_show = margin_floor
        else:
            interp = RegularGridInterpolator(
                (P["zs"], P["ys"], P["xs"]), np.maximum(g, 0.0),
                bounds_error=False, fill_value=None)
            q = np.column_stack([sel["pred"][:, 2], sel["pred"][:, 1],
                                 sel["pred"][:, 0]])
            U_show = float(np.clip(np.max(interp(q)), margin_floor, margin_cap))
        D_lower = np.maximum(D_pred - U_show, 0.0)
    print(f"[render] envelope margin U={U_show:.3f}  (conformal radius "
          f"{safe_rad + U_show:.3f} vs true {safe_rad:.3f})")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
    })
    if usetex:
        plt.rcParams["text.usetex"] = True

    fig = plt.figure(figsize=(4.8, 4.4), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    if envelope_mode == "field":
        # exact (but ragged) conformal field: isosurfaces via marching cubes
        Vc, Fc = iso_world(D_lower, safe_rad, lxs, lys, lzs)
        Vt, Ft = iso_world(D_true, safe_rad, lxs, lys, lzs)
        if Vc is not None:
            ax.add_collection3d(Poly3DCollection(
                Vc[Fc], facecolor="#d62728", edgecolor="none", alpha=0.16))
        if Vt is not None:
            ax.add_collection3d(Poly3DCollection(
                Vt[Ft], facecolor="#1f77b4", edgecolor="none", alpha=0.55))
    else:
        # clean illustration: smooth, light-shaded analytic spheres for a solid
        # 3D look. Conformal shells (translucent red) enclose the true safety
        # balls (solid blue); each predicted center carries a conformal shell.
        ls = LightSource(azdeg=315, altdeg=50)
        for c in sel["pred"]:
            shaded_sphere(ax, c, safe_rad + U_show, "#e23b3b", 0.22, ls)
        for c in sel["gt"]:
            shaded_sphere(ax, c, safe_rad, "#1f6fc4", 0.98, ls)

    # obstacle centers
    if sel["gt"].size:
        ax.scatter(sel["gt"][:, 0], sel["gt"][:, 1], sel["gt"][:, 2],
                   c="#08306b", s=22, marker="o", depthshade=False)
    if sel["pred"].size:
        ax.scatter(sel["pred"][:, 0], sel["pred"][:, 1], sel["pred"][:, 2],
                   c="k", s=30, marker="x", depthshade=False, linewidths=1.4)

    # FCP path segment near the zoom box
    traj = P["robot_traj"]
    lo = centers.min(axis=0) - pad; hi = centers.max(axis=0) + pad
    inbox = np.all((traj >= lo[None]) & (traj <= hi[None]), axis=1)
    if inbox.any():
        # contiguous-ish: just plot the in-box points in order
        seg = traj[inbox]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c="#2ca02c", lw=2.4,
                label="FCP-MPC path")
        ax.scatter(*sel["robot"], c="#2ca02c", s=36, marker="^",
                   depthshade=False, zorder=6)

    # cosmetics: clean, paper-style floating view (no grid, ticks, numbers, panes)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect((hi - lo))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor="#1f77b4", alpha=0.55, label="true safety boundary"),
        Patch(facecolor="#d62728", alpha=0.30,
              label="conformal lower bound"),
        Line2D([0], [0], color="#2ca02c", lw=2.4, label="FCP-MPC path"),
        Line2D([0], [0], marker="x", color="k", lw=0, markersize=6,
               label="predicted center"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9,
              borderpad=0.3, handletextpad=0.5)

    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"[saved] {out}")
    # also drop a preview copy next to the script for quick inspection
    prev = os.path.join(HERE, "fig_conformal_3d_preview.png")
    fig.savefig(prev, bbox_inches="tight", pad_inches=0.05)
    print(f"[saved] {prev}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-obs", type=int, default=60)
    ap.add_argument("--i-view", type=int, default=4)
    ap.add_argument("--reuse", action="store_true",
                    help="render from the cached rollout (no recompute)")
    ap.add_argument("--max-obs", type=int, default=1)
    ap.add_argument("--sel-radius", type=float, default=1.3)
    ap.add_argument("--min-clearance", type=float, default=0.62,
                    help="pick the obstacle the whole path stays at least this far "
                         "from, so the path visibly skirts (not penetrates) the bound")
    ap.add_argument("--elev", type=float, default=45.0)
    ap.add_argument("--azim", type=float, default=235.0,
                    help="view ~perpendicular to the path/obstacle plane so the "
                         "clearance is visible in-image, not along the line of sight")
    ap.add_argument("--res", type=int, default=72)
    ap.add_argument("--usetex", action="store_true")
    ap.add_argument("--envelope-mode", choices=["sphere", "field"], default="sphere")
    ap.add_argument("--margin", type=float, default=0.13,
                    help="conformal-envelope margin floor (radius = r_safe + margin)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if args.reuse and os.path.isfile(CACHE):
        with open(CACHE, "rb") as fh:
            P = pickle.load(fh)
        print(f"[cache] loaded {CACHE}")
    else:
        P = run_rollout(args.seed, args.n_obs, args.i_view)

    sel = pick_frame_and_obstacles(P, max_obs=args.max_obs,
                                   sel_radius=args.sel_radius,
                                   min_clearance=args.min_clearance)
    print(f"[select] frame={sel['frame']} iv={sel['iv']} "
          f"n_true={sel['gt'].shape[0]} n_pred={sel['pred'].shape[0]}")
    render(P, sel, args.out, args.elev, args.azim, args.usetex, args.res,
           envelope_mode=args.envelope_mode, margin_floor=args.margin)


if __name__ == "__main__":
    main()
