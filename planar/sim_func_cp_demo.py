from __future__ import annotations

import os as _os, sys as _sys
# Shared controllers/, cp/, utils.py and viz_traj.py live at the repo root (alongside
# quadrotor/); put it on sys.path so those absolute imports resolve from anywhere.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _os, _sys

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import build_grid, distance_field_points
from cp.functional_cp import get_envelopes_value_and_function, CPStepParameters
from controllers.func_cp_mpc import FunctionalCPMPC
import time

# ==============================================================================
# 0) Constants (aligned to your main runner)
# ==============================================================================

DT = 0.4

ROBOT_RAD = 0.4
OBSTACLE_RAD = 1.0 / np.sqrt(2.0)

MAX_LINEAR_X = 0.8
MIN_LINEAR_X = -0.8
MAX_ANGULAR_Z = 0.7
MIN_ANGULAR_Z = -0.7

EVAL_TASK_CONFIGS = {
    "zara1": {"init_robot_pose": np.array([12.0, 5.0, np.pi], dtype=np.float32), "goal_pos": np.array([3.0, 6.0], dtype=np.float32)},
    "zara2": {"init_robot_pose": np.array([1.0, 6.0, 0.0], dtype=np.float32), "goal_pos": np.array([14.0, 5.0], dtype=np.float32)},
    "hotel": {"init_robot_pose": np.array([-1.5, 0.0, -np.pi / 2.0], dtype=np.float32), "goal_pos": np.array([2.0, -6.0], dtype=np.float32)},
    "eth":   {"init_robot_pose": np.array([5.0, 1.0, np.pi / 2.0], dtype=np.float32), "goal_pos": np.array([3.0, 10.0], dtype=np.float32)},
    "univ":  {"init_robot_pose": np.array([3.5, 2.0, np.pi / 4.0], dtype=np.float32), "goal_pos": np.array([11.5, 8.5], dtype=np.float32)},
}

SCENARIOS = {
    "zara1": [100, 200, 300],
    "zara2": [100, 200, 300],
    "eth": [100, 200, 300],
    "hotel": [100, 200, 300],
    "univ": [100],
}

INIT_FRAMES = {
    "zara1": 0,
    "zara2": 1,
    "eth": 78,
    "hotel": 0,
    "univ": 0,
}

MAX_N_STEPS = {
    "zara1": 100,
    "zara2": 100,
    "eth": 100,
    "hotel": 100,
    "univ": 300,
}


def unicycle_step(xy: np.ndarray, th: float, v: float, w: float, dt: float):
    x = xy[0] + dt * v * np.cos(th)
    y = xy[1] + dt * v * np.sin(th)
    th = th + dt * w
    return np.array([x, y], dtype=np.float32), float(th)


def min_dist_robot_to_peds(robot_xy: np.ndarray, peds_xy: np.ndarray) -> float:
    if peds_xy.size == 0:
        return float("inf")
    d = peds_xy - robot_xy[None, :]
    return float(np.sqrt(np.sum(d * d, axis=1)).min())


def compute_lower_field_single_step(
    obst_pred_step: np.ndarray,
    obst_mask_step: np.ndarray,
    Xg: np.ndarray,
    Yg: np.ndarray,
    g_up: Optional[np.ndarray],
    world_center: np.ndarray,
) -> np.ndarray:
    if g_up is None:
        return np.full(Xg.shape, np.inf, dtype=np.float32)

    lower = np.full(Xg.shape, np.inf, dtype=np.float32)
    for m in range(obst_pred_step.shape[0]):
        if not bool(obst_mask_step[m]):
            continue
        y_hat = obst_pred_step[m] - world_center
        d_pred = distance_field_points(y_hat, Xg, Yg)
        lower_m = np.maximum(d_pred - g_up, 0.0)
        lower = np.minimum(lower, lower_m)

    if not np.isfinite(lower).any():
        return np.full(Xg.shape, 20.0, dtype=np.float32)
    return lower


# ==============================================================================
# 1) PKL loader (same convention as your main runner)
# ==============================================================================

def load_prediction_results(dataset: str) -> dict:
    pkl_path = os.path.join(os.path.dirname(__file__), "predictions", f"{dataset}.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    for k in ("prediction", "history", "future"):
        if k not in res:
            raise ValueError(f"PKL is missing key '{k}'")
    return res


# ==============================================================================
# 2) World bounds inference from PKL (CRITICAL FIX)
# ==============================================================================

def _collect_points_for_bounds(
    pred_all: dict,
    hist_all: dict,
    scenario_begin: int,
    n_steps: int,
    horizon: int,
    init_pose: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    pts = [init_pose[:2][None, :], goal[None, :]]

    for k in range(n_steps):
        ts_key = scenario_begin + k

        if ts_key in hist_all:
            for traj in hist_all[ts_key].values():
                arr = np.asarray(traj, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                    pts.append(arr[-1:])

        if ts_key in pred_all:
            for pred_traj in pred_all[ts_key].values():
                arr = np.asarray(pred_traj, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                    take = min(horizon, arr.shape[0])
                    pts.append(arr[:take])

    return np.vstack(pts) if len(pts) > 0 else np.zeros((0, 2), dtype=np.float32)


def _infer_world(points_xy: np.ndarray, margin: float = 2.0):
    if points_xy.size == 0:
        world_center = np.array([0.0, 0.0], dtype=np.float32)
        box = 40.0
        bounds = (-20.0, 20.0, -20.0, 20.0)
        return world_center, box, bounds

    xmin = float(np.min(points_xy[:, 0])) - margin
    xmax = float(np.max(points_xy[:, 0])) + margin
    ymin = float(np.min(points_xy[:, 1])) - margin
    ymax = float(np.max(points_xy[:, 1])) + margin

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    xspan = xmax - xmin
    yspan = ymax - ymin
    box = float(max(xspan, yspan))  # make square for grid

    world_center = np.array([cx, cy], dtype=np.float32)
    return world_center, box, (xmin, xmax, ymin, ymax)


# ==============================================================================
# 3) Offline calibration from file: build residual tensor (N, H, Hg, Wg)
#    (Already aligned to reference: pred[i] <-> future[i])
# ==============================================================================

def build_training_residuals_from_file(
    all_data_dict: dict,
    scene_ids: List[int],
    Xg: np.ndarray,
    Yg: np.ndarray,
    world_center: np.ndarray,
    time_horizon: int,
) -> np.ndarray:
    """
    Build residual tensor using MIN-over-pedestrians SDF residual:

        S_i(x) = D_pred_i(x) - D_true_i(x)
              = min_m ||x - yhat_i^m|| - min_m ||x - y_i^m||

    Returns: (N_scenes, H, Hg, Wg)
    """
    pred_dict = all_data_dict["prediction"]
    fut_dict = all_data_dict["future"]

    Hh = int(time_horizon)
    Hg, Wg = Xg.shape

    residuals: List[np.ndarray] = []

    for sid in scene_ids:
        if sid not in pred_dict or sid not in fut_dict:
            continue

        p_scene = pred_dict[sid]  # {pid: (H,2)}
        f_scene = fut_dict[sid]   # {pid: (H,2)}

        # Use only peds that exist in both pred & GT
        pids = [pid for pid in p_scene.keys() if pid in f_scene]
        if len(pids) == 0:
            continue

        res_i = np.zeros((Hh, Hg, Wg), dtype=np.float32)

        last_valid_i: Optional[int] = None

        for i in range(Hh):
            pred_pts = []
            true_pts = []

            for pid in pids:
                y_pred_traj = np.asarray(p_scene[pid], dtype=np.float32)
                y_true_traj = np.asarray(f_scene[pid], dtype=np.float32)

                # sanity
                if y_pred_traj.ndim != 2 or y_pred_traj.shape[1] != 2:
                    continue
                if y_true_traj.ndim != 2 or y_true_traj.shape[1] != 2:
                    continue
                if i >= y_pred_traj.shape[0] or i >= y_true_traj.shape[0]:
                    continue

                pred_pts.append(y_pred_traj[i])
                true_pts.append(y_true_traj[i])

            # If no valid peds at this horizon step, hold previous residual (or zeros if first)
            if len(pred_pts) == 0 or len(true_pts) == 0:
                if last_valid_i is not None:
                    res_i[i] = res_i[last_valid_i]
                else:
                    res_i[i] = 0.0
                continue

            pred_pts = (np.asarray(pred_pts, dtype=np.float32) - world_center)  # (M,2)
            true_pts = (np.asarray(true_pts, dtype=np.float32) - world_center)  # (M,2)

            # MIN-over-peds distance field
            sdf_pred = distance_field_points(pred_pts, Xg, Yg)  # (Hg,Wg)
            sdf_true = distance_field_points(true_pts, Xg, Yg)  # (Hg,Wg)

            res_i[i] = sdf_pred - sdf_true
            last_valid_i = i

        residuals.append(res_i)

    if len(residuals) == 0:
        raise RuntimeError("No valid residual samples built from file. (Check PKL contents.)")

    return np.stack(residuals, axis=0)


def calibrate_cp_from_file(
    *,
    all_data: dict,
    Xg: np.ndarray,
    Yg: np.ndarray,
    world_center: np.ndarray,
    time_horizon: int,
    p_base: int,
    k_mix: int,
    alpha: float,
    test_size: float,
    random_state: int,
    n_jobs: int,
    backend: str,
) -> Tuple[np.ndarray, List[CPStepParameters]]:
    all_scenes = sorted(list(all_data["prediction"].keys()))
    residuals = build_training_residuals_from_file(
        all_data_dict=all_data,
        scene_ids=all_scenes,
        Xg=Xg,
        Yg=Yg,
        world_center=world_center,
        time_horizon=time_horizon,
    )

    print(f"[offline] residual tensor: {residuals.shape} (N_samples, H, Hg, Wg)")
    print("[offline] calibrating CP envelopes...")

    g_upper_grid, cp_params = get_envelopes_value_and_function(
        residuals_train=residuals,
        p_base=p_base,
        K=k_mix,
        alpha=alpha,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        backend=backend,
    )

    return g_upper_grid.astype(np.float32), cp_params


# ==============================================================================
# 4) Helpers: controller input from PKL prediction dict
# ==============================================================================

def stack_pred_from_p_dict(p_dict: Dict, horizon: int) -> Tuple[np.ndarray, np.ndarray, List]:
    # p_dict[pid][i] is prediction at (t+1+i) in reference convention
    pids = list(p_dict.keys())
    M = len(pids)
    Hh = int(horizon)

    pred = np.zeros((Hh, M, 2), dtype=np.float32)
    mask = np.zeros((Hh, M), dtype=bool)

    for j, pid in enumerate(pids):
        arr = np.asarray(p_dict[pid], dtype=np.float32)
        take = min(Hh, arr.shape[0])
        if take > 0:
            pred[:take, j] = arr[:take]
            mask[:take, j] = True

    return pred, mask, pids


def get_current_obs_from_history(h_dict: Dict) -> np.ndarray:
    # h_dict[pid][-1] is observation at current time t
    if len(h_dict) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([traj[-1] for traj in h_dict.values()], dtype=np.float32)


def get_future_obs_from_future_dict(f_dict: Dict, i_view: int) -> np.ndarray:
    """
    Reference-aligned 'true' positions for comparison with pred[i_view]:
        pred[i_view]  ~ y_{t+1+i_view}
        future[i_view] = y_{t+1+i_view}
    """
    if len(f_dict) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    for pid, traj in f_dict.items():
        arr = np.asarray(traj, dtype=np.float32)  # (H,2)
        if arr.ndim == 2 and arr.shape[1] == 2 and i_view < arr.shape[0]:
            pts.append(arr[i_view])
    return np.asarray(pts, dtype=np.float32) if len(pts) > 0 else np.zeros((0, 2), dtype=np.float32)


def _timing_stats_ms(arr: List[float]) -> Dict[str, float]:
    if len(arr) == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


# ==============================================================================
# 5) One-episode visualization (file-based) with AUTO world/grid
# ==============================================================================

def run_one_episode_visual_from_file(
    dataset: str,
    *,
    scenario_idx: int = 0,
    time_horizon: int = 12,
    grid_H: int = 16,
    grid_W: int = 16,
    alpha: float = 0.05,
    p_base: int = 4,
    k_mix: int = 5,
    test_size: float = 0.30,
    random_state: int = 0,
    n_jobs: int = 4,
    backend: str = "loky",
    n_skip: int = 4,
    n_paths: int = 1200,
    max_steps: Optional[int] = None,
    max_tracking_error: float = 0.1,
    i_view: int = 0,
    CP: bool = False,
):
    if dataset not in EVAL_TASK_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")

    init_robot_pose = EVAL_TASK_CONFIGS[dataset]["init_robot_pose"].copy()
    goal_pos = EVAL_TASK_CONFIGS[dataset]["goal_pos"].copy()
    scenario_begin = SCENARIOS[dataset][scenario_idx]
    _ = INIT_FRAMES[dataset]
    max_n_steps = MAX_N_STEPS[dataset] if max_steps is None else int(max_steps)

    # Load PKL
    all_data = load_prediction_results(dataset)
    pred_all = all_data["prediction"]
    hist_all = all_data["history"]
    fut_all = all_data["future"]

    # --------- AUTO world/grid inference (CRITICAL FIX) ----------
    points_xy = _collect_points_for_bounds(
        pred_all=pred_all,
        hist_all=hist_all,
        scenario_begin=scenario_begin,
        n_steps=min(max_n_steps, 200),
        horizon=time_horizon,
        init_pose=init_robot_pose,
        goal=goal_pos,
    )
    world_center, box, (xmin, xmax, ymin, ymax) = _infer_world(points_xy, margin=2.0)

    xs, ys, Xg, Yg = build_grid(box, grid_H, grid_W)
    extent = [
        xs[0] + world_center[0], xs[-1] + world_center[0],
        ys[0] + world_center[1], ys[-1] + world_center[1],
    ]

    print(f"[world] inferred bounds: x=[{xmin:.2f},{xmax:.2f}] y=[{ymin:.2f},{ymax:.2f}]")
    print(f"[world] inferred box={box:.2f}, world_center={world_center.tolist()}")

    # Offline calibration (same world/grid!)

    if CP:
        g_upper_grid, cp_params = calibrate_cp_from_file(
            all_data=all_data,
            Xg=Xg,
            Yg=Yg,
            world_center=world_center,
            time_horizon=time_horizon,
            p_base=p_base,
            k_mix=k_mix,
            alpha=alpha,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            backend=backend,
        )
    else:
        g_upper_grid = None
        cp_params = None
    

    # Controller init (use inferred box/world_center)
    ctrl = FunctionalCPMPC(
        cp_params=cp_params,
        box=box,
        world_center=world_center,
        n_steps=time_horizon,
        dt=DT,
        n_skip=n_skip,
        robot_rad=ROBOT_RAD,
        obstacle_rad=OBSTACLE_RAD,
        min_linear_x=MIN_LINEAR_X,
        max_linear_x=MAX_LINEAR_X,
        min_angular_z=MIN_ANGULAR_Z,
        max_angular_z=MAX_ANGULAR_Z,
        n_paths=n_paths,
        seed=0,
        grid_H=grid_H,
        grid_W=grid_W,
        CP = CP,
    )

    robot_xy = init_robot_pose[:2].astype(np.float32)
    robot_th = float(init_robot_pose[2])
    goal = goal_pos.astype(np.float32)

    robot_traj = [robot_xy.copy()]
    collision_count = 0
    infeasible_count = 0
    ctrl_times_ms: List[float] = []  # store per-step controller call time (ms)

    # timestep-wise global coverage: 1{ true_unsafe ⊆ cp_unsafe } averaged over t
    eval_steps = 0
    ok_steps = 0
    current_cov_val = 0.0
    per_step_ok: List[bool] = []


    fig, axs = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    im_heat = axs[0].imshow(
        np.zeros((grid_H, grid_W), dtype=np.float32),
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=10,
    )
    plt.colorbar(im_heat, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title("CP lower-bound field", fontsize=18)

    axs[1].set_title(f"Unsafe region: True vs CP (horizon={i_view+1})", fontsize=18)
    im_true = axs[1].imshow(np.zeros((grid_H, grid_W)), extent=extent, origin="lower", cmap="Blues", vmin=0, vmax=1, alpha=0.4)
    im_cp = axs[1].imshow(np.zeros((grid_H, grid_W)), extent=extent, origin="lower", cmap="Reds", vmin=0, vmax=1, alpha=0.5)

    robot_dots, traj_lines, plan_lines, peds_scatters, pred_scatters = [], [], [], [], []
    for ax in axs:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.plot(goal[0], goal[1], "gx", ms=12, mew=2)

        dot, = ax.plot([], [], "bo", ms=7, zorder=5)
        line, = ax.plot([], [], "r-", lw=5, alpha=0.9)
        plan_p, = ax.plot([], [], "c--", lw=1.5, zorder=4)
        peds = ax.scatter([], [], s=25, edgecolors="black", zorder=6)
        pred_sc = ax.scatter([], [], s=40, marker="x", zorder=7)

        robot_dots.append(dot)
        traj_lines.append(line)
        plan_lines.append(plan_p)
        peds_scatters.append(peds)
        pred_scatters.append(pred_sc)

    status_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10, fontweight="bold")

    steps = max_n_steps
    anim = None

    def _stop_animation(final_text: str, color: str):
        nonlocal anim
        status_text.set_text(final_text)
        status_text.set_color(color)
        if anim is not None and anim.event_source is not None:
            anim.event_source.stop()
        fig.canvas.draw_idle()

    def update(k: int):
        nonlocal robot_xy, robot_th, collision_count, infeasible_count, max_tracking_error
        nonlocal eval_steps, ok_steps, current_cov_val, per_step_ok, steps, i_view, anim
        ts_key = scenario_begin + k
        if ts_key not in pred_all or ts_key not in hist_all or ts_key not in fut_all:
            status_text.set_text(
                f"Ended | timestep={k} | Cov={current_cov_val:.2%} | infeasible={infeasible_count} | collisions={collision_count} | colision_rate={collision_count / max(1, k-14):.2f} | infeas_rate={infeasible_count / max(1, k-14):.2f}"
            )
            print(f"Ended | timestep={k} | Cov={current_cov_val:.2%} | infeasible={infeasible_count} | collisions={collision_count} | colision_rate={collision_count / max(1, k-14):.2f} | infeas_rate={infeasible_count / max(1, k-14):.2f}")
            stats = _timing_stats_ms(ctrl_times_ms)
            print(
                f"[timing] controller: "
                f"mean={stats['mean']:.3f} ms | p50={stats['p50']:.3f} | "
                f"p90={stats['p90']:.3f} | p99={stats['p99']:.3f} | max={stats['max']:.3f}"
            )
            _stop_animation(
                f"Ended | timestep={k} | Cov={current_cov_val:.2%} | infeasible={infeasible_count} | collisions={collision_count} | colision_rate={collision_count / max(1, k-14):.2f} | infeas_rate={infeasible_count / max(1, k-14):.2f}",
                "red",
            )
            return []

        dist_to_goal = float(np.linalg.norm(robot_xy - goal))
        if dist_to_goal <= 0.6:
            status_text.set_text(
                f"GOAL reached | timestep={k} | Cov={current_cov_val:.2%} | dist={dist_to_goal:.2f} | infeasible={infeasible_count} | collisions={collision_count} | colision_rate={collision_count / max(1, k-14):.2f} | infeas_rate={infeasible_count / max(1, k-14):.2f}"
            )
            print(f"GOAL reached | timestep={k} | Cov={current_cov_val:.2%} | dist={dist_to_goal:.2f} | infeasible={infeasible_count} | collisions={collision_count} | colision_rate={collision_count / max(1, k-14):.2f} | infeas_rate={infeasible_count / max(1, k-14):.2f}")
            stats = _timing_stats_ms(ctrl_times_ms)
            print(
                f"[timing] controller: "
                f"mean={stats['mean']:.3f} ms | p50={stats['p50']:.3f} | "
                f"p90={stats['p90']:.3f} | p99={stats['p99']:.3f} | max={stats['max']:.3f}"
            )
            _stop_animation(
                f"GOAL reached | timestep={k} | Cov={current_cov_val:.2%} | dist={dist_to_goal:.2f} | infeasible={infeasible_count} | collisions={collision_count} | colision_rate={collision_count / max(1, k-14):.2f} | infeas_rate={infeasible_count / max(1, k-14):.2f}",
                "green",
            )
            return []

        p_dict = pred_all[ts_key]   # predictions at t: give (t+1..t+H)
        h_dict = hist_all[ts_key]   # history up to t: last = y_t
        f_dict = fut_all[ts_key]    # future at t: gives (t+1..t+H) GT

        # Current observed positions at time t (used for collision check only)
        p_now = get_current_obs_from_history(h_dict)

        # Choose which future step to visualize/compare

        p_future = get_future_obs_from_future_dict(f_dict, i_view)


        # Build pred tensor (H, M, 2): pred[i] corresponds to t+1+i
        pred, obst_mask, _ = stack_pred_from_p_dict(p_dict, horizon=time_horizon)

        # Predicted points at target time (t+1+i_view)
        pred_i = pred[i_view]              # (M,2)
        mask_i = obst_mask[i_view]         # (M,)
        pred_pts = pred_i[mask_i]          # only valid predicted agents

        # CP lower-bound field at the SAME target time (t+1+i_view)
        lower_field = compute_lower_field_single_step(
            pred_i,
            mask_i,
            Xg,
            Yg,
            g_upper_grid[i_view],
            world_center,
        )

        safe_thresh = ROBOT_RAD + OBSTACLE_RAD

        dist_now = distance_field_points(p_future - world_center, Xg, Yg) if p_future.size > 0 else np.full((grid_H, grid_W), np.inf, dtype=np.float32)

        im_true.set_data((dist_now < safe_thresh).astype(float))
        im_cp.set_data((lower_field < safe_thresh).astype(float))
        im_heat.set_data(lower_field)

        # =========================
        # timestep-wise global coverage (event indicator)
        # event: true unsafe region is contained in CP unsafe region
        #    ∀x: [D_true(x) < r_safe] ⇒ [D_lower(x) < r_safe]
        # equivalently: no grid cell where true_unsafe is 1 but cp_unsafe is 0
        # =========================
        true_unsafe = (dist_now < safe_thresh)
        cp_unsafe   = (lower_field < safe_thresh)

        # if there are no pedestrians (dist_now is inf everywhere), then true_unsafe is all False
        # => containment holds trivially
        ok_t = (not np.any(true_unsafe & (~cp_unsafe)))

        per_step_ok.append(bool(ok_t))
        if k > i_view:
            ok_steps += int(ok_t)
            current_cov_val = ok_steps / (k - i_view)
        else:
            current_cov_val = 0.0

        # Control uses full horizon pred (still reference-aligned)
        t0 = time.perf_counter()
        act, info = ctrl(
            pos_x=float(robot_xy[0]),
            pos_y=float(robot_xy[1]),
            orientation_z=float(robot_th),
            boxes=[],
            obst_pred_traj=pred,
            obst_mask=obst_mask,
            goal=goal,
        )
        t1 = time.perf_counter()
        ctrl_times_ms.append((t1 - t0) * 1000.0)

        safety_weight = info.get("safety_weight", 0.0)
        feasible = bool(info.get("feasible", False))
        if not feasible:
            if k > 15:
                infeasible_count += 1
            v, w = 0.0, 0.0
        else:
            v, w = float(act[0]), float(act[1])
        if k <= 15:
            v, w = 0.0, 0.0

        plan = info.get("final_path", None)

        # Collision check MUST use current observed positions at time t
        # (not future positions)

        if p_now.size == 0:
            dmin = np.inf
        else:
            dmin = float(min_dist_robot_to_peds(robot_xy, p_now))
        is_coll = dmin < safe_thresh
        if is_coll and k > 15:
            collision_count += 1

        traj = np.stack(robot_traj)

        for ax_i in range(2):
            robot_dots[ax_i].set_data([robot_xy[0]], [robot_xy[1]])
            robot_dots[ax_i].set_color("red" if is_coll else "blue")
            traj_lines[ax_i].set_data(traj[:, 0], traj[:, 1])

            if plan is not None:
                plan_lines[ax_i].set_data(plan[:, 0], plan[:, 1])

            # For visualization:
            # - show current observed pedestrians at time t
            peds_scatters[ax_i].set_offsets(p_future if p_future.size else np.zeros((0, 2), dtype=np.float32))
            pred_scatters[ax_i].set_offsets(pred_i if pred_i.size else np.zeros((0, 2), dtype=np.float32))

        status_text.set_text(
            f"dataset={dataset} | step={k} | Cov={current_cov_val:.2%} |"
            f"v={v:.2f} | w={w:.2f} | safety_w={safety_weight:.2f} | "
            f"collisions={collision_count} | infeasible={infeasible_count} | "
            f"dist_goal={float(np.linalg.norm(robot_xy - goal)):.2f}"
        )


        status_text.set_color("red" if (is_coll or not feasible) else "black")

        # Update robot
        robot_xy, robot_th = unicycle_step(robot_xy, robot_th, v, w, DT)
        robot_traj.append(robot_xy.copy())
        if k == 49:
            video_filename = f"{dataset}_iview_{i_view}_CP_{CP}.png"
            save_path = os.path.join(os.path.dirname(__file__), video_filename)
            plt.savefig(save_path, dpi=150)



        return []

    #anim = FuncAnimation(fig, update, frames=97, interval=100, blit=False, repeat=False)
    #video_filename = f"{dataset}_iview_{i_view}_CP_{CP}.mp4"
    #save_path = os.path.join(os.path.dirname(__file__), video_filename)
    #anim.save(save_path, writer='ffmpeg', fps=10, dpi=150)

    anim = FuncAnimation(fig, update, frames=max_n_steps, interval=100, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    DATASET = "zara1"
    run_one_episode_visual_from_file(
        dataset=DATASET,
        scenario_idx=0,
        time_horizon=12,
        grid_H=128,
        grid_W=128,
        alpha=0.1,
        p_base=6,
        k_mix=7,
        test_size=0.30,
        random_state=0,
        n_jobs=max(1, (os.cpu_count() or 4) - 2),
        backend="loky",
        n_skip=2,
        n_paths=1200,
        max_tracking_error=0.05,
        i_view=0,
        CP = True,
    )
