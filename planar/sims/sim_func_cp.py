from __future__ import annotations

import os as _os, sys as _sys
# Shared controllers/, cp/ and utils.py live at the repo root (two levels up, alongside
# quadrotor/); put it on sys.path so those absolute imports resolve.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
del _os, _sys

import os
import pickle
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import time

from utils import build_grid, distance_field_points
from sims.sim_utils import min_dist_robot_to_peds, unicycle_step
from cp.functional_cp import CPStepParameters, get_envelopes_value_and_function
from controllers.func_cp_mpc import FunctionalCPMPC, FuncMPCWeights


# -------------------------
# Helpers from your visual code
# -------------------------
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
    box = float(max(xspan, yspan))

    world_center = np.array([cx, cy], dtype=np.float32)
    return world_center, box, (xmin, xmax, ymin, ymax)


def build_training_residuals_from_file(
    all_data_dict: dict,
    scene_ids: List[int],
    Xg: np.ndarray,
    Yg: np.ndarray,
    world_center: np.ndarray,
    time_horizon: int,
) -> np.ndarray:
    pred_dict = all_data_dict["prediction"]
    fut_dict = all_data_dict["future"]

    Hh = int(time_horizon)
    Hg, Wg = Xg.shape

    residuals: List[np.ndarray] = []

    for sid in scene_ids:
        if sid not in pred_dict or sid not in fut_dict:
            continue

        p_scene = pred_dict[sid]
        f_scene = fut_dict[sid]

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

                if y_pred_traj.ndim != 2 or y_pred_traj.shape[1] != 2:
                    continue
                if y_true_traj.ndim != 2 or y_true_traj.shape[1] != 2:
                    continue
                if i >= y_pred_traj.shape[0] or i >= y_true_traj.shape[0]:
                    continue

                pred_pts.append(y_pred_traj[i])
                true_pts.append(y_true_traj[i])

            if len(pred_pts) == 0 or len(true_pts) == 0:
                if last_valid_i is not None:
                    res_i[i] = res_i[last_valid_i]
                else:
                    res_i[i] = 0.0
                continue

            pred_pts = (np.asarray(pred_pts, dtype=np.float32) - world_center)
            true_pts = (np.asarray(true_pts, dtype=np.float32) - world_center)

            sdf_pred = distance_field_points(pred_pts, Xg, Yg)
            sdf_true = distance_field_points(true_pts, Xg, Yg)

            res_i[i] = sdf_pred - sdf_true
            last_valid_i = i

        residuals.append(res_i)

    if len(residuals) == 0:
        raise RuntimeError("No valid residual samples built from file. (Check PKL contents.)")

    return np.stack(residuals, axis=0)


def calibrate_or_load_cp(
    *,
    cache_path: str,
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
    def _fit_and_cache() -> Tuple[np.ndarray, List[CPStepParameters]]:
        all_scenes = sorted(list(all_data["prediction"].keys()))
        residuals = build_training_residuals_from_file(
            all_data_dict=all_data,
            scene_ids=all_scenes,
            Xg=Xg,
            Yg=Yg,
            world_center=world_center,
            time_horizon=time_horizon,
        )

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

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "g_upper_grid": g_upper_grid.astype(np.float32),
                    "cp_params": cp_params,
                },
                f,
            )

        return g_upper_grid.astype(np.float32), cp_params

    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)

        g_upper_grid = obj.get("g_upper_grid")
        cp_params = obj.get("cp_params")

        if g_upper_grid is not None and cp_params is not None:
            return np.asarray(g_upper_grid, dtype=np.float32), cp_params

        # Legacy cache missing cp_params – recompute to populate new format.
        return _fit_and_cache()

    return _fit_and_cache()


def stack_pred_from_p_dict(p_dict: Dict, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
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

    return pred, mask


def get_current_obs_from_history(h_dict: Dict) -> np.ndarray:
    if len(h_dict) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([traj[-1] for traj in h_dict.values()], dtype=np.float32)


# -------------------------
# Main entry: run_fcp_mpc
# -------------------------
def run_fcp_mpc(
    *,
    dataset: str,
    scenarios: List[int],
    max_linear_x: float,
    min_linear_x: float,
    max_angular_z: float,
    min_angular_z: float,
    predictions: dict,   # loaded pkl dict containing prediction/history/future
    dt: float,
    init_frame: int,     # unused here but keep API parity
    visualize: bool,     # ignore; keep API parity
    asset_dir: Any,      # ignore
    robot_img: Any,      # ignore
    max_n_steps: int,
    robot_rad: float,
    obstacle_rad: float,
    init_robot_pose: np.ndarray,
    goal_pos: np.ndarray,
    # ---- knobs mapped from "controller_configs"
    target_miscoverage_level: float = 0.1,  # -> alpha
    step_size: float = 10.0,                # -> eta (safety weight update)
    adaptive: bool = True,                  # True = ACP online update; False = fixed offline coeffs
    safety_mode: str = "hard",              # "hard" = filter paths; "soft" = penalty in cost
    w_safety: float = 50.0,                 # soft-mode safety-shaping weight (declared hyperparameter)
):
    # ----- fixed controller settings -----
    # Horizon unified to 12 = the predictor's horizon = the baselines' horizon, for a clean
    # fair comparison. (Earlier FCP used 17, but steps 13-17 had masked obstacle predictions,
    # so this matches the obstacle-aware horizon and only drops the goal-seeking lookahead.)
    time_horizon = 12
    grid_H = 128
    grid_W = 128
    n_skip = 2
    n_paths = 1200

    p_base = 5
    k_mix = 7
    test_size = 0.30
    random_state = 0
    n_jobs = max(1, (os.cpu_count() or 4) - 2)
    backend = "loky"

    alpha = float(target_miscoverage_level)

    pred_all = predictions["prediction"]
    hist_all = predictions["history"]
    fut_all = predictions["future"]  # not required for control loop, only for calibration residual build

    metric_dict: Dict[int, dict] = {}
    traj_out: List[np.ndarray] = []  # list of trajectories per scenario (variable length)

    safe_thresh = float(robot_rad + obstacle_rad)

    for scene_begin in scenarios:
        # --- infer world bounds from this scenario (same as your visual code) ---
        points_xy = _collect_points_for_bounds(
            pred_all=pred_all,
            hist_all=hist_all,
            scenario_begin=int(scene_begin),
            n_steps=min(int(max_n_steps), 200),
            horizon=int(time_horizon),
            init_pose=np.asarray(init_robot_pose, dtype=np.float32),
            goal=np.asarray(goal_pos, dtype=np.float32),
        )
        world_center, box, _ = _infer_world(points_xy, margin=2.0)
        xs, ys, Xg, Yg = build_grid(float(box), int(grid_H), int(grid_W))

        # --- CP calibration cache path (per dataset + scenario + key hyperparams) ---
        cache_dir = os.path.join(os.path.dirname(__file__), "cp_cache")
        cache_path = os.path.join(
            cache_dir,
            f"{dataset}_H{time_horizon}_G{grid_H}x{grid_W}_a{alpha:.3f}_p{p_base}_k{k_mix}.pkl",
        )

        _, cp_params = calibrate_or_load_cp(
            cache_path=cache_path,
            all_data=predictions,
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

        # --- controller ---
        ctrl = FunctionalCPMPC(
            cp_params=cp_params,
            box=float(box),
            world_center=world_center,
            grid_H=int(grid_H),
            grid_W=int(grid_W),
            n_steps=int(time_horizon),
            dt=float(dt),
            n_skip=int(n_skip),
            robot_rad=float(robot_rad),
            obstacle_rad=float(obstacle_rad),
            min_linear_x=float(min_linear_x),
            max_linear_x=float(max_linear_x),
            min_angular_z=float(min_angular_z),
            max_angular_z=float(max_angular_z),
            n_paths=int(n_paths),
            seed=0,
            CP=True,
            adaptive=bool(adaptive),
            safety_mode=str(safety_mode),
            weights=FuncMPCWeights(w_safety=float(w_safety)),
        )

        robot_xy = np.asarray(init_robot_pose[:2], dtype=np.float32).copy()
        robot_th = float(init_robot_pose[2])
        goal = np.asarray(goal_pos, dtype=np.float32).copy()

        traj = [robot_xy.copy()]
        collisions = []
        infeasible = []
        costs = []

        ctrl_times_ms: List[float] = []
        loop_times_ms: List[float] = []

        # evaluation loop
        for k in range(int(max_n_steps)):
            ts_key = int(scene_begin) + k
            if ts_key not in pred_all or ts_key not in hist_all:
                break

            # stop if reached goal
            if float(np.linalg.norm(robot_xy - goal)) <= 0.6:
                break

            # predictions for this timestep
            p_dict = pred_all[ts_key]
            h_dict = hist_all[ts_key]

            pred, obst_mask = stack_pred_from_p_dict(p_dict, horizon=time_horizon)

            t0 = time.perf_counter()
            act, info = ctrl(
                pos_x=float(robot_xy[0]),
                pos_y=float(robot_xy[1]),
                orientation_z=float(robot_th),
                boxes=[],
                obst_pred_traj=pred,
                obst_mask=obst_mask,
                goal=goal,
                # Realized obstacle positions for the online ACP coefficient update.
                # Without this, CPOnlineAdapter never sees ground truth and the
                # `adaptive` flag is a no-op (adaptive == fixed). See func_cp_mpc.py.
                observations=h_dict,
            )
            t1 = time.perf_counter()
            ctrl_times_ms.append((t1 - t0) * 1000.0)

            feasible = bool(info.get("feasible", False))
            infeasible.append(0 if feasible else 1)

            if feasible:
                v, w = float(act[0]), float(act[1])
                costs.append(float(info.get("cost", 0.0)))
            else:
                v, w = 0.0, 0.0
                costs.append(float("nan"))

            # collision check uses CURRENT observed positions at time t
            p_now = get_current_obs_from_history(h_dict)
            if p_now.size == 0:
                dmin = np.inf
            else:
                dmin = float(min_dist_robot_to_peds(robot_xy, p_now))
            is_coll = (dmin < safe_thresh)
            collisions.append(1 if is_coll else 0)

            # step
            robot_xy, robot_th = unicycle_step(robot_xy, robot_th, v, w, float(dt))
            traj.append(robot_xy.copy())

        traj_arr = np.stack(traj, axis=0).astype(np.float32)
        traj_out.append(traj_arr)

        # pack metrics in the format your main runner expects
        metric_dict[int(scene_begin)] = {
            "collisions": np.asarray(collisions, dtype=np.int32),
            "infeasible": np.asarray(infeasible, dtype=np.int32),
            "costs": np.asarray(costs, dtype=np.float32),
            "exit_time": int(len(collisions)),  # steps executed
            "timing_ctrl_ms": np.asarray(ctrl_times_ms, dtype=np.float32),
            "timing_loop_ms": np.asarray(loop_times_ms, dtype=np.float32),
        }

    # to keep parity: return a single ndarray. easiest: pad to max length
    maxL = max(t.shape[0] for t in traj_out) if traj_out else 0
    traj_pad = np.zeros((len(traj_out), maxL, 2), dtype=np.float32)
    for i, t in enumerate(traj_out):
        traj_pad[i, : t.shape[0], :] = t

    return metric_dict, traj_pad
