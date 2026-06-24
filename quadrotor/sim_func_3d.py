from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
from skimage import measure

# ---- 3D env/utils ----
from quad_env import QuadWorldEnv3D
from quad_env import (
    build_grid_3d,
    distance_field_points_3d,
)

from cp.functional_cp import get_envelopes_value_and_function
from controllers.func_3d_mpc import FunctionalCPMPC3D

import rerun as rr


DT = 0.4
ROBOT_RAD = 0.1
OBSTACLE_RAD = 0.2

MAX_LINEAR_VEL = 3.0
MAX_ANGULAR_Z = 0.7
MIN_ANGULAR_Z = -0.7
MAX_VZ = 0.7

def rr_set_step_time(k: int, timeline: str = "step") -> None:
    """
    Rerun version compatibility wrapper.
    Prefer sequence timeline if available; fallback to seconds timeline.
    """
    # Newer rerun:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence(timeline, k)
        return

    # Some versions expose set_time_seconds:
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds(timeline, float(k))
        return

    # Some versions expose set_time_nanos:
    if hasattr(rr, "set_time_nanos"):
        rr.set_time_nanos(timeline, int(k))
        return

    # Worst-case: no timeline support; do nothing (logs still appear, just not time-scrubbable)
    return

# =============================================================================
# Data collection: residuals for CP calibration
# =============================================================================
def build_training_residuals_from_env_3d(
    env,
    n_samples,
    X,
    Y,
    Z,
    time_horizon,
    episode_len=60,
    group_by_episode=True,
    v_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL),
    yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z),
    vz_lim=(-MAX_VZ, MAX_VZ),
):
    residuals = []
    groups = []

    ep_idx = 0
    obs = env.reset()

    while len(residuals) < n_samples:
        p_dict = obs.get("prediction", {})
        f_dict = obs.get("future", {})

        pids = [pid for pid in p_dict.keys() if pid in f_dict]

        if len(pids) > 0:
            res_t = np.zeros((time_horizon, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
            valid = True

            for h in range(time_horizon):
                pts_pred, pts_true = [], []
                for pid in pids:
                    pr = p_dict[pid]
                    gt = f_dict[pid]
                    if h < len(pr) and h < len(gt):
                        pts_pred.append(pr[h])
                        pts_true.append(gt[h])

                if not pts_pred:
                    valid = False
                    break

                pts_pred = np.array(pts_pred)
                pts_true = np.array(pts_true)

                sdf_pred = distance_field_points_3d(pts_pred, X, Y, Z)
                sdf_true = distance_field_points_3d(pts_true, X, Y, Z)
                res_t[h] = sdf_pred - sdf_true  # Residual: Pred - True

            if valid:
                residuals.append(res_t)
                groups.append(ep_idx if group_by_episode else (len(residuals) - 1))

        # 2) Random Step
        robot_xyz = obs["robot_xyz"]

        vx = np.random.uniform(v_lim[0], v_lim[1])
        vy = np.random.uniform(v_lim[0], v_lim[1])
        w = np.random.uniform(yaw_rate_lim[0], yaw_rate_lim[1])
        vz = np.random.uniform(vz_lim[0], vz_lim[1])

        target_pos = robot_xyz + np.array([vx, vy, vz], dtype=np.float32) * float(env.dt)
        target_pos = np.array(
            [
                np.clip(target_pos[0], env.xlim[0], env.xlim[1]),
                np.clip(target_pos[1], env.ylim[0], env.ylim[1]),
                np.clip(target_pos[2], env.zlim[0], env.zlim[1]),
            ],
            dtype=np.float32,
        )

        cmd = (vx, vy, vz, w)  # w = yaw_rate
        obs = env.step(target_pos, cmd)

        if env.step_counter % episode_len == 0:
            ep_idx += 1
            obs = env.reset()

    return np.stack(residuals), np.array(groups)


# =============================================================================
# Helper: prediction dict -> (H, M, 3) + mask
# =============================================================================
def stack_pred3d_from_p_dict(p_dict: Dict[int, np.ndarray], horizon: int):
    pids = list(p_dict.keys())
    M = len(pids)
    Hh = int(horizon)

    pred = np.zeros((Hh, M, 3), dtype=np.float32)
    mask = np.zeros((Hh, M), dtype=bool)

    for j, pid in enumerate(pids):
        arr = np.asarray(p_dict[pid], dtype=np.float32)
        take = min(Hh, arr.shape[0])
        if take > 0:
            pred[:take, j, :] = arr[:take, :]
            mask[:take, j] = True
    return pred, mask, pids


def _deepcopy_obs_dict(obs: Dict) -> Dict:
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, dict):
            out[k] = _deepcopy_obs_dict(v)
        elif isinstance(v, list):
            out[k] = [vv.copy() if isinstance(vv, np.ndarray) else vv for vv in v]
        else:
            out[k] = v
    return out


def _get_obs_positions_from_history(obs_dict) -> np.ndarray:
    """obs['history']에서 현재(가장 최근) 장애물 위치들을 추출"""
    h_dict = obs_dict.get("history", {})
    if not h_dict:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray([traj[-1] for traj in h_dict.values()], dtype=np.float32)

def _min_dist_robot_to_points(robot_xyz: np.ndarray, pts_xyz: np.ndarray) -> float:
    if pts_xyz.size == 0:
        return float("inf")
    d = np.linalg.norm(pts_xyz - robot_xyz[None, :], axis=1)
    return float(np.min(d))


# =============================================================================
# Rerun mesh helper: bool volume -> (V,F)
# =============================================================================
def mesh_from_bool_zyx(vol_bool_zyx: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    """
    vol_bool_zyx: (nz, ny, nx) boolean
    returns:
      V: (Nv, 3) float32 world coords
      F: (Nf, 3) int32 triangle indices
    """
    if vol_bool_zyx is None or vol_bool_zyx.ndim != 3 or not np.any(vol_bool_zyx):
        return None, None

    vol = vol_bool_zyx.astype(np.float32)
    try:
        verts_zyx, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    except Exception:
        return None, None

    # marching_cubes verts are in (z, y, x) index space
    vz_idx, vy_idx, vx_idx = verts_zyx[:, 0], verts_zyx[:, 1], verts_zyx[:, 2]

    vx = np.interp(vx_idx, np.arange(xs.size), xs)
    vy = np.interp(vy_idx, np.arange(ys.size), ys)
    vz = np.interp(vz_idx, np.arange(zs.size), zs)

    V = np.stack([vx, vy, vz], axis=1).astype(np.float32)
    F = faces.astype(np.int32)
    return V, F


# =============================================================================
# Rerun visualization
# =============================================================================

def rr_log_boxes(
    path: str,
    boxes: List[List[float]],
    color=(120, 120, 120, 80),
):
    centers = []
    half_sizes = []

    for b in boxes:
        x0, x1, y0, y1, z0, z1 = b
        centers.append([
            0.5 * (x0 + x1),
            0.5 * (y0 + y1),
            0.5 * (z0 + z1),
        ])
        half_sizes.append([
            0.5 * (x1 - x0),
            0.5 * (y1 - y0),
            0.5 * (z1 - z0),
        ])

    rr.log(
        path,
        rr.Boxes3D(
            centers=np.asarray(centers, dtype=np.float32),
            half_sizes=np.asarray(half_sizes, dtype=np.float32),
            colors=[color],
        ),
        static=True,
    )

def visualize_with_rerun(
    *,
    episode_history: List[Dict],
    goal: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    g_upper_grid: np.ndarray,
    safe_rad: float,
    nx: int,
    ny: int,
    nz: int,
    i_view: int,
    mc_stride: int,
    CP: bool,
    save_rrd: bool,
    rrd_path: str,
    only_log_every: int = 1, 
):
    # init rerun
    rr.init("cp_mpc_3d", spawn=(not save_rrd))
    if save_rrd:
        rr.save(rrd_path) 

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/goal", rr.Points3D(goal.reshape(1, 3), radii=0.08, colors=[0, 0, 0]), static=True)
    start = np.asarray(episode_history[0]["robot"], dtype=np.float32).reshape(3,)

    rr.log(
        "world/start",
        rr.Points3D(
            start.reshape(1, 3),
            radii=0.12,
            colors=[30, 144, 255],  
        ),
        static=True,
    )

    robot_traj: List[np.ndarray] = []

    total_frames = len(episode_history)
    for k in range(total_frames):
        if only_log_every > 1 and (k % only_log_every != 0) and (k != total_frames - 1):
            continue

        rr_set_step_time(k, timeline="step")
        data = episode_history[k]
        obs_k = data["obs"]
        robot = np.asarray(data["robot"], dtype=np.float32).reshape(3,)
        robot_traj.append(robot.copy())
        tr = np.asarray(robot_traj, dtype=np.float32)

        # obstacles now
        obs_now = _get_obs_positions_from_history(obs_k)

        # prediction (i_view)
        pred_full = np.asarray(data["pred"], dtype=np.float32)          # (H,M,3)
        pred_mask_full = np.asarray(data["pred_mask"], dtype=bool)      # (H,M)

        if pred_full.size:
            i_view_clamped = int(np.clip(i_view, 0, pred_full.shape[0] - 1))
            pred_i = pred_full[i_view_clamped]
            mask_i = pred_mask_full[i_view_clamped]
            pred_i_vis = pred_i[mask_i] if pred_i.size else np.zeros((0, 3), dtype=np.float32)
        else:
            i_view_clamped = 0
            pred_i_vis = np.zeros((0, 3), dtype=np.float32)

        # GT future (k + i_view + 1)
        future_idx = k + (i_view_clamped + 1)
        gt_i_vis = np.zeros((0, 3), dtype=np.float32)
        if future_idx < total_frames:
            gt_i_vis = _get_obs_positions_from_history(episode_history[future_idx]["obs"])

        # ---- distance fields -> unsafe volumes ----
        D_pred = (
            distance_field_points_3d(pred_i_vis, X, Y, Z)
            if pred_i_vis.size else np.full((nz, ny, nx), np.inf, dtype=np.float32)
        )

        U_i = g_upper_grid[i_view_clamped] if CP else 0.0
        U_i = np.maximum(U_i, 0.0)
        D_lower = np.maximum(D_pred - U_i, 0.0)
        cp_unsafe = (D_lower < safe_rad)  # (nz,ny,nx)

        true_unsafe = None
        if gt_i_vis.size:
            D_true = distance_field_points_3d(gt_i_vis, X, Y, Z)
            true_unsafe = (D_true < safe_rad)

        # ---- downsample for meshing ----
        s = int(max(1, mc_stride))
        cp_ds = cp_unsafe[::s, ::s, ::s]
        xs_ds = xs[np.arange(0, xs.size, s)]
        ys_ds = ys[np.arange(0, ys.size, s)]
        zs_ds = zs[np.arange(0, zs.size, s)]

        Vcp, Fcp = mesh_from_bool_zyx(cp_ds, xs_ds, ys_ds, zs_ds)
        if Vcp is not None:
            rr.log(
                "world/unsafe/cp",
                rr.Mesh3D(
                    vertex_positions=Vcp,
                    triangle_indices=Fcp,
                    albedo_factor=[220, 60, 60, 70],
                ),
            )
        else:
            rr.log("world/unsafe/cp", rr.Clear(recursive=True))

        if true_unsafe is not None:
            true_ds = true_unsafe[::s, ::s, ::s]
            Vt, Ft = mesh_from_bool_zyx(true_ds, xs_ds, ys_ds, zs_ds)
            if Vt is not None:
                rr.log(
                    "world/unsafe/true",
                    rr.Mesh3D(
                        vertex_positions=Vt,
                        triangle_indices=Ft,
                        albedo_factor=[60, 90, 240, 80],
                    ),
                )
            else:
                rr.log("world/unsafe/true", rr.Clear(recursive=True))
        else:
            rr.log("world/unsafe/true", rr.Clear(recursive=True))

        # ---- robot + traj ----
        rr.log("world/robot", rr.Points3D(robot.reshape(1, 3), radii=ROBOT_RAD, colors=[255, 217, 0]))
        if tr.shape[0] >= 2:
            rr.log("world/robot/traj", rr.LineStrips3D([tr], radii=ROBOT_RAD, colors=[255, 217, 0]))

        # ---- obstacle points ----
        if obs_now.size:
            rr.log("world/obstacles/now", rr.Points3D(obs_now, radii=OBSTACLE_RAD, colors=[30, 30, 30]))
        else:
            rr.log("world/obstacles/now", rr.Clear(recursive=True))

        if pred_i_vis.size:
            rr.log("world/obstacles/pred", rr.Points3D(pred_i_vis, radii=OBSTACLE_RAD, colors=[220, 60, 60]))
        else:
            rr.log("world/obstacles/pred", rr.Clear(recursive=True))

        if gt_i_vis.size:
            rr.log("world/obstacles/gt", rr.Points3D(gt_i_vis, radii=OBSTACLE_RAD, colors=[60, 90, 240]))
        else:
            rr.log("world/obstacles/gt", rr.Clear(recursive=True))

        # ---- small status text ----
        rr.log(
            "world/status",
            rr.TextLog(
                "\n".join([
                    f"step={k}/{total_frames-1}   i_view={i_view_clamped}",
                    f"feasible={bool(data.get('feasible', False))}",
                    f"CP={CP}",
                ])
            ),
        )


# =============================================================================
# Main: run simulation then visualize with rerun
# =============================================================================
def run_one_episode_visual_3d(
    env: QuadWorldEnv3D,
    *,
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    time_horizon: int = 12,
    alpha: float = 0.10,
    p_base: int = 5,
    k_mix: int = 5,
    test_size: float = 0.30,
    random_state: int = 0,
    n_jobs: int = 4,
    backend: str = "loky",
    n_skip: int = 4,
    n_paths: int = 600,
    max_steps: int = 2500,
    i_view: int = 3,
    n_calib_samples: int = 120,
    goal_finish_dist: float = 0.8,
    mc_stride: int = 2,
    CP: bool = False,
    save_rrd: bool = False,
    rrd_path: str = "cp_mpc_3d.rrd",
    only_log_every: int = 1,
    visualize: bool = False,
    save_traj_img: bool = False,
    traj_img_path: str = "traj_3d/fcp.png",
    method_name: str = "FCP-MPC",
    break_on_collision: bool = False,
    capture_history: bool = False,
    return_fields: bool = False,
    safety_mode: str = "hard",
    w_safety: float = 300.0,
):
    safe_rad = ROBOT_RAD + OBSTACLE_RAD

    # ----------------------------
    # 1) Grid & CP Setup
    # ----------------------------
    xlim, ylim, zlim = env.xlim, env.ylim, env.zlim
    xs, ys, zs, X, Y, Z = build_grid_3d(xlim, ylim, zlim, nx, ny, nz)

    margin = 5.0  # wall thickness
    cov_min, cov_max = -50.0, 50.0
    x_min, x_max = xlim
    y_min, y_max = ylim
    z_min, z_max = zlim

    wall_boxes = [
        [x_min - margin, x_min, cov_min, cov_max, cov_min, cov_max],
        [x_max, x_max + margin, cov_min, cov_max, cov_min, cov_max],
        [cov_min, cov_max, y_min - margin, y_min, cov_min, cov_max],
        [cov_min, cov_max, y_max, y_max + margin, cov_min, cov_max],
        [cov_min, cov_max, cov_min, cov_max, z_min - margin, z_min],
        [cov_min, cov_max, cov_min, cov_max, z_max, z_max + margin],
    ]

    if n_calib_samples > 0:
        residuals, _ = build_training_residuals_from_env_3d(
            env,
            n_samples=n_calib_samples,
            X=X,
            Y=Y,
            Z=Z,
            time_horizon=time_horizon,
            episode_len=1000,
            group_by_episode=False,
            v_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL),
            yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z),
            vz_lim=(-MAX_VZ, MAX_VZ),
        )
    else:
        residuals = np.zeros((0, time_horizon, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)

    g_upper_grid = None
    cp_params = None

    if CP and residuals.shape[0] > 0:
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

    # ----------------------------
    # 2) Controller Setup
    # ----------------------------
    ctrl = FunctionalCPMPC3D(
        cp_params=cp_params,
        xs=xs,
        ys=ys,
        zs=zs,
        n_steps=time_horizon,
        dt=env.dt,
        n_skip=n_skip,
        robot_rad=ROBOT_RAD,
        obstacle_rad=OBSTACLE_RAD,
        v_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL),
        yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z),
        vz_lim=(-MAX_VZ, MAX_VZ),
        n_paths=n_paths,
        seed=0,
        CP=CP,
        safety_mode=safety_mode,
        w_safety=w_safety,
    )

    # ----------------------------
    # 3) Record Phase (Run Simulation First)
    # ----------------------------
    print(f"Running simulation for max {max_steps} steps to record Ground Truth...")
    obs = env.reset()
    goal = np.asarray(obs.get("goal_xyz", [0, 0, 0]), dtype=np.float32).reshape(3,)

    episode_history: List[Dict] = [] if (visualize or capture_history) else None
    timing = {"ctrl_ms": [], "step_ms": [], "loop_ms": []}
    vx_global, vy_global, vz_global, yaw_rate = 0.0, 0.0, 0.0, 0.0

    n_collisions = 0
    n_infeasible = 0
    reached_goal = False
    steps = 0

    robot_traj: List[np.ndarray] = []        # always recorded for the static traj image
    last_obs_now = np.zeros((0, 3), dtype=np.float32)

    for k in range(max_steps):
        t_loop0 = time.perf_counter()

        robot = np.asarray(obs["robot_xyz"], dtype=np.float32).reshape(3,)
        yaw = float(obs["robot_yaw"])
        robot_traj.append(robot.copy())

        if np.linalg.norm(robot - goal) <= goal_finish_dist:
            reached_goal = True
            break

        if break_on_collision and robot[2] <= z_min + ROBOT_RAD:
            break

        obs_now = _get_obs_positions_from_history(obs)                           # current obstacles (after step)
        if obs_now.size:
            last_obs_now = obs_now

        dmin_now = _min_dist_robot_to_points(robot, obs_now) if obs_now.size else float("inf")
        if dmin_now < safe_rad:
            n_collisions += 1

        pred, pred_mask, _ = stack_pred3d_from_p_dict(obs.get("prediction", {}), horizon=time_horizon)

        t0 = time.perf_counter()
        act, info = ctrl(
            robot_xyz=robot,
            robot_yaw=yaw,
            goal_xyz=goal,
            pred_xyz=pred,
            pred_mask=pred_mask,
            boxes_3d=wall_boxes,
            robot_vel=(vx_global, vy_global, vz_global),
            observed_xyz=obs_now,
        )
        t1 = time.perf_counter()

        if act is None:
            target = robot.copy()
            vx_global, vy_global, vz_global, yaw_rate = 0.0, 0.0, 0.0, 0.0
            act_to_store = None
        else:
            pos, vel = act
            target = np.asarray(pos, dtype=np.float32).reshape(3,)
            vel = np.asarray(vel, dtype=np.float32).reshape(4,)
            vx_global, vy_global, vz_global, yaw_rate = map(float, vel)
            act_to_store = (target.copy(), vel.copy())

        cmd = (vx_global, vy_global, vz_global, yaw_rate)
        is_feasible = bool(info.get("feasible", False))

        if not is_feasible:
            n_infeasible += 1

        t2 = time.perf_counter()
        obs = env.step(target, cmd)
        t3 = time.perf_counter()

        ctrl_ms = (t1 - t0) * 1000.0
        step_ms = (t3 - t2) * 1000.0
        loop_ms = (t3 - t_loop0) * 1000.0

        timing["ctrl_ms"].append(ctrl_ms)
        timing["step_ms"].append(step_ms)
        timing["loop_ms"].append(loop_ms)

        if visualize or capture_history:

            episode_history.append(
                {
                    "step": k,
                    "obs": _deepcopy_obs_dict(obs),
                    "robot": robot,
                    "yaw": yaw,
                    "act": act_to_store,
                    "feasible": bool(is_feasible),
                    "pred": pred,
                    "pred_mask": pred_mask,
                    "timing": {"ctrl_ms": ctrl_ms, "step_ms": step_ms, "loop_ms": loop_ms},
                }
            )
        steps +=1

    total_frames = steps

    if visualize:

        print(f"{n_collisions} collisions, {n_infeasible} infeasibility {total_frames} step")

        def _summ(x: List[float], name: str):
            arr = np.asarray(x, dtype=np.float64)
            if arr.size == 0:
                print(f"[timing] {name}: no data")
                return
            p50, p90, p99 = np.percentile(arr, [50, 90, 99])
            print(
                f"[timing] {name}: mean={arr.mean():.3f} ms | "
                f"p50={p50:.3f} | p90={p90:.3f} | p99={p99:.3f} | max={arr.max():.3f}"
            )

        print("\n==== Online compute timing (ms) ====")
        _summ(timing["ctrl_ms"], "controller (FunctionalCPMPC3D)")
        _summ(timing["step_ms"], "env.step (physics)")
        _summ(timing["loop_ms"], "total loop")
        print("===================================\n")
        print(f"Simulation finished. Total frames: {total_frames}. Starting Rerun Visualization...")

    # ----------------------------
    # 4) Rerun Visualization (Replay)
    # ----------------------------

        visualize_with_rerun(
            episode_history=episode_history,
            goal=goal,
            xs=xs,
            ys=ys,
            zs=zs,
            X=X,
            Y=Y,
            Z=Z,
            g_upper_grid=g_upper_grid,
            safe_rad=safe_rad,
            nx=nx,
            ny=ny,
            nz=nz,
            i_view=i_view,
            mc_stride=mc_stride,
            CP=CP,
            save_rrd=save_rrd,
            rrd_path=rrd_path,
            only_log_every=only_log_every,
        )

    robot_traj_arr = np.asarray(robot_traj, dtype=np.float32).reshape(-1, 3)

    # ----------------------------
    # 5) Static trajectory image (headless, no Rerun needed)
    # ----------------------------
    if save_traj_img:
        from viz_traj import save_traj_image_3d
        title = (f"{method_name} | steps={total_frames} coll={n_collisions} "
                 f"infeas={n_infeasible} reached={reached_goal}")
        saved = save_traj_image_3d(
            robot_traj=robot_traj_arr,
            goal=goal,
            start=robot_traj_arr[0] if robot_traj_arr.size else None,
            obstacles=last_obs_now,
            bounds=(env.xlim, env.ylim, env.zlim),
            title=title,
            out_path=traj_img_path,
        )
        if saved:
            print(f"[traj-img] saved {saved}")

    out = {
        "reached_goal": bool(reached_goal),
        "steps": int(total_frames),
        "collisions": int(n_collisions),
        "infeasible_steps": int(n_infeasible),
        "ctrl_times_ms": list(timing["ctrl_ms"]),
        "loop_times_ms": list(timing["loop_ms"]),
        "robot_traj": robot_traj_arr.tolist(),
    }
    if return_fields:
        # Extra payload for offline figure generation (e.g. the Fig.4 conformal
        # safety-bound illustration). Only populated on demand; the controller loop
        # and metrics above are unaffected.
        out["fields"] = {
            "episode_history": episode_history,
            "g_upper_grid": g_upper_grid,
            "xs": xs, "ys": ys, "zs": zs,
            "X": X, "Y": Y, "Z": Z,
            "safe_rad": float(safe_rad),
            "goal": np.asarray(goal, dtype=np.float32),
            "robot_rad": float(ROBOT_RAD),
            "obstacle_rad": float(OBSTACLE_RAD),
        }
    return out


if __name__ == "__main__":
    env = QuadWorldEnv3D(
        dt=0.1,
        horizon=20,
        n_obs=280,
        world_bounds_xyz=((-3, 7), (-3, 7), (0.0, 8.0)),
        seed=2,

        pred_model_noise=0.20,

        obs_process_noise=0.22,
        gt_future_noise=0.20,

        mode_switch_p=0.95,
        mode_min_ttl=1,
        mode_max_ttl=6,
        turn_rate_std=3.0,
        stop_go_p=0.6,
        gui = False
    )

    run_one_episode_visual_3d(
        env,
        nx=40,
        ny=40,
        nz=40,
        time_horizon=12,
        alpha=0.10,
        p_base=5,
        k_mix=10,
        test_size=0.30,
        random_state=0,
        n_jobs=max(1, (os.cpu_count() or 4) - 2),
        backend="loky",
        n_skip=4,
        n_paths=2000,
        max_steps=250,
        i_view=0,
        n_calib_samples=120,
        goal_finish_dist=0.3,
        mc_stride=2,
        CP=True,
        save_rrd=False,
        rrd_path="cp_mpc_3d.rrd",
        only_log_every=1,
        visualize=True,
    )
