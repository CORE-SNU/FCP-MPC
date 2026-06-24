from __future__ import annotations

import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import rerun as rr

from quad_env import QuadWorldEnv3D
from controllers.ecp_mpc_3d import EgocentricCPMPC3D, MPC3DWeights

ROBOT_RAD = 0.1
OBSTACLE_RAD = 0.2

MAX_LINEAR_VEL = 3.0
MAX_VZ = 0.7
MAX_ANGULAR_Z = 0.7
MIN_ANGULAR_Z = -0.7


# ----------------------------
# Rerun helper
# ----------------------------
def rr_set_step_time(k: int, timeline: str = "step") -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence(timeline, k)
    elif hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds(timeline, float(k))
    elif hasattr(rr, "set_time_nanos"):
        rr.set_time_nanos(timeline, int(k))


# ----------------------------
# Helpers
# ----------------------------
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
    return pred, mask


def _get_obs_positions_from_history(obs_dict) -> np.ndarray:
    h_dict = obs_dict.get("history", {})
    if not h_dict:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray([np.asarray(traj, dtype=np.float32)[-1] for traj in h_dict.values()], dtype=np.float32)


def _min_dist_robot_to_points(robot_xyz: np.ndarray, pts_xyz: np.ndarray) -> float:
    if pts_xyz.size == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(pts_xyz - robot_xyz[None, :], axis=1)))


def _make_wall_boxes(xlim, ylim, zlim):
    margin = 5.0
    cov_min, cov_max = -50.0, 50.0
    x_min, x_max = xlim
    y_min, y_max = ylim
    z_min, z_max = zlim
    return [
        [x_min - margin, x_min, cov_min, cov_max, cov_min, cov_max],
        [x_max, x_max + margin, cov_min, cov_max, cov_min, cov_max],
        [cov_min, cov_max, y_min - margin, y_min, cov_min, cov_max],
        [cov_min, cov_max, y_max, y_max + margin, cov_min, cov_max],
        [cov_min, cov_max, cov_min, cov_max, z_min - margin, z_min],
        [cov_min, cov_max, cov_min, cov_max, z_max, z_max + margin],
    ]


# ----------------------------
# Main (ECP + Rerun + scalable)
# ----------------------------
def run_one_episode_ecp_3d_rerun(
    env: QuadWorldEnv3D,
    *,
    # controller params
    time_horizon: int = 12,
    n_skip: int = 4,
    robot_rad: float = ROBOT_RAD,
    obstacle_rad: float = OBSTACLE_RAD,
    v_lim: Tuple[float, float] = (-MAX_LINEAR_VEL, MAX_LINEAR_VEL),
    vz_lim: Tuple[float, float] = (-MAX_VZ, MAX_VZ),
    yaw_rate_lim: Tuple[float, float] = (MIN_ANGULAR_Z, MAX_ANGULAR_Z),
    v_points: Tuple[float, ...] = (-1.0, 0.0, 1.0),
    w_points: Tuple[float, ...] = (-1.0, 0.0, 1.0),
    vz_points: Tuple[float, ...] = (-1.0, 0.0, 1.0),
    n_paths: int = 2000,
    seed: int = 0,
    calibration_set_size: int = 15,
    miscoverage_level: float = 0.10,
    step_size: float = 0.05,
    weights: Optional[MPC3DWeights] = None,
    # sim params
    max_steps: int = 250,
    warmup_steps: int = 0,
    goal_finish_dist: float = 0.3,
    break_on_collision: bool = False,
    # rerun params
    visualize: bool = True,
    log_pred: bool = False,
    pred_view_idx: int = 0,
    only_log_every: int = 1,         # rerun log downsample
    save_rrd: bool = False,
    rrd_path: str = "quad_ecp_3d.rrd",
    # perf params
    collect_timing: bool = True,
    # static traj image
    save_traj_img: bool = False,
    traj_img_path: str = "traj_3d/ecp.png",
    method_name: str = "ECP-MPC",
):
    if weights is None:
        weights = MPC3DWeights(w_terminal=10.0, w_intermediate=1.0, w_control=0.001)

    safe_rad = float(robot_rad + obstacle_rad)

    ctrl = EgocentricCPMPC3D(
        n_steps=int(time_horizon),
        dt=float(env.dt),
        n_skip=int(n_skip),
        robot_rad=float(robot_rad),
        obstacle_rad=float(obstacle_rad),
        v_lim=tuple(map(float, v_lim)),
        vz_lim=tuple(map(float, vz_lim)),
        yaw_rate_lim=tuple(map(float, yaw_rate_lim)),
        v_points=tuple(map(float, v_points)),
        w_points=tuple(map(float, w_points)),
        vz_points=tuple(map(float, vz_points)),
        n_paths=int(n_paths),
        seed=int(seed),
        calibration_set_size=int(calibration_set_size),
        miscoverage_level=float(miscoverage_level),
        step_size=float(step_size),
        weights=weights,
    )

    wall_boxes = _make_wall_boxes(env.xlim, env.ylim, env.zlim)

    # ---- rerun init ----
    if visualize:
        rr.init("quad_ecp_3d", spawn=(not save_rrd))
        if save_rrd:
            rr.save(rrd_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    obs = env.reset()
    goal = np.asarray(obs.get("goal_xyz", [0, 0, 0]), dtype=np.float32).reshape(3,)

    if visualize:
        rr.log("world/goal", rr.Points3D(goal.reshape(1, 3), radii=0.10, colors=[0, 0, 0]), static=True)

    # stats
    n_collisions = 0
    n_infeasible = 0
    reached_goal = False
    steps = 0

    timing_ctrl_ms: List[float] = []
    timing_step_ms: List[float] = []
    timing_loop_ms: List[float] = []

    # for env.step cmd
    vx_global, vy_global, vz_global, yaw_rate = 0.0, 0.0, 0.0, 0.0

    # rerun trajectory
    robot_traj: List[np.ndarray] = []
    last_obs_now = np.zeros((0, 3), dtype=np.float32)

    for k in range(int(max_steps)):
        t_loop0 = time.perf_counter()

        if visualize and (only_log_every <= 1 or k % only_log_every == 0):
            rr_set_step_time(k, "step")

        robot = np.asarray(obs["robot_xyz"], dtype=np.float32).reshape(3,)
        yaw = float(obs["robot_yaw"])
        robot_vel = np.asarray(obs.get("robot_vel", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(3,)
        robot_traj.append(robot.copy())

        # stop
        if float(np.linalg.norm(robot - goal)) <= float(goal_finish_dist):
            reached_goal = True
            break

        if break_on_collision and robot[2] <= env.zlim[0] + ROBOT_RAD:
            break

        # collision (current)
        obs_now = _get_obs_positions_from_history(obs)
        if obs_now.size:
            last_obs_now = obs_now
        dmin_now = _min_dist_robot_to_points(robot, obs_now) if obs_now.size else float("inf")
        if dmin_now < safe_rad:
            n_collisions += 1

        # ECP uses history internally
        ctrl.update_observations(obs.get("history", {}))

        # pred for controller + optional vis
        p_dict = obs.get("prediction", {})
        pred_xyz, pred_mask = stack_pred3d_from_p_dict(p_dict, horizon=ctrl.n_steps)

        # controller timing
        t0 = time.perf_counter()
        act, info = ctrl(
            robot_xyz=robot,
            robot_yaw=yaw,
            goal_xyz=goal,
            pred_xyz=pred_xyz,
            pred_mask=pred_mask,
            boxes_3d=wall_boxes,
            robot_vel=robot_vel,
        )
        t1 = time.perf_counter()

        # keep your behavior
        ctrl.update_predictions(p_dict)

        feasible = bool(info.get("feasible", False)) if isinstance(info, dict) else (act is not None)
        feasible = feasible and (act is not None)

        # warmup / infeasible policy
        if k < int(warmup_steps):
            feasible = True
            target_pos = robot.copy()
            target_vel = np.zeros((4,), dtype=np.float32)
        else:
            if not feasible:
                n_infeasible += 1
                target_pos = robot.copy()
                target_vel = np.zeros((4,), dtype=np.float32)
            else:
                target_pos, target_vel = act
                target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3,)
                target_vel = np.asarray(target_vel, dtype=np.float32).reshape(4,)

        vx_global, vy_global, vz_global, yaw_rate = map(float, target_vel)

        # env.step timing
        t2 = time.perf_counter()
        obs = env.step(target_pos, (vx_global, vy_global, vz_global, yaw_rate))
        t3 = time.perf_counter()

        if collect_timing:
            ctrl_ms = (t1 - t0) * 1000.0
            step_ms = (t3 - t2) * 1000.0
            loop_ms = (t3 - t_loop0) * 1000.0
            # Exclude warmup/learning steps from the control-time benchmark: during warmup the
            # robot is stationary and ECP's online calibration set is still filling, so those
            # steps are cheap and unrepresentative of steady-state control cost. Counting them
            # biases the per-step timing downward for episodes that crash during warmup.
            if k >= int(warmup_steps):
                timing_ctrl_ms.append(ctrl_ms)
                timing_step_ms.append(step_ms)
                timing_loop_ms.append(loop_ms)
        else:
            ctrl_ms = step_ms = loop_ms = 0.0

        # ---- rerun logging (cheap) ----
        if visualize and (only_log_every <= 1 or k % only_log_every == 0):
            tr = np.asarray(robot_traj, dtype=np.float32)

            rr.log("world/robot", rr.Points3D(robot.reshape(1, 3), radii=robot_rad, colors=[255, 217, 0]))
            if tr.shape[0] >= 2:
                rr.log("world/robot/traj", rr.LineStrips3D([tr], radii=robot_rad * 0.6, colors=[255, 217, 0]))

            if obs_now.size:
                rr.log("world/obstacles/now", rr.Points3D(obs_now, radii=obstacle_rad, colors=[30, 30, 30]))
            else:
                rr.log("world/obstacles/now", rr.Clear(recursive=True))

            if log_pred:
                # visualize one horizon slice
                if pred_xyz.size:
                    i = int(np.clip(pred_view_idx, 0, pred_xyz.shape[0] - 1))
                    pts = pred_xyz[i][pred_mask[i]]
                    if pts.size:
                        rr.log("world/obstacles/pred", rr.Points3D(pts, radii=obstacle_rad, colors=[220, 60, 60]))
                    else:
                        rr.log("world/obstacles/pred", rr.Clear(recursive=True))
                else:
                    rr.log("world/obstacles/pred", rr.Clear(recursive=True))

            rr.log(
                "world/status",
                rr.TextLog(
                    "\n".join([
                        f"step={k} feasible={feasible}",
                        f"collisions={n_collisions} infeasible={n_infeasible}",
                        f"dmin_now={dmin_now:.3f} r_safe={safe_rad:.3f}",
                        f"ctrl_ms={ctrl_ms:.2f} step_ms={step_ms:.2f} loop_ms={loop_ms:.2f}" if collect_timing else "",
                    ])
                ),
            )

        steps += 1

    print(f"{n_collisions} collisions, {n_infeasible} infeasibility {steps} step")

    if collect_timing and len(timing_ctrl_ms) > 0:
        def _summ(arr: List[float], name: str):
            a = np.asarray(arr, dtype=np.float64)
            p50, p90, p99 = np.percentile(a, [50, 90, 99])
            print(
                f"[timing] {name}: mean={a.mean():.3f} ms | "
                f"p50={p50:.3f} | p90={p90:.3f} | p99={p99:.3f} | max={a.max():.3f}"
            )

        print("\n==== Online compute timing (ms) ====")
        _summ(timing_ctrl_ms, "controller (ECP)")
        _summ(timing_step_ms, "env.step (physics)")
        _summ(timing_loop_ms, "total loop")
        print("===================================\n")

    robot_traj_arr = np.asarray(robot_traj, dtype=np.float32).reshape(-1, 3)

    # ---- static trajectory image (headless, no Rerun needed) ----
    if save_traj_img:
        from viz_traj import save_traj_image_3d
        title = (f"{method_name} | steps={steps} coll={n_collisions} "
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

    return {
        "reached_goal": bool(reached_goal),
        "steps": int(steps),
        "collisions": int(n_collisions),
        "infeasible_steps": int(n_infeasible),
        "ctrl_times_ms": list(timing_ctrl_ms) if collect_timing else [],
        "loop_times_ms": list(timing_loop_ms) if collect_timing else [],
        "robot_traj": robot_traj_arr.tolist(),
    }


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    env = QuadWorldEnv3D(
        dt=0.1,
        horizon=20,
        n_obs=2,
        world_bounds_xyz=((-3.0, 7.0), (-3.0, 7.0), (0.0, 8.0)),
        seed=2,
        pred_model_noise=0.20,
        obs_process_noise=0.22,
        gt_future_noise=0.20,
        mode_switch_p=0.95,
        mode_min_ttl=1,
        mode_max_ttl=6,
        turn_rate_std=3.0,
        stop_go_p=0.6,
        gui=False,
    )

    weights = MPC3DWeights(w_terminal=10.0, w_intermediate=1.0, w_control=0.001)

    run_one_episode_ecp_3d_rerun(
        env,
        time_horizon=12,
        n_skip=4,
        max_steps=250,
        warmup_steps=15,
        goal_finish_dist=0.3,
        visualize=True,  
        log_pred=True,
        pred_view_idx=0,
        only_log_every=1,
        save_rrd=False,
        weights=weights,
        collect_timing=True,
    )