from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import rerun as rr

from quad_env import QuadWorldEnv3D
from controllers.cp_3d_mpc import ConformalController3D

ROBOT_RAD = 0.1
OBSTACLE_RAD = 0.2

MAX_LINEAR_VEL = 3.0
MAX_VZ = 0.7
MAX_ANGULAR_Z = 0.7
MIN_ANGULAR_Z = -0.7


def rr_set_step_time(k: int, timeline: str = "step") -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence(timeline, k)
    elif hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds(timeline, float(k))
    elif hasattr(rr, "set_time_nanos"):
        rr.set_time_nanos(timeline, int(k))


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
    return np.asarray([traj[-1] for traj in h_dict.values()], dtype=np.float32)

def _min_dist_robot_to_points(robot_xyz: np.ndarray, pts_xyz: np.ndarray) -> float:
    if pts_xyz.size == 0:
        return float("inf")
    d = np.linalg.norm(pts_xyz - robot_xyz[None, :], axis=1)
    return float(np.min(d))


def run_one_episode_rerun_simple(
    env: QuadWorldEnv3D,
    *,
    time_horizon: int = 12,
    n_skip: int = 4,
    n_paths: int = 2000,
    seed: int = 0,
    max_steps: int = 250,
    goal_finish_dist: float = 0.3,
    log_pred: bool = False,
    pred_view_idx: int = 0,
    save_rrd: bool = False,
    rrd_path: str = "quad_cc_3d_simple.rrd",
    visualize: bool = True,
    save_traj_img: bool = False,
    traj_img_path: str = "traj_3d/cc.png",
    method_name: str = "CC-MPC",
    break_on_collision: bool = False,
):
    safe_rad = ROBOT_RAD + OBSTACLE_RAD

    ctrl = ConformalController3D(
        n_steps=time_horizon,
        dt=env.dt,
        n_skip=n_skip,
        n_paths=n_paths,
        seed=seed,
        v_xy_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL),
        vz_lim=(-MAX_VZ, MAX_VZ),
        yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z),
        robot_rad=ROBOT_RAD,
        obstacle_rad=OBSTACLE_RAD,
        w_terminal=10.0,
        w_intermediate=1.0,
        w_control=0.001,
        use_dynamic=True,
    )

    # ---- rerun init ----
    if visualize:
        rr.init("quad_cc_3d_simple", spawn=(not save_rrd))
        if save_rrd:
            rr.save(rrd_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    obs = env.reset()
    goal = np.asarray(obs.get("goal_xyz", [0, 0, 0]), dtype=np.float32).reshape(3,)

    if visualize:
        rr.log("world/goal", rr.Points3D(goal.reshape(1, 3), radii=0.10, colors=[0, 0, 0]), static=True)

    # ---- stats (match your example style) ----
    n_collisions = 0
    n_infeasible = 0
    total_frames = 0
    reached_goal = False

    timing_ctrl_ms: List[float] = []
    timing_step_ms: List[float] = []
    timing_loop_ms: List[float] = []

    robot_traj: List[np.ndarray] = []
    last_obs_now = np.zeros((0, 3), dtype=np.float32)
    vx_global, vy_global, vz_global, yaw_rate = 0.0, 0.0, 0.0, 0.0

    for k in range(max_steps):
        t_loop0 = time.perf_counter()

        if visualize:
            rr_set_step_time(k, "step")

        robot = np.asarray(obs["robot_xyz"], dtype=np.float32).reshape(3,)
        yaw = float(obs["robot_yaw"])
        robot_traj.append(robot.copy())

        # ---- stop condition (same timing as your example: check before stepping) ----
        if np.linalg.norm(robot - goal) <= goal_finish_dist:
            reached_goal = True
            break

        if break_on_collision and robot[2] <= env.zlim[0] + ROBOT_RAD:
            break

        # ---- collision counting (same as your example) ----
        obs_now = _get_obs_positions_from_history(obs)
        if obs_now.size:
            last_obs_now = obs_now
        dmin_now = _min_dist_robot_to_points(robot, obs_now) if obs_now.size else float("inf")
        if dmin_now < safe_rad:
            n_collisions += 1

        # ---- pred (needed for controller; optional for visualization) ----
        pred, pred_mask = stack_pred3d_from_p_dict(obs.get("prediction", {}), horizon=time_horizon)

        # ---- controller timing ----
        t0 = time.perf_counter()
        act, info = ctrl(
            robot_xyz=robot,
            robot_yaw=yaw,
            goal_xyz=goal,
            pred_xyz=pred,
            pred_mask=pred_mask,
            boxes_3d=None,
        )
        t1 = time.perf_counter()

        is_feasible = bool(info.get("feasible", False))
        if not is_feasible:
            n_infeasible += 1

        # ---- choose action (match your example’s behavior when act is None) ----
        if act is None:
            target_pos = robot.copy()
            vx_global, vy_global, vz_global, yaw_rate = 0.0, 0.0, 0.0, 0.0
        else:
            target_pos, target_vel = act
            target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3,)
            target_vel = np.asarray(target_vel, dtype=np.float32).reshape(4,)
            vx_global, vy_global, vz_global, yaw_rate = map(float, target_vel)

        # ---- env.step timing ----
        t2 = time.perf_counter()
        obs = env.step(target_pos, (vx_global, vy_global, vz_global, yaw_rate))
        t3 = time.perf_counter()

        ctrl_ms = (t1 - t0) * 1000.0
        step_ms = (t3 - t2) * 1000.0
        loop_ms = (t3 - t_loop0) * 1000.0
        timing_ctrl_ms.append(ctrl_ms)
        timing_step_ms.append(step_ms)
        timing_loop_ms.append(loop_ms)

        # ---- update total_frames exactly like "len(episode_history)" would ----
        total_frames += 1

        if visualize:
        # ---- rerun logging: drone + obstacles + traj only ----
            tr = np.asarray(robot_traj, dtype=np.float32)

            rr.log("world/robot", rr.Points3D(robot.reshape(1, 3), radii=ROBOT_RAD, colors=[255, 217, 0]))
            if tr.shape[0] >= 2:
                rr.log("world/robot/traj", rr.LineStrips3D([tr], radii=ROBOT_RAD * 0.6, colors=[255, 217, 0]))

            if obs_now.size:
                rr.log("world/obstacles/now", rr.Points3D(obs_now, radii=OBSTACLE_RAD, colors=[30, 30, 30]))
            else:
                rr.log("world/obstacles/now", rr.Clear(recursive=True))

            if log_pred:
                if pred.size:
                    i = int(np.clip(pred_view_idx, 0, pred.shape[0] - 1))
                    pts = pred[i][pred_mask[i]]
                    if pts.size:
                        rr.log("world/obstacles/pred", rr.Points3D(pts, radii=OBSTACLE_RAD, colors=[220, 60, 60]))
                    else:
                        rr.log("world/obstacles/pred", rr.Clear(recursive=True))
                else:
                    rr.log("world/obstacles/pred", rr.Clear(recursive=True))

            rr.log(
                "world/status",
                rr.TextLog(
                    f"step={k} feasible={is_feasible} "
                    f"collisions={n_collisions} infeasible={n_infeasible} "
                    f"ctrl_ms={ctrl_ms:.2f} step_ms={step_ms:.2f}"
                ),
            )


    if visualize and len(timing_ctrl_ms) > 0:
        print(f"{n_collisions} collisions, {n_infeasible} infeasibility {total_frames} step")
        def _summ(arr: List[float], name: str):
            a = np.asarray(arr, dtype=np.float64)
            p50, p90, p99 = np.percentile(a, [50, 90, 99])
            print(
                f"[timing] {name}: mean={a.mean():.3f} ms | "
                f"p50={p50:.3f} | p90={p90:.3f} | p99={p99:.3f} | max={a.max():.3f}"
            )

        print("\n==== Online compute timing (ms) ====")
        _summ(timing_ctrl_ms, "controller")
        _summ(timing_step_ms, "env.step (physics)")
        _summ(timing_loop_ms, "total loop")
        print("===================================\n")

    robot_traj_arr = np.asarray(robot_traj, dtype=np.float32).reshape(-1, 3)

    # ---- static trajectory image (headless, no Rerun needed) ----
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

    return {
        "reached_goal": bool(reached_goal),
        "steps": int(total_frames),
        "collisions": int(n_collisions),
        "infeasible_steps": int(n_infeasible),
        "ctrl_times_ms": list(timing_ctrl_ms),
        "loop_times_ms": list(timing_loop_ms),
        "robot_traj": robot_traj_arr.tolist(),
    }

if __name__ == "__main__":
    env = QuadWorldEnv3D(
        dt=0.1,
        horizon=20,
        n_obs=280,
        world_bounds_xyz=((-3, 7), (-3, 7), (0.0, 8.0)),
        seed=8,
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

    run_one_episode_rerun_simple(
        env,
        time_horizon=12,
        n_skip=4,
        max_steps=250,
        goal_finish_dist=0.3,
        log_pred=True,    
        save_rrd=False,    
        visualize=True,
    )