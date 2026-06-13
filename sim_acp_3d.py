"""3D quadrotor episode runner for ACP-MPC (Adaptive Conformal Prediction MPC).

Mirrors the closed-loop structure of ``sim_func_3d.run_one_episode_visual_3d`` so
that timing and metrics are directly comparable, but swaps in the holonomic
AdaptiveCPMPC3D controller. Registered in runner_3d.py as the ``acp`` method.
"""
from __future__ import annotations

import time
import numpy as np

from sim_func_3d import (
    stack_pred3d_from_p_dict,
    _get_obs_positions_from_history,
    _min_dist_robot_to_points,
)
from controllers.acp_3d_mpc import AdaptiveCPMPC3D

ROBOT_RAD = 0.1
OBSTACLE_RAD = 0.2
MAX_LINEAR_VEL = 3.0
MAX_VZ = 0.7
MAX_ANGULAR_Z = 0.7
MIN_ANGULAR_Z = -0.7


def run_one_episode_acp_3d(
    env,
    *,
    time_horizon: int = 12,
    n_skip: int = 4,
    n_paths: int = 2000,
    max_steps: int = 2500,
    goal_finish_dist: float = 0.8,
    target_miscoverage_level: float = 0.10,
    step_size: float = 0.05,
    seed: int = 0,
    break_on_collision: bool = False,
    **_ignore,
):
    safe_rad = ROBOT_RAD + OBSTACLE_RAD

    ctrl = AdaptiveCPMPC3D(
        n_steps=time_horizon,
        dt=env.dt,
        v_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL),
        vz_lim=(-MAX_VZ, MAX_VZ),
        yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z),
        n_skip=n_skip,
        robot_rad=ROBOT_RAD,
        obstacle_rad=OBSTACLE_RAD,
        n_paths=n_paths,
        seed=seed,
        target_miscoverage_level=target_miscoverage_level,
        step_size=step_size,
    )

    obs = env.reset()
    goal = np.asarray(obs.get("goal_xyz", [0, 0, 0]), dtype=np.float32).reshape(3)

    timing = {"ctrl_ms": [], "loop_ms": []}
    n_collisions = 0
    n_infeasible = 0
    reached_goal = False
    steps = 0
    robot_traj = []

    for k in range(int(max_steps)):
        t_loop0 = time.perf_counter()

        robot = np.asarray(obs["robot_xyz"], dtype=np.float32).reshape(3)
        robot_traj.append(robot.copy())

        if np.linalg.norm(robot - goal) <= goal_finish_dist:
            reached_goal = True
            break

        if break_on_collision and robot[2] <= env.zlim[0] + ROBOT_RAD:
            break

        obs_now = _get_obs_positions_from_history(obs)
        dmin = _min_dist_robot_to_points(robot, obs_now) if obs_now.size else float("inf")
        if dmin < safe_rad:
            n_collisions += 1

        p_dict = obs.get("prediction", {})
        h_dict = obs.get("history", {})
        pred, pred_mask, _ = stack_pred3d_from_p_dict(p_dict, horizon=time_horizon)

        # control = online ACI update + sampled MPC (both counted as control time)
        t0 = time.perf_counter()
        intervals, _ = ctrl.update_cp(h_dict, p_dict)
        act, info = ctrl(
            robot_xyz=robot,
            goal_xyz=goal,
            pred_xyz=pred,
            pred_mask=pred_mask,
            intervals=intervals,
        )
        t1 = time.perf_counter()

        if act is None:
            target = robot.copy()
            cmd = (0.0, 0.0, 0.0, 0.0)
        else:
            pos, vel = act
            target = np.asarray(pos, dtype=np.float32).reshape(3)
            vel = np.asarray(vel, dtype=np.float32).reshape(4)
            cmd = tuple(map(float, vel))

        if not bool(info.get("feasible", False)):
            n_infeasible += 1

        t2 = time.perf_counter()
        obs = env.step(target, cmd)
        t3 = time.perf_counter()

        timing["ctrl_ms"].append((t1 - t0) * 1000.0)
        timing["loop_ms"].append((t3 - t_loop0) * 1000.0)
        steps += 1

    robot_traj_arr = np.asarray(robot_traj, dtype=np.float32).reshape(-1, 3)
    return {
        "reached_goal": bool(reached_goal),
        "steps": int(steps),
        "collisions": int(n_collisions),
        "infeasible_steps": int(n_infeasible),
        "ctrl_times_ms": list(timing["ctrl_ms"]),
        "loop_times_ms": list(timing["loop_ms"]),
        "robot_traj": robot_traj_arr.tolist(),
    }
