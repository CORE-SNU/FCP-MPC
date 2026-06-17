from __future__ import annotations
import numpy as np
import pybullet as p
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# Gym Pybullet Drones
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
import pybullet_data
from gymnasium import spaces
import time


# -----------------------------------------------------------------------------
# [1] Helper Utils & Predictors
# -----------------------------------------------------------------------------

def build_grid_3d(xlim, ylim, zlim, nx, ny, nz):
    """3D Grid 생성 함수 (시각화 코드에서 import하여 사용)"""
    xs = np.linspace(xlim[0], xlim[1], nx, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], ny, dtype=np.float32)
    zs = np.linspace(zlim[0], zlim[1], nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
    return xs, ys, zs, np.transpose(X, (2, 0, 1)), np.transpose(Y, (2, 0, 1)), np.transpose(Z, (2, 0, 1))

def distance_field_points_3d(points_xyz, X, Y, Z):
    """SDF 계산 함수"""
    if points_xyz.size == 0:
        return np.full_like(X, np.inf, dtype=np.float32)
    pts = points_xyz.astype(np.float32)
    # Broadcasting: (nz,ny,nx,1) - (1,1,1,M)
    dx = X[..., None] - pts[:, 0][None, None, None, :]
    dy = Y[..., None] - pts[:, 1][None, None, None, :]
    dz = Z[..., None] - pts[:, 2][None, None, None, :]
    d2 = dx*dx + dy*dy + dz*dz
    return np.sqrt(np.min(d2, axis=-1)).astype(np.float32)




@dataclass
class ObstacleAgent:
    pid: int
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)
    mode: int = 0    # 0: CV, 1: Turn, 2: Wander, 3: Stop-Go
    mode_ttl: int = 0
    goal: Optional[np.ndarray] = None  # if set, agent walks toward this goal (crossing pedestrian)
    speed: float = 0.0                 # preferred cruising speed for goal-directed agents

class CVPredictor3D:
    def __init__(self, dt, horizon, process_noise_std=0.0):
        self.dt = dt
        self.H = horizon
        self.noise = process_noise_std

    def predict(self, pos, vel, rng):
        # (H, 3) trajectory
        traj = np.zeros((self.H, 3), dtype=np.float32)
        p = pos.copy()
        v = vel.copy()
        for i in range(self.H):
            # Noisy prediction model
            v_eff = v + rng.normal(0, self.noise, size=3) if self.noise > 0 else v
            p = p + v_eff * self.dt
            traj[i] = p
        return traj

# -----------------------------------------------------------------------------
# [2] Environment Class (Physics + Logic)
# -----------------------------------------------------------------------------

class QuadWorldEnv3D(BaseAviary):
    """
    PyBullet Physics + Custom Obstacle Logic + Velocity Control Interface
    """
    def __init__(
        self,
        dt: float = 0.2,            # Simulation step (Low frequency for high-level planning)
        horizon: int = 12,
        n_obs: int = 10,
        world_bounds_xyz: Tuple = ((-5, 15), (-5, 15), (0, 20)),
        seed: int = 0,
        # Noise Params
        pred_model_noise: float = 0.1,
        obs_process_noise: float = 0.1,
        gt_future_noise: float = 0.0,
        start_xyz_yaw: Optional[np.ndarray] = None,   # (4,)
        goal_xyz: Optional[np.ndarray] = None,  
        # Obstacle Logic Params
        mode_switch_p: float = 0.05,
        mode_min_ttl: int = 5,
        mode_max_ttl: int = 25,
        turn_rate_std: float = 1.0,
        stop_go_p: float = 0.1,
        # Goal-directed (crossing-pedestrian) obstacles
        goal_directed_frac: float = 0.0,        # fraction of obstacles that walk toward a goal
        goal_speed_range: Tuple = (0.3, 0.7),   # cruising speed sampled per goal-directed agent
        goal_reach_thresh: float = 1.0,         # distance at which a new crossing goal is drawn
        goal_steer_gain: float = 0.3,           # how fast velocity turns toward the goal direction
        # GUI
        gui: bool = False,
    ):
        self.dt = float(dt)
        self.H = int(horizon)
        self.xlim, self.ylim, self.zlim = world_bounds_xyz
        
        # Physics setup
        # BaseAviary runs at 240Hz by default. We will step it multiple times 
        # to match our requested dt.
        self.sim_freq = 240 
        self.aggr_phy_steps = int(self.dt * self.sim_freq)
        
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=10,
            physics=Physics.PYB,
            pyb_freq=self.sim_freq,
            ctrl_freq=self.sim_freq,  
            gui=gui,
            user_debug_gui=False

        )
        
        # Control
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        
        # RNG. Keep the base seed so reset() can reproduce the SAME obstacle field
        # every episode regardless of how much RNG was consumed beforehand (e.g. FCP's
        # offline calibration steps the env, which would otherwise advance self.rng and
        # give FCP a different obstacle realization than the baselines at the same seed).
        self._seed0 = seed
        self.rng = np.random.default_rng(seed)
        self.oracle_rng = np.random.default_rng(seed + 9999) # Separate stream for GT
        
        # Parameters
        self.pred_model_noise = pred_model_noise
        self.obs_process_noise = obs_process_noise
        self.gt_future_noise = gt_future_noise
        
        self.mode_switch_p = mode_switch_p
        self.mode_min_ttl = mode_min_ttl
        self.mode_max_ttl = mode_max_ttl
        self.turn_rate_std = turn_rate_std
        self.stop_go_p = stop_go_p

        self.goal_directed_frac = float(goal_directed_frac)
        self.goal_speed_range = goal_speed_range
        self.goal_reach_thresh = float(goal_reach_thresh)
        self.goal_steer_gain = float(goal_steer_gain)

        # fixed start/goal
        if start_xyz_yaw is None:
            sx = 0.2 * self.xlim[1] + 0.8 * self.xlim[0]
            sy = 0.2 * self.ylim[1] + 0.8 * self.ylim[0]
            sz = 0.2 * self.zlim[1] + 0.8 * self.zlim[0]
            syaw = 0.0
            start_xyz_yaw = np.array([sx, sy, sz, syaw], dtype=np.float32)
        self.start_xyz_yaw = np.asarray(start_xyz_yaw, dtype=np.float32)

        if goal_xyz is None:
            gx = 0.8 * self.xlim[1] + 0.2 * self.xlim[0]
            gy = 0.8 * self.ylim[1] + 0.2 * self.ylim[0]
            gz = 0.8 * self.zlim[1] + 0.2 * self.zlim[0]
            goal_xyz = np.array([gx, gy, gz], dtype=np.float32)
        self.goal_xyz = np.asarray(goal_xyz, dtype=np.float32)
        
        # Components
        self.predictor = CVPredictor3D(dt=self.dt, horizon=self.H, process_noise_std=pred_model_noise)
        self.obstacles: List[ObstacleAgent] = []
        self.obs_ids: List[int] = [] # PyBullet IDs
        self.history_xyz: Dict[int, List[np.ndarray]] = {}
        
        self.step_counter = 0
        self.n_obs_target = n_obs
        
        # Init Obstacles
        self._init_obstacles(n_obs)

        self.gui_enabled = bool(gui)

        # --- Debug viz handles (GUI only) ---
        self._goal_uid = None
        self._goal_text_uid = None

        self._path_line_uids = []   # list[int]
        self._cmd_line_uid = None   # optional: current step command

        self._safety_sphere_uid = None

        # camera follow params (tweak as you like)
        self._cam_dist = 6.0
        self._cam_yaw = 45.0
        self._cam_pitch = -25.0

    def _debug_init_goal_marker(self):
        if not self.gui_enabled:
            return

        # remove old
        if self._goal_uid is not None:
            try: 
                p.removeBody(self._goal_uid)
            except: 
                pass
            self._goal_uid = None
        if self._goal_text_uid is not None:
            try: 
                p.removeUserDebugItem(self._goal_text_uid)
            except: 
                pass
            self._goal_text_uid = None

        vis_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.25,
            rgbaColor=[0, 1, 0, 0.85]
        )
        self._goal_uid = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis_id,
            baseCollisionShapeIndex=-1,
            basePosition=self.goal_xyz.tolist()
        )
        self._goal_text_uid = p.addUserDebugText(
            "GOAL",
            self.goal_xyz.tolist(),
            textColorRGB=[0, 1, 0],
            textSize=1.4
        )

    def _debug_update_goal_marker(self):
        if not self.gui_enabled:
            return
        if self._goal_uid is not None:
            p.resetBasePositionAndOrientation(self._goal_uid, self.goal_xyz.tolist(), [0,0,0,1])
        if self._goal_text_uid is not None:
            self._goal_text_uid = p.addUserDebugText(
                "GOAL",
                self.goal_xyz.tolist(),
                textColorRGB=[0, 1, 0],
                textSize=1.4,
                replaceItemUniqueId=self._goal_text_uid
            )


    def _debug_follow_camera(self, target_pos_xyz):
        if not self.gui_enabled:
            return
        p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=target_pos_xyz.tolist()
        )

    def _debug_draw_cmd_line(self, p_curr, p_cmd):
        if not self.gui_enabled:
            return
        if self._cmd_line_uid is None:
            self._cmd_line_uid = p.addUserDebugLine(p_curr.tolist(), p_cmd.tolist(), [1,1,1], 3)
        else:
            self._cmd_line_uid = p.addUserDebugLine(
                p_curr.tolist(), p_cmd.tolist(),
                [1,1,1], 3,
                replaceItemUniqueId=self._cmd_line_uid
            )


    def _actionSpace(self):
        """BaseAviary 초기화 시 호출됨: 행동 공간 정의"""
        return spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        """BaseAviary 초기화 시 호출됨: 관측 공간 정의"""
        return spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    def _computeObs(self):
        """매 step마다 호출됨: 현재 드론 상태 반환"""
        # 부모 클래스의 기본 state vector 반환 (필수 구현 요건 충족)
        return self._getDroneStateVector(0)

    def _preprocessAction(self, action):
        """매 step마다 호출됨: 입력을 RPM 등으로 변환"""
        # super().step()을 호출할 때만 사용됨.
        # 현재 코드 구조상 직접 step을 오버라이딩해서 쓰고 계시다면 이 부분은 호출되지 않을 수 있으나, 구현은 필수입니다.
        return action
    
    def _computeInfo(self):
        """BaseAviary 필수 구현: 매 step/reset 마다 Info 딕셔너리 반환"""
        # 사용자 환경에서 별도로 info를 구성하시므로, 부모 클래스 호출용으로는 빈 dict를 반환합니다.
        return {}

    def _computeReward(self):
        """BaseAviary 필수 구현: Reward 반환"""
        # 사용자 정의 step()을 쓰시므로, 부모 클래스 통과용 더미 값을 반환합니다.
        return -1

    def _computeTerminated(self):
        """BaseAviary 필수 구현: 종료(Terminated) 여부"""
        return False
    
    def _computeTruncated(self):
        """BaseAviary 필수 구현: 시간 초과(Truncated) 여부"""
        return False

    def _init_obstacles(self, n_obs):
        # Clear existing
        for uid in self.obs_ids:
            p.removeBody(uid)
        self.obs_ids = []
        self.obstacles = []
        self.history_xyz = {}
        
        # Create visual shape only once (Optimization)
        col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 0.6])
        
        n_goal_directed = int(round(n_obs * self.goal_directed_frac))
        for pid in range(n_obs):
            pos = np.array([
                self.rng.uniform(*self.xlim),
                self.rng.uniform(*self.ylim),
                self.rng.uniform(*self.zlim)
            ], dtype=np.float32)

            if pid < n_goal_directed:
                # Crossing pedestrian: head toward a goal on the far side at a steady speed.
                speed = float(self.rng.uniform(*self.goal_speed_range))
                goal = self._new_crossing_goal(pos)
                direction = goal - pos
                direction /= (np.linalg.norm(direction) + 1e-6)
                vel = (direction * speed).astype(np.float32)
                agent = ObstacleAgent(pid=pid, pos=pos, vel=vel, goal=goal, speed=speed)
            else:
                vel = self.rng.normal(0, 0.5, size=3).astype(np.float32)
                vel[2] *= 0.2
                agent = ObstacleAgent(pid=pid, pos=pos, vel=vel)
            self.obstacles.append(agent)
            self.history_xyz[pid] = [pos.copy()]
            
            # Create PyBullet Body (Kinematic)
            uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=pos)
            self.obs_ids.append(uid)

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.ctrl.reset()

        # Reseed deterministically so the obstacle field is identical for every method
        # at a given seed (fairness): otherwise any RNG consumed before this reset
        # (notably FCP's offline calibration rollouts) would shift the obstacle layout
        # and trajectories, letting FCP fly a different scene than the baselines.
        base = seed if seed is not None else self._seed0
        self._seed0 = base
        self.rng = np.random.default_rng(base)
        self.oracle_rng = np.random.default_rng(base + 9999)

        # Set Drone Initial Position (Randomized within bounds or fixed safe area)
        start_pos = np.array([0.0, 0.0, 1.0]) # Default start
        p.resetBasePositionAndOrientation(self.DRONE_IDS[0], start_pos, [0,0,0,1])
        
        # Reset Obstacles
        self._init_obstacles(self.n_obs_target)
        
        self.step_counter = 0

        if self.gui_enabled:
            self._debug_init_goal_marker()

        return self._make_obs()

    def _new_crossing_goal(self, pos):
        """Draw a goal on the far side of the workspace so the agent traverses it
        (one horizontal axis is pushed to the opposite extreme; the rest is random)."""
        g = np.array([
            self.rng.uniform(*self.xlim),
            self.rng.uniform(*self.ylim),
            self.rng.uniform(*self.zlim),
        ], dtype=np.float32)
        axis = int(self.rng.integers(0, 2))   # cross along x or y
        lim = [self.xlim, self.ylim][axis]
        mid = 0.5 * (lim[0] + lim[1])
        g[axis] = lim[0] if pos[axis] > mid else lim[1]
        return g

    def _step_obstacles_logic(self):
        """Update obstacle positions: goal-directed agents steer toward a crossing
        goal; the rest follow the random multi-mode (CV/Turn/Wander/Stop-Go) walk."""
        for i, ob in enumerate(self.obstacles):
            if ob.goal is not None:
                # Goal-directed crossing pedestrian
                to_goal = ob.goal - ob.pos
                dist = float(np.linalg.norm(to_goal))
                if dist < self.goal_reach_thresh:
                    ob.goal = self._new_crossing_goal(ob.pos)
                    to_goal = ob.goal - ob.pos
                    dist = float(np.linalg.norm(to_goal))
                desired = (to_goal / (dist + 1e-6)) * ob.speed
                v = ob.vel + (desired - ob.vel) * self.goal_steer_gain
                v += self.rng.normal(0, self.obs_process_noise, size=3) * self.dt
                ob.vel = v
                next_pos = ob.pos + ob.vel * self.dt
                # Stay inside the workspace; re-target next step if it hits a wall.
                for dim, lim in enumerate([self.xlim, self.ylim, self.zlim]):
                    next_pos[dim] = np.clip(next_pos[dim], lim[0], lim[1])
                ob.pos = next_pos
            else:
                # 1. Mode Switch
                if ob.mode_ttl <= 0 or self.rng.random() < self.mode_switch_p:
                    ob.mode = self.rng.integers(0, 4)
                    ob.mode_ttl = self.rng.integers(self.mode_min_ttl, self.mode_max_ttl)
                ob.mode_ttl -= 1

                # 2. Dynamics
                acc = self.rng.normal(0, self.obs_process_noise, size=3)
                acc[2] *= 0.2

                v = ob.vel.copy()

                if ob.mode == 0: # CV
                    pass
                elif ob.mode == 1: # Turn
                    ang = self.rng.normal(0, self.turn_rate_std) * self.dt
                    c, s = np.cos(ang), np.sin(ang)
                    v[0], v[1] = c*v[0] - s*v[1], s*v[0] + c*v[1]
                elif ob.mode == 2: # Wander
                    target = self.rng.normal(0, 1.0, size=3)
                    v += (target - v) * 0.2
                elif ob.mode == 3: # Stop-Go
                    if self.rng.random() < self.stop_go_p:
                        v *= 0.1

                v += acc * self.dt
                ob.vel = v
                next_pos = ob.pos + ob.vel * self.dt

                # 3. Bounds Reflection
                for dim, lim in enumerate([self.xlim, self.ylim, self.zlim]):
                    if not (lim[0] < next_pos[dim] < lim[1]):
                        ob.vel[dim] *= -1.0
                        next_pos[dim] = np.clip(next_pos[dim], lim[0], lim[1])

                ob.pos = next_pos

            # 4. History
            hist = self.history_xyz[ob.pid]
            hist.append(ob.pos.copy())
            if len(hist) > 5:
                hist.pop(0)

            # 5. Sync PyBullet
            p.resetBasePositionAndOrientation(self.obs_ids[i], ob.pos, [0,0,0,1])

    def _oracle_future(self, pos, vel, rng):
        """Ground Truth Future Trajectory (Simulation)"""
        traj = np.zeros((self.H, 3), dtype=np.float32)
        p_curr = pos.copy()
        v_curr = vel.copy()
        
        for t in range(self.H):
            # Simple reflection logic, maybe with less noise than predictor
            if self.gt_future_noise > 0:
                v_curr += rng.normal(0, self.gt_future_noise, size=3) * self.dt
            
            p_curr += v_curr * self.dt
            
            for dim, lim in enumerate([self.xlim, self.ylim, self.zlim]):
                if not (lim[0] < p_curr[dim] < lim[1]):
                    v_curr[dim] *= -1
                    p_curr[dim] = np.clip(p_curr[dim], lim[0], lim[1])
            
            traj[t] = p_curr
        return traj

    def _make_obs(self):
        # 1. Drone State
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        
        # 2. Obstacle Data
        h_dict = {pid: np.array(hist, dtype=np.float32) for pid, hist in self.history_xyz.items()}
        p_dict = {}
        f_dict = {}
        
        for ob in self.obstacles:
            # Prediction
            p_dict[ob.pid] = self.predictor.predict(ob.pos, ob.vel, self.rng)
            
            # Oracle Future (Deterministic seed per step/agent for consistency if needed, but here we roll)
            # Use separate oracle_rng
            seed_pid = self.oracle_rng.integers(0, 100000)
            rng_temp = np.random.default_rng(seed_pid)
            f_dict[ob.pid] = self._oracle_future(ob.pos, ob.vel, rng_temp)
            
        return {
            "robot_xyz": pos.astype(np.float32),
            "robot_yaw": float(rpy[2]),
            "robot_vel": vel.astype(np.float32),
            "goal_xyz": self.goal_xyz.copy().astype(np.float32),
            "history": h_dict,
            "prediction": p_dict,
            "future": f_dict,
        }


    def step(self, target_pos_xyz, target_vel):
        """
        High-Level Position Interface:
          target_pos_xyz: (3,) desired position in world frame
          target_yaw: desired yaw angle
        """
        inner_steps = self.aggr_phy_steps
        sub_dt = self.dt / inner_steps

        target_vx, target_vy, target_vz, target_yaw_rate = target_vel

        target_pos_xyz = np.asarray(target_pos_xyz, dtype=np.float32).reshape(3,)
        target_pos_xyz = np.array([
            np.clip(target_pos_xyz[0], self.xlim[0], self.xlim[1]),
            np.clip(target_pos_xyz[1], self.ylim[0], self.ylim[1]),
            np.clip(target_pos_xyz[2], self.zlim[0], self.zlim[1]),
        ], dtype=np.float32)

        for _ in range(inner_steps):
            state = self._getDroneStateVector(0)
            pos = state[0:3]
            vel = state[10:13]
            ang_vel = state[13:16]

            current_yaw = float(state[9])

            target_yaw = current_yaw + target_yaw_rate * sub_dt
            target_rpy = np.array([0.0, 0.0, target_yaw], dtype=np.float32)

            target_vel_world = np.array([target_vx, target_vy, target_vz], dtype=np.float32)

            rpm, _, _ = self.ctrl.computeControl(
                control_timestep=sub_dt,
                cur_pos=pos,
                cur_quat=state[3:7],
                cur_vel=vel,
                cur_ang_vel=ang_vel,
                target_pos=target_pos_xyz,
                target_vel=target_vel_world,
                target_rpy=target_rpy,


            )
            super().step(rpm.reshape(1, 4))

        self._step_obstacles_logic()
        self.step_counter += 1


        if self.gui_enabled:
            state = self._getDroneStateVector(0)
            self._debug_follow_camera(state[0:3].astype(np.float32))
            self._debug_draw_cmd_line(state[0:3].astype(np.float32), np.asarray(target_pos_xyz, np.float32))

        return self._make_obs()
