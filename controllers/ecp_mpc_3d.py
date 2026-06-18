from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import time
import math

from controllers.utils import sample_random_paths_3d

DISTANCE_BOUND = 10000.0


# ============================================================
# 3D utilities (dict <-> numpy)
# ============================================================
def process_obs_3d(o: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """history dict(pid -> (L,3)) -> pid -> (3,) (most recent)"""
    return {pid: np.asarray(v[-1], dtype=np.float32) for pid, v in o.items()}


def to_numpy_3d(o: Dict[int, np.ndarray], pids: List[int]) -> np.ndarray:
    """dict(pid -> (3,)) -> (|pids|, 3)"""
    if len(pids) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack([np.asarray(o[pid], dtype=np.float32) for pid in pids], axis=0)


def pred_dict_to_stacked(pred_dict: Dict[int, np.ndarray], horizon: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    dict(pid -> (H,3)) -> pred_xyz: (H, M, 3), pred_mask: (H, M), pids
    """
    pids = list(pred_dict.keys())
    M = len(pids)
    H = int(horizon)
    pred_xyz = np.zeros((H, M, 3), dtype=np.float32)
    pred_mask = np.zeros((H, M), dtype=bool)
    for j, pid in enumerate(pids):
        arr = np.asarray(pred_dict[pid], dtype=np.float32)
        take = min(H, arr.shape[0])
        if take > 0:
            pred_xyz[:take, j, :] = arr[:take, :]
            pred_mask[:take, j] = True
    return pred_xyz, pred_mask, pids


# ============================================================
# pairwise distance + min-dist helpers
# ============================================================
def compute_pairwise_distances_3d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    x: (N,3), y: (M,3) -> (N,M)
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.full((x.shape[0], 0), np.inf, dtype=np.float32)
    d = x[:, None, :] - y[None, :, :]
    return np.linalg.norm(d, axis=-1)


def compute_min_dist_paths_to_predxyz(paths_xyz: np.ndarray, pred_xyz: np.ndarray, pred_mask: Optional[np.ndarray]) -> np.ndarray:
    """
    paths_xyz: (P, T, 3)   (NOTE: usually paths[:,1:,:])
    pred_xyz : (T_pred, M, 3)
    pred_mask: (T_pred, M) bool or None

    return: (P, T_use) where T_use=min(T, T_pred)
    """
    P, T, _ = paths_xyz.shape
    T_pred = int(pred_xyz.shape[0])
    T_use = min(T, T_pred)
    out = np.full((P, T_use), np.inf, dtype=np.float32)

    pred = np.asarray(pred_xyz[:T_use], dtype=np.float32)            # (T_use, M, 3)
    if pred.shape[1] == 0:
        return out
    mask = None if pred_mask is None else np.asarray(pred_mask[:T_use], dtype=bool)  # (T_use, M)

    # Vectorized over t (exact same per-element norm as the original loop), with a
    # path-axis chunk to bound peak memory for the large lattice (P up to ~2e4).
    chunk_P = 4096
    for s in range(0, P, chunk_P):
        e = min(P, s + chunk_P)
        diff = paths_xyz[s:e, :T_use, None, :] - pred[None, :, :, :]  # (c, T_use, M, 3)
        d = np.linalg.norm(diff, axis=-1)                            # (c, T_use, M)
        if mask is not None:
            d = np.where(mask[None, :, :], d, np.inf)
        out[s:e] = d.min(axis=-1)
    return out


# ============================================================
# quantile helper (like "higher" quantile, but per-(P,T) level)
# ============================================================
def quantile_higher_per_entry(scores: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    scores: (P, T, B)
    levels: (P, T) in [0,1]
    returns: (P, T) with "higher" quantile (ceil)
    """
    scores = np.asarray(scores, dtype=np.float32)
    levels = np.asarray(levels, dtype=np.float32)

    P, T, B = scores.shape
    if B <= 0:
        return np.full((P, T), np.inf, dtype=np.float32)

    # sort along batch axis
    s = np.sort(scores, axis=-1)  # (P,T,B)

    # index = ceil(level*B) - 1  (clipped)
    idx = np.ceil(levels * B).astype(np.int32) - 1
    idx = np.clip(idx, 0, B - 1)

    # gather
    # build advanced indices
    p_idx = np.arange(P)[:, None]
    t_idx = np.arange(T)[None, :]
    return s[p_idx, t_idx, idx].astype(np.float32)


# ============================================================
# Cost weights (match FunctionalCPMPC3D style)
# ============================================================
@dataclass
class MPC3DWeights:
    w_terminal: float = 10.0
    w_intermediate: float = 1.0
    w_control: float = 0.001


# ============================================================
# EgocentricCPMPC3D (ECP + ACI, env-compatible call)
# ============================================================
class EgocentricCPMPC3D:
    """
    3D version of your 2D EgocentricCPMPC:
      - keeps the SAME mechanism:
          * prediction queue (_prediction_queue)
          * observation queue (_track_queue)
          * path_history, quantile_history
          * update_observations() updates alpha_t (ACI)
          * evaluate_scores() computes per-path per-step quantiles from batch
          * filter_unsafe uses (safe_rad + quantiles)
      - BUT public __call__ is env-compatible with FunctionalCPMPC3D:
          input: robot_xyz, robot_yaw, goal_xyz, pred_xyz, pred_mask, boxes_3d, robot_vel
          output: (next_pos, target_vel_env), info
    """

    def __init__(
        self,
        n_steps: int = 12,
        dt: float = 0.4,
        n_skip: int = 4,
        robot_rad: float = 0.4,
        obstacle_rad: float = 1.0 / np.sqrt(2.0),
        # limits (env/functional)
        v_lim: Tuple[float, float] = (-0.8, 0.8),         # v_xy
        vz_lim: Tuple[float, float] = (-0.3, 0.3),
        yaw_rate_lim: Tuple[float, float] = (-0.7, 0.7),
        # discretization points (kept for API parity; unused by the sampling rollout)
        v_points: Tuple[float, ...] = (-1.0, 0.0, 1.0),
        w_points: Tuple[float, ...] = (-1.0, 0.0, 1.0),
        vz_points: Tuple[float, ...] = (-1.0, 0.0, 1.0),
        # sampling-based ("MPPI-style") rollout, matching FCP-3D
        n_paths: int = 2000,
        seed: int = 0,
        # ACI / online calibration
        calibration_set_size: int = 15,
        miscoverage_level: float = 0.10,
        step_size: float = 0.05,
        weights: Optional[MPC3DWeights] = None,
    ):
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.n_skip = int(n_skip)

        self.robot_rad = float(robot_rad)
        self.obstacle_rad = float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad

        self.vmin, self.vmax = map(float, v_lim)
        self.wmin, self.wmax = map(float, yaw_rate_lim)
        self.vzmin, self.vzmax = map(float, vz_lim)

        # Sampling-based rollout: the candidate set is re-drawn each step (instead of a
        # fixed (v, w, vz) meshgrid). The egocentric ACI keeps a per-path adapted
        # miscoverage level; with resampling a given row no longer tracks a fixed control
        # sequence, so the per-row adaptation behaves as a noisy global adaptation.
        self.n_paths = int(n_paths)
        self.rng = np.random.default_rng(int(seed))

        # Global per-horizon ACI level (one scalar per horizon step), shared across candidates.
        # Per-path alpha_t is ill-defined under MPPI resampling (paths redrawn each step), so a
        # single alpha_t[i] is the correct egocentric ACI under resampling (see 2D ecp_mpc.py).
        self._miscoverage_level = float(miscoverage_level)
        self.alpha_t = self._miscoverage_level * np.ones(self.n_steps, dtype=np.float32)
        self._gamma = float(step_size)

        self.calibration_set_size = int(calibration_set_size)

        self.weights = weights or MPC3DWeights()

        # queues / history (same idea as 2D)
        self._prediction_queue: List[Dict[int, np.ndarray]] = []   # list of dict(pid -> (H,3))
        self._track_queue: List[Dict[int, np.ndarray]] = []        # list of dict(pid -> (3,))
        self.path_history: List[np.ndarray] = []                   # list of (P,T+1,3)
        self.quantile_history: List[np.ndarray] = []               # list of (P,T)

        self.last_best_vels: Optional[np.ndarray] = None

    # ---------------------------------------------------------
    # queue APIs (same semantics as 2D)
    # ---------------------------------------------------------
    def update_predictions(self, prediction_result: Dict[int, np.ndarray]) -> None:
        """
        prediction_result: dict(pid -> (H,3)) at current time t
        You should call this AFTER ctrl(...) in the loop,
        consistent with your 2D code.
        """
        self._prediction_queue.append(prediction_result)

    def update_observations(self, obs_history: Dict[int, np.ndarray]) -> np.ndarray:
        """
        obs_history: dict(pid -> (L,3)) (trajectories)
        returns err: (P,T) miscoverage indicator used in ACI update
        """
        n_paths = self.n_paths

        if not obs_history:
            self._track_queue.append({})
            return np.zeros((n_paths, self.n_steps), dtype=np.float32)

        obs = process_obs_3d(obs_history)  # pid -> (3,)

        quantiles = []
        min_dist_obs = []
        min_dist_pred = []

        n_data = len(self.quantile_history)  # stored from t0..t-1 (like 2D)
        max_n_steps = min(self.n_steps + 1, n_data)

        for i in range(1, max_n_steps):
            # predictions stored up to t-i (queue includes up to t-1 if called outer-loop style)
            if len(self._prediction_queue) < i:
                # not enough prediction history yet
                min_dist_obs_i = np.full((n_paths,), DISTANCE_BOUND, dtype=np.float32)
                min_dist_pred_i = np.full((n_paths,), DISTANCE_BOUND, dtype=np.float32)
            else:
                pred = self._prediction_queue[-i]  # dict(pid -> (H,3)) at time (t-i)

                obs_nodes = set(obs.keys())
                pred_nodes = set(pred.keys())
                common = list(obs_nodes & pred_nodes)

                if common:
                    obs_np = to_numpy_3d(obs, common)  # (|V|,3)

                    paths_prev = self.path_history[-i]      # (P,T+1,3) planned at time (t-i)
                    x_i = paths_prev[:, i, :]               # (P,3) state at time t predicted from (t-i)

                    # obs distance
                    d_obs = compute_pairwise_distances_3d(x_i, obs_np)  # (P,|V|)
                    min_dist_obs_i = np.min(d_obs, axis=-1).astype(np.float32)

                    # pred distance (use step i-1 from pred trajectories)
                    pred_i_pts = np.stack([np.asarray(pred[pid][i - 1], dtype=np.float32) for pid in common], axis=0)  # (|V|,3)
                    d_pred = compute_pairwise_distances_3d(x_i, pred_i_pts)
                    min_dist_pred_i = np.min(d_pred, axis=-1).astype(np.float32)
                else:
                    min_dist_obs_i = np.full((n_paths,), DISTANCE_BOUND, dtype=np.float32)
                    min_dist_pred_i = np.full((n_paths,), DISTANCE_BOUND, dtype=np.float32)

            min_dist_obs.append(min_dist_obs_i)
            min_dist_pred.append(min_dist_pred_i)

            q_i = self.quantile_history[-i][:, i - 1]  # (P,)
            quantiles.append(q_i)

        if n_data > 1 and len(quantiles) > 0:
            min_dist_obs = np.stack(min_dist_obs, axis=-1)   # (P, L)
            min_dist_pred = np.stack(min_dist_pred, axis=-1) # (P, L)
            quantiles = np.stack(quantiles, axis=-1)         # (P, L)

            err = (quantiles < (min_dist_pred - min_dist_obs)).astype(np.float32)  # (P,L)

            # ACI update
            self.alpha_t[:max_n_steps - 1] += self._gamma * (self._miscoverage_level - err.mean(axis=0))

            # pad to (P,T) for consistent return
            if (max_n_steps - 1) < self.n_steps:
                pad = self.n_steps - (max_n_steps - 1)
                err = np.hstack([err, np.zeros((n_paths, pad), dtype=np.float32)])
        else:
            err = np.zeros((n_paths, self.n_steps), dtype=np.float32)

        self._track_queue.append(obs)  # store processed obs (pid->(3,))
        return err

    # ---------------------------------------------------------
    # batch loaders (same indexing idea as 2D)
    # ---------------------------------------------------------
    def load_recent_obs(self, batch_size: int) -> List[Dict[int, np.ndarray]]:
        return self._track_queue[-batch_size:]

    def load_recent_pred(self, batch_size: int, step: int) -> List[Dict[int, np.ndarray]]:
        """
        prediction queue stores results up to t-1 if called outer-loop style.
        Returns list length=batch_size, each element is dict(pid -> (3,)) of predicted position at that 'step'.
        """
        assert step >= 1
        idx_begin = -batch_size + 1 - step
        idx_end = len(self._prediction_queue) if step == 1 else 1 - step

        pred_batch = []
        for idx in range(idx_begin, idx_end):
            pred_t = self._prediction_queue[idx]  # dict(pid -> (H,3))
            pred_i = {pid: np.asarray(v[step - 1], dtype=np.float32) for pid, v in pred_t.items()}
            pred_batch.append(pred_i)
        return pred_batch

    # ---------------------------------------------------------
    # evaluate_scores (online quantiles) - SAME idea as 2D
    # ---------------------------------------------------------
    def evaluate_scores(self, paths: np.ndarray) -> np.ndarray:
        """
        paths: (P,T+1,3)
        returns quantiles: (P,T)
        """
        n_data = len(self._track_queue)  # stores obs up to current time (if update_observations called first)
        P = paths.shape[0]

        if n_data >= self.n_steps + 1:
            batch_size = min(n_data - self.n_steps, self.calibration_set_size)

            obs_batch = self.load_recent_obs(batch_size=batch_size)

            min_dist_obs = np.zeros((P, self.n_steps, batch_size), dtype=np.float32)
            min_dist_pred = np.zeros((P, self.n_steps, batch_size), dtype=np.float32)

            for i in range(1, self.n_steps + 1):
                x_i = paths[:, i, :]  # (P,3)
                pred_batch = self.load_recent_pred(batch_size=batch_size, step=i)

                for b, (obs_j, pred_j) in enumerate(zip(obs_batch, pred_batch)):
                    obs_nodes = set(obs_j.keys())
                    pred_nodes = set(pred_j.keys())
                    common = list(obs_nodes & pred_nodes)

                    if common:
                        obs_np = to_numpy_3d(obs_j, common)    # (|V|,3)
                        pred_np = to_numpy_3d(pred_j, common)  # (|V|,3)

                        d_obs = compute_pairwise_distances_3d(x_i, obs_np)   # (P,|V|)
                        d_pred = compute_pairwise_distances_3d(x_i, pred_np)

                        min_dist_obs[:, i - 1, b] = np.min(d_obs, axis=-1)
                        min_dist_pred[:, i - 1, b] = np.min(d_pred, axis=-1)
                    else:
                        min_dist_obs[:, i - 1, b] = DISTANCE_BOUND
                        min_dist_pred[:, i - 1, b] = DISTANCE_BOUND

            scores = np.clip(min_dist_pred - min_dist_obs, a_min=0.0, a_max=None)  # (P,T,B)

            levels = 1.0 - np.tile(self.alpha_t, (scores.shape[0], 1))  # global per-step level -> (P,T)
            quantiles = quantile_higher_per_entry(scores, levels)  # (P,T)

            # same spirit as 2D: cap inf quantiles by a growing bound
            max_scores = (0.5 * self.dt * np.arange(1, self.n_steps + 1, dtype=np.float32))  # (T,)
            max_scores = np.tile(max_scores[None, :], (P, 1))
            quantiles = np.where(np.isposinf(quantiles), max_scores, quantiles).astype(np.float32)

            return quantiles

        # not enough data yet: return conservative increasing bound
        max_scores = (1.5 * self.dt * np.arange(1, self.n_steps + 1, dtype=np.float32))
        return np.tile(max_scores[None, :], (P, 1)).astype(np.float32)

    # ---------------------------------------------------------
    # path generation (3D grid actions)
    # ---------------------------------------------------------
    def generate_paths_3d(
        self,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        *,
        n_skip: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns:
          paths: (P, T+1, 3)
          vels:  (P, T, 3) where last dim is (v_xy, yaw_rate, vz)

        Uses the same sampling-based ("MPPI-style") rollout as FCP-3D so the action
        search matches the proposed method instead of the legacy (v, w, vz) meshgrid.
        The number of returned paths is fixed at ``self.n_paths`` to stay aligned with
        the per-path ``alpha_t`` buffer used by the egocentric ACI update.
        """
        return sample_random_paths_3d(
            np.asarray(robot_xyz, dtype=np.float32),
            float(robot_yaw),
            n_steps=self.n_steps,
            n_skip=n_skip,
            dt=self.dt,
            v_lim=(self.vmin, self.vmax),
            w_lim=(self.wmin, self.wmax),
            vz_lim=(self.vzmin, self.vzmax),
            n_paths=self.n_paths,
            rng=self.rng,
            last_best_vels=self.last_best_vels,
        )

    # ---------------------------------------------------------
    # scoring (same weights as Functional)
    # ---------------------------------------------------------
    def score_paths(self, paths: np.ndarray, vels: np.ndarray, goal_xyz: np.ndarray) -> Tuple[int, float]:
        goal = np.asarray(goal_xyz, dtype=np.float32).reshape(3,)

        term_cost = self.weights.w_terminal * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        inter_cost = self.weights.w_intermediate * np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-1, -2))
        ctrl_cost = self.weights.w_control * np.sum(vels ** 2, axis=(-1, -2))
        total = term_cost + inter_cost + ctrl_cost

        best = int(np.argmin(total))
        return best, float(total[best])

    # ---------------------------------------------------------
    # unsafe filter using CURRENT prediction + quantiles
    # ---------------------------------------------------------
    def filter_unsafe_paths_predxyz(
        self,
        *,
        paths: np.ndarray,          # (P, T+1, 3)
        vels: np.ndarray,           # (P, T, 3)
        pred_xyz: np.ndarray,       # (T_pred, M, 3)
        pred_mask: Optional[np.ndarray],  # (T_pred, M)
        quantiles: np.ndarray,      # (P, T)
        boxes_3d: List[Any],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # NOTE: static boxes are ignored here (same as your previous 3D baseline)
        # If you need, you can add 3D box collision checks later.

        P, T1, _ = paths.shape
        T = T1 - 1

        # min dist per (P,t)
        min_dist = compute_min_dist_paths_to_predxyz(paths[:, 1:, :], pred_xyz, pred_mask)  # (P, T_use)
        T_use = min_dist.shape[1]

        # compare with safe_rad + quantiles
        q_use = quantiles[:, :T_use]
        unsafe_dyn = np.any(min_dist < (self.safe_rad + q_use), axis=-1)  # (P,)

        safe = ~unsafe_dyn
        if not np.any(safe):
            return None, None
        return paths[safe], vels[safe]

    # ---------------------------------------------------------
    # PUBLIC API (env-compatible)
    # ---------------------------------------------------------
    def __call__(
        self,
        *,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        goal_xyz: np.ndarray,
        pred_xyz: np.ndarray,
        pred_mask: Optional[np.ndarray],
        boxes_3d: Optional[List[Any]] = None,
        robot_vel: Optional[np.ndarray] = None,  # unused, for signature parity
    ):
        """
        IMPORTANT usage pattern (same as 2D ECP):
          - outer loop should call:
              err = ctrl.update_observations(obs["history"])
              (then) act, info = ctrl(...)
              (then) ctrl.update_predictions(obs["prediction"])
        """
        boxes_3d = boxes_3d or []
        t0 = time.perf_counter()

        # 1) candidates
        paths, vels = self.generate_paths_3d(
            np.asarray(robot_xyz, dtype=np.float32),
            float(robot_yaw),
            n_skip=self.n_skip,
        )
        self.path_history.append(paths)

        # 2) online quantiles (needs queues populated)
        quantiles = self.evaluate_scores(paths)  # (P,T)
        self.quantile_history.append(quantiles)

        # 3) filter unsafe with CURRENT prediction
        safe_paths, safe_vels = self.filter_unsafe_paths_predxyz(
            paths=paths,
            vels=vels,
            pred_xyz=np.asarray(pred_xyz, dtype=np.float32),
            pred_mask=np.asarray(pred_mask, dtype=bool) if pred_mask is not None else None,
            quantiles=quantiles,
            boxes_3d=boxes_3d,
        )

        if safe_paths is None:
            return None, {
                "feasible": False,
                "quantiles": quantiles,
                "final_path": None,
                "timing_ms": (time.perf_counter() - t0) * 1000.0,
            }

        # 4) pick best
        best_idx, best_cost = self.score_paths(
            safe_paths, safe_vels, np.asarray(goal_xyz, dtype=np.float32)
        )
        best_path = safe_paths[best_idx]
        best_vel_traj = safe_vels[best_idx]
        self.last_best_vels = best_vel_traj.copy()

        next_pos = best_path[1]  # (3,)

        # Convert (v_xy, yaw_rate, vz) -> env command (vx,vy,vz,yaw_rate)
        v_xy, w, vz = best_vel_traj[0]
        yaw = float(robot_yaw)

        vx = float(v_xy * np.cos(yaw))
        vy = float(v_xy * np.sin(yaw))

        vx = float(np.clip(vx, self.vmin, self.vmax))
        vy = float(np.clip(vy, self.vmin, self.vmax))
        vz = float(np.clip(vz, self.vzmin, self.vzmax))
        w = float(np.clip(w, self.wmin, self.wmax))

        target_vel_env = np.array([vx, vy, vz, w], dtype=np.float32)

        info = {
            "feasible": True,
            "quantiles": quantiles,
            "final_path": best_path,
            "cost": best_cost,
            "timing_ms": (time.perf_counter() - t0) * 1000.0,
            "target_pos": next_pos,
            "target_vel": target_vel_env,
            "n_paths": int(paths.shape[0]),
            "n_safe": int(safe_paths.shape[0]),
        }

        act = (next_pos, target_vel_env)
        return act, info