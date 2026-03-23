# controllers/func_cp_mpc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import math
import time

import numpy as np


def min_dist_robot_to_peds(robot_xy, peds_xy):
    if peds_xy.size == 0:
        return float("inf")
    d = peds_xy - robot_xy[None, :]
    return float(np.sqrt(np.sum(d * d, axis=1)).min())

# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class FuncMPCWeights:
    """
    Weights for the MPC objective.

    The controller uses:
      - goal tracking (intermediate + terminal),
      - control effort penalty,
      - soft safety shaping (optional), with a separate adaptive multiplier.

    Parameters
    ----------
    w_terminal : float
        Terminal goal tracking weight.
    w_intermediate : float
        Intermediate goal tracking weight (sum along the horizon).
    w_control : float
        Control effort weight.
    w_safety : float
        Adaptive multiplier (updated online) for the soft safety shaping accumulator.
    w_margin : float
        Weight for the soft barrier against safety margin violation.
    safety_scale : float
        Reserved knob (kept for compatibility); can be used to scale the safety term.
    """
    w_terminal: float = 10.0
    w_intermediate: float = 1.0
    w_control: float = 0.001

    # soft-safety shaping
    w_safety: float = 10.0
    w_margin: float = 10.0
    safety_scale: float = 0.2

# =============================================================================
# Functional CP-MPC Controller
# =============================================================================

class FunctionalCPMPC:
    """
    Functional CP-informed Monte Carlo MPC controller.

    Summary of the online logic:
      1) Sample candidate control sequences.
      2) Roll out unicycle dynamics to generate candidate paths.
      3) Filter infeasible paths using:
           - static obstacle collision checks, and
           - CP-conformalized distance lower bound to predicted dynamic obstacles.
      4) Score the remaining feasible paths with an MPC objective
         (goal tracking + control effort + optional soft safety shaping),
         and select the first control of the best plan.

    Key idea:
      - Offline, you precompute (per horizon index i) a parameterized upper envelope U_i(x)
        for the residual field, and cache its parameters.
      - Online, you only evaluate U_i(x) at the finite set of rollout states.
    """

    # ---------------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------------

    def __init__(
        self,
        *,
        cp_upper_grid: Optional[np.ndarray],
        box: float,
        world_center: np.ndarray,
        grid_H: int,
        grid_W: int,
        n_steps: int,
        dt: float,
        n_skip: int,
        robot_rad: float,
        obstacle_rad: float,
        min_linear_x: float,
        max_linear_x: float,
        min_angular_z: float,
        max_angular_z: float,
        n_paths: int,
        seed: int = 0,
        weights: Optional[FuncMPCWeights] = None,
        risk_level: float = 1.0,
        step_size: float = 9.,
        CP: bool = True,
    ):
        # Workspace/grid configuration
        self.box = float(box)
        self.world_center = np.asarray(world_center, dtype=np.float32)
        self.grid_H, self.grid_W = int(grid_H), int(grid_W)

        if cp_upper_grid is None:
            self.U_grid = None
        else:
            U = np.asarray(cp_upper_grid, dtype=np.float32)
            if U.ndim != 3:
                raise ValueError(f"cp_upper_grid must have shape (T,H,W), got {U.shape}")
            if U.shape[1] != self.grid_H or U.shape[2] != self.grid_W:
                raise ValueError(
                    f"cp_upper_grid H,W mismatch: grid=({self.grid_H},{self.grid_W}), U=({U.shape[1]},{U.shape[2]})"
                )
            self.U_grid = U

        # MPC rollout configuration
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.n_skip = int(n_skip)

        # Robot and safety geometry
        self.robot_rad = float(robot_rad)
        self.obstacle_rad = float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad

        # Control bounds
        self.min_v, self.max_v = float(min_linear_x), float(max_linear_x)
        self.min_w, self.max_w = float(min_angular_z), float(max_angular_z)

        # Monte Carlo sampling
        self.n_paths = int(n_paths)
        self.rng = np.random.default_rng(int(seed))
        self.weights = weights or FuncMPCWeights()
        self.last_best_vels: Optional[np.ndarray] = None

        # Adaptive soft-safety tuning
        self._target_slack = float(risk_level)
        self._eta = float(step_size)

        self.CP = CP

        # -----------------------------
        # Online CP calibration (Plan A)
        # -----------------------------

        self.beta_maps: List[Dict[int, float]] = [dict() for _ in range(self.n_steps)]

        # update step size
        self.beta_eta = 0.05

        # clip range for beta
        self.beta_min = -1.5
        self.beta_max = 1.5

        # optional mild decay toward zero
        self.beta_decay = 0.999

        self.theta_t = np.zeros(self.n_steps, dtype=np.float32)
        self.gamma_t = np.ones(self.n_steps, dtype=np.float32)

        # Target miscoverage level (e.g., 0.1 => 90% coverage)
        self.cp_alpha = 0.1

        # Online step size (learning rate)
        self.cp_eta = 0.01

        # Stability clipping range for theta (distance units)
        self.theta_min = -2.0
        self.theta_max = 2.0

        # Simulation timestep counter
        self.step_count = 0

        # Store past prediction tensors for delayed updates.
        # Each entry: {"time": int, "pred": np.ndarray(H, M, 2)}
        self.prediction_history = deque(maxlen=self.n_steps + 1)

        # Debug prints
        self._cp_debug = True

    # ---------------------------------------------------------------------
    # Grid geometry helpers (world <-> grid)
    # ---------------------------------------------------------------------

    def _world_to_grid_ij(self, pos_world: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Map a world coordinate (x,y) to the nearest grid index (i,j).

        The grid represents coordinates in:
          rel = pos_world - world_center
          rel_x, rel_y in [-box/2, box/2].
        """
        rel = np.asarray(pos_world, dtype=np.float32) - self.world_center
        u = (rel[0] + self.box / 2.0) / self.box * (self.grid_W - 1)
        v = (rel[1] + self.box / 2.0) / self.box * (self.grid_H - 1)

        if not (0.0 <= u <= (self.grid_W - 1) and 0.0 <= v <= (self.grid_H - 1)):
            return None

        j = int(np.rint(u))
        i = int(np.rint(v))
        return (i, j)

    def _grid_flat_index(self, ij: Tuple[int, int]) -> int:
        """Flatten (i,j) index into [0, H*W)."""
        i, j = ij
        return i * self.grid_W + j
    
    def _world_to_grid_ij_batch(self, x_world: np.ndarray):
        x = np.asarray(x_world, dtype=np.float32)
        rel = x - self.world_center[None, :]

        u = (rel[:, 0] + self.box / 2.0) / self.box * (self.grid_W - 1)
        v = (rel[:, 1] + self.box / 2.0) / self.box * (self.grid_H - 1)

        inside = (u >= 0.0) & (u <= (self.grid_W - 1)) & (v >= 0.0) & (v <= (self.grid_H - 1))

        j = np.rint(u).astype(np.int32)
        i = np.rint(v).astype(np.int32)

        j = np.clip(j, 0, self.grid_W - 1)
        i = np.clip(i, 0, self.grid_H - 1)
        return i, j, inside
    

    def _lookup_beta_batch(
        self,
        t_idx: int,
        flat_idx: np.ndarray,
        inside: np.ndarray,
    ) -> np.ndarray:
        """
        Return beta values for queried grid cells at horizon t_idx.
        Only inside-grid points receive learned beta; outside gets 0.
        """
        beta = np.zeros(flat_idx.shape[0], dtype=np.float32)
        beta_map = self.beta_maps[t_idx]
        if np.any(inside):
            vals = [beta_map.get(int(f), 0.0) for f in flat_idx[inside]]
            beta[inside] = np.asarray(vals, dtype=np.float32)
        return beta
    # ---------------------------------------------------------------------
    # Public MPC API
    # ---------------------------------------------------------------------

    def __call__(
        self,
        pos_x: float,
        pos_y: float,
        orientation_z: float,
        boxes=None,
        predictions=None,
        goal=None,
        *,
        obst_pred_traj: Optional[np.ndarray] = None,
        obst_mask: Optional[np.ndarray] = None,
        current_obs: Optional[np.ndarray] = None,
    ):
        """
        Compute the control action [v, w] for the current state.

        Inputs
        ------
        pos_x, pos_y, orientation_z
            Current robot pose.
        goal
            Goal position as (2,) array-like in world coordinates.
        boxes
            Optional list of static obstacle boxes with fields: pos, w, h, rad.
        predictions
            Optional dict-format dynamic predictions:
              {agent_id: np.ndarray(T_pred, 2)}.

        Alternative input format (trajectory + mask)
        -------------------------------------------
        obst_pred_traj:
            Array with shape (H, M, 2) (or (H,2) for single obstacle).
        obst_mask:
            Visibility mask with shape (H, M) (or (H,) for single).
            Invisible steps are treated as "far away" and thus ignored.
        """
        if goal is None:
            raise ValueError("goal must be provided.")
        goal = np.asarray(goal, dtype=np.float32)

        # Normalize dynamic predictions into dict format if obst_pred_traj was provided.
        if obst_pred_traj is not None:
            predictions = self._normalize_predictions(obst_pred_traj, obst_mask)

        if predictions is None:
            predictions = {}

        boxes = boxes or []

        t0 = time.perf_counter()

        self._update_local_beta_from_observation(current_obs)

        # 1) Sample candidate controls and roll out dynamics
        t_roll0 = time.perf_counter()
        paths, vels = self.generate_paths_random(pos_x, pos_y, orientation_z)
        t_roll1 = time.perf_counter()

        # 2) Hard feasibility filtering (static obstacles + CP-safe distance checks)
        t_filt0 = time.perf_counter()
        safe_paths, safe_vels, cp_violation, unsafe_pts, visited_cells = self.filter_unsafe_paths(
            paths, vels, boxes, predictions
        )
        t_filt1 = time.perf_counter()

        stats = {
            "n_paths": int(paths.shape[0]),
            "n_safe": int(0 if safe_paths is None else safe_paths.shape[0]),
            "cp_violation": float(cp_violation),
        }

        if safe_paths is None or safe_vels is None or safe_vels.shape[0] == 0:
            return None, {
                "feasible": False,
                "final_path": None,
                "cost": None,
                "timing": {
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                    "rollout_ms": (t_roll1 - t_roll0) * 1000.0,
                    "filter_ms": (t_filt1 - t_filt0) * 1000.0,
                },
                "counts": stats,
            }

        # 3) Score feasible candidates and pick the best
        t_score0 = time.perf_counter()
        best_idx, best_cost= self.score_paths(safe_paths, safe_vels, goal, predictions)

        self.last_best_vels = safe_vels[best_idx].copy()
        t_score1 = time.perf_counter()

        # Apply the first control in the best plan
        act = np.asarray(safe_vels[best_idx, 0], dtype=np.float32)
        self.last_best_vels = safe_vels[best_idx].copy()

        if predictions:
            self.prediction_history.append({
                "time": int(self.step_count),
                "visited_cells": visited_cells
            })

        info = {
            "feasible": True,
            "final_path": safe_paths[best_idx],
            "cost": float(best_cost),
            "timing": {
                "total_ms": (time.perf_counter() - t0) * 1000.0,
                "rollout_ms": (t_roll1 - t_roll0) * 1000.0,
                "filter_ms": (t_filt1 - t_filt0) * 1000.0,
                "score_ms": (t_score1 - t_score0) * 1000.0,
            },
            "counts": stats,
            "safety_weight": float(self.weights.w_safety),
        }
        self.step_count += 1

        return act, info

    # ---------------------------------------------------------------------
    # Prediction normalization helper
    # ---------------------------------------------------------------------

    def _normalize_predictions(
        self,
        obst_pred_traj: np.ndarray,
        obst_mask: Optional[np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """
        Convert (H, M, 2) (+ mask) predictions into dict format:
          {m: (H,2)} in world coordinates.

        Invisible steps are set to a very large value so they do not constrain distances.
        """
        pred_arr = np.asarray(obst_pred_traj, dtype=np.float32)

        # Allow (H,2) for single obstacle
        if pred_arr.ndim == 2 and pred_arr.shape[-1] == 2:
            pred_arr = pred_arr[:, None, :]

        if pred_arr.ndim != 3 or pred_arr.shape[-1] != 2:
            raise ValueError("obst_pred_traj must have shape (H,M,2) or (H,2).")

        H, M, _ = pred_arr.shape

        if obst_mask is None:
            mask = np.ones((H, M), dtype=bool)
        else:
            mask = np.asarray(obst_mask, dtype=bool)
            if mask.ndim == 1 and M == 1 and mask.shape[0] == H:
                mask = mask[:, None]
            if mask.shape != (H, M):
                raise ValueError(f"obst_mask must have shape {(H, M)}, got {mask.shape}.")

        pred_dict: Dict[int, np.ndarray] = {}
        for m in range(M):
            traj_m = pred_arr[:, m, :].copy()
            invis = ~mask[:, m]
            if np.any(invis):
                traj_m[invis] = 1e9  # effectively removes obstacle at those steps
            pred_dict[m] = traj_m

        return pred_dict
    

    def _update_local_beta_from_observation(
        self,
        current_obstacles_xy: Optional[np.ndarray],
    ):
        """
        Update beta only on grid cells that were actually queried in previous rollouts.

        current_obstacles_xy: (M_now, 2) actual observed obstacle positions at current time.
        This is used as "truth" for delayed horizon updates.
        """
        if current_obstacles_xy is None:
            return

        obs = np.asarray(current_obstacles_xy, dtype=np.float32)
        if obs.size == 0:
            return
        if obs.ndim == 1:
            obs = obs[None, :]

        # optional decay toward zero
        for t_idx in range(self.n_steps):
            beta_map = self.beta_maps[t_idx]
            if len(beta_map) == 0:
                continue
            for k in list(beta_map.keys()):
                beta_map[k] *= self.beta_decay
                if abs(beta_map[k]) < 1e-6:
                    del beta_map[k]

        # delayed update:
        # if a plan was made at time tau, then at current step_count = k
        # horizon index matched to current observation is h = k - tau - 1
        for record in self.prediction_history:
            tau = int(record["time"])
            h = self.step_count - tau - 1
            if h < 0 or h >= self.n_steps:
                continue

            step_cells = record["visited_cells"][h]
            if len(step_cells) == 0:
                continue

            beta_map = self.beta_maps[h]

            for flat, (x_ref, planned_d_lower) in step_cells.items():
                true_d = min_dist_robot_to_peds(x_ref, obs)

                conf_safe = (planned_d_lower >= self.safe_rad)
                true_safe = (true_d >= self.safe_rad)

                beta_old = beta_map.get(flat, 0.0)

                # rule:
                # conformal safe but actually unsafe  -> reduce d_lower -> beta down
                # conformal unsafe but actually safe -> increase d_lower -> beta up
                if conf_safe and (not true_safe):
                    beta_new = beta_old - self.beta_eta
                elif (not conf_safe) and true_safe:
                    beta_new = beta_old + self.beta_eta
                else:
                    beta_new = beta_old

                beta_map[flat] = float(np.clip(beta_new, self.beta_min, self.beta_max))

    # ---------------------------------------------------------------------
    # Safety filtering (hard constraints)
    # ---------------------------------------------------------------------

    def filter_unsafe_paths(
        self,
        paths: np.ndarray,           # (P, T+1, 2)
        vels: np.ndarray,            # (P, T, 2)
        boxes: List[Any],
        predictions: Dict[Any, np.ndarray],
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        float,
        Optional[List[Tuple[int, np.ndarray]]],
        List[Dict[int, Tuple[np.ndarray, float]]],
    ]:
        P, T1, _ = paths.shape
        T = T1 - 1

        boxes = boxes or []
        predictions = predictions or {}

        # (A) Static obstacles
        mask_static_unsafe = np.zeros(P, dtype=bool)
        if len(boxes) > 0:
            for box in boxes:
                center = box.pos
                sz = np.array([box.w, box.h], dtype=np.float32)
                th = float(box.rad)

                c, s = np.cos(th), np.sin(th)
                R = np.array([[c, -s], [s, c]], dtype=np.float32)

                lb = -0.5 * sz - self.robot_rad
                ub = 0.5 * sz + self.robot_rad

                transformed = (paths[:, 1:, :] - center) @ R
                coll = np.any(
                    np.all((transformed >= lb) & (transformed <= ub), axis=-1),
                    axis=-1,
                )
                mask_static_unsafe |= coll

        # (B) Dynamic obstacles
        mask_dynamic_unsafe = np.zeros(P, dtype=bool)
        cp_violation = 0.0
        unsafe_pts: List[Tuple[int, np.ndarray]] = []

        # visited_cells[t][flat_idx] = (representative_world_xy, planned_d_lower)
        visited_cells: List[Dict[int, Tuple[np.ndarray, float]]] = [dict() for _ in range(T)]

        if len(predictions) > 0:
            pred_list = list(predictions.values())
            pred_arr = np.asarray(pred_list, dtype=np.float32)  # (M, T_pred, 2)
            if pred_arr.ndim != 3:
                raise ValueError("predictions must be a dict of arrays shaped (T_pred, 2).")
            pred_arr = pred_arr.transpose(1, 0, 2)  # (T_pred, M, 2)

            T_use = min(T, pred_arr.shape[0])
            alive = ~mask_dynamic_unsafe

            for t in range(T_use):
                if not np.any(alive):
                    break

                x_t = paths[alive, t + 1, :]    # (Palive, 2)
                obs_t = pred_arr[t]             # (M, 2)

                diff = x_t[:, None, :] - obs_t[None, :, :]
                d_nom = np.min(np.linalg.norm(diff, axis=-1), axis=1)  # (Palive,)

                d_lower = d_nom.copy()

                if self.CP and (self.U_grid is not None):
                    tU = min(max(int(t), 0), int(self.U_grid.shape[0]) - 1)

                    i, j, inside = self._world_to_grid_ij_batch(x_t)
                    flat_idx = i * self.grid_W + j

                    U_vec = np.zeros((x_t.shape[0],), dtype=np.float32)
                    if np.any(inside):
                        U_vec[inside] = self.U_grid[tU, i[inside], j[inside]]

                    beta_vec = self._lookup_beta_batch(tU, flat_idx, inside)

                    # adaptive lower bound
                    d_lower = np.maximum(d_nom - U_vec + beta_vec, 0.0)

                    step_cells = visited_cells[t]
                    inside_idx = np.flatnonzero(inside)
                    for k in inside_idx:
                        f = int(flat_idx[k])
                        cand = (x_t[k].copy(), float(d_lower[k]))
                        old = step_cells.get(f, None)
                        if old is None or cand[1] < old[1]:
                            step_cells[f] = cand

                hit = d_lower < self.safe_rad
                if np.any(hit):
                    idx_alive = np.flatnonzero(alive)
                    hit_global = idx_alive[hit]
                    mask_dynamic_unsafe[hit_global] = True
                    alive[hit_global] = False

                    # optional debugging points
                    local_hit_idx = np.flatnonzero(hit)
                    for kk in local_hit_idx[:5]:
                        unsafe_pts.append((t, x_t[kk].copy()))

        mask_safe = ~(mask_static_unsafe | mask_dynamic_unsafe)

        if np.any(mask_safe):
            return (
                paths[mask_safe],
                vels[mask_safe],
                float(cp_violation),
                unsafe_pts,
                visited_cells,
            )

        return None, None, float(cp_violation), (unsafe_pts if len(unsafe_pts) > 0 else None), visited_cells

    # ---------------------------------------------------------------------
    # Candidate generation (Monte Carlo control sampling)
    # ---------------------------------------------------------------------

    def generate_paths_random(
        self,
        pos_x: float,
        pos_y: float,
        orientation_z: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample random piecewise-constant control sequences and roll out unicycle dynamics.

        Implementation details:
          - Control blocking: every n_skip steps share the same (v, w).
          - Warm start: if last_best_vels exists, reuse it (shifted) as the first candidate.
        """
        n_steps = int(self.n_steps)
        n_skip = int(self.n_skip)
        if n_steps <= 0:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        # Number of control epochs after blocking
        n_epochs = int(math.ceil(n_steps / max(1, n_skip)))

        # Sample epoch-wise controls
        v_epoch = self.rng.uniform(self.min_v, self.max_v, size=(self.n_paths, n_epochs)).astype(np.float32)
        w_epoch = self.rng.uniform(self.min_w, self.max_w, size=(self.n_paths, n_epochs)).astype(np.float32)

        # Warm start (first candidate)
        if self.last_best_vels is not None and self.last_best_vels.shape[0] >= 2:
            v_warm = np.append(self.last_best_vels[1:, 0], self.rng.uniform(self.min_v, self.max_v))
            w_warm = np.append(self.last_best_vels[1:, 1], self.rng.uniform(self.min_w, self.max_w))
            v_epoch[0, :] = v_warm[:n_epochs]
            w_epoch[0, :] = w_warm[:n_epochs]

        # Expand to per-step controls and truncate to horizon length
        v = np.repeat(v_epoch, n_skip, axis=1)[:, :n_steps]  # (P, n_steps)
        w = np.repeat(w_epoch, n_skip, axis=1)[:, :n_steps]  # (P, n_steps)

        # Roll out positions
        paths = np.zeros((self.n_paths, n_steps + 1, 2), dtype=np.float32)
        paths[:, 0, 0] = float(pos_x)
        paths[:, 0, 1] = float(pos_y)

        th = np.full((self.n_paths,), float(orientation_z), dtype=np.float32)
        dt = float(self.dt)

        for t in range(n_steps):
            paths[:, t + 1, 0] = paths[:, t, 0] + dt * v[:, t] * np.cos(th)
            paths[:, t + 1, 1] = paths[:, t, 1] + dt * v[:, t] * np.sin(th)
            th = th + dt * w[:, t]

        vels = np.stack([v, w], axis=-1).astype(np.float32)  # (P, n_steps, 2)
        return paths, vels

    # ---------------------------------------------------------------------
    # Scoring (MPC objective over feasible paths)
    # ---------------------------------------------------------------------

    def score_paths(
        self,
        paths: np.ndarray,                # (P, T+1, 2)
        vels: np.ndarray,                 # (P, T, 2)
        goal: np.ndarray,                 # (2,)
        predictions: Dict[Any, np.ndarray],
    ) -> Tuple[int, float]:
        """
        Score feasible candidate paths.

        Base costs (standard MPC):
          - intermediate goal tracking,
          - terminal goal tracking,
          - control effort.

        Optional soft safety shaping (only if predictions exist):
          - penalize safety margin violations using a smooth barrier on (safe_rad - d_lower),
          - include an urgency weight to emphasize earlier steps,
          - compute a "soft minimum slack" summary used to adapt w_safety online.

        Returns
        -------
        best_idx : int
            Index of the minimum-cost path.
        best_cost : float
            Minimum total cost value.
        best_min_slack : float
            Slack-like scalar for the selected path (used for adaptive tuning).
        """
        P, T1, _ = paths.shape
        T = T1 - 1

        # Standard MPC terms
        intermediate = self.weights.w_intermediate * np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2, -1))
        terminal = self.weights.w_terminal * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        control = self.weights.w_control * np.sum(vels ** 2, axis=(-2, -1))
        total_cost = intermediate + terminal + control

        best_idx = int(np.argmin(total_cost))
        best_cost = float(total_cost[best_idx])

        return best_idx, best_cost
