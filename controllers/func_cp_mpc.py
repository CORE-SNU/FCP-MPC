# controllers/func_cp_mpc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import copy

import math
import time

import numpy as np

from cp.functional_cp import CPStepParameters
from utils import build_grid, distance_field_points


class CPOnlineAdapter:
    """
    Maintains residual buffers and adapts coefficient quantiles using ACP.
    """

    def __init__(
        self,
        cp_params: List[CPStepParameters],
        *,
        world_center: np.ndarray,
        box: float,
        grid_H: int,
        grid_W: int,
        target_violation: float = 0.1,
        eta: float = 0.15,
        warmup_frames: int = 15,
        coeff_limits: Tuple[float, float] = (1e-4, 10.0),
    ):
        if cp_params is None or len(cp_params) == 0:
            raise ValueError("CPOnlineAdapter requires non-empty cp_params.")

        self.base_params = sorted(copy.deepcopy(cp_params), key=lambda p: int(p.t_idx))
        self.idx_map = {int(p.t_idx): i for i, p in enumerate(self.base_params)}
        self.horizon = len(self.base_params)
        self.world_center = np.asarray(world_center, dtype=np.float32).reshape(2,)
        self.box = float(box)
        self.grid_H = int(grid_H)
        self.grid_W = int(grid_W)
        _, _, self.Xg, self.Yg = build_grid(self.box, self.grid_H, self.grid_W)
        self.vector_size = int(self.grid_H * self.grid_W)

        self.target_violation = float(target_violation)
        self.eta = float(eta)
        self.warmup_frames = max(0, int(warmup_frames))
        self.coeff_limits = (float(coeff_limits[0]), float(coeff_limits[1]))

        self.current_coeffs: List[np.ndarray] = [
            np.asarray(p.coeff_upper, dtype=np.float32).copy() for p in self.base_params
        ]

        self.pending_preds: Dict[int, Dict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    def step(
        self,
        frame_idx: int,
        observations: Optional[Dict[Any, np.ndarray]],
        predictions: Optional[Dict[Any, np.ndarray]],
    ) -> bool:
        self._register_predictions(frame_idx, predictions)
        actual_pts = self._extract_actual_points(observations)
        matured = self._pop_predictions(frame_idx)

        if actual_pts is None or actual_pts.size == 0 or not matured:
            self._cleanup(frame_idx)
            return False

        updated = False
        for step_idx, pred_pts in matured.items():
            field = self._build_residual_field(pred_pts, actual_pts)
            if field is None:
                continue
            coeff = self._project_coefficients(step_idx, field)
            if coeff is None:
                continue
            if frame_idx >= self.warmup_frames:
                updated |= self._apply_update(step_idx, coeff)

        self._cleanup(frame_idx)
        return updated

    def snapshot(self) -> List[CPStepParameters]:
        params = []
        for p in self.base_params:
            idx = self.idx_map[int(p.t_idx)]
            q = copy.deepcopy(p)
            q.coeff_upper = self.current_coeffs[idx].copy()
            params.append(q)
        return params

    # --- helpers ---

    def _register_predictions(self, frame_idx: int, predictions: Optional[Dict[Any, np.ndarray]]) -> None:
        if not predictions:
            return
        for traj in predictions.values():
            if traj is None:
                continue
            arr = np.asarray(traj, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            steps = min(arr.shape[0], self.horizon)
            for step_idx in range(steps):
                target_frame = frame_idx + step_idx + 1
                self.pending_preds[target_frame][step_idx].append(arr[step_idx, :2].copy())

    def _pop_predictions(self, frame_idx: int) -> Dict[int, np.ndarray]:
        frame_store = self.pending_preds.pop(frame_idx, None)
        if not frame_store:
            return {}
        out: Dict[int, np.ndarray] = {}
        for step_idx, pts in frame_store.items():
            if len(pts) == 0:
                continue
            out[step_idx] = np.asarray(pts, dtype=np.float32)
        return out

    def _extract_actual_points(self, observations: Optional[Dict[Any, np.ndarray]]) -> Optional[np.ndarray]:
        if not observations:
            return None
        pts = []
        for traj in observations.values():
            if traj is None or len(traj) == 0:
                continue
            arr = np.asarray(traj, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                pts.append(arr[-1, :2])
        if not pts:
            return None
        return np.asarray(pts, dtype=np.float32)

    def _build_residual_field(self, pred_pts: np.ndarray, actual_pts: np.ndarray) -> Optional[np.ndarray]:
        if pred_pts is None or pred_pts.size == 0:
            return None
        if actual_pts is None or actual_pts.size == 0:
            return None
        pred_rel = pred_pts - self.world_center[None, :]
        true_rel = actual_pts - self.world_center[None, :]
        sdf_pred = distance_field_points(pred_rel, self.Xg, self.Yg)
        sdf_true = distance_field_points(true_rel, self.Xg, self.Yg)
        return (sdf_pred - sdf_true).astype(np.float32)

    def _project_coefficients(self, step_idx: int, field: np.ndarray) -> Optional[np.ndarray]:
        idx = self.idx_map.get(int(step_idx))
        if idx is None:
            return None
        vec = field.reshape(-1).astype(np.float32)
        if vec.shape[0] != self.vector_size:
            return None
        params = self.base_params[idx]
        centered = vec - params.mean.reshape(-1)
        coeff = centered @ params.phi_basis.T
        return coeff.astype(np.float32)

    def _apply_update(self, step_idx: int, coeff: np.ndarray) -> bool:
        idx = self.idx_map.get(int(step_idx))
        if idx is None:
            return False

        current = self.current_coeffs[idx]
        indicator = (np.abs(coeff) > current).astype(np.float32)
        delta = self.eta * (indicator - self.target_violation)
        if not np.any(np.abs(delta) > 1e-6):
            return False

        updated = current + delta
        updated = np.clip(updated, self.coeff_limits[0], self.coeff_limits[1])
        self.current_coeffs[idx] = updated.astype(np.float32)
        return True

    def _cleanup(self, frame_idx: int) -> None:
        stale = [k for k in list(self.pending_preds.keys()) if k + self.horizon < frame_idx]
        for key in stale:
            self.pending_preds.pop(key, None)

# NOTE:
# - CPStepParameters is assumed to be produced offline for each horizon index (time-to-go),
#   and cached for online pointwise queries along rollouts.

# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class FuncMPCWeights:
    """
    Weights for the MPC objective.

    Parameters
    ----------
    w_terminal : float
        Terminal goal tracking weight.
    w_intermediate : float
        Intermediate goal tracking weight (sum along the horizon).
    w_control : float
        Control effort weight.
    w_safety : float
        Weight for the soft safety penalty (used only when safety_mode="soft").
    """
    w_terminal: float = 10.0
    w_intermediate: float = 1.0
    w_control: float = 0.001
    w_safety: float = 50.0


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
        cp_params: Optional[List[CPStepParameters]] = None,
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
        CP: bool = True,
        adaptive: bool = True,
        safety_mode: str = "hard",
    ):
        """
        Parameters
        ----------
        cp_params : list
            Offline-cached parameters for the CP envelope, one per horizon index i.
        adaptive : bool
            If True, use CPOnlineAdapter to update coeff quantiles online (ACP).
            If False, use fixed offline coefficients.
        safety_mode : str
            "hard" — filter out candidate paths that violate the CP safety constraint.
            "soft" — skip dynamic-obstacle filtering; instead add a soft penalty in scoring.
        """
        # Workspace/grid configuration
        self.box = float(box)
        self.world_center = np.asarray(world_center, dtype=np.float32)
        self.grid_H, self.grid_W = int(grid_H), int(grid_W)

        self.params: Dict[int, CPStepParameters] = {}
        self.U_grid: Optional[np.ndarray] = None

        if cp_params is None or len(cp_params) == 0:
            raise ValueError("FunctionalCPMPC requires a non-empty cp_params list.")
        self.set_cp_params(cp_params)

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
        self.last_best_vels: Optional[np.ndarray] = None  # warm-start storage

        self.CP = CP
        self.adaptive = bool(adaptive)
        self.safety_mode = str(safety_mode).lower()
        if self.safety_mode not in ("hard", "soft"):
            raise ValueError(f"safety_mode must be 'hard' or 'soft', got '{safety_mode}'")

        self._frame_idx = 0
        self._cp_adapter: Optional[CPOnlineAdapter] = None
        if self.CP and self.adaptive:
            self._cp_adapter = CPOnlineAdapter(
                self.get_cp_params(),
                world_center=self.world_center,
                box=self.box,
                grid_H=self.grid_H,
                grid_W=self.grid_W,
                target_violation=0.1,
                eta=0.05,
                warmup_frames=self.n_steps,
            )

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

    def set_cp_params(self, params_list: List[CPStepParameters]) -> None:
        """
        Cache CPStepParameters for online envelope evaluations.
        """
        if params_list is None or len(params_list) == 0:
            raise ValueError("cp_params must be a non-empty list.")
        for p in params_list:
            if not hasattr(p, "mean"):
                raise ValueError(
                    "CPStepParameters missing PCA mean; regenerate the CP cache with the new format."
                )
        sorted_params = sorted(params_list, key=lambda p: int(p.t_idx))
        self.params = {int(p.t_idx): p for p in sorted_params}
        self.U_grid = self._build_cp_grid_from_params(sorted_params)

    def get_cp_params(self) -> List[CPStepParameters]:
        """
        Return the cached CPStepParameters sorted by horizon index.
        """
        return [self.params[k] for k in sorted(self.params.keys())]

    def _build_cp_grid_from_params(self, params_list: List[CPStepParameters]) -> np.ndarray:
        """
        Reconstruct the envelope grid from CPStepParameters.
        """
        H, W = self.grid_H, self.grid_W
        D = H * W
        grids: List[np.ndarray] = []

        for p in params_list:
            phi = np.asarray(p.phi_basis, dtype=np.float32)
            if phi.shape[1] != D:
                raise ValueError(
                    f"phi_basis dimension mismatch: expected {D}, got {phi.shape[1]}"
                )
            mean_vec = np.asarray(p.mean, dtype=np.float32).reshape(D,)
            AT = phi.T  # (D, p_eff)

            coeff = np.asarray(p.coeff_upper, dtype=np.float32).reshape(-1)
            g_upper_vec = mean_vec + AT @ coeff + float(p.epsilon)
            grids.append(g_upper_vec.reshape(H, W))

        return np.stack(grids, axis=0).astype(np.float32)

    # ---------------------------------------------------------------------
    # Functional CP envelope evaluation: U_i(x)
    # ---------------------------------------------------------------------

    def evaluate_U(self, pos_world: np.ndarray, t_idx: int) -> float:
        """
        Evaluate U_i(x) either from cached CPStepParameters or from the grid.
        """
        batch = self.evaluate_U_batch(np.asarray(pos_world, dtype=np.float32)[None, :], t_idx)
        return float(batch[0])

    def evaluate_U_batch(self, pos_world: np.ndarray, t_idx: int) -> np.ndarray:
        """
        pos_world: (P,2)
        returns: (P,) U_i(x)
        """
        X = np.asarray(pos_world, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("pos_world must have shape (P, 2).")

        if self.U_grid is not None:
            return self._lookup_from_grid(X, t_idx)
        if self.params:
            return self._evaluate_from_params_batch(X, t_idx)
        return np.ones((X.shape[0],), dtype=np.float32)

    def _evaluate_from_params_batch(self, pos_world: np.ndarray, t_idx: int) -> np.ndarray:
        p = self.params.get(int(t_idx))
        if p is None:
            return np.ones((pos_world.shape[0],), dtype=np.float32)

        rel = pos_world - self.world_center[None, :]
        u = (rel[:, 0] + self.box / 2.0) / self.box * (self.grid_W - 1)
        v = (rel[:, 1] + self.box / 2.0) / self.box * (self.grid_H - 1)

        inside = (u >= 0.0) & (u <= (self.grid_W - 1)) & (v >= 0.0) & (v <= (self.grid_H - 1))
        out = np.ones((pos_world.shape[0],), dtype=np.float32)
        if not np.any(inside):
            return out

        j = np.rint(u[inside]).astype(np.int32)
        i = np.rint(v[inside]).astype(np.int32)
        idx = (i * self.grid_W + j).astype(np.int32)

        phi = p.phi_basis[:, idx].astype(np.float32, copy=False)
        coeff = p.coeff_upper.astype(np.float32, copy=False)
        mean_vec = np.asarray(p.mean, dtype=np.float32).reshape(-1)
        mean_vals = mean_vec[idx]
        Ui = (mean_vals + coeff @ phi + float(p.epsilon)).astype(np.float32)
        out[inside] = Ui
        return out

    def _lookup_from_grid(self, pos_world: np.ndarray, t_idx: int) -> np.ndarray:
        if self.U_grid is None:
            return np.ones((pos_world.shape[0],), dtype=np.float32)
        T = int(self.U_grid.shape[0])
        idx_t = min(max(int(t_idx), 0), T - 1)

        rel = pos_world - self.world_center[None, :]
        u = (rel[:, 0] + self.box / 2.0) / self.box * (self.grid_W - 1)
        v = (rel[:, 1] + self.box / 2.0) / self.box * (self.grid_H - 1)
        inside = (u >= 0.0) & (u <= (self.grid_W - 1)) & (v >= 0.0) & (v <= (self.grid_H - 1))
        out = np.ones((pos_world.shape[0],), dtype=np.float32)
        if not np.any(inside):
            return out

        j = np.rint(u[inside]).astype(np.int32)
        i = np.rint(v[inside]).astype(np.int32)
        out[inside] = self.U_grid[idx_t, i, j]
        return out

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
        observations: Optional[Dict[Any, np.ndarray]] = None,
        *,
        obst_pred_traj: Optional[np.ndarray] = None,  # (H, M, 2) or (H,2)
        obst_mask: Optional[np.ndarray] = None,       # (H, M) or (H,)
    ):
        """
        Compute the control action [v, w] for the current state.

        Inputs
        ------
        pos_x, pos_y, orientation_z
            Current robot pose.
        goal
            Goal position as (2,) array-like in world coordinates.
        observations
            Optional dict of tracked agent trajectories used for online CP adaptation.
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
        if observations is None:
            obs_dict: Dict[Any, np.ndarray] = {}
        else:
            obs_dict = observations

        self._frame_idx += 1
        if self.CP and self.adaptive and self._cp_adapter is not None:
            updated = self._cp_adapter.step(self._frame_idx, obs_dict, predictions)
            if updated:
                self.set_cp_params(self._cp_adapter.snapshot())

        boxes = boxes or []

        t0 = time.perf_counter()

        # 1) Sample candidate controls and roll out dynamics
        t_roll0 = time.perf_counter()
        paths, vels = self.generate_paths_random(pos_x, pos_y, orientation_z)
        t_roll1 = time.perf_counter()

        # 2) Feasibility filtering
        #    hard mode: strict dynamic-obstacle filter (paths violating CP safety removed)
        #    soft mode: only static-obstacle filter; dynamic safety handled as cost penalty
        t_filt0 = time.perf_counter()
        safe_paths, safe_vels, cp_violation = self.filter_unsafe_paths(paths, vels, boxes, predictions)
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
        best_idx, best_cost = self.score_paths(safe_paths, safe_vels, goal, predictions)
        self.last_best_vels = safe_vels[best_idx].copy()
        t_score1 = time.perf_counter()

        # Apply the first control in the best plan
        act = np.asarray(safe_vels[best_idx, 0], dtype=np.float32)

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
        }
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

    # ---------------------------------------------------------------------
    # Safety filtering (hard constraints)
    # ---------------------------------------------------------------------

    def filter_unsafe_paths(
        self,
        paths: np.ndarray,           # (P, T+1, 2)
        vels: np.ndarray,            # (P, T, 2)
        boxes: List[Any],
        predictions: Dict[Any, np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Hard feasibility filtering.

        A candidate path is rejected if:
          (A) It collides with any static box obstacle (expanded by robot radius), OR
          (B) It violates dynamic safety when compared against:
                d_lower(x) = max(d_nom(x) - U_i(x), 0).

        Notes
        -----
        - This stage is intentionally strict: it defines the feasible set.
        - Soft safety shaping (used in scoring) is only for ranking within feasible candidates.
        """
        P, T1, _ = paths.shape
        T = T1 - 1

        boxes = boxes or []
        predictions = predictions or {}

        # (A) Static obstacles: OBB collision check in obstacle local frame
        mask_static_unsafe = np.zeros(P, dtype=bool)
        if len(boxes) > 0:
            for box in boxes:
                center = box.pos
                sz = np.array([box.w, box.h], dtype=np.float32)
                th = float(box.rad)

                c, s = np.cos(th), np.sin(th)
                R = np.array([[c, -s], [s, c]], dtype=np.float32)

                # Expand box by robot radius (Minkowski sum approximation for axis-aligned bounds in local frame)
                lb = -0.5 * sz - self.robot_rad
                ub = 0.5 * sz + self.robot_rad

                # Transform candidate points into obstacle local coordinates
                transformed = (paths[:, 1:, :] - center) @ R  # (P, T, 2)

                coll = np.any(
                    np.all((transformed >= lb) & (transformed <= ub), axis=-1),
                    axis=-1,
                )
                mask_static_unsafe |= coll

        # (B) Dynamic obstacles: CP-conformalized distance lower bound checks
        # In soft mode, skip hard filtering — safety is handled as a cost penalty in score_paths.
        mask_dynamic_unsafe = np.zeros(P, dtype=bool)
        cp_violation = 0.0

        if self.safety_mode == "soft":
            mask_safe = ~mask_static_unsafe
            if np.any(mask_safe):
                return paths[mask_safe], vels[mask_safe], cp_violation
            return None, None, cp_violation

        if len(predictions) > 0:
            pred_list = list(predictions.values())
            pred_arr = np.asarray(pred_list, dtype=np.float32)  # (M, T_pred, 2)
            if pred_arr.ndim != 3:
                raise ValueError("predictions must be a dict of arrays shaped (T_pred, 2).")
            pred_arr = pred_arr.transpose(1, 0, 2)  # (T_pred, M, 2)

            T_use = min(T, pred_arr.shape[0])

            # Track which paths are still "alive" (not yet marked unsafe)
            alive = ~mask_dynamic_unsafe  # (P,)

            for t in range(T_use):
                if not np.any(alive):
                    break

                x_t = paths[alive, t + 1, :]               # (Palive, 2)
                obs_t = pred_arr[t]                        # (M, 2)

                # d_nom for all alive paths: min_m ||x - obs||
                diff = x_t[:, None, :] - obs_t[None, :, :] # (Palive, M, 2)
                d_nom = np.min(np.linalg.norm(diff, axis=-1), axis=1)  # (Palive,)

                d_lower = d_nom
                if self.CP:
                    U_vec = self.evaluate_U_batch(x_t, t)  # (Palive,)
                    d_lower = np.maximum(d_nom - U_vec, 0.0)
                
                # ICS relaxation

                a_lat_max = self.max_v * self.max_w
                
                elapsed_time = t * self.dt
                delta_evade = 0.5 * a_lat_max * (elapsed_time ** 2)
                
                effective_r_safe = max(0.0, self.safe_rad - delta_evade)

                hit = d_lower < effective_r_safe

                if np.any(hit):
                    idx_alive = np.flatnonzero(alive)      # indices in [0,P)
                    mask_dynamic_unsafe[idx_alive[hit]] = True
                    alive[idx_alive[hit]] = False

                mask_safe = ~(mask_static_unsafe | mask_dynamic_unsafe)

        if np.any(mask_safe):
            return paths[mask_safe], vels[mask_safe], float(cp_violation)

        return None, None, float(cp_violation)

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
          - Control blocking: every `n_skip` steps share the same (v, w).
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
    
    def generate_paths_guided(self, pos_x, pos_y, orientation_z):
        n_steps = self.n_steps
        n_skip = self.n_skip
        n_epochs = int(math.ceil(n_steps / max(1, n_skip)))
        
        n_explo = int(self.n_paths * 0.7)
        n_explor = self.n_paths - n_explo

        v_explor = self.rng.uniform(self.min_v, self.max_v, size=(n_explor, n_epochs))
        w_explor = self.rng.uniform(self.min_w, self.max_w, size=(n_explor, n_epochs))

        if self.last_best_vels is not None:
            prev_v = np.append(self.last_best_vels[1:, 0], self.last_best_vels[-1, 0])[:n_epochs]
            prev_w = np.append(self.last_best_vels[1:, 1], self.last_best_vels[-1, 1])[:n_epochs]
            
            v_explo = self.rng.normal(loc=prev_v, scale=(self.max_v - self.min_v)*0.1, size=(n_explo, n_epochs))
            w_explo = self.rng.normal(loc=prev_w, scale=(self.max_w - self.min_w)*0.1, size=(n_explo, n_epochs))
            
            v_explo = np.clip(v_explo, self.min_v, self.max_v)
            w_explo = np.clip(w_explo, self.min_w, self.max_w)
            
            v_epoch = np.vstack([v_explo, v_explor])
            w_epoch = np.vstack([w_explo, w_explor])
        else:
            v_epoch = self.rng.uniform(self.min_v, self.max_v, size=(self.n_paths, n_epochs))
            w_epoch = self.rng.uniform(self.min_w, self.max_w, size=(self.n_paths, n_epochs))

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
        predictions: Optional[Dict] = None,
    ) -> Tuple[int, float]:

        P, T1, _ = paths.shape
        T = T1 - 1

        intermediate = self.weights.w_intermediate * np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2, -1))
        terminal = self.weights.w_terminal * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        control = self.weights.w_control * np.sum(vels ** 2, axis=(-2, -1))
        total_cost = intermediate + terminal + control

        # Soft safety penalty: sum over horizon of max(0, r_safe - d_lower)^2
        if self.safety_mode == "soft" and predictions and len(predictions) > 0:
            pred_list = list(predictions.values())
            pred_arr = np.asarray(pred_list, dtype=np.float32)   # (M, T_pred, 2)
            if pred_arr.ndim == 3:
                pred_arr = pred_arr.transpose(1, 0, 2)           # (T_pred, M, 2)
                T_use = min(T, pred_arr.shape[0])
                safety_pen = np.zeros(P, dtype=np.float32)

                for t in range(T_use):
                    x_t = paths[:, t + 1, :]                     # (P, 2)
                    obs_t = pred_arr[t]                           # (M, 2)
                    diff = x_t[:, None, :] - obs_t[None, :, :]   # (P, M, 2)
                    d_nom = np.min(np.linalg.norm(diff, axis=-1), axis=1)  # (P,)

                    if self.CP:
                        U_vec = self.evaluate_U_batch(x_t, t)    # (P,)
                        d_lower = np.maximum(d_nom - U_vec, 0.0)
                    else:
                        d_lower = d_nom

                    # ICS-relaxed effective safety radius
                    a_lat_max = self.max_v * self.max_w
                    delta_evade = 0.5 * a_lat_max * (t * self.dt) ** 2
                    effective_r_safe = max(0.0, self.safe_rad - delta_evade)

                    violation = np.maximum(0.0, effective_r_safe - d_lower)
                    safety_pen += violation ** 2

                total_cost = total_cost + self.weights.w_safety * safety_pen

        best_idx = int(np.argmin(total_cost))
        best_cost = float(total_cost[best_idx])
        return best_idx, best_cost
