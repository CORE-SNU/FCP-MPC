from __future__ import annotations

import os as _os, sys as _sys
# Shared utils.py lives at the repo root (two levels up, alongside quadrotor/); put it
# on sys.path so the absolute `from utils import` resolves from anywhere.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
del _os, _sys

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from utils import reflect_to_box 


@dataclass
class ConstantVelocityPredictor:
    """
    Constant-Velocity predictor with per-step reflection.

    Parameters
    ----------
    box : float
        Side length of the square domain [0, box]^2 used for reflection.
    """
    box: float = 100.0

    # --------------- core API ---------------

    def predict_h(
        self,
        y_t: np.ndarray,          # shape (2,)
        y_tminus1: np.ndarray,    # shape (2,)
        h: int
    ) -> np.ndarray:
        """
        Predict the position at (t+h) by repeating the current velocity h times.
        Reflection is applied at each step.

        y_{t+1|t} = reflect(y_t + (y_t - y_{t-1}))
        ...
        y_{t+h|t} repeats the same velocity and reflects per-step.
        """
        if h < 1:
            raise ValueError("h must be >= 1")

        v = np.asarray(y_t, float) - np.asarray(y_tminus1, float)
        y_hat = np.asarray(y_t, float).copy()
        for _ in range(h):
            y_hat = y_hat + v
            y_hat = reflect_to_box(y_hat, self.box)
        return y_hat

    # --------------- helpers ---------------

    def predict_one_step(
        self,
        y_t: np.ndarray,
        y_tminus1: np.ndarray
    ) -> np.ndarray:
        """Shorthand for predict_h(..., h=1)."""
        return self.predict_h(y_t, y_tminus1, h=1)

    def predict_over_trajectory(
        self,
        traj: np.ndarray,   # shape (T, 2)
        h: int,
        t_start: int = 1,
        t_end: Optional[int] = None,
    ) -> np.ndarray:
        """
        Rolling h-step prediction across a single trajectory.

        For each decision time t in [t_start .. t_end-1], we return y_{t+h|t}.
        Validity requires indices (t-1, t, t+h) to exist.

        Returns
        -------
        preds : np.ndarray, shape (T, 2), filled with NaNs except valid t+h
            preds[t+h] = y_{t+h|t}
        """
        T = int(traj.shape[0])
        if t_end is None:
            t_end = T - h  # exclusive upper bound for t
        if t_start < 1 or t_end > T - h:
            raise ValueError("Invalid range: ensure t in [1 .. T-h-1] is valid.")

        preds = np.full((T, 2), np.nan, dtype=float)
        for t in range(t_start, t_end):
            y_tm1, y_t = traj[t - 1], traj[t]
            preds[t + h] = self.predict_h(y_t, y_tm1, h)
        return preds

    def batch_predict_over_trajectories(
        self,
        trajs: np.ndarray,  # shape (N, T, 2)
        h: int,
        t_start: int = 1,
        t_end: Optional[int] = None,
    ) -> np.ndarray:
        """
        Batched variant of predict_over_trajectory.

        Returns
        -------
        preds : np.ndarray, shape (N, T, 2)
            For each i and each valid t: preds[i, t+h] = y_{t+h|t}
            Non-valid indices are NaN.
        """
        N, T, _ = trajs.shape
        preds = np.full((N, T, 2), np.nan, dtype=float)
        for i in range(N):
            preds[i] = self.predict_over_trajectory(trajs[i], h=h, t_start=t_start, t_end=t_end)
        return preds
