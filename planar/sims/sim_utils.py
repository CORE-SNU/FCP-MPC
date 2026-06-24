import os as _os, sys as _sys
# Shared utils.py lives at the repo root (two levels up, alongside quadrotor/); put it
# on sys.path so the absolute `from utils import` resolves.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
del _os, _sys

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Local module imports
from utils import build_grid, reflect_vectorized, distance_field_points


def reflect_to_box_xy(xy: np.ndarray, box: float) -> np.ndarray:
    xy = xy.astype(np.float32, copy=True)
    for i in range(2):
        while xy[i] < 0 or xy[i] > box:
            if xy[i] < 0: xy[i] = -xy[i]
            if xy[i] > box: xy[i] = 2 * box - xy[i]
    return xy

def unicycle_step(xy: np.ndarray, th: float, v: float, w: float, dt: float):
    x = xy[0] + dt * v * np.cos(th)
    y = xy[1] + dt * v * np.sin(th)
    th = th + dt * w
    return np.array([x, y], dtype=np.float32), float(th)

def align_mask_dims(mask, target_T, target_M):
    if mask.ndim == 3 and mask.shape[1] == target_M and mask.shape[2] == target_T:
        return np.transpose(mask, (0, 2, 1))
    if mask.ndim == 3 and mask.shape[1] == target_T and mask.shape[2] == target_M:
        return mask
    if mask.ndim == 2 and mask.shape[1] == target_M:
        return np.repeat(mask[:, None, :], target_T, axis=1)
    if mask.ndim == 2 and mask.shape[1] == target_T:
        return np.repeat(mask[:, :, None], target_M, axis=2)
    raise ValueError(f"Unsupported mask shape: {mask.shape}")

def build_cv_predictions(trajs_true, box, horizon):
    trajs_pred = trajs_true.copy().astype(np.float32)
    N, T = trajs_true.shape[:2]
    if T <= horizon + 1: return trajs_pred
    for t in range(horizon + 1, T):
        base = t - horizon
        v = trajs_true[:, base] - trajs_true[:, base - 1]
        curr = trajs_true[:, base].copy()
        for _ in range(horizon):
            curr = reflect_vectorized(curr + v, box)
        trajs_pred[:, t] = curr
    return trajs_pred

def build_cv_predictions_leadtime(trajs_true: np.ndarray, box: float, horizon: int) -> np.ndarray:
    """
    Returns:
        pred: (N, T, time_horizon, M, 2)
        pred[n, t, i-1, m] = y_hat_{t+i|t} for agent m
    """
    if trajs_true.ndim == 3:
        trajs_true = trajs_true[:, :, None, :]  # (N,T,1,2)

    N, T, M, _ = trajs_true.shape
    pred = np.zeros((N, T, horizon, M, 2), dtype=np.float32)

    for t in range(1, T):  # need t-1
        y_tm1 = trajs_true[:, t-1]  # (N,M,2)
        y_t   = trajs_true[:, t]    # (N,M,2)
        v = (y_t - y_tm1).astype(np.float32)
        curr = y_t.copy().astype(np.float32)
        for i in range(horizon):
            curr = reflect_vectorized(curr + v, box)
            pred[:, t, i] = curr

    return pred

def make_cv_prediction_from_two_points(y_tm1, y_t, n_steps, box):
    v = (y_t - y_tm1).astype(np.float32)
    pred = np.zeros((n_steps, y_t.shape[0], 2), dtype=np.float32)
    curr = y_t.copy().astype(np.float32)
    for t in range(n_steps):
        curr = reflect_vectorized(curr + v, box)
        pred[t] = curr
    return pred

def compute_lower_field_single_step(obst_pred_step, obst_mask_step, Xg, Yg, g_up, world_center):
    lower = np.full(Xg.shape, np.inf, dtype=np.float32)
    for m in range(obst_pred_step.shape[0]):
        if not bool(obst_mask_step[m]): continue
        y_hat = obst_pred_step[m] - world_center
        d_pred = distance_field_points(y_hat, Xg, Yg)
        lower_m = np.maximum(d_pred - g_up, 0.0)
        lower = np.minimum(lower, lower_m)
    return lower if np.isfinite(lower).any() else np.full(Xg.shape, 20.0, dtype=np.float32)

def min_dist_robot_to_peds(robot_xy, peds_xy):
    if peds_xy.size == 0:
        return float("inf")
    d = peds_xy - robot_xy[None, :]
    return float(np.sqrt(np.sum(d * d, axis=1)).min())