from __future__ import annotations

import os as _os, sys as _sys
# Shared controllers/, cp/, utils.py and viz_traj.py live at the repo root (alongside
# quadrotor/); put it on sys.path so those absolute imports resolve from anywhere.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _os, _sys

import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_eth_pickle_preprocessed
from cp.functional_cp import compute_cp_upper_envelopes
from utils import build_grid, reflect_vectorized, animate_cp_comparison_multi, distance_field_points, render_cp_comparison_paper_multi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = "zara1"

H, W = 128, 128
BOX = 100.0
SCALING = None

WORLD_CENTER = np.array([BOX / 2.0, BOX / 2.0], dtype=np.float32)

EGO_BOX = 60.0

TSTEPS = 12
N_TRAIN = 10000
SEED = 2023
TIME_HORIZON = 2
EVAL_SEEDS = np.arange(SEED + 100, SEED + 110)

SAFE_THRESHOLD = 4

P_BASE = 5
K_MIX = 4
ALPHA = 0.05
TEST_SIZE = 0.30
RANDOM_STATE = 0
N_JOBS = max(1, (os.cpu_count() or 4) - 2)
BACKEND = "loky"

AGENT_SOURCE = "random_walk"
AGENT_FIXED_XY = (BOX / 2, BOX / 2)

NUM_PEDS = 20



def predict_h_step_cv(y_t: np.ndarray, y_tm1: np.ndarray, h: int, box: float) -> np.ndarray:
    v = y_t - y_tm1
    y_hat = np.array(y_t, dtype=float)
    curr = y_hat.copy()
    for _ in range(h):
        curr = curr + v
        curr = reflect_vectorized(curr, box)
    return curr.astype(np.float32)


def build_cv_predictions(trajs_true: np.ndarray, box: float, horizon: int) -> np.ndarray:
    if trajs_true.ndim not in [3, 4]:
        raise ValueError("trajs_true must be (N, T, 2) or (N, T, M, 2)")
    
    N = trajs_true.shape[0]
    T = trajs_true.shape[1]
    trajs_pred = trajs_true.copy().astype(np.float32)
    
    if T <= horizon + 1:
        return trajs_pred

    for t in range(horizon + 1, T):
        base = t - horizon
        if base - 1 < 0: continue
        
        y_base = trajs_true[:, base]
        y_prev = trajs_true[:, base - 1]
        
        v = y_base - y_prev
        
        curr = y_base.copy()
        for _ in range(horizon):
            curr = curr + v
            curr = reflect_vectorized(curr, box)
            
        trajs_pred[:, t] = curr

    return trajs_pred


def build_agent_traj(T: int, source: str, box: float, obst_ref: np.ndarray | None = None):
    if source == "fixed":
        return np.tile(np.asarray(AGENT_FIXED_XY, dtype=np.float32), (T, 1))
    if source == "copy_obst" and obst_ref is not None:
        return obst_ref.copy()
    if source == "random_walk":
        rng = np.random.default_rng(42)
        xy = np.array(AGENT_FIXED_XY, dtype=np.float32)
        traj = [xy.copy()]
        for _ in range(T - 1):
            step = rng.normal(scale=1.0, size=2).astype(np.float32)
            xy = reflect_vectorized(xy + step, box)
            traj.append(xy.copy())
        return np.stack(traj, axis=0)
    raise ValueError(f"Unsupported AGENT_SOURCE={source}")


def align_mask_dims(mask: np.ndarray, target_T: int, target_M: int) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[1] == target_M and mask.shape[2] == target_T:
        return np.transpose(mask, (0, 2, 1))
    
    if mask.ndim == 3 and mask.shape[1] == target_T and mask.shape[2] == target_M:
        return mask

    if mask.ndim == 2 and mask.shape[1] == target_M:
        mask_exp = np.expand_dims(mask, axis=1)
        return np.repeat(mask_exp, target_T, axis=1)

    if mask.ndim == 2 and mask.shape[1] == target_T:
        mask_exp = np.expand_dims(mask, axis=2)
        return np.repeat(mask_exp, target_M, axis=2)

    raise ValueError(f"Unexpected mask shape: {mask.shape}. Expected compatible with (N, {target_T}, {target_M}).")


def build_training_residuals_valid_only(
    obst_true: np.ndarray,
    obst_pred: np.ndarray,
    masks: np.ndarray,
    Xg: np.ndarray,
    Yg: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, int]:
    if obst_true.ndim == 3:
        obst_true = obst_true[:, :, None, :]
        obst_pred = obst_pred[:, :, None, :]
        masks = masks[:, :, None]

    N, T, M, _ = obst_true.shape
    H_grid, W_grid = Xg.shape
    t_offset = horizon + 1
    T_eff = T - t_offset

    valid_residuals = []
    
    for n in range(N):
        for m in range(M):
            mask_slice = masks[n, t_offset:, m]
            
            if not np.all(mask_slice):
                continue

            traj_true = obst_true[n, t_offset:, m, :]
            traj_pred = obst_pred[n, t_offset:, m, :]
            
            res_m = np.zeros((T_eff, H_grid, W_grid), dtype=np.float32)
            
            for t_idx in range(T_eff):
                y_true = traj_true[t_idx] - WORLD_CENTER
                y_hat = traj_pred[t_idx] - WORLD_CENTER
                
                F_true = distance_field_points(y_true, Xg, Yg)
                F_pred = distance_field_points(y_hat, Xg, Yg)
                
                res_m[t_idx] = F_pred - F_true
            
            valid_residuals.append(res_m)

    if not valid_residuals:
        raise ValueError("No valid residuals found! Check masks or horizon.")

    return np.stack(valid_residuals, axis=0), t_offset


def compute_g_upper_unified(
    residuals: np.ndarray,
    p_base: int,
    K_mix: int,
    alpha: float,
    test_size: float,
    random_state: int,
    n_jobs: int,
    backend: str,
) -> np.ndarray:
    g_upper = compute_cp_upper_envelopes(
        residuals_train=residuals,
        p_base=p_base,
        K=K_mix,
        alpha=alpha,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        backend=backend,
    )
    return g_upper


def build_lower_forecast_fields_unified(
    obst_pred: np.ndarray,
    masks: np.ndarray,
    Xg: np.ndarray,
    Yg: np.ndarray,
    g_upper: np.ndarray,
) -> np.ndarray:
    
    T_eff, M, _ = obst_pred.shape
    H_grid, W_grid = Xg.shape

    lower_multi = np.full((T_eff, H_grid, W_grid), np.inf, dtype=np.float32)

    for t in range(T_eff):
        g_up_t = g_upper[t]
        
        for m in range(M):
            if not masks[t, m]:
                continue
            
            y_hat = obst_pred[t, m] - WORLD_CENTER
            d_pred = distance_field_points(y_hat, Xg, Yg)
            
            lower_m = np.maximum(d_pred - g_up_t, 0.0)
            lower_multi[t] = np.minimum(lower_multi[t], lower_m)

    return lower_multi


def evaluate_coverage_model_multi(
    test_true: np.ndarray,
    test_pred: np.ndarray,
    test_mask: np.ndarray,
    Xg: np.ndarray,
    Yg: np.ndarray,
    g_upper: np.ndarray,
    safe_threshold: float,
):
    E_test, T_eff, M, _ = test_true.shape
    episode_success = []
    timestep_success = []

    for e in range(E_test):
        obst_true_e = test_true[e]
        obst_pred_e = test_pred[e]
        mask_e = test_mask[e]

        lower_multi = build_lower_forecast_fields_unified(
            obst_pred=obst_pred_e,
            masks=mask_e,
            Xg=Xg, 
            Yg=Yg, 
            g_upper=g_upper
        )

        ep_has_unsafe_event = False
        ep_covered = True

        for t in range(T_eff):
            field_true = np.full(Xg.shape, np.inf, dtype=np.float32)
            valid_gt_exists = False
            for m in range(M):
                if mask_e[t, m]:
                    y_true = obst_true_e[t, m] - WORLD_CENTER
                    d = distance_field_points(y_true, Xg, Yg)
                    field_true = np.minimum(field_true, d)
                    valid_gt_exists = True
            
            if not valid_gt_exists:
                continue

            mask_true_unsafe = field_true < safe_threshold
            if not np.any(mask_true_unsafe):
                continue

            ep_has_unsafe_event = True

            mask_pred_unsafe = lower_multi[t] < safe_threshold
            missed = np.logical_and(mask_true_unsafe, np.logical_not(mask_pred_unsafe))
            
            success_t = not np.any(missed)
            timestep_success.append(1.0 if success_t else 0.0)
            
            if not success_t:
                ep_covered = False

        if ep_has_unsafe_event:
            episode_success.append(1.0 if ep_covered else 0.0)
        else:
            episode_success.append(1.0)

    return np.mean(episode_success), np.std(episode_success), np.mean(timestep_success), np.std(timestep_success)


def main():
    rng = np.random.default_rng(SEED)

    train_true_raw, _, train_mask, test_true_raw, _, test_mask = load_eth_pickle_preprocessed(
        dataset=DATASET, box=BOX, T=TSTEPS, split_ratio=0.8,
        scaling=SCALING, seed=SEED, base_dir=BASE_DIR, num_peds=NUM_PEDS,
    )
    
    train_true = np.transpose(train_true_raw, (0, 2, 1, 3))
    test_true = np.transpose(test_true_raw, (0, 2, 1, 3))
    
    _, T_dim, M_dim, _ = train_true.shape

    train_mask = align_mask_dims(train_mask, target_T=T_dim, target_M=M_dim)
    test_mask = align_mask_dims(test_mask, target_T=T_dim, target_M=M_dim)

    _, _, Xg, Yg = build_grid(BOX, H, W)

    N_train_all = train_true.shape[0]
    N_use = min(N_TRAIN, N_train_all)
    idx_sel = rng.choice(N_train_all, size=N_use, replace=False)
    
    train_true_sel = train_true[idx_sel]
    train_mask_sel = train_mask[idx_sel]

    train_pred_sel = build_cv_predictions(train_true_sel, box=BOX, horizon=TIME_HORIZON)

    print("Building residuals...")
    residuals_flat, t_offset = build_training_residuals_valid_only(
        obst_true=train_true_sel,
        obst_pred=train_pred_sel,
        masks=train_mask_sel,
        Xg=Xg, Yg=Yg, horizon=TIME_HORIZON
    )
    print(f"Residuals collected. Shape: {residuals_flat.shape}")

    print("Computing CP Envelope...")
    g_upper = compute_g_upper_unified(
        residuals=residuals_flat,
        p_base=P_BASE, K_mix=K_MIX, alpha=ALPHA,
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
        n_jobs=N_JOBS, backend=BACKEND
    )

    test_pred = build_cv_predictions(test_true, box=BOX, horizon=TIME_HORIZON)
    
    test_true_valid = test_true[:, t_offset:, :, :]
    test_pred_valid = test_pred[:, t_offset:, :, :]
    test_mask_valid = test_mask[:, t_offset:, :]

    print("Evaluating...")
    mean_ep, std_ep, mean_ts, std_ts = evaluate_coverage_model_multi(
        test_true=test_true_valid,
        test_pred=test_pred_valid,
        test_mask=test_mask_valid,
        Xg=Xg, Yg=Yg,
        g_upper=g_upper,
        safe_threshold=SAFE_THRESHOLD
    )

    print(f"[CP] Episode Coverage : {mean_ep*100:.2f}%")
    print(f"[CP] Timestep Coverage: {mean_ts*100:.2f}%")

    ep_idx = int(rng.integers(test_true.shape[0]))
    obst_vis_true = test_true_valid[ep_idx]
    obst_vis_pred = test_pred_valid[ep_idx]
    mask_vis = test_mask_valid[ep_idx]
    
    agent_vis = np.zeros((obst_vis_true.shape[0], 2)) 

    lower_fields = build_lower_forecast_fields_unified(
        obst_pred=obst_vis_pred,
        masks=mask_vis,
        Xg=Xg, Yg=Yg,
        g_upper=g_upper
    )
    
    true_fields = np.zeros_like(lower_fields)
    for t in range(true_fields.shape[0]):
        f = np.full(Xg.shape, np.inf)
        for m in range(obst_vis_true.shape[1]):
            if mask_vis[t, m]:
                d = distance_field_points(obst_vis_true[t, m] - WORLD_CENTER, Xg, Yg)
                f = np.minimum(f, d)
        true_fields[t] = f

    ani = render_cp_comparison_paper_multi(
        agent_traj=agent_vis,
        obst_traj=obst_vis_true,
        obst_pred_traj=obst_vis_pred,
        true_fields_ego=true_fields,
        lower_fields_ego=lower_fields,
        box=BOX,
        ego_box=EGO_BOX,
        safe_threshold=SAFE_THRESHOLD,
        h=TIME_HORIZON,
        headings=None,
        interval=150,
    )
    plt.show()


if __name__ == "__main__":
    main()