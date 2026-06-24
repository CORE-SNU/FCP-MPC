import os
import argparse
import numpy as np
import pickle
import json

from sims.sim_acp_mpc import run_acp_mpc
from sims.sim_cc import run_cc
from sims.sim_ecp_mpc import run_ecp_mpc
from sims.sim_func_cp import run_fcp_mpc
import csv

seed = 0
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="id of dataset to evaluate on", type=str, required=True)
parser.add_argument("--controller", help="control method to use", type=str, required=True)
parser.add_argument("--visualize", help="visualize rollout", action='store_true')
parser.add_argument("--asset_dir", help="asset dirpath for visualization", type=str)
args = parser.parse_args()


def load_prediction_results(dataset):
    with open(os.path.join(os.path.dirname(__file__), f'predictions/{dataset}.pkl'), 'rb') as f:
        res = pickle.load(f)
    return res


def to_json_safe(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    return x


def stats_ms(arr):
    """Return JSON-safe timing stats from a list/np array in ms."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return None
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
        "n": int(a.size),
    }


if __name__ == "__main__":

    # simulation step
    dt = 0.4

    # robot parameters
    robot_rad = 0.4
    obstacle_rad = 1. / np.sqrt(2.)
    max_linear_x = 0.8
    min_linear_x = -0.8
    max_angular_z = 0.7
    min_angular_z = -0.7

    if args.visualize:
        if not args.asset_dir or (not os.path.isdir(args.asset_dir)):
            raise OSError(
                "A valid asset directory path must be provided for visualization. "
                "If you do not have one, please run video_parser.py first."
            )
        from PIL import Image
        asset_dir = args.asset_dir
        print('The dataset frames will be loaded from', asset_dir)
        robot_img = Image.open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets/robot.png"))
    else:
        asset_dir = None
        robot_img = None

    # controller-specific arguments
    controller_configs = {
        'acp-mpc': {'target_miscoverage_level': 0.2, 'step_size': 0.05},
        'ecp-mpc': {'target_miscoverage_level': 0.1, 'step_size': 0.02},
        'cc': {'risk_level': -2., 'step_size': 5000},
        # Legacy alias (hard + adaptive)
        'fcp-mpc': {'target_miscoverage_level': 0.1, 'step_size': 10.0, 'adaptive': True, 'safety_mode': 'hard'},
        # Four explicit variants for the ablation study
        'fcp-hard-adaptive':    {'target_miscoverage_level': 0.1, 'step_size': 10.0, 'adaptive': True,  'safety_mode': 'hard'},
        'fcp-hard-nonadaptive': {'target_miscoverage_level': 0.1, 'step_size': 10.0, 'adaptive': False, 'safety_mode': 'hard'},
        'fcp-soft-adaptive':    {'target_miscoverage_level': 0.1, 'step_size': 10.0, 'adaptive': True,  'safety_mode': 'soft', 'w_safety': 100.0},
        'fcp-soft-nonadaptive': {'target_miscoverage_level': 0.1, 'step_size': 10.0, 'adaptive': False, 'safety_mode': 'soft', 'w_safety': 100.0},
    }

    eval_functions = {
        'cc': run_cc,
        'acp-mpc': run_acp_mpc,
        'ecp-mpc': run_ecp_mpc,
        'fcp-mpc':              run_fcp_mpc,
        'fcp-hard-adaptive':    run_fcp_mpc,
        'fcp-hard-nonadaptive': run_fcp_mpc,
        'fcp-soft-adaptive':    run_fcp_mpc,
        'fcp-soft-nonadaptive': run_fcp_mpc,
    }

    eval_task_configs = {
        'zara1': {'init_robot_pose': np.array([12., 5., np.pi]), 'goal_pos': np.array([3., 6.])},
        'zara2': {'init_robot_pose': np.array([1., 6., 0.]), 'goal_pos': np.array([14., 5.])},
        'eth': {'init_robot_pose': np.array([5., 1.0, np.pi / 2.]), 'goal_pos': np.array([3., 10.])},
        'hotel': {'init_robot_pose': np.array([-1.5, 0., -np.pi / 2.]), 'goal_pos': np.array([2., -6.])},
        'univ': {'init_robot_pose': np.array([3.5, 2., np.pi / 4.]), 'goal_pos': np.array([11.5, 8.5])},
    }

    scenarios = {
        'zara1': [100, 200, 300],
        'zara2': [100, 200, 300],
        # eth's old [100,200,300] landed in short/empty data windows; use scene
        # starts with long consecutive prediction runs instead.
        'eth': [732, 339, 653],
        'hotel': [1001, 1245, 1582],
        'univ': [100]
    }

    init_frames = {
        'zara1': 0,
        'zara2': 1,
        'eth': 78,
        'hotel': 0,
        'univ': 0,
    }

    max_n_steps = {
        'zara1': 100,
        'zara2': 100,
        'eth': 100,
        'hotel': 100,
        'univ': 300
    }

    # ---- validate keys early (helps debugging) ----
    if args.dataset not in eval_task_configs:
        raise KeyError(f"Unknown dataset: {args.dataset}. Available: {list(eval_task_configs.keys())}")
    if args.controller not in eval_functions:
        raise KeyError(f"Unknown controller: {args.controller}. Available: {list(eval_functions.keys())}")

    task_kwargs = eval_task_configs[args.dataset]
    eval_func = eval_functions[args.controller]
    ctrl_kwargs = controller_configs.get(args.controller, {})

    predictions = load_prediction_results(args.dataset)

    metric_dict, trajectories = eval_func(
        dataset=args.dataset,
        scenarios=scenarios[args.dataset],
        max_linear_x=max_linear_x,
        min_linear_x=min_linear_x,
        max_angular_z=max_angular_z,
        min_angular_z=min_angular_z,
        predictions=predictions,
        dt=dt,
        init_frame=init_frames[args.dataset],
        visualize=args.visualize,
        asset_dir=asset_dir,
        robot_img=robot_img,
        max_n_steps=max_n_steps[args.dataset],
        robot_rad=robot_rad,
        obstacle_rad=obstacle_rad,
        **task_kwargs,
        **ctrl_kwargs
    )

    os.makedirs(os.path.join(os.path.dirname(__file__), 'traj'), exist_ok=True)
    np.save(os.path.join(os.path.dirname(__file__), f'traj/{args.dataset}_{args.controller}.npy'), trajectories)

    # ---- static trajectory image (asset-free, generated on every run) ----
    # Output dir is overridable via FCP_FIG_DIR so a run can write its figures
    # straight into the paper folder (e.g. FCP_FIG_DIR=T_RO2026/figures/2d).
    try:
        from viz_traj import save_traj_image_2d
        fig_dir = os.environ.get('FCP_FIG_DIR', os.path.join(os.path.dirname(__file__), 'traj'))
        os.makedirs(fig_dir, exist_ok=True)
        img_path = os.path.join(fig_dir, f'{args.dataset}_{args.controller}.png')
        save_traj_image_2d(
            trajectories=trajectories,
            goal=task_kwargs.get('goal_pos'),
            start=task_kwargs.get('init_robot_pose', np.zeros(3))[:2],
            title=f'{args.dataset} / {args.controller}',
            out_path=img_path,
        )
        print(f'[traj-img] saved {img_path}')
    except Exception as e:
        print(f'[traj-img] skipped ({e})')

    # ---- JSON to save ----
    dict_to_save = {
        "dataset": args.dataset,
        "controller": args.controller,

        # per-scene scalars
        "collision": [],
        "cost": [],
        "time": [],         # exit steps per scene
        "infeasible": [],
        "miscoverage": [],

        # optional: keep per-scene timing stats if available
        "timing_ctrl_ms": [],   # list of dict stats, one per scene (or None)
        "timing_loop_ms": [],

        # optional: to debug mapping
        "scene_ids": [],
    }

    print(f'dataset: {args.dataset} / controller: {args.controller}')
    for scene_idx, eval_metric in metric_dict.items():
        print(f'-------- scene {scene_idx} --------')
        dict_to_save["scene_ids"].append(int(scene_idx))

        # collisions
        col_arr = np.asarray(eval_metric.get('collisions', []), dtype=np.float64)
        collision_ratio = float(np.sum(col_arr) / max(1, col_arr.size))
        print(f'* collision_ratio={collision_ratio}')
        dict_to_save['collision'].append(collision_ratio)

        # cost
        cost_arr = np.asarray(eval_metric.get('costs', []), dtype=np.float64)
        avg_cost = float(np.nanmean(cost_arr)) if cost_arr.size > 0 else float("nan")
        print(f'* avg cost={avg_cost:.4f}')
        dict_to_save['cost'].append(avg_cost)

        # exit time
        exit_time = eval_metric.get('exit_time', max_n_steps[args.dataset])

        # inf / nan 방지
        if exit_time is None:
            exit_time = max_n_steps[args.dataset]

        # numpy scalar / python float 모두 처리
        exit_time_f = float(exit_time)
        if not np.isfinite(exit_time_f):
            exit_time = max_n_steps[args.dataset]
        else:
            exit_time = int(exit_time_f)

        exit_time = min(exit_time, max_n_steps[args.dataset])
        print(f'* exit time={exit_time}')
        dict_to_save['time'].append(exit_time)

        # infeasible
        if 'infeasible' in eval_metric:
            infeas_arr = np.asarray(eval_metric['infeasible'], dtype=np.float64)
            infeasible_ratio = float(np.sum(infeas_arr) / max(1, infeas_arr.size))
            print(f'* infeasible_ratio={infeasible_ratio}')
            dict_to_save['infeasible'].append(infeasible_ratio)
        else:
            dict_to_save['infeasible'].append(float("nan"))

        # miscoverage
        if 'miscoverage' in eval_metric:
            mis_arr = np.asarray(eval_metric['miscoverage'], dtype=np.float64)
            miscoverage_ratio = float(np.sum(mis_arr) / max(1, mis_arr.size))
            print(f'* asymptotic miscoverage={miscoverage_ratio:.4f}')
            dict_to_save['miscoverage'].append(miscoverage_ratio)
        else:
            dict_to_save['miscoverage'].append(float("nan"))

        # timing (optional)
        t_ctrl = eval_metric.get("timing_ctrl_ms", None)
        t_loop = eval_metric.get("timing_loop_ms", None)
        dict_to_save["timing_ctrl_ms"].append(stats_ms(t_ctrl))
        dict_to_save["timing_loop_ms"].append(stats_ms(t_loop))

    # ---- final JSON-safe conversion (extra safety) ----
    dict_to_save = to_json_safe(dict_to_save)

    save_dir = os.path.join(os.path.dirname(__file__), 'metric')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f'{args.dataset}_{args.controller}.json'), 'w') as f:
        json.dump(dict_to_save, f, ensure_ascii=False, indent=4)