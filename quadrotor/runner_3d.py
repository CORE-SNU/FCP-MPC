from __future__ import annotations

import os
import csv
import json
import time
import math
import traceback
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Callable
import inspect
import argparse

import numpy as np
from quad_env import QuadWorldEnv3D

from sim_func_3d import run_one_episode_visual_3d as run_one_episode_func_3d
from sim_cp_3d import run_one_episode_rerun_simple as run_one_episode_cp_3d
from sim_ecp_3d import run_one_episode_ecp_3d_rerun as run_one_episode_ecp_3d
from sim_acp_3d import run_one_episode_acp_3d


# ------------------------------------------------------------
# Helpers: timing stats
# ------------------------------------------------------------
def _stats_ms(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {"mean": math.nan, "p50": math.nan, "p90": math.nan, "p99": math.nan, "max": math.nan}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def _safe_mkdir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _short_err(s: str, limit: int = 2000) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    return s[:limit]


def _safe_rate(num: float, den: int) -> float:
    if den <= 0:
        return math.nan
    return float(num) / float(den)


def to_json_safe(x):
    """Recursively convert to JSON-serializable Python types."""
    if isinstance(x, dict):
        return {str(k): to_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if (not math.isfinite(v)) else v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float):
        return None if (not math.isfinite(x)) else x
    return x


# ------------------------------------------------------------
# Standardized outcome labels
# ------------------------------------------------------------
@dataclass
class EpisodeResult:
    # identifiers
    method: str
    seed: int

    # environment summary
    dt: float
    horizon: int
    n_obs: int
    world_bounds_xyz: str

    # knobs (log everything you care about)
    alpha: float
    cp_enabled: bool
    backend: str
    n_paths: int
    n_skip: int
    time_horizon: int

    pred_model_noise: float
    obs_process_noise: float
    gt_future_noise: float

    mode_switch_p: float
    mode_min_ttl: int
    mode_max_ttl: int
    turn_rate_std: float
    stop_go_p: float

    # episode outcomes
    status: str                  # success | collision | timeout | compute_fail | error
    reached_goal: int            # 0/1
    steps: int
    collisions: int              # collision steps (or events)
    infeasible_steps: int

    # NEW: rates (per-step)
    collision_rate: float        # collisions / steps (NaN if steps==0)
    infeas_rate: float           # infeasible_steps / steps (NaN if steps==0)

    # compute constraint stats
    ctrl_mean_ms: float
    ctrl_p50_ms: float
    ctrl_p90_ms: float
    ctrl_p99_ms: float
    ctrl_max_ms: float
    ctrl_over_dt_rate: float     # fraction of steps where ctrl_time > dt

    loop_mean_ms: float
    loop_p50_ms: float
    loop_p90_ms: float
    loop_p99_ms: float
    loop_max_ms: float
    loop_over_dt_rate: float     # loop_time > dt

    # misc
    runtime_sec: float
    error_msg: str = ""


def stats_from_row(row: EpisodeResult, which: str) -> Optional[Dict[str, Any]]:
    """2D runner의 stats_ms처럼 row에서 timing stats dict를 만들어줌."""
    if which == "ctrl":
        mean, p50, p90, p99, mx = row.ctrl_mean_ms, row.ctrl_p50_ms, row.ctrl_p90_ms, row.ctrl_p99_ms, row.ctrl_max_ms
    elif which == "loop":
        mean, p50, p90, p99, mx = row.loop_mean_ms, row.loop_p50_ms, row.loop_p90_ms, row.loop_p99_ms, row.loop_max_ms
    else:
        raise ValueError("which must be 'ctrl' or 'loop'")

    vals = [mean, p50, p90, p99, mx]
    if all((isinstance(v, float) and (not math.isfinite(v))) for v in vals):
        return None

    return to_json_safe({"mean": mean, "p50": p50, "p90": p90, "p99": p99, "max": mx})


def build_json_dump(
    rows: List[EpisodeResult],
    env_base: Dict[str, Any],
    exp_base: Dict[str, Any],
    methods: Dict[str, Dict[str, Any]],
    seeds: List[int],
) -> Dict[str, Any]:
    by_method: Dict[str, Dict[int, EpisodeResult]] = {}
    for r in rows:
        by_method.setdefault(r.method, {})[r.seed] = r

    out: Dict[str, Any] = {
        "env_base": to_json_safe(env_base),
        "exp_base": to_json_safe(exp_base),
        "methods": to_json_safe(methods),
        "seeds": to_json_safe(seeds),
        "results": {},
    }

    for method_name in methods.keys():
        per_seed = by_method.get(method_name, {})

        status = []
        reached_goal = []
        steps = []
        collisions = []
        infeasible_steps = []
        collision_rate = []
        infeas_rate = []
        runtime_sec = []
        error_msg = []

        timing_ctrl_ms = []
        timing_loop_ms = []
        miss_ctrl = []
        miss_loop = []

        for s in seeds:
            r = per_seed.get(s, None)
            if r is None:
                status.append("missing")
                reached_goal.append(0)
                steps.append(0)
                collisions.append(0)
                infeasible_steps.append(0)
                collision_rate.append(None)
                infeas_rate.append(None)
                runtime_sec.append(None)
                error_msg.append("missing row")
                timing_ctrl_ms.append(None)
                timing_loop_ms.append(None)
                miss_ctrl.append(None)
                miss_loop.append(None)
                continue

            status.append(r.status)
            reached_goal.append(int(r.reached_goal))
            steps.append(int(r.steps))
            collisions.append(int(r.collisions))
            infeasible_steps.append(int(r.infeasible_steps))
            collision_rate.append(r.collision_rate)
            infeas_rate.append(r.infeas_rate)
            runtime_sec.append(r.runtime_sec)
            error_msg.append(r.error_msg or "")

            timing_ctrl_ms.append(stats_from_row(r, "ctrl"))
            timing_loop_ms.append(stats_from_row(r, "loop"))
            miss_ctrl.append(r.ctrl_over_dt_rate)
            miss_loop.append(r.loop_over_dt_rate)

        out["results"][method_name] = to_json_safe({
            "status": status,
            "reached_goal": reached_goal,
            "steps": steps,
            "collisions": collisions,
            "infeasible_steps": infeasible_steps,
            "collision_rate": collision_rate,
            "infeas_rate": infeas_rate,
            "runtime_sec": runtime_sec,
            "error_msg": error_msg,
            "timing_ctrl_ms": timing_ctrl_ms,
            "timing_loop_ms": timing_loop_ms,
            "miss_ctrl_over_dt_rate": miss_ctrl,
            "miss_loop_over_dt_rate": miss_loop,
        })

    return out


# ------------------------------------------------------------
# CSV writer (append-safe)
# ------------------------------------------------------------
def append_csv(path: str, rows: List[EpisodeResult]) -> None:
    if not rows:
        return
    _safe_mkdir_for_file(path)
    write_header = (not os.path.exists(path))
    fieldnames = [f.name for f in fields(EpisodeResult)]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


# ------------------------------------------------------------
# Core: run a single episode (wrap your existing call)
# ------------------------------------------------------------
def run_one_episode_wrapper(
    *,
    QuadWorldEnv3D,                            # class
    run_fn: Callable[..., Dict[str, Any]],     # fn(env, **kwargs)->dict
    env_kwargs: Dict[str, Any],
    exp_kwargs: Dict[str, Any],
    method: str,
    seed: int,
    out_dt_fail_frac: float = 0.1,
    fail_on: str = "loop",                     # "loop" or "ctrl"
) -> EpisodeResult:
    t0 = time.time()

    env_kwargs = dict(env_kwargs)
    env_kwargs["seed"] = seed
    env = QuadWorldEnv3D(**env_kwargs)

    result = None
    err = ""
    try:
        result = run_fn(env, **exp_kwargs)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        result = None

    runtime = time.time() - t0

    # error path
    if result is None:
        return EpisodeResult(
            method=method,
            seed=seed,
            dt=float(env_kwargs["dt"]),
            horizon=int(env_kwargs["horizon"]),
            n_obs=int(env_kwargs["n_obs"]),
            world_bounds_xyz=str(env_kwargs.get("world_bounds_xyz", "")),
            alpha=float(exp_kwargs.get("alpha", math.nan)),
            cp_enabled=bool(exp_kwargs.get("CP", False)),
            backend=str(exp_kwargs.get("backend", "")),
            n_paths=int(exp_kwargs.get("n_paths", -1)),
            n_skip=int(exp_kwargs.get("n_skip", -1)),
            time_horizon=int(exp_kwargs.get("time_horizon", -1)),
            pred_model_noise=float(env_kwargs.get("pred_model_noise", math.nan)),
            obs_process_noise=float(env_kwargs.get("obs_process_noise", math.nan)),
            gt_future_noise=float(env_kwargs.get("gt_future_noise", math.nan)),
            mode_switch_p=float(env_kwargs.get("mode_switch_p", math.nan)),
            mode_min_ttl=int(env_kwargs.get("mode_min_ttl", -1)),
            mode_max_ttl=int(env_kwargs.get("mode_max_ttl", -1)),
            turn_rate_std=float(env_kwargs.get("turn_rate_std", math.nan)),
            stop_go_p=float(env_kwargs.get("stop_go_p", math.nan)),
            status="error",
            reached_goal=0,
            steps=0,
            collisions=0,
            infeasible_steps=0,
            collision_rate=math.nan,
            infeas_rate=math.nan,
            ctrl_mean_ms=math.nan, ctrl_p50_ms=math.nan, ctrl_p90_ms=math.nan, ctrl_p99_ms=math.nan, ctrl_max_ms=math.nan,
            ctrl_over_dt_rate=math.nan,
            loop_mean_ms=math.nan, loop_p50_ms=math.nan, loop_p90_ms=math.nan, loop_p99_ms=math.nan, loop_max_ms=math.nan,
            loop_over_dt_rate=math.nan,
            runtime_sec=float(runtime),
            error_msg=_short_err(err),
        )

    reached_goal = bool(result.get("reached_goal", False))
    steps = int(result.get("steps", 0))
    collisions = int(result.get("collisions", 0))
    infeasible_steps = int(result.get("infeasible_steps", 0))

    # NEW: rates
    coll_rate = _safe_rate(collisions, steps)
    infeas_rate = _safe_rate(infeasible_steps, steps)

    ctrl_times_ms = list(result.get("ctrl_times_ms", []))
    loop_times_ms = list(result.get("loop_times_ms", []))

    ctrl_stats = _stats_ms(ctrl_times_ms)
    loop_stats = _stats_ms(loop_times_ms)

    dt_ms = float(env_kwargs["dt"]) * 1000.0
    ctrl_over_dt_rate = float(np.mean([x > dt_ms for x in ctrl_times_ms])) if ctrl_times_ms else math.nan
    loop_over_dt_rate = float(np.mean([x > dt_ms for x in loop_times_ms])) if loop_times_ms else math.nan

    if fail_on not in ("loop", "ctrl"):
        fail_on = "loop"

    miss_rate = loop_over_dt_rate if fail_on == "loop" else ctrl_over_dt_rate
    compute_fail = (not math.isnan(miss_rate)) and (miss_rate > out_dt_fail_frac)

    max_steps = int(exp_kwargs.get("max_steps", steps))

    if compute_fail:
        status = "compute_fail"
    elif collisions > 0:
        status = "collision"
    elif reached_goal:
        status = "success"
    else:
        # ended without reaching: either hit the step budget (timeout) or
        # terminated early via break-on-floor-crash (crash)
        status = "timeout" if steps >= max_steps else "crash"

    return EpisodeResult(
        method=method,
        seed=seed,
        dt=float(env_kwargs["dt"]),
        horizon=int(env_kwargs["horizon"]),
        n_obs=int(env_kwargs["n_obs"]),
        world_bounds_xyz=str(env_kwargs.get("world_bounds_xyz", "")),
        alpha=float(exp_kwargs.get("alpha", math.nan)),
        cp_enabled=bool(exp_kwargs.get("CP", False)),
        backend=str(exp_kwargs.get("backend", "")),
        n_paths=int(exp_kwargs.get("n_paths", -1)),
        n_skip=int(exp_kwargs.get("n_skip", -1)),
        time_horizon=int(exp_kwargs.get("time_horizon", -1)),
        pred_model_noise=float(env_kwargs.get("pred_model_noise", math.nan)),
        obs_process_noise=float(env_kwargs.get("obs_process_noise", math.nan)),
        gt_future_noise=float(env_kwargs.get("gt_future_noise", math.nan)),
        mode_switch_p=float(env_kwargs.get("mode_switch_p", math.nan)),
        mode_min_ttl=int(env_kwargs.get("mode_min_ttl", -1)),
        mode_max_ttl=int(env_kwargs.get("mode_max_ttl", -1)),
        turn_rate_std=float(env_kwargs.get("turn_rate_std", math.nan)),
        stop_go_p=float(env_kwargs.get("stop_go_p", math.nan)),
        status=status,
        reached_goal=int(reached_goal),
        steps=steps,
        collisions=collisions,
        infeasible_steps=infeasible_steps,
        collision_rate=coll_rate,
        infeas_rate=infeas_rate,
        ctrl_mean_ms=ctrl_stats["mean"],
        ctrl_p50_ms=ctrl_stats["p50"],
        ctrl_p90_ms=ctrl_stats["p90"],
        ctrl_p99_ms=ctrl_stats["p99"],
        ctrl_max_ms=ctrl_stats["max"],
        ctrl_over_dt_rate=float(ctrl_over_dt_rate) if not math.isnan(ctrl_over_dt_rate) else math.nan,
        loop_mean_ms=loop_stats["mean"],
        loop_p50_ms=loop_stats["p50"],
        loop_p90_ms=loop_stats["p90"],
        loop_p99_ms=loop_stats["p99"],
        loop_max_ms=loop_stats["max"],
        loop_over_dt_rate=float(loop_over_dt_rate) if not math.isnan(loop_over_dt_rate) else math.nan,
        runtime_sec=float(runtime),
        error_msg="",
    )


# ------------------------------------------------------------
# Runner registry + method presets
# ------------------------------------------------------------
RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "func": run_one_episode_func_3d,  # from sim_func_3d
    "cp": run_one_episode_cp_3d,      # from sim_cp_3d
    "ecp": run_one_episode_ecp_3d,    # from sim_ecp_3d
    "acp": run_one_episode_acp_3d,    # from sim_acp_3d
}

METHOD_PRESETS: Dict[str, Dict[str, Any]] = {
    # nocp: skip SDF calibration entirely (CP=False, so residuals are never used)
    "nocp": {"runner": "func", "over": {"CP": False, "n_calib_samples": 0}},
    "cc":   {"runner": "cp",   "over": {"CP": False}},
    "fcp":  {"runner": "func", "over": {"CP": True, "alpha": 0.10, "n_calib_samples": 20}},
    "ecp":  {"runner": "ecp",  "over": {}},
    "acp":  {"runner": "acp",  "over": {"target_miscoverage_level": 0.10, "step_size": 0.05}},
}


# ------------------------------------------------------------
# Main suite runner
# ------------------------------------------------------------
def run_suite(
    *,
    QuadWorldEnv3D,
    env_base: Dict[str, Any],
    exp_base: Dict[str, Any],
    methods: Dict[str, Dict[str, Any]],   # method_name -> {"runner": str, "over": dict}
    seeds: List[int],
    csv_path: str,
    out_dt_fail_frac: float = 0.05,
    fail_on: str = "loop",
    dump_json: bool = True,
    save_traj_img: bool = False,
    traj_img_dir: str = "traj_3d",
    traj_img_max_seeds: int = 3,
) -> None:
    _safe_mkdir_for_file(csv_path)
    all_rows: List[EpisodeResult] = []

    for method_name, spec in methods.items():
        runner_key = spec.get("runner", "func")
        run_episode = RUNNERS[runner_key]

        allowed = set(inspect.signature(run_episode).parameters.keys())

        def run_fn(env, **exp_kwargs):
            clean = {k: v for k, v in exp_kwargs.items() if k in allowed}
            return run_episode(env, **clean)

        exp_over = dict(spec.get("over", {}))

        for seed_idx, seed in enumerate(seeds):
            env_kwargs = dict(env_base)
            exp_kwargs = dict(exp_base)
            exp_kwargs.update(exp_over)

            exp_kwargs.setdefault("visualize", False)
            exp_kwargs.setdefault("save_rrd", False)

            # Only dump trajectory images for the first few seeds: enough to show
            # the method works across seeds without one PNG per sweep entry.
            if save_traj_img and seed_idx < traj_img_max_seeds:
                exp_kwargs["save_traj_img"] = True
                exp_kwargs["traj_img_path"] = os.path.join(
                    traj_img_dir, f"{method_name}_seed{seed}.png"
                )
                exp_kwargs["method_name"] = method_name

            row = run_one_episode_wrapper(
                QuadWorldEnv3D=QuadWorldEnv3D,
                run_fn=run_fn,
                env_kwargs=env_kwargs,
                exp_kwargs=exp_kwargs,
                method=method_name,
                seed=seed,
                out_dt_fail_frac=out_dt_fail_frac,
                fail_on=fail_on,
            )

            all_rows.append(row)
            append_csv(csv_path, [row])

            miss = row.loop_over_dt_rate if fail_on == "loop" else row.ctrl_over_dt_rate
            print(
                f"[saved] {method_name} seed={seed} status={row.status} "
                f"coll={row.collisions} (rate={row.collision_rate:.3f}) "
                f"infeas={row.infeasible_steps} (rate={row.infeas_rate:.3f}) "
                f"steps={row.steps} "
                f"ctrl_p99={row.ctrl_p99_ms:.1f}ms loop_p99={row.loop_p99_ms:.1f}ms "
                f"miss({fail_on})={miss:.3f}"
            )

    if dump_json:
        json_path = os.path.splitext(csv_path)[0] + ".json"
        _safe_mkdir_for_file(json_path)
        dump_obj = build_json_dump(
            rows=all_rows,
            env_base=env_base,
            exp_base=exp_base,
            methods=methods,
            seeds=seeds,
        )
        with open(json_path, "w") as f:
            json.dump(dump_obj, f, ensure_ascii=False, indent=2)
        print(f"[done] wrote {len(all_rows)} rows to {csv_path} and {json_path}")
    else:
        print(f"[done] wrote {len(all_rows)} rows to {csv_path}")


def _parse_methods_arg(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", type=str, default="nocp,cp,cc,fcp,ecp", help="comma-separated")
    ap.add_argument("--seed-from", type=int, default=20)
    ap.add_argument("--seed-to", type=int, default=39, help="inclusive")
    ap.add_argument("--csv-path", type=str, default="metric_3d/quad_suite.csv")
    ap.add_argument("--out-dt-fail-frac", type=float, default=0.10)
    ap.add_argument("--fail-on", type=str, default="loop", choices=["loop", "ctrl"])
    ap.add_argument("--dump-json", action="store_true", default=True)
    ap.add_argument("--n-obs", type=int, default=280, help="Number of dynamic obstacles")
    ap.add_argument("--max-steps", type=int, default=500,
                    help="Cap episode length; timing-only sweeps (e.g. scalability) can use a small value")
    ap.add_argument("--save-traj-img", action="store_true", default=False,
                    help="Dump a static 3D trajectory PNG per (method, seed)")
    ap.add_argument("--traj-img-dir", type=str, default="traj_3d",
                    help="Output dir for trajectory PNGs")
    ap.add_argument("--traj-img-max-seeds", type=int, default=3,
                    help="Only dump trajectory PNGs for the first N seeds")
    args = ap.parse_args()

    selected = _parse_methods_arg(args.methods)
    methods: Dict[str, Dict[str, Any]] = {}
    for m in selected:
        if m not in METHOD_PRESETS:
            raise ValueError(f"Unknown method '{m}'. Known: {list(METHOD_PRESETS.keys())}")
        methods[m] = METHOD_PRESETS[m]

    seeds = list(range(args.seed_from, args.seed_to + 1))

    env_base = dict(
        dt=0.1, horizon=20, n_obs=args.n_obs,
        world_bounds_xyz=((-3, 7), (-3, 7), (0, 8)),
        pred_model_noise=0.20, obs_process_noise=0.22, gt_future_noise=0.20,
        mode_switch_p=0.95, mode_min_ttl=1, mode_max_ttl=6,
        turn_rate_std=3.0, stop_go_p=0.6, gui=False,
    )

    exp_base = dict(
        nx=40, ny=40, nz=40,
        time_horizon=12,
        n_skip=4,
        n_paths=2000,
        max_steps=args.max_steps,
        backend="loky",
        visualize=False,
        save_rrd=False,
    )

    run_suite(
        QuadWorldEnv3D=QuadWorldEnv3D,
        env_base=env_base,
        exp_base=exp_base,
        methods=methods,
        seeds=seeds,
        csv_path=args.csv_path,
        out_dt_fail_frac=args.out_dt_fail_frac,
        fail_on=args.fail_on,
        dump_json=args.dump_json,
        save_traj_img=args.save_traj_img,
        traj_img_dir=args.traj_img_dir,
        traj_img_max_seeds=args.traj_img_max_seeds,
    )


if __name__ == "__main__":
    main()