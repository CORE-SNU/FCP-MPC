#!/usr/bin/env python3
"""Unified 3D experiment driver: ONE run -> table + trajectory figure + scalability.

Runs every controller (CC / ECP / ACP / FCP) on the SAME seeds and configuration so
the paper's 3D table, the qualitative trajectory figure (Fig. traj_3d), and the
scalability plot (Fig. scalability_3d) are mutually consistent --- previously these
came from three separate runs with different seeds/data files, which is why their
numbers disagreed.

Independent episodes (method x seed x n_obs) are run in parallel across CPU cores,
each pinned to a single BLAS thread to avoid nested-parallelism oversubscription.

Outputs:
  metric_3d/results_3d.{csv,json}   per-(method,seed) metrics @ n_obs_main  -> table
  metric_3d/scalability_3d.csv      per-(method,seed,n_obs) ctrl timing     -> Fig.7
  T_RO2026/traj_3d_seeds.png        Fig.6 trajectory overlay @ n_obs_main
  T_RO2026/control_time_3d.png      Fig.7 scalability vs n_obs

A trajectory+metric cache is written so --replot re-renders figures without re-sim.

Usage:
  conda run -n cp python make_3d_results.py                  # full run (parallel)
  conda run -n cp python make_3d_results.py --replot         # re-render from cache
  conda run -n cp python make_3d_results.py --smoke          # tiny end-to-end check
"""
from __future__ import annotations

# --- limit per-process BLAS threads BEFORE numpy is imported (children inherit) ---
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import inspect
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from make_figs_3d import (ENV_KWARGS, EXP_BASE, METHODS, build_env, make_figure,
                          PAPER_DIR, run_fcp)

HERE = os.path.dirname(os.path.abspath(__file__))
METRIC_DIR = os.path.join(HERE, "metric_3d")
TABLE_CSV = os.path.join(METRIC_DIR, "results_3d.csv")
TABLE_JSON = os.path.join(METRIC_DIR, "results_3d.json")
SCAL_CSV = os.path.join(METRIC_DIR, "scalability_3d.csv")
CACHE = os.path.join(PAPER_DIR, "results_3d_cache.pkl")
TRAJ_OUT = os.path.join(PAPER_DIR, "traj_3d_seeds.png")
SCAL_OUT = os.path.join(PAPER_DIR, "control_time_3d.png")

# label -> (run_fn, extras); reuse the exact method configs from make_figs_3d
METHOD_MAP = {m[0]: (m[1], m[2]) for m in METHODS}
METHOD_LABELS = [m[0] for m in METHODS]
# Nominal MPC (no conformal calibration) -- included in table/scalability only;
# make_figure iterates make_figs_3d.METHODS (4) so it is excluded from Fig.6.
METHOD_MAP["Nominal MPC"] = (run_fcp, {"CP": False, "n_calib_samples": 0, "break_on_collision": True})
METHOD_LABELS.append("Nominal MPC")


# ----------------------------------------------------------------------------- metrics
def _stats_ms(xs):
    if not xs:
        nan = float("nan")
        return dict(mean=nan, p50=nan, p90=nan, p99=nan, max=nan)
    a = np.sort(np.asarray(xs, dtype=np.float64))
    n = a.size

    def q(p):
        return float(a[min(n - 1, max(0, int(np.ceil(p * n)) - 1))])
    return dict(mean=float(a.mean()), p50=q(0.5), p90=q(0.9), p99=q(0.99), max=float(a[-1]))


def compute_metrics(result, dt, max_steps, out_dt_fail_frac=0.10):
    steps = int(result.get("steps", 0))
    reached = bool(result.get("reached_goal", False))
    coll = int(result.get("collisions", 0))
    infeas = int(result.get("infeasible_steps", 0))
    ctrl = list(result.get("ctrl_times_ms", []))
    loop = list(result.get("loop_times_ms", []))
    cs, ls = _stats_ms(ctrl), _stats_ms(loop)
    dt_ms = float(dt) * 1000.0
    over = float(np.mean([x > dt_ms for x in loop])) if loop else float("nan")
    compute_fail = (not np.isnan(over)) and (over > out_dt_fail_frac)
    if compute_fail:
        status = "compute_fail"
    elif coll > 0:
        status = "collision"
    elif reached:
        status = "success"
    else:
        status = "timeout" if steps >= max_steps else "crash"
    return dict(
        status=status, reached_goal=int(reached), steps=steps,
        collisions=coll, collision_rate=(coll / steps if steps else 0.0),
        infeasible_steps=infeas, infeas_rate=(infeas / steps if steps else 0.0),
        ctrl_mean_ms=cs["mean"], ctrl_p50_ms=cs["p50"], ctrl_p90_ms=cs["p90"],
        ctrl_p99_ms=cs["p99"], ctrl_max_ms=cs["max"],
        loop_mean_ms=ls["mean"], loop_p99_ms=ls["p99"], loop_over_dt_rate=over,
    )


# ----------------------------------------------------------------------------- worker
def run_one_job(job):
    """One independent episode. Picklable in/out (runs in a worker process)."""
    label, seed, n_obs, exp_base, want_traj = job
    run_fn, extras = METHOD_MAP[label]
    env = build_env(seed, n_obs)
    start = np.asarray(env.start_xyz_yaw[:3], dtype=np.float32)
    goal = np.asarray(env.goal_xyz, dtype=np.float32)
    obs0 = np.zeros((0, 3), dtype=np.float32)
    if want_traj:
        try:
            if getattr(env, "obstacles", None):
                obs0 = np.array([ob.pos for ob in env.obstacles], dtype=np.float32)
        except Exception:
            pass
    exp = dict(exp_base, **extras)
    allowed = set(inspect.signature(run_fn).parameters.keys())
    exp_clean = {k: v for k, v in exp.items() if k in allowed}
    t0 = time.perf_counter()
    result = run_fn(env, **exp_clean)
    elapsed = time.perf_counter() - t0
    traj = np.asarray(result["robot_traj"], dtype=np.float32).reshape(-1, 3)
    metrics = compute_metrics(result, ENV_KWARGS["dt"],
                              int(exp_clean.get("max_steps", len(traj))))
    return dict(label=label, seed=int(seed), n_obs=int(n_obs),
                traj=(traj if want_traj else None),
                start=start, goal=goal, obs=obs0, metrics=metrics, elapsed=elapsed)


def run_jobs(jobs, workers):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            m = r["metrics"]
            print(f"  [{done:>3}/{len(jobs)}] {r['label']:16s} seed={r['seed']} "
                  f"n_obs={r['n_obs']:>3} status={m['status']:>11} steps={m['steps']:>3} "
                  f"ctrl_mean={m['ctrl_mean_ms']:.1f}ms elapsed={r['elapsed']:.1f}s", flush=True)
            out.append(r)
    return out


def run_timing_sequential(seeds, n_obs_list, timing_steps, exp_base):
    """Clean per-step control timing WITHOUT CPU contention: episodes run one at a
    time (single process), capped to `timing_steps` steps. The simulation is
    deterministic, so outcomes match the parallel pass; only the wall-clock timing
    differs, and here it is contention-free and therefore trustworthy."""
    eb = dict(exp_base)
    eb["max_steps"] = int(timing_steps)
    jobs = [(lab, s, no, eb, False)
            for no in sorted(set(n_obs_list)) for s in seeds for lab in METHOD_LABELS]
    rows = []
    for i, j in enumerate(jobs, 1):
        r = run_one_job(j)
        m = r["metrics"]
        print(f"  [timing {i:>3}/{len(jobs)}] {r['label']:16s} seed={r['seed']} "
              f"n_obs={r['n_obs']:>3} steps={m['steps']:>3} "
              f"ctrl_mean={m['ctrl_mean_ms']:.1f}ms ctrl_p99={m['ctrl_p99_ms']:.1f}ms "
              f"over_dt={m['loop_over_dt_rate']:.2f}", flush=True)
        rows.append(r)
    return rows


def _agg(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def clean_ctrl_by_method(timing_rows, n_obs):
    """Per-method clean ctrl stats at a given n_obs, averaged over timing seeds."""
    out = {}
    for label in METHOD_LABELS:
        rs = [r["metrics"] for r in timing_rows
              if r["label"] == label and r["n_obs"] == n_obs]
        if not rs:
            continue
        out[label] = {
            "ctrl_mean_ms": _agg([m["ctrl_mean_ms"] for m in rs]),
            "ctrl_p99_ms": _agg([m["ctrl_p99_ms"] for m in rs]),
            "loop_over_dt_rate": _agg([m["loop_over_dt_rate"] for m in rs]),
        }
    return out


# ----------------------------------------------------------------------------- outputs
def write_table(outcome_results, clean_ctrl, seeds):
    """Per-(method,seed) outcomes (deterministic; from the parallel pass) joined with
    clean single-thread control timing (from the sequential pass). Outcome and timing
    are reported as separate fields rather than a single conflated 'status'."""
    os.makedirs(METRIC_DIR, exist_ok=True)
    cols = ["method", "seed", "n_obs", "reached_goal", "steps", "collisions",
            "collision_rate", "infeasible_steps", "infeas_rate"]
    with open(TABLE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in outcome_results:
            row = {"method": r["label"], "seed": r["seed"], "n_obs": r["n_obs"]}
            row.update({k: r["metrics"].get(k) for k in cols if k in r["metrics"]})
            w.writerow(row)
    agg = {}
    for label in METHOD_LABELS:
        rs = [r["metrics"] for r in outcome_results if r["label"] == label]
        if not rs:
            continue
        ct = clean_ctrl.get(label, {})
        over = ct.get("loop_over_dt_rate", float("nan"))
        agg[label] = {
            "n_seeds": len(rs),
            "reached_rate": float(np.mean([m["reached_goal"] for m in rs])),
            "mean_collision_rate": float(np.mean([m["collision_rate"] for m in rs])),
            "mean_infeas_rate": float(np.mean([m["infeas_rate"] for m in rs])),
            "mean_steps": float(np.mean([m["steps"] for m in rs])),
            "ctrl_mean_ms": ct.get("ctrl_mean_ms", float("nan")),
            "ctrl_p99_ms": ct.get("ctrl_p99_ms", float("nan")),
            "realtime_feasible": bool((not np.isnan(over)) and (over <= 0.10)),
        }
    json.dump({"seeds": list(seeds), "methods": METHOD_LABELS, "agg": agg,
               "note": "outcomes from full parallel episodes; control timing from a "
                       "sequential single-thread pass (contention-free)"},
              open(TABLE_JSON, "w"), indent=2)
    print(f"[saved] table -> {TABLE_CSV} , {TABLE_JSON}")


def write_scalability(results_all):
    os.makedirs(METRIC_DIR, exist_ok=True)
    cols = ["method", "seed", "n_obs", "status", "steps", "ctrl_mean_ms", "ctrl_p50_ms",
            "ctrl_p90_ms", "ctrl_p99_ms", "ctrl_max_ms", "loop_mean_ms", "loop_over_dt_rate"]
    with open(SCAL_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results_all:
            row = {"method": r["label"], "seed": r["seed"], "n_obs": r["n_obs"]}
            row.update({k: r["metrics"].get(k) for k in cols if k in r["metrics"]})
            w.writerow(row)
    print(f"[saved] scalability -> {SCAL_CSV}")


def render_traj(results_main, seeds_for_fig):
    data = {}
    for seed in seeds_for_fig:
        rs = [r for r in results_main if r["seed"] == seed]
        if not rs:
            continue
        trajs = {r["label"]: r["traj"] for r in rs if r["traj"] is not None}
        ref = rs[0]
        data[seed] = dict(start=ref["start"], goal=ref["goal"], obs=ref["obs"], trajs=trajs)
    make_figure(data, TRAJ_OUT)


def render_scalability():
    # reuse plot_n_obs by pointing it at our unified CSV
    os.environ["FCP_NOBS_CSV"] = SCAL_CSV
    os.environ["FCP_NOBS_OUT"] = SCAL_OUT
    import importlib
    import plot_n_obs
    importlib.reload(plot_n_obs)
    plot_n_obs.main()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22, 23, 24])
    ap.add_argument("--traj-seeds", type=int, nargs="+", default=[20, 21, 22],
                    help="subset of --seeds shown as Fig.6 panels (a)-(c)")
    ap.add_argument("--n-obs-main", type=int, default=280)
    ap.add_argument("--n-obs-sweep", type=int, nargs="+",
                    default=[10, 50, 100, 150, 200, 280])
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--timing-steps", type=int, default=40,
                    help="capped #steps per episode for the sequential timing pass")
    ap.add_argument("--timing-seeds", type=int, nargs="+", default=[20, 21, 22])
    ap.add_argument("--replot", action="store_true",
                    help="reuse cached outcomes AND timing; re-render figures/table only")
    ap.add_argument("--fix-timing", action="store_true",
                    help="reuse cached outcomes; re-measure control timing sequentially")
    ap.add_argument("--reuse-timing", action="store_true",
                    help="run outcomes fresh but reuse cached timing (skip re-measuring)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    global CACHE
    if args.smoke:
        CACHE = CACHE.replace(".pkl", "_smoke.pkl")
        args.seeds = [25]; args.traj_seeds = [25]
        args.n_obs_main = 20; args.n_obs_sweep = [10, 20]
        args.max_steps = 20; args.timing_seeds = [25]; args.timing_steps = 12

    exp_base = dict(EXP_BASE)
    exp_base["max_steps"] = args.max_steps
    exp_base["n_jobs"] = 1  # single-thread per episode (fair, contention-free timing)
    n_obs_all = sorted(set(args.n_obs_sweep) | {args.n_obs_main})

    cache = {}
    if (args.replot or args.fix_timing or args.reuse_timing) and os.path.isfile(CACHE):
        cache = pickle.load(open(CACHE, "rb"))

    # --- outcomes + trajectories (PARALLEL; deterministic, so contention is harmless) ---
    if (args.replot or args.fix_timing) and "main" in cache:
        results_main, seeds = cache["main"], cache["seeds"]
        print(f"[outcomes] reused {len(results_main)} cached episodes")
    else:
        jobs_main = [(lab, s, args.n_obs_main, exp_base, True)
                     for s in args.seeds for lab in METHOD_LABELS]
        print(f"[outcomes] {len(jobs_main)} episodes (parallel) on {args.workers} workers",
              flush=True)
        t0 = time.perf_counter()
        results_main = run_jobs(jobs_main, args.workers)
        seeds = args.seeds
        print(f"[outcomes] done in {time.perf_counter() - t0:.0f}s", flush=True)

    # --- control timing (SEQUENTIAL, contention-free; this is what the paper reports) ---
    if (args.replot or args.reuse_timing) and "timing" in cache:
        timing_rows = cache["timing"]
        print(f"[timing] reused {len(timing_rows)} cached timing episodes")
    else:
        print(f"[timing] sequential: {len(args.timing_seeds)} seeds x {len(n_obs_all)} "
              f"n_obs x {len(METHOD_LABELS)} methods, {args.timing_steps} steps each",
              flush=True)
        t1 = time.perf_counter()
        timing_rows = run_timing_sequential(args.timing_seeds, n_obs_all,
                                            args.timing_steps, exp_base)
        print(f"[timing] done in {time.perf_counter() - t1:.0f}s", flush=True)

    os.makedirs(PAPER_DIR, exist_ok=True)
    pickle.dump({"main": results_main, "seeds": seeds, "timing": timing_rows},
                open(CACHE, "wb"))
    print(f"[cached] -> {CACHE}")

    clean = clean_ctrl_by_method(timing_rows, args.n_obs_main)
    write_table(results_main, clean, seeds)
    write_scalability(timing_rows)
    traj_seeds = [s for s in args.traj_seeds if s in seeds] or list(seeds)[:3]
    render_traj(results_main, traj_seeds)
    render_scalability()
    print("[done] table + traj + scalability (outcomes parallel, timing sequential)")


if __name__ == "__main__":
    main()
