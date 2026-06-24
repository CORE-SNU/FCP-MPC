#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import numpy as np

# ----------------------------
# Config
# ----------------------------
METRIC_DIR = "metric"          # runner_2d.py writes per-(dataset, controller) JSON here
OUT_DIR = "tables"
# The paper \inputs the 2D tables from T_RO2026/, so write there too — otherwise
# tables/ holds the fresh (MPPI) numbers while the paper keeps stale ones. Writing
# both keeps T_RO2026/ in lock-step with the regenerated tables/.
PAPER_DIR = "T_RO2026"
OUT_MAIN_TEX = "table_2d_results.tex"      # baselines vs FCP-MPC (our full / adaptive method)
OUT_ABLATION_TEX = "table_2d_ablation.tex" # FCP-MPC internal ablation: online adaptation effect

DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]   # standard ETH-UCY order

# ---- Main results table ----
# Baselines keep their \cite; FCP rows are the core method = the *offline-calibrated*
# (non-adaptive) envelope, shown for the hard and soft constraint modes. Online
# adaptation is an optional add-on, so its effect is isolated in the ablation table
# below; the main table compares like-for-like (offline envelopes) and stays compact.
# Ordered by development year: ACP (2023), CC (2024), ECP (2025), FCP (ours, 2026).
MAIN_CONTROLLERS = [
    ("acp-mpc",              r"ACP-MPC~\cite{dixit2023adaptive}"),
    ("cc",                   r"CC-MPC~\cite{lekeufack2024decision}"),
    ("ecp-mpc",              r"ECP-MPC~\cite{shin2025egocentric}"),
    ("fcp-hard-nonadaptive", r"FCP-MPC (hard)"),
    ("fcp-soft-nonadaptive", r"FCP-MPC (soft)"),
]

# ---- Ablation table: effect of online coefficient adaptation ----
# Each entry: (controller key, constraint-mode label, online-adaptation label).
# Bolding is computed within each constraint-mode pair so the table reads as a
# direct fixed-vs-online comparison.
ABLATION_ROWS = [
    ("fcp-hard-nonadaptive", "Hard", "No"),
    ("fcp-hard-adaptive",    "Hard", "Yes"),
    ("fcp-soft-nonadaptive", "Soft", "No"),
    ("fcp-soft-adaptive",    "Soft", "Yes"),
]

MAX_N_STEPS = {
    "zara1": 100,
    "zara2": 100,
    "eth": 100,
    "hotel": 100,
    "univ": 300,
}

TIMEOUT_EPS = 0.5
NA = "--"


# ----------------------------
# Helpers
# ----------------------------
def safe_float(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else float("nan")
    except Exception:
        return float("nan")


def nanmean_or_nan(vals):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def valid_scene_mask(d):
    """A scene actually ran iff its per-scene controller-timing entry is a dict.

    Degenerate scenes (e.g. eth scenes 200/300 with 0 steps) come back with
    timing_ctrl_ms == null, cost == NaN and time == 0; including them poisons
    the averages, so we drop them from every statistic.
    """
    timing = d.get("timing_ctrl_ms", []) or []
    return [isinstance(t, dict) for t in timing]


def masked(lst, mask):
    """Select entries of lst at positions where mask is True (length-tolerant)."""
    out = []
    for i, m in enumerate(mask):
        if m and i < len(lst):
            out.append(lst[i])
    return out


def ctrl_time_mean(d, mask):
    """Average the per-scene mean controller time over valid scenes only."""
    timing = d.get("timing_ctrl_ms", []) or []
    vals = []
    for i, m in enumerate(mask):
        if m and i < len(timing) and isinstance(timing[i], dict):
            vals.append(safe_float(timing[i].get("mean")))
    return nanmean_or_nan(vals)


def format_steps(val, max_steps):
    if not math.isfinite(val):
        return NA
    if val >= max_steps - TIMEOUT_EPS:
        return "timeout"
    return f"{val:.1f}"


def fmt(x, nd=3):
    return NA if not math.isfinite(x) else f"{x:.{nd}f}"


def bold(s):
    return r"\textbf{" + s + "}"


# ----------------------------
# Metric loading
# ----------------------------
def load_results(dataset, controller_keys):
    """Return (present_keys, results) where results[key] holds masked metric means.

    Controllers whose JSON is missing are simply skipped, so partial/failed runs
    drop out of the table instead of crashing the build.
    """
    present = []
    results = {}
    for key in controller_keys:
        path = os.path.join(METRIC_DIR, f"{dataset}_{key}.json")
        if not os.path.isfile(path):
            continue
        present.append(key)

        d = read_json(path)
        mask = valid_scene_mask(d)
        if not any(mask):
            results[key] = {"collision": float("nan"), "infeasible": float("nan"),
                            "steps": float("nan"), "ctrl_ms": float("nan")}
            continue

        results[key] = {
            "collision": nanmean_or_nan(masked(d.get("collision", []), mask)),
            "infeasible": nanmean_or_nan(masked(d.get("infeasible", []), mask)),
            "steps": nanmean_or_nan(masked(d.get("time", []), mask)),
            "ctrl_ms": ctrl_time_mean(d, mask),
        }
    return present, results


# Soft/penalty methods have no hard safety constraint, so infeasibility cannot occur
# by construction -> reported as N/A and excluded from the best-infeasible comparison.
SOFT_NA_KEYS = {"cc", "fcp-soft-adaptive", "fcp-soft-nonadaptive"}


def best_values(dataset, results, keys):
    """Best (min) value per column over `keys`; steps ignores timeouts."""
    def best(metric, ignore_timeout=False, exclude=()):
        vals = []
        for k in keys:
            if k in exclude:
                continue
            v = results[k][metric]
            if not math.isfinite(v):
                continue
            if ignore_timeout and v >= MAX_N_STEPS[dataset] - TIMEOUT_EPS:
                continue
            vals.append(v)
        return min(vals) if vals else float("nan")

    return {
        "collision": best("collision"),
        "infeasible": best("infeasible", exclude=SOFT_NA_KEYS),
        "steps": best("steps", ignore_timeout=True),
        "ctrl_ms": best("ctrl_ms"),
    }


def render_metric_cells(dataset, r, best, infeas_na=False):
    """Format the four metric cells for one row, bolding column-best values.
    ``infeas_na`` marks soft/penalty methods whose infeasibility is N/A by construction."""
    c, i, s, t = r["collision"], r["infeasible"], r["steps"], r["ctrl_ms"]

    c_str = fmt(c, 3)
    i_str = "N/A" if infeas_na else fmt(i, 3)
    s_str = format_steps(s, MAX_N_STEPS[dataset])
    t_str = fmt(t, 2)

    if math.isfinite(c) and abs(c - best["collision"]) < 1e-9:
        c_str = bold(c_str)
    if (not infeas_na) and math.isfinite(i) and abs(i - best["infeasible"]) < 1e-9:
        i_str = bold(i_str)
    if s_str not in (NA, "timeout") and math.isfinite(s) and abs(s - best["steps"]) < 1e-9:
        s_str = bold(s_str)
    if math.isfinite(t) and abs(t - best["ctrl_ms"]) < 1e-9:
        t_str = bold(t_str)

    return c_str, i_str, s_str, t_str


# ----------------------------
# Main results table
# ----------------------------
def build_main_table():
    lines = []
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\hline")
    lines.append(
        r"Dataset & Method & Collision rate $\downarrow$ & "
        r"Infeasible rate $\downarrow$ & "
        r"Steps to goal $\downarrow$ & "
        r"Ctrl.\ time (ms) $\downarrow$ \\"
    )
    lines.append(r"\hline")

    keys = [k for k, _ in MAIN_CONTROLLERS]
    names = dict(MAIN_CONTROLLERS)

    for dataset in DATASETS:
        present, results = load_results(dataset, keys)
        if not present:
            continue
        best = best_values(dataset, results, present)

        for j, key in enumerate(present):
            c_str, i_str, s_str, t_str = render_metric_cells(
                dataset, results[key], best, infeas_na=key in SOFT_NA_KEYS)
            dataset_str = dataset if j == 0 else ""
            lines.append(
                f"{dataset_str} & {names[key]} & {c_str} & {i_str} & {s_str} & {t_str} \\\\"
            )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ----------------------------
# Ablation table (online adaptation effect)
# ----------------------------
def build_ablation_table():
    lines = []
    lines.append(r"\begin{tabular}{lllcccc}")
    lines.append(r"\hline")
    lines.append(
        r"Dataset & Constraint & Online adapt. & Collision rate $\downarrow$ & "
        r"Infeasible rate $\downarrow$ & "
        r"Steps to goal $\downarrow$ & "
        r"Ctrl.\ time (ms) $\downarrow$ \\"
    )
    lines.append(r"\hline")

    keys = [k for k, _, _ in ABLATION_ROWS]

    for dataset in DATASETS:
        present, results = load_results(dataset, keys)
        if not present:
            continue
        # Bold the column-best within each constraint mode (fixed vs online pair).
        present_set = set(present)
        for j, (key, mode, adapt) in enumerate(ABLATION_ROWS):
            if key not in present_set:
                continue
            pair = [k for k, m, _ in ABLATION_ROWS if m == mode and k in present_set]
            best = best_values(dataset, results, pair)
            c_str, i_str, s_str, t_str = render_metric_cells(
                dataset, results[key], best, infeas_na=key in SOFT_NA_KEYS)
            dataset_str = dataset if j == 0 else ""
            lines.append(
                f"{dataset_str} & {mode} & {adapt} & {c_str} & {i_str} & {s_str} & {t_str} \\\\"
            )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ----------------------------
# Entry point
# ----------------------------
def build_table():
    os.makedirs(OUT_DIR, exist_ok=True)

    main_tex = build_main_table()
    ablation_tex = build_ablation_table()

    # Write the tables into both tables/ (reference) and T_RO2026/ (what the paper
    # \inputs) so the two never desync.
    out_dirs = [OUT_DIR, PAPER_DIR]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, OUT_MAIN_TEX), "w", encoding="utf-8") as f:
            f.write(main_tex)
        with open(os.path.join(d, OUT_ABLATION_TEX), "w", encoding="utf-8") as f:
            f.write(ablation_tex)

    print(f"[saved] main results table  -> {[os.path.join(d, OUT_MAIN_TEX) for d in out_dirs]}")
    print(main_tex)
    print(f"\n[saved] ablation table      -> {[os.path.join(d, OUT_ABLATION_TEX) for d in out_dirs]}")
    print(ablation_tex)


if __name__ == "__main__":
    build_table()
