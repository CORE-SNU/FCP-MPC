from __future__ import annotations

import os
import json
import math
import numpy as np

# ----------------------------
# Config
# ----------------------------
# NEW: suite json 하나만 읽는다 (runner가 csv_path 옆에 생성)
SUITE_JSON = "results/quad_suite.json"

OUT_DIR = "tables"
OUT_TEX = "table_3d_results.tex"

METHODS = ["nocp", "cc", "ecp", "fcp"]
METHOD_NAME = {
    "nocp": "Nominal MPC",
    "cc": r"CC-MPC~\cite{lekeufack2024decision}",
    "ecp": r"ECP-MPC~\cite{shin2025egocentricconformalpredictionsafe}",
    "fcp": r"FCP-MPC (ours)",
}

MAX_STEPS = 500
TIMEOUT_EPS = 0.5


# ----------------------------
# Helpers
# ----------------------------
def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def safe_float(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else float("nan")
    except Exception:
        return float("nan")


def mean_or_nan(arr):
    if not arr:
        return float("nan")
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    return float(np.nanmean(a))


def fmt(x, nd=3):
    if not math.isfinite(x):
        return "N/A"
    return f"{x:.{nd}f}"


def bold(s):
    return r"\textbf{" + s + "}"


def _is_finite_num(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _as_list(x):
    return x if isinstance(x, list) else []


def _pick_by_seed(suite: dict, method: str, field: str):
    """
    suite["results"][method][field] is a list aligned with suite["seeds"].
    Returns that list (possibly empty).
    """
    return _as_list(suite.get("results", {}).get(method, {}).get(field, []))


def _timing_mean_list(suite: dict, method: str, which: str):
    """
    which in {"ctrl","loop"}.
    suite["results"][method][f"timing_{which}_ms"] is list of dict|None.
    We extract dict["mean"] -> float, else NaN.
    """
    xs = _pick_by_seed(suite, method, f"timing_{which}_ms")
    out = []
    for v in xs:
        if isinstance(v, dict):
            out.append(safe_float(v.get("mean", float("nan"))))
        else:
            out.append(float("nan"))
    return out


# ----------------------------
# Main
# ----------------------------
def build_table():
    os.makedirs(OUT_DIR, exist_ok=True)

    suite = read_json(SUITE_JSON)

    seeds = _as_list(suite.get("seeds", []))
    if not seeds:
        raise RuntimeError(f"No seeds found in suite json: {SUITE_JSON}")

    rows = {}

    # ---------- load & aggregate ----------
    for m in METHODS:
        # arrays aligned with seeds
        status = _pick_by_seed(suite, m, "status")
        steps = [safe_float(x) for x in _pick_by_seed(suite, m, "steps")]
        infeasible_steps = [safe_float(x) for x in _pick_by_seed(suite, m, "infeasible_steps")]
        ctrl_ms = _timing_mean_list(suite, m, "ctrl")

        # if method missing entirely
        if (not status) and (not steps):
            rows[m] = None
            continue

        # collision rate: fraction of seeds whose status == "collision"
        collision_rate = mean_or_nan([1.0 if s == "collision" else 0.0 for s in status])

        # infeasible rate (seed-level): any infeasible step occurred
        infeasible_rate = mean_or_nan([1.0 if (x > 0) else 0.0 for x in infeasible_steps])

        # steps-to-goal: only successful runs (and not near timeout)
        steps_success = [
            steps[i]
            for i in range(min(len(status), len(steps)))
            if (status[i] == "success") and (steps[i] < MAX_STEPS - TIMEOUT_EPS)
        ]

        rows[m] = {
            "collision": collision_rate,
            "infeasible": infeasible_rate,
            "steps": mean_or_nan(steps_success),
            "steps_success_count": len(steps_success),
            "ctrl_ms": mean_or_nan(ctrl_ms),
        }

    # ---------- mins for bold ----------
    def min_val(key):
        vals = [
            rows[m][key]
            for m in METHODS
            if rows.get(m) is not None and _is_finite_num(rows[m][key])
        ]
        return min(vals) if vals else float("nan")

    min_collision = min_val("collision")
    min_infeasible = min_val("infeasible")
    min_steps = min_val("steps")
    min_ctrl = min_val("ctrl_ms")

    # ---------- LaTeX ----------
    lines = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{3D quadrotor navigation results with 280 dynamic obstacles.}",
        r"\label{tab:pybullet_results}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & "
        r"Collision rate $\downarrow$ & "
        r"Infeasible rate $\downarrow$ & "
        r"Steps to goal $\downarrow$ & "
        r"Ctrl.\ time (ms) $\downarrow$ \\",
        r"\hline",
    ]

    for m in METHODS:
        rrow = rows.get(m)
        name = METHOD_NAME[m]

        if rrow is None:
            lines.append(f"{name} & N/A & N/A & N/A & N/A \\\\")
            continue

        c, i, s, t = rrow["collision"], rrow["infeasible"], rrow["steps"], rrow["ctrl_ms"]

        c_str = fmt(c)
        i_str = fmt(i)
        t_str = fmt(t, 2)

        if rrow["steps_success_count"] == 0:
            s_str = "timeout (crashed)"
        else:
            s_str = fmt(s, 1)

        if math.isfinite(c) and abs(c - min_collision) < 1e-12:
            c_str = bold(c_str)
        if math.isfinite(i) and abs(i - min_infeasible) < 1e-12:
            i_str = bold(i_str)
        if rrow["steps_success_count"] > 0 and math.isfinite(s) and abs(s - min_steps) < 1e-12:
            s_str = bold(s_str)
        if math.isfinite(t) and abs(t - min_ctrl) < 1e-12:
            t_str = bold(t_str)

        lines.append(f"{name} & {c_str} & {i_str} & {s_str} & {t_str} \\\\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path = os.path.join(OUT_DIR, OUT_TEX)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[saved] LaTeX table -> {out_path}")
    print(f"[info] read suite json -> {SUITE_JSON}")


if __name__ == "__main__":
    build_table()