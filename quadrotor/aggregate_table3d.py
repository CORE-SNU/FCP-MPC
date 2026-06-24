#!/usr/bin/env python3
"""Aggregate metric_3d/table3d.csv into Table III numbers and patch main.tex inline table.

Usage:
    python aggregate_table3d.py [--csv metric_3d/table3d.csv] [--tex T_RO2026/main.tex]

ECP is not in the CSV (too slow to run); its numbers come from the scalability sweep:
  ctrl ≈ 10.6 s/step, steps = timeout (crashed).
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import re
from typing import Dict, List, Optional


# ── ECP constants from scalability sweep ─────────────────────────────────────
ECP_CTRL_MS  = 10600.0   # ~10.6 s/step (from n-obs=280 point in scalability fig)
ECP_STEPS    = "timeout (crashed)"
ECP_COLL_RATE  = 0.016   # from prior seeds (ref in HANDOFF)
ECP_INFEAS_RATE = 0.027  # from prior seeds

# FCP-offline 5-seed reference (HANDOFF):
# coll 0.020 / infeas 0.069 / steps 82.2 / ctrl ~54ms

METHOD_ORDER = ["nocp", "acp", "cc", "ecp", "fcp"]
METHOD_LABEL = {
    "nocp": "Nominal MPC",
    "acp":  r"ACP-MPC~\cite{dixit2023adaptive}",
    "cc":   r"CC-MPC~\cite{lekeufack2024decision}",
    "ecp":  r"ECP-MPC~\cite{shin2025egocentric}",  # confirmed key
    "fcp":  r"FCP-MPC (ours)",
}


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def load_csv(path: str) -> Dict[str, List[dict]]:
    by_method: Dict[str, List[dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            m = row.get("method", "")
            by_method.setdefault(m, []).append(row)
    return by_method


def mean_finite(vals: List[float]) -> float:
    v = [x for x in vals if math.isfinite(x)]
    return sum(v) / len(v) if v else math.nan


def aggregate(rows: List[dict], max_steps: int = 250) -> dict:
    coll_rates   = [_safe_float(r.get("collision_rate", "nan")) for r in rows]
    infeas_rates = [_safe_float(r.get("infeas_rate",    "nan")) for r in rows]
    steps_vals   = [_safe_float(r.get("steps",          "nan")) for r in rows]
    ctrl_vals    = [_safe_float(r.get("ctrl_mean_ms",   "nan")) for r in rows]
    statuses     = [r.get("status", "") for r in rows]

    # steps-to-goal: include all finished (success OR collision—they at least finished)
    # but treat timeout (steps >= max_steps) as timeout
    steps_for_avg = []
    for s, st in zip(steps_vals, statuses):
        if math.isfinite(s) and s < max_steps and st not in ("error", "missing"):
            steps_for_avg.append(s)

    return {
        "collision_rate": mean_finite(coll_rates),
        "infeas_rate":    mean_finite(infeas_rates),
        "steps":          mean_finite(steps_for_avg),
        "ctrl_mean_ms":   mean_finite(ctrl_vals),
        "n_timeout":      sum(1 for s in statuses if s in ("timeout", "compute_fail")),
        "n_seeds":        len(rows),
    }


def fmt3(x: float) -> str:
    return f"{x:.3f}" if math.isfinite(x) else "N/A"


def fmt1(x: float) -> str:
    return f"{x:.1f}" if math.isfinite(x) else "N/A"


def fmt_ms(x: float) -> str:
    return f"{x:.1f}" if math.isfinite(x) else "N/A"


def bold(s: str) -> str:
    return r"\textbf{" + s + r"}"


def build_rows(by_method: Dict[str, List[dict]]) -> Dict[str, dict]:
    rows = {}
    for m in METHOD_ORDER:
        if m == "ecp":
            rows["ecp"] = {
                "collision_rate": ECP_COLL_RATE,
                "infeas_rate":    ECP_INFEAS_RATE,
                "steps":          math.nan,
                "ctrl_mean_ms":   ECP_CTRL_MS,
                "n_timeout":      5,
                "n_seeds":        5,
                "is_ecp":         True,
            }
        elif m in by_method:
            rows[m] = aggregate(by_method[m])
            rows[m]["is_ecp"] = False
        else:
            rows[m] = {
                "collision_rate": math.nan,
                "infeas_rate":    math.nan,
                "steps":          math.nan,
                "ctrl_mean_ms":   math.nan,
                "n_timeout":      0,
                "n_seeds":        0,
                "is_ecp":         False,
            }
    return rows


def best_val(rows: Dict[str, dict], key: str) -> float:
    vals = [rows[m][key] for m in METHOD_ORDER if math.isfinite(rows[m].get(key, math.nan))]
    return min(vals) if vals else math.nan


def render_table(rows: Dict[str, dict]) -> str:
    best_coll  = best_val(rows, "collision_rate")
    best_inf   = best_val(rows, "infeas_rate")
    best_ctrl  = best_val(rows, "ctrl_mean_ms")

    # steps best among non-ECP non-timeout
    steps_vals = [rows[m]["steps"] for m in METHOD_ORDER
                  if not rows[m].get("is_ecp") and math.isfinite(rows[m]["steps"])]
    best_steps = min(steps_vals) if steps_vals else math.nan

    lines = []
    for m in METHOD_ORDER:
        r      = rows[m]
        label  = METHOD_LABEL[m]
        coll   = r["collision_rate"]
        inf    = r["infeas_rate"]
        steps  = r["steps"]
        ctrl   = r["ctrl_mean_ms"]

        c_s = fmt3(coll)
        i_s = fmt3(inf)
        s_s = ECP_STEPS if r.get("is_ecp") else (
              "timeout" if not math.isfinite(steps)
              else fmt1(steps))
        t_s = fmt_ms(ctrl)

        if math.isfinite(coll)  and abs(coll  - best_coll)  < 1e-9: c_s = bold(c_s)
        if math.isfinite(inf)   and abs(inf   - best_inf)   < 1e-9: i_s = bold(i_s)
        if math.isfinite(ctrl)  and abs(ctrl  - best_ctrl)  < 1e-9: t_s = bold(t_s)
        if not r.get("is_ecp") and math.isfinite(steps) and abs(steps - best_steps) < 0.05:
            s_s = bold(s_s)

        lines.append(f"{label}\n& {c_s}\n& {i_s}\n& {s_s}\n& {t_s} \\\\")

    return "\n\n".join(lines)


TABLE_START = r"\begin{tabular}{lcccc}"
TABLE_END   = r"\end{tabular}"
HEADER = (
    r"\hline" + "\n"
    r"Method &" + "\n"
    r"Collision rate $\downarrow$ &" + "\n"
    r"Infeasible rate $\downarrow$ &" + "\n"
    r"Steps to goal $\downarrow$ &" + "\n"
    r"Ctrl.\ time (ms) $\downarrow$ \\" + "\n"
    r"\hline"
)


def patch_tex(tex_path: str, body: str) -> None:
    with open(tex_path, "r") as f:
        src = f.read()

    # Find the tabular block containing this table
    start = src.find(TABLE_START)
    if start < 0:
        raise RuntimeError("Could not find table tabular block in main.tex")
    end = src.find(TABLE_END, start)
    if end < 0:
        raise RuntimeError("Could not find end of tabular in main.tex")
    end += len(TABLE_END)

    new_tabular = (
        TABLE_START + "\n"
        + HEADER + "\n\n"
        + body + "\n\n"
        + r"\hline" + "\n"
        + TABLE_END
    )

    new_src = src[:start] + new_tabular + src[end:]
    with open(tex_path, "w") as f:
        f.write(new_src)
    print(f"[patched] {tex_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="metric_3d/table3d.csv")
    ap.add_argument("--tex",  default="T_RO2026/main.tex")
    ap.add_argument("--max-steps", type=int, default=250)
    args = ap.parse_args()

    by_method = load_csv(args.csv)
    rows = build_rows(by_method)

    print("\n=== Table III aggregated numbers ===")
    for m in METHOD_ORDER:
        r = rows[m]
        print(
            f"  {m:6s}: coll={fmt3(r['collision_rate'])}  infeas={fmt3(r['infeas_rate'])}"
            f"  steps={fmt1(r['steps'])}  ctrl={fmt_ms(r['ctrl_mean_ms'])} ms"
            f"  (timeout {r['n_timeout']}/{r['n_seeds']})"
        )
    print()

    body = render_table(rows)
    print("=== LaTeX rows ===")
    print(body)
    print()

    patch_tex(args.tex, body)
    print("[done]")


if __name__ == "__main__":
    main()
