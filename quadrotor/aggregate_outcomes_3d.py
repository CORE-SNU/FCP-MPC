#!/usr/bin/env python3
"""Aggregate the fixed-reset outcome re-run (outcomes_3d_fixed.pkl) into the two 3D
paper tables, PRESERVING the existing contention-free control-time column (the
start-altitude fix changes outcomes, not per-step compute cost).

Mirrors make_3d_results.write_latex_table exactly (mean$\\pm$std cells, best-value
bolding, crash/timeout from step counts ignoring the parallel-pass compute_fail
label). Writes table_3d_results.tex (N=280) and table_3d_sparse.tex (N=50) and
prints the per-method aggregates used in the Results/Discussion prose.
"""
from __future__ import annotations
import os
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "T_RO2026")
TABLE_ORDER = ["ACP-MPC", "CC-MPC", "ECP-MPC", "FCP-MPC (hard)", "FCP-MPC (soft)"]
TABLE_CITE = {"ACP-MPC": r"~\cite{dixit2023adaptive}",
              "CC-MPC": r"~\cite{lekeufack2024decision}",
              "ECP-MPC": r"~\cite{shin2025egocentric}",
              "FCP-MPC (hard)": "", "FCP-MPC (soft)": ""}
SOFT_INFEAS_NA = {"CC-MPC", "FCP-MPC (soft)"}

# Preserved control-time (ms) from the contention-free sequential timing pass
# (unchanged by the start-altitude fix). Keyed by density.
CTRL_MS = {
    280: {"ACP-MPC": 215.9, "CC-MPC": 209.0, "ECP-MPC": 1282.9,
          "FCP-MPC (hard)": 40.2, "FCP-MPC (soft)": 160.7},
    50:  {"ACP-MPC": 42.2, "CC-MPC": 40.8, "ECP-MPC": 198.8,
          "FCP-MPC (hard)": 25.9, "FCP-MPC (soft)": 38.2},
}
OUT_TEX = {280: os.path.join(PAPER, "table_3d_results.tex"),
           50: os.path.join(PAPER, "table_3d_sparse.tex")}


def _steps_cell(rs, max_steps):
    reached = [m for m in rs if m["reached_goal"]]
    if reached:
        vs = [m["steps"] for m in reached]
        v, sd = float(np.mean(vs)), float(np.std(vs))
        cell = f"${v:.1f}\\pm{sd:.1f}$" if len(vs) > 1 else f"{v:.1f}"
        return cell, v
    n_timeout = sum(1 for m in rs if m["steps"] >= max_steps)
    return ("timeout" if n_timeout * 2 >= len(rs) else "crashed"), None


def build_table(results, density):
    ctrl = CTRL_MS[density]
    max_steps = max((m["steps"] for m in results), default=0)
    rows = {}
    for label in TABLE_ORDER:
        rs = [m for m in results if m["label"] == label]
        if not rs:
            continue
        rs_reached = [m for m in rs if m["reached_goal"]]
        if rs_reached:
            sv = [m["steps"] for m in rs_reached]
            steps_val, steps_std, steps_label = float(np.mean(sv)), float(np.std(sv)), None
        else:
            nt = sum(1 for m in rs if m["steps"] >= max_steps)
            steps_val, steps_std, steps_label = None, None, ("timeout" if nt * 2 >= len(rs) else "crashed")
        coll_v = [m["collision_rate"] for m in rs]
        infeas_v = [m["infeas_rate"] for m in rs]
        rows[label] = dict(
            coll=float(np.mean(coll_v)), coll_std=float(np.std(coll_v)),
            infeas=(None if label in SOFT_INFEAS_NA else float(np.mean(infeas_v))),
            infeas_std=(None if label in SOFT_INFEAS_NA else float(np.std(infeas_v))),
            ctrl=ctrl.get(label, float("nan")),
            steps_val=steps_val, steps_std=steps_std, steps_label=steps_label, n=len(rs),
            reached_rate=float(np.mean([m["reached_goal"] for m in rs])),
        )

    def _colmin(key):
        vals = [v[key] for v in rows.values()
                if v.get(key) is not None and not (isinstance(v[key], float) and np.isnan(v[key]))]
        return min(vals) if vals else None

    # Column-best = the genuine min over ALL methods (the Goal-reached column now
    # contextualizes a low collision rate that comes paired with a crash). Ties are
    # all bolded.
    best_coll, best_infeas, best_ctrl = _colmin("coll"), _colmin("infeas"), _colmin("ctrl")
    best_steps = _colmin("steps_val")
    best_reach = max((v["reached_rate"] for v in rows.values()), default=0.0)

    def _isbest(v, best):
        return (best is not None and isinstance(v, (int, float)) and not np.isnan(v)
                and abs(v - best) < 1e-9)

    def _f(label, key, fmt, best):
        v = rows[label][key]
        if isinstance(v, float) and np.isnan(v):
            return "--"
        s = fmt.format(v)
        return rf"\textbf{{{s}}}" if _isbest(v, best) else s

    def _fpm(label, key, nd, best):
        # Bold only the MEAN (via \mathbf, which propagates inside math mode); the
        # \pm std stays at normal weight so the highlight reads cleanly.
        m = rows[label][key]
        sd = rows[label].get(key + "_std")
        mean_s = rf"\mathbf{{{m:.{nd}f}}}" if _isbest(m, best) else f"{m:.{nd}f}"
        body = mean_s if (sd is None or rows[label]['n'] <= 1) else f"{mean_s}\\pm{sd:.{nd}f}"
        return f"${body}$"

    lines = [r"\begin{tabular}{lccccc}", r"\hline",
             "Method &", r"Collision rate $\downarrow$ &",
             r"Infeasible rate $\downarrow$ &", r"Steps to goal $\downarrow$ &",
             r"Ctrl.\ time (ms) $\downarrow$ &", r"Goal reached $\uparrow$ \\", r"\hline"]
    for label in TABLE_ORDER:
        if label not in rows:
            continue
        r = rows[label]
        if r["steps_label"] is not None:
            steps = r["steps_label"]
        else:
            mean_s = (rf"\mathbf{{{r['steps_val']:.1f}}}"
                      if _isbest(r["steps_val"], best_steps) else f"{r['steps_val']:.1f}")
            steps = (f"${mean_s}\\pm{r['steps_std']:.1f}$" if r["n"] > 1 else f"${mean_s}$")
        infeas_cell = ("N/A" if r["infeas"] is None
                       else _fpm(label, "infeas", 3, best_infeas))
        reach_pct = f"{r['reached_rate']*100:.0f}\\%"
        reach_cell = (rf"\textbf{{{reach_pct}}}"
                      if abs(r["reached_rate"] - best_reach) < 1e-9 else reach_pct)
        lines += [
            f"{label}{TABLE_CITE.get(label, '')}",
            f"& {_fpm(label, 'coll', 3, best_coll)}",
            f"& {infeas_cell}",
            f"& {steps}",
            f"& {_f(label, 'ctrl', '{:.1f}', best_ctrl)}",
            rf"& {reach_cell} \\",
        ]
    lines += [r"\hline", r"\end{tabular}"]
    with open(OUT_TEX[density], "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n===== N_obs={density}  ->  {OUT_TEX[density]}")
    for label in TABLE_ORDER:
        if label not in rows:
            continue
        r = rows[label]
        inf = "N/A" if r["infeas"] is None else f"{r['infeas']:.3f}"
        st = r["steps_label"] if r["steps_label"] else f"{r['steps_val']:.1f}±{r['steps_std']:.1f}"
        print(f"  {label:16} coll={r['coll']:.3f}±{r['coll_std']:.3f}  infeas={inf}  "
              f"steps={st:>14}  reached={r['reached_rate']*100:.0f}%  ctrl={r['ctrl']}ms  n={r['n']}")
    print("\n".join(lines))


def main():
    d = pickle.load(open(os.path.join(HERE, "outcomes_3d_fixed.pkl"), "rb"))
    results = [{"label": r["label"], "n_obs": r["n_obs"], **r["metrics"]} for r in d["main"]]
    for density in (280, 50):
        sub = [m for m in results if m["n_obs"] == density]
        if sub:
            build_table(sub, density)


if __name__ == "__main__":
    main()
