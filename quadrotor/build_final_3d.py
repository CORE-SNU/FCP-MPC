"""Assemble the final 3D table + trajectory figure from already-computed results.
No new episodes are run.

Table (6 methods): ACP, CC, ECP, Nominal MPC (no conformal field), FCP-MPC (hard),
FCP-MPC (soft). The same offline-calibrated field is deployed two ways (hard
filter / soft penalty); Nominal is that same controller with the field removed.
Trajectory figure: FCP-soft (the practical headline) vs baselines.
"""
import pickle
import make_3d_results as D

N_OBS = 280
sf = pickle.load(open("soft3d_full.pkl", "rb"))            # ACP/CC/ECP/Nominal/FCP-soft(as "ours")
sc = pickle.load(open("traj_scan_results.pkl", "rb"))      # FCP-hard (as "ours")
cache = pickle.load(open(f"{D.PAPER_DIR}/results_3d_cache.pkl", "rb"))


def relabel(rows, frm, to):
    return [dict(r, label=to) if r["label"] == frm else r for r in rows]


# --- outcomes ---
results = relabel(list(sf["results"]), "FCP-MPC (ours)", "FCP-MPC (soft)")
fcp_hard = [dict(r, label="FCP-MPC (hard)") for r in sc if r["label"] == "FCP-MPC (ours)"]
all_results = results + fcp_hard

# --- timing (contention-free; n_obs=280) ---
timing = relabel(list(sf["timing"]), "FCP-MPC (ours)", "FCP-MPC (soft)")
hard_tim = [dict(r, label="FCP-MPC (hard)") for r in cache["timing"]
            if r["label"] == "FCP-MPC (ours)" and r["n_obs"] == N_OBS]
all_timing = timing + hard_tim

# --- table ---
# Narrative focuses on FCP's flexible hard/soft field usage vs the baselines;
# Nominal is dropped (not needed for that story). Both FCP variants shown.
D.TABLE_ORDER = ["ACP-MPC", "CC-MPC", "ECP-MPC", "FCP-MPC (hard)", "FCP-MPC (soft)"]
D.TABLE_CITE.update({"FCP-MPC (hard)": "", "FCP-MPC (soft)": ""})
D.METHOD_LABELS = list(D.TABLE_ORDER)
clean = D.clean_ctrl_by_method(all_timing, N_OBS)
D.write_latex_table(all_results, clean)

# --- trajectory figure: FCP-soft (kept as "FCP-MPC (ours)" so make_figure plots it) ---
D.render_traj(sf["results"], [20, 22, 30])

# --- console summary ---
import numpy as np
print("\n=== final 3D table aggregates (17 seeds) ===")
for lab in D.TABLE_ORDER:
    rs = [r["metrics"] for r in all_results if r["label"] == lab]
    if not rs:
        continue
    print(f"  {lab:16s} reach={np.mean([m['reached_goal'] for m in rs]):.2f} "
          f"coll={np.mean([m['collision_rate'] for m in rs]):.3f} "
          f"infeas={np.mean([m['infeas_rate'] for m in rs]):.3f} "
          f"ctrl={clean.get(lab, {}).get('ctrl_mean_ms', float('nan')):.1f}ms")
