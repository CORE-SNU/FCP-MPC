"""Second 3D results table at a LOWER obstacle count (N_obs=50), all five methods,
to complement the dense (N_obs=280) table -- showing the comparison across obstacle
densities. Same columns/format/timing methodology as the main table. In-process so
the FCP hard/soft overrides apply.
"""
from __future__ import annotations
import os, pickle, time
import numpy as np
import make_3d_results as D
from sim_func_3d import run_one_episode_visual_3d as run_fcp

N_OBS = 50
SEEDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
LABELS = ["ACP-MPC", "CC-MPC", "ECP-MPC", "FCP-MPC (hard)", "FCP-MPC (soft)"]
OUT_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_3d_results.pkl")

D.METHOD_MAP["FCP-MPC (hard)"] = (run_fcp, {"CP": True, "safety_mode": "hard", "break_on_collision": True})
D.METHOD_MAP["FCP-MPC (soft)"] = (run_fcp, {"CP": True, "safety_mode": "soft", "break_on_collision": True})


def main():
    eb = dict(D.EXP_BASE); eb["max_steps"] = 250; eb["n_jobs"] = 1
    print(f"[sparse N_obs={N_OBS}] {len(SEEDS)*len(LABELS)} episodes (sequential)", flush=True)
    results = []
    for s in SEEDS:
        for lab in LABELS:
            t0 = time.perf_counter()
            r = D.run_one_job((lab, s, N_OBS, eb, False))
            m = r["metrics"]
            print(f"  {lab:16s} seed={s:>3} reach={m['reached_goal']} "
                  f"coll={m['collision_rate']:.3f} infeas={m['infeas_rate']:.3f} "
                  f"steps={m['steps']:>3} ({time.perf_counter()-t0:.0f}s)", flush=True)
            results.append(r)

    D.METHOD_LABELS = LABELS
    # Use all seeds for the timing pass too (cheap at N_obs=50) so ECP has post-warmup
    # samples after warmup-step exclusion; otherwise its control-time cell can be empty.
    timing = D.run_timing_sequential(SEEDS, [N_OBS], 40, eb)
    pickle.dump({"results": results, "timing": timing}, open(OUT_PKL, "wb"))

    # write the sparse table to its own file
    D.TABLE_TEX = os.path.join(D.PAPER_DIR, "table_3d_sparse.tex")
    D.TABLE_ORDER = LABELS
    D.TABLE_CITE.update({"FCP-MPC (hard)": "", "FCP-MPC (soft)": ""})
    D.METHOD_LABELS = LABELS
    clean = D.clean_ctrl_by_method(timing, N_OBS)
    D.write_latex_table(results, clean)

    print("\n=== sparse (N_obs=50) aggregates ===", flush=True)
    for lab in LABELS:
        rs = [r["metrics"] for r in results if r["label"] == lab]
        print(f"  {lab:16s} reach={np.mean([m['reached_goal'] for m in rs]):.2f} "
              f"coll={np.mean([m['collision_rate'] for m in rs]):.3f} "
              f"infeas={np.mean([m['infeas_rate'] for m in rs]):.3f} "
              f"ctrl={clean.get(lab, {}).get('ctrl_mean_ms', float('nan')):.1f}ms", flush=True)


if __name__ == "__main__":
    main()
