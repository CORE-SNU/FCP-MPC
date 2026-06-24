"""Full 3D evaluation with the soft conformal variant as the headline method, plus
a Nominal-MPC ablation (same soft controller, conformal bound OFF). Reuses the
ACP/ECP/CC results + timing already computed; (re)computes FCP-soft and Nominal
over all 17 seeds. Regenerates the 3D table (adds Nominal) and the trajectory
figure (with FCP-soft). Everything runs in-process so the method overrides apply.
"""
from __future__ import annotations
import os, pickle, time
import numpy as np

import make_3d_results as D
from sim_func_3d import run_one_episode_visual_3d as run_fcp

SEEDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
N_OBS = 280
TRAJ_SEEDS = [20, 22, 30]
SCAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traj_scan_results.pkl")
CACHE = os.path.join(D.PAPER_DIR, "results_3d_cache.pkl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soft3d_full.pkl")
BASE_LABELS = ("ACP-MPC", "CC-MPC", "ECP-MPC")

# --- method overrides (visible because everything below runs in-process) ---
D.METHOD_MAP["FCP-MPC (ours)"] = (run_fcp, {"CP": True, "safety_mode": "soft",
                                            "break_on_collision": True})
D.METHOD_MAP["Nominal MPC"] = (run_fcp, {"CP": False, "safety_mode": "soft",
                                         "break_on_collision": True})
NEW_LABELS = ["Nominal MPC", "FCP-MPC (ours)"]


def main():
    eb = dict(D.EXP_BASE); eb["max_steps"] = 250; eb["n_jobs"] = 1

    # 1) outcomes (sequential, in-process) for the two new methods, with trajectories
    print(f"[outcomes] {len(SEEDS)*len(NEW_LABELS)} episodes (sequential)", flush=True)
    new_results = []
    for s in SEEDS:
        for lab in NEW_LABELS:
            t0 = time.perf_counter()
            r = D.run_one_job((lab, s, N_OBS, eb, True))
            m = r["metrics"]
            print(f"  {lab:16s} seed={s:>3} reach={m['reached_goal']} "
                  f"coll_rate={m['collision_rate']:.3f} infeas={m['infeas_rate']:.3f} "
                  f"steps={m['steps']:>3} ({time.perf_counter()-t0:.0f}s)", flush=True)
            new_results.append(r)

    # 2) contention-free control timing (same methodology as the table) at n_obs=280
    D.METHOD_LABELS = NEW_LABELS
    new_timing = D.run_timing_sequential([30, 31, 32], [N_OBS], 40, eb)

    # 3) reuse ACP/ECP/CC outcomes (scan) + timing (cache)
    scan = pickle.load(open(SCAN, "rb"))
    base_results = [r for r in scan if r["label"] in BASE_LABELS]
    cache = pickle.load(open(CACHE, "rb"))
    base_timing = [r for r in cache["timing"]
                   if r["label"] in BASE_LABELS and r["n_obs"] == N_OBS]

    all_results = base_results + new_results
    all_timing = base_timing + new_timing
    pickle.dump({"results": all_results, "timing": all_timing}, open(OUT, "wb"))

    # 4) regenerate table (5 methods, Nominal added) and trajectory figure (FCP-soft)
    D.TABLE_ORDER = ["ACP-MPC", "CC-MPC", "ECP-MPC", "Nominal MPC", "FCP-MPC (ours)"]
    D.TABLE_CITE["Nominal MPC"] = ""
    D.METHOD_LABELS = list(D.TABLE_ORDER)
    clean = D.clean_ctrl_by_method(all_timing, N_OBS)
    D.write_latex_table(all_results, clean)

    traj_results = base_results + [r for r in new_results if r["label"] == "FCP-MPC (ours)"]
    D.render_traj(traj_results, TRAJ_SEEDS)

    # 5) aggregate summary
    print("\n=== aggregate (17 seeds) ===", flush=True)
    for lab in D.TABLE_ORDER:
        rs = [r["metrics"] for r in all_results if r["label"] == lab]
        if not rs:
            continue
        reach = np.mean([m["reached_goal"] for m in rs])
        coll = np.mean([m["collision_rate"] for m in rs])
        inf = np.mean([m["infeas_rate"] for m in rs])
        ct = clean.get(lab, {}).get("ctrl_mean_ms", float("nan"))
        print(f"  {lab:16s} reach={reach:.2f} coll={coll:.3f} infeas={inf:.3f} ctrl={ct:.1f}ms",
              flush=True)
    print(f"[done] table + traj regenerated; combined -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
