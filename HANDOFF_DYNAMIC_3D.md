# Handoff — Dynamic-env 3D experiments (run on desktop)

Local machine is slow for the 3D run (ECP is ~1 s/step at n_obs=280). Run the
commands below on the desktop (conda env `cp`), then commit the regenerated
tables/figures. Everything except the final 3D table/figure regeneration is
already done and committed.

---

## ⮞ Copy-paste prompt for Claude Code on the desktop

> All code is committed (HEAD `b8c2bd1`). First `git pull`. Then regenerate the
> **dynamic-environment** 3D paper tables and figures and commit them.
>
> The environment now has goal-directed crossing-pedestrian obstacles
> (`goal_directed_frac=0.5` in `make_figs_3d.py::ENV_KWARGS`). `make_3d_results.py`
> was refactored to a single 5-method pass (ACP/CC/ECP/FCP-hard/FCP-soft over the
> same 17 seeds) — there is no longer any scan/soft3d/build_final chain.
>
> Run (conda env `cp`):
> ```bash
> conda run -n cp python make_3d_results.py \
>   --seeds 20 21 22 23 24 30 31 32 33 34 35 36 37 38 39 40 41 --traj-seeds 20 22 30
> conda run -n cp python run_sparse_3d.py
> ```
> Then:
> 1. Check `T_RO2026/table_3d_results.tex` is 5 rows; under crossing traffic the
>    baselines (ACP/ECP) should crash/timeout more while FCP (hard/soft) still
>    reaches the goal. If FCP also fails badly, lower `goal_directed_frac` (e.g.
>    0.3) or `goal_speed_range` in `make_figs_3d.py::ENV_KWARGS` and re-run.
> 2. Look at `T_RO2026/traj_3d_seeds.png`. If a panel is ~empty (a baseline died at
>    step ~1), re-pick `--traj-seeds <a> <b> <c>` — seeds where FCP reaches AND the
>    baselines visibly travel before crashing — and re-run `make_3d_results.py`.
> 3. Commit `T_RO2026/table_3d_results.tex`, `table_3d_sparse.tex`,
>    `traj_3d_seeds.png`, `control_time_3d.png`
>    (`metric_3d/*` is gitignored). Do not commit `T_RO2026/main.*` build artifacts,
>    `main.tex`, `reference.bib`, `make_figs_2d.py` (pre-existing edits).
>
> Sanity-check first with `conda run -n cp python make_3d_results.py --smoke`.

---

## Goal
Produce the **dynamic-environment** 3D tables & figures: obstacles now include
goal-directed *crossing pedestrians* (traverse the workspace) instead of only a
random-walk-with-wall-bounce gas. This dynamic setting is the paper's main topic.

## Commands to run on desktop (env `cp`)

### 3D — dense table + trajectory figure + scalability (one-shot, 5 methods, 17 seeds)
```bash
conda run -n cp python make_3d_results.py \
  --seeds 20 21 22 23 24 30 31 32 33 34 35 36 37 38 39 40 41 \
  --traj-seeds 20 22 30
```
Outputs: `T_RO2026/table_3d_results.tex` (5 rows: ACP/CC/ECP/FCP-hard/FCP-soft),
`T_RO2026/traj_3d_seeds.png`, `T_RO2026/control_time_3d.png`, `metric_3d/*`.

### 3D — sparse table (N_obs=50)
```bash
conda run -n cp python run_sparse_3d.py
```
Outputs: `T_RO2026/table_3d_sparse.tex`.

### Sanity check first (fast)
```bash
conda run -n cp python make_3d_results.py --smoke
```

### 2D — already done & committed; only if you want to re-run
```bash
for ds in zara1 zara2 eth hotel univ; do
  for c in cc acp-mpc ecp-mpc fcp-hard-adaptive fcp-soft-adaptive \
           fcp-hard-nonadaptive fcp-soft-nonadaptive; do
    conda run -n cp python runner_2d.py --dataset $ds --controller $c
  done
done
conda run -n cp python make_table_2d.py
conda run -n cp python make_figs_2d.py
```

## What this session changed

### Already committed (3246739, 6d56b3e)
- **2D**: all CP baselines (`cc`, `acp_mpc`, `ecp_mpc`) switched from the fixed
  3x3 (v,w) grid to **MPPI-style sampling** via the shared
  `controllers/utils.py::sample_random_paths`. Conditions unified: `n_paths=1200`,
  `n_skip=2`, horizon=12 (= prediction horizon) for all; FCP keeps a 17-step
  planning horizon (extra steps are masked goal-seeking lookahead). 2D tables/figs
  regenerated.
- **3D ECP** (`ecp_mpc_3d.py`): grid → MPPI sampling
  (`controllers/utils.py::sample_random_paths_3d`), `n_paths=2000`.

### In the commit that ships with THIS file (code only)
- **`quad_env.py`** — goal-directed *crossing pedestrian* obstacle mode:
  `ObstacleAgent.goal/speed`, `_new_crossing_goal()`, a goal-directed branch in
  `_step_obstacles_logic()`, and env params
  `goal_directed_frac` / `goal_speed_range` / `goal_reach_thresh` / `goal_steer_gain`.
  Default `goal_directed_frac=0.0` preserves the old behavior.
- **`make_figs_3d.py`** `ENV_KWARGS`: `goal_directed_frac=0.5` (paper's dynamic
  setting — half crossing pedestrians, half random-walk).
- **`make_3d_results.py`** — refactored to a **5-method one-shot**
  (ACP / CC / ECP / FCP-hard / FCP-soft over the same 17 seeds) → the 5-row paper
  table + figure + scalability in a single run. This **removes the brittle
  `scan_traj_seeds` → `run_soft3d_full` → `build_final_3d` chain** that caused the
  seed-mismatch/duplication bug (baselines were aggregated over a different seed
  set than FCP). The trajectory figure maps FCP-soft to the "FCP-MPC (ours)"
  headline; FCP-hard is table-only.
- **`sim_cp_3d.py`** — CC-3D now accepts `n_paths`/`seed` and runs at **2000**
  (was the controller default 512), so CC matches ACP/ECP/FCP.

## Key settings / decisions
- **3D action search uniform across all methods**: `n_paths=2000`, `n_skip=4`,
  horizon=12 (from `make_figs_3d.py::EXP_BASE` + the per-method run wrappers).
- **FCP-3D is non-adaptive** (offline-calibrated field only). Online adaptation is
  optional and the 2D ablation already covers that story; skipped in 3D for speed.
- **Dynamic env**: `goal_directed_frac=0.5`. If metrics look too easy/hard, tune
  `goal_directed_frac` and `goal_speed_range` in `make_figs_3d.py::ENV_KWARGS`.

## After running on desktop — verify, then commit
1. `table_3d_results.tex` is 5 rows; baselines (ACP/ECP) should crash/timeout more
   under crossing traffic while FCP (hard/soft) still reaches the goal.
2. `traj_3d_seeds.png` panels = seeds [20,22,30]; if a panel looks empty (baseline
   died at step ~1), re-pick with `--traj-seeds <a> <b> <c>` choosing seeds where
   FCP reaches AND baselines visibly travel before crashing.
3. Commit `T_RO2026/table_3d_results.tex`, `table_3d_sparse.tex`,
   `traj_3d_seeds.png`, `control_time_3d.png` (metric_3d/* is gitignored).

## Notes
- ECP-3D control time is high (~1 s/step at n_obs=280) — inherent to its per-path
  online calibration, not a misconfiguration (sparse N_obs=50 → ~18 ms).
- Experiment data (`metric*/`, `*.pkl/json/csv/npy`) is gitignored; only code and
  `T_RO2026/` assets (`*.tex`, `*.png`) are tracked.

---

# Spatial-uncertainty "why a functional spatial bound" figure — status + rounD/SDD handoff

## Verdict on ETH-UCY (measured, not assumed)
Goal: show prediction uncertainty is spatially structured (motivates the functional,
space-dependent bound). Ran `analyze_spatial_uncertainty.py` (Trajectron++ ADE per cell
vs turning vs density) and the confound check `diagnose_spatial_uncertainty.py`
(full-length futures only + per-cell count ≥ 20 + interior trim):

| scene | corr(err,turn) RAW → controlled | corr(err,density) controlled | err hi-dens vs lo-dens |
|-------|--------------------------------|------------------------------|------------------------|
| zara1 | +0.57 → **+0.32** | −0.38 | 0.41 vs 0.47 |
| zara2 | +0.44 → **+0.15** | −0.64 | 0.37 vs 0.54 |
| univ  | +0.66 → **+0.35** | −0.53 | 0.58 vs 0.71 |

- **Weak claim holds**: the error field is spatially non-uniform AND uncertainty is
  *not* visitation-density (it is anti-correlated; high-traffic corridors are the
  lowest-error cells). This is enough for FCP's load-bearing premise.
- **Strong claim (curve/decision-point → uncertainty) is NOT supported on ETH-UCY**:
  the turning correlation roughly halves under controls (zara2 collapses to +0.15), the
  raw hot cells sit on frame edges / next to parked cars & buildings (boundary +
  low-sample + occlusion artifacts), and zara has no curve. **Pulled from `main.tex`**
  (the ETH-UCY overlay is NOT a paper figure).
- Per-cell stats + raw trajectories exported for re-analysis:
  `spatial_cells_{zara1,zara2,univ}.npz` (err/turn/count grids + raw futures).

## TODO on desktop — show the strong claim on rounD / SDD (geometry-rich, fixed)
Download there, then run the (already-written) `analyze_spatial_uncertainty_ext.py`.

**SDD (free)** — pixel coords + reference.jpg; `deathCircle` IS a roundabout:
- Get annotations + reference images: https://cvgl.stanford.edu/projects/uav_data/
  Layout expected: `<data_dir>/annotations/<scene>/<video>/annotations.txt` (+ `reference.jpg`).
- Run: `conda run -n cp python analyze_spatial_uncertainty_ext.py --dataset sdd \
    --data-dir <SDD> --scene deathCircle --video video0`
  (also try `--scene hyang`/`gates` for intersections.)

**rounD (license-gated — download with your levelXdata account):**
- https://levelxdata.com/round-dataset/  → `<data_dir>/<rec>_tracks.csv`,
  `<rec>_recordingMeta.csv`, `<rec>_background.png`.
- Run: `conda run -n cp python analyze_spatial_uncertainty_ext.py --dataset round \
    --data-dir <rounD/data> --recording 00`
- Vehicles curving on the roundabout give a strong, geometry-fixed turning signal with
  full top-down coverage (little frame-edge truncation) — exactly what ETH-UCY lacked.

## Make the rounD/SDD figure reviewer-proof (apply BEFORE trusting it)
`analyze_spatial_uncertainty_ext.py` currently does CV-residual ADE + grid + overlay.
Harden it (these are methodological, not cosmetic):
1. **Count-mask + interior/FOV trim** (drop low-sample + boundary cells); plot the
   count map alongside.
2. **Full-length futures only** (no truncated ADE).
3. **Density control / per-capita** — show uncertainty is not just 1/visitation.
4. **Use the score field, not ADE**: signed mean `S = D_pred − D_true` (systematic bias,
   e.g. corner-cutting) and per-location across-episode variance (the real envelope
   width). Cleanest evidence = **FPCA eigenfunction φ₁(x) + eigenvalue decay** of the
   residual field (this is the object actually calibrated; reuse
   `sims/sim_func_cp.py::build_training_residuals_from_file` →
   `get_envelopes_value_and_function`).

## 3D framing (decided)
- Air-lanes / drone crowd-control corridors are fixed → persistent spatial uncertainty →
  good fit. To DEMONSTRATE a 3D spatial map, make obstacles follow fixed corridors
  (goal-directed crossers on fixed axes); otherwise keep 3D as the dynamic-control
  efficacy result.

## Scene videos for the ETH-UCY overlay (if needed)
- `.avi` are gitignored (≈222 MB). Re-fetch on desktop: `bash assets/download_videos.sh`
  (uses gdown). Homographies (`assets/homographies/*.txt`, image→world) ARE tracked.
- Regenerate exploratory overlay: `conda run -n cp python overlay_spatial_uncertainty.py`.
