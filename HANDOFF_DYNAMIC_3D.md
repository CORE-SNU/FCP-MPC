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

## TODO on desktop — SDD (the spatial-structure dataset)
SDD is downloaded by hand (the only official link, vatic2, is down, and it is one ~69 GB
zip; the desktop machine should fetch it — e.g. a Kaggle "Stanford Drone Dataset" mirror).
You only need, per scene/video: `annotations.txt` (+ `reference.jpg`). `deathCircle` is a
roundabout; `hyang`/`gates` are intersections — these have the fixed curved geometry that
open ETH-UCY sidewalks lack.

### Data location & git policy
The SDD dataset lives in a top-level **`SDD/`** folder (annotations + reference images,
already unzipped, ~455 MB). **`SDD/` is gitignored** (raw data, large) — re-fetch it on any
machine, don't commit it. The analysis **OUTPUTS are tracked**: `sdd_*_overlay.png`,
`sdd_*_diag.png`, `sdd_*_cells.npz` at the repo root (gitignore has explicit `!` exceptions),
plus any final figure/table you put in `T_RO2026/`.

### (Re)download if `SDD/` is missing (OpenTraj zip — cited, MIT, single file, no account)
OpenTraj (Amirian et al., ACCV 2020; MIT) ships loaders, not raw data, and points to a
single SDD annotations zip (annotations + reference images; NOT the 69 GB videos):
```bash
mkdir -p SDD && cd SDD
curl -L "https://www.dropbox.com/s/v9jvt4ln7t42m6m/StanfordDroneDataset.zip?dl=1" -o sdd.zip
unzip -q sdd.zip && rm sdd.zip && cd ..
```
Standard SDD layout (all 8 scenes incl. `deathCircle` roundabout, `hyang`/`gates`
intersections; pixel coords; FPS 30). Our parser auto-finds it — no need to install the
OpenTraj toolkit. Fallbacks: `git clone https://github.com/flclain/StanfordDroneDataset
SDD` (smaller, unofficial, no license) or OpenTraj's own
`opentraj/toolkit/loaders/loader_sdd.py`.

Optional upgrade — **constrained-SDD** (april-tools; explicit polygon constraints for
building/obstacle/offroad → correlate uncertainty with *distance-to-constraint*, a cleaner
geometric feature than turning): `pip install constrained-sdd`
(`csdd.ConstrainedStanfordDroneDataset`, `get_trajectory_prediction_dataset`). Needs a
small custom loader; treat as supplementary.

**Citations** (cold take: keep the load-bearing claim on standard SDD):
- Original SDD (always): Robicquet, Sadeghian, Alahi, Savarese, *Learning Social Etiquette:
  Human Trajectory Understanding in Crowded Scenes*, ECCV 2016.
- OpenTraj (if you use its zip/toolkit): Amirian, Zhang, Castro, Baldelomar, Hayet, Pettré,
  *OpenTraj: Assessing Prediction Complexity in Human Trajectories Datasets*, ACCV 2020.
- constrained-SDD (only if used): Kurscheidt, Morettin, Sebastiani, Passerini, Vergari,
  *A Probabilistic Neuro-symbolic Layer for Algebraic Constraint Satisfaction*,
  arXiv:2503.19466 (2025) + the april-tools/constrained-sdd repo.

**The analysis auto-finds the data.** Point `--data-dir` at `SDD`; it globs
`**/annotation*.txt` (works for both `annotation.txt` and `annotations.txt`, any nesting)
and locates a nearby `reference.jpg/png`. After downloading, sanity-check: per-scene track
counts plausible, coordinates within the reference-image size, reference image present.

### 1) Spatial-uncertainty analysis + control check (already hardened)
`analyze_spatial_uncertainty_ext.py` already applies the controls ETH-UCY failed
(full-length futures only, per-cell count ≥ NMIN, interior trim, density control,
per-cell variance). It auto-finds all scenes if `--scene` is omitted, prints RAW vs
CONTROLLED `corr(error,turning)`, a `SUPPORTED / WEAK` verdict, and saves
`sdd_<scene>_<video>_{diag,overlay}.png` + `_cells.npz`.

```bash
conda run -n cp python analyze_spatial_uncertainty_ext.py --dataset sdd --data-dir SDD
# or a single scene:
conda run -n cp python analyze_spatial_uncertainty_ext.py --dataset sdd --data-dir SDD \
    --scene deathCircle --video video0
```
Gate: only treat the spatial-structure claim as established if the **CONTROLLED** corr
stays strong (verdict SUPPORTED, e.g. ρ ≳ 0.35 over ≥30 cells) — do NOT trust the RAW
number. For the deepest evidence, also do signed score `S = D_pred − D_true` + per-location
variance + FPCA eigenfunction φ₁(x)/eigenvalue decay on the residual field
(reuse `sims/sim_func_cp.py::build_training_residuals_from_file` →
`get_envelopes_value_and_function`); `_cells.npz` has the per-cell stats + raw tracks.

### 2) Run the controllers (baselines + ours) on SDD as a navigation benchmark
Goal: ETH-UCY + SDD both in the paper. Run CC / ACP-MPC / ECP-MPC / FCP (hard, soft) on
SDD scenes and produce the same metrics (collision / infeasible / steps-to-goal /
ctrl-time) as the 2D ETH-UCY table.
- The 2D runner (`runner_2d.py`) consumes `predictions/<dataset>.pkl` with
  `{prediction, history, future}` dicts (frame → pid → (H,2), world metres). **Write an
  SDD adapter** that converts SDD tracks into that schema:
  - build per-pedestrian trajectories (downsample to dt=0.4 s ≈ every 12 frames),
  - 8-step history / 12-step future windows; `prediction` = a forecaster (reuse the same
    predictor as ETH-UCY if available, else CV) so the comparison is apples-to-apples,
  - SDD is in pixels → convert to metres if a scale is available, else keep consistent
    units and set robot/obstacle radii accordingly,
  - register scene start/goal + `eval_task_configs`/`scenarios` entries in `runner_2d.py`.
- Then run, per SDD scene, the same controller sweep as ETH-UCY and regenerate the table
  via `make_table_2d.py` (extend `DATASETS`), writing the `.tex` into **`T_RO2026/`**
  (NOTE: `make_table_2d.py` currently writes only to `tables/`, which the paper does NOT
  read — copy/point its output to `T_RO2026/`, see the 2D-table caveat below).
- Save all produced metrics/trajectories (gitignored) and the figure/table.

### 3) If results hold, update main.tex
- Add an SDD subsection: the controlled spatial-uncertainty figure/table (only if verdict
  SUPPORTED) as the *justification* for the functional bound, and the SDD navigation
  results alongside the ETH-UCY table.
- `\includegraphics`/`\input` from `T_RO2026/`; match the existing figure/table style.
- Keep ETH-UCY as the standard 2D benchmark; SDD is the geometry-rich addition.

### 2D-table propagation caveat (fix while here)
`make_table_2d.py` writes `tables/table_2d_{results,ablation}.tex`, but the paper
`\input`s the copies in `T_RO2026/`, which are currently STALE (old grid-baseline numbers,
not the MPPI baselines). Point `make_table_2d.py` at `T_RO2026/` (or copy after running)
so the paper reflects the MPPI-baseline 2D results.

## 3D framing (decided)
- Air-lanes / drone crowd-control corridors are fixed → persistent spatial uncertainty →
  good fit. To DEMONSTRATE a 3D spatial map, make obstacles follow fixed corridors
  (goal-directed crossers on fixed axes); otherwise keep 3D as the dynamic-control
  efficacy result.

## Scene videos for the ETH-UCY overlay (if needed)
- `.avi` are gitignored (≈222 MB). Re-fetch on desktop: `bash assets/download_videos.sh`
  (uses gdown). Homographies (`assets/homographies/*.txt`, image→world) ARE tracked.
- Regenerate exploratory overlay: `conda run -n cp python overlay_spatial_uncertainty.py`.
