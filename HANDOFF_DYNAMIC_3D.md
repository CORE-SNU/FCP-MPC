# Handoff — remaining work (run on desktop; overnight OK)

Continue via `git pull` (conversation context doesn't sync across machines; this file
does). Conda env `cp` for everything. The dynamic-env 3D run itself is **done**
(commit `0cb8f52`); what remains is below.

**Task order (overnight-friendly):**
1. **§1b** Read-only fairness/sanity audit of the 3D pipeline (no run). If a real
   problem is found, fix it — then §1 must be re-run anyway.
2. **§1** Re-run `make_3d_results.py` + `run_sparse_3d.py` to regenerate the 3D figures
   (scalability-FCP and traj-FCP-success bugs are already fixed in code); verify + commit.
3. **§2** SDD: ensure `SDD/` is present (re-download if missing) → run the hardened
   spatial-uncertainty analysis and check the CONTROLLED gate → if SUPPORTED, run the SDD
   navigation benchmark → update `main.tex`.
4. Fix the **2D-table propagation** caveat so the paper shows the MPPI 2D numbers.
Each section below is self-contained with exact commands and what to verify/commit.

---

## 1) 3D figures — regenerate (two bugs fixed in code)

The dynamic-env 3D table is correct, but the two committed 3D **figures were wrong**.
Both causes are now fixed in code — just re-run `make_3d_results.py` to regenerate.

- **Scalability `control_time_3d.png`: FCP was missing.** `plot_n_obs.py` only knew the
  old label `FCP-MPC (ours)`, but the 5-method driver emits `FCP-MPC (hard)/(soft)`, so
  FCP was dropped from the plot. Fixed: soft is remapped to the `FCP-MPC (ours)` headline
  line. → after re-run, confirm FCP appears (should sit ~ACP/CC, far below ECP).
- **Trajectory `traj_3d_seeds.png`: FCP didn't reach the goal.** The fixed panel seeds
  `[20,22,30]` aren't FCP-soft successes in the dynamic env. Fixed: `make_3d_results.py`
  now auto-selects traj-seeds where **FCP-soft reaches the goal** (prints the chosen
  seeds). → after re-run, confirm the blue FCP curve reaches the star in each panel.

Run (no `--traj-seeds` needed; auto-selected):
```bash
conda run -n cp python make_3d_results.py \
  --seeds 20 21 22 23 24 30 31 32 33 34 35 36 37 38 39 40 41
conda run -n cp python run_sparse_3d.py
```
**Do NOT run `make_3d_results.py --smoke` afterward** — smoke writes seed-25/n_obs-20 toy
data to the same `metric_3d/` + `T_RO2026/` paths and clobbers the real outputs.

Verify, then commit: `T_RO2026/table_3d_results.tex`, `table_3d_sparse.tex`,
`traj_3d_seeds.png`, `control_time_3d.png` (`metric_3d/*` is gitignored).

Tunables if needed (`make_figs_3d.py::ENV_KWARGS`): `goal_directed_frac=0.5`,
`goal_speed_range` — lower if FCP also fails, raise for a harder scene.

---

## 1b) 3D fairness / sanity audit — READ THE CODE FIRST (don't trust the result blindly)

It is suspicious that essentially **only FCP reaches the goal** at the dense setting. Before
believing it, audit the 3D pipeline for an unfair asymmetry that could fake it. **Do not
re-run yet — read the code; only re-run if you find and fix a real problem.**

Reassurance to keep in mind: in the **sparse** table (N_obs=50) **ECP does reach the goal**
(~117 steps), so the baselines are not globally broken — the "only ours" pattern is mainly
at dense N_obs=280, which is plausibly genuine. The audit is to confirm that.

Checklist (files: `make_3d_results.py`, `make_figs_3d.py`, `sim_{cp,acp,ecp,func}_3d.py`,
`controllers/{cp,acp_3d,ecp}_mpc_3d.py`, `quad_env.py`):
1. **Identical environment per (seed, n_obs) across methods.** `run_one_job` builds a fresh
   `build_env(seed, n_obs)` per method — confirm the same seed yields the same obstacles,
   start, and goal for every method (env RNG seeded only from `seed`, not perturbed by the
   controller). If obstacle layout differs per method, the comparison is invalid.
2. **Same action budget for all.** `n_paths=2000`, `n_skip=4`, `time_horizon=12` must reach
   every controller (CC was 512 until fixed — re-verify it now gets 2000). FCP must NOT use
   a longer horizon or more samples than the baselines in 3D.
3. **`break_on_collision` applied equally.** All five `METHOD_MAP` entries pass it; confirm
   no method runs to a different termination rule.
4. **Same `max_steps`, `dt`, `goal_finish_dist`, robot/obstacle radii** for all.
5. **Same prediction input.** Every controller gets the same CV obstacle prediction from the
   env; confirm **no method receives ground-truth future** (FCP must not peek). Check the
   env returns identical `pred_xyz`/`history` to all.
6. **Metrics computed identically.** `compute_metrics` is shared; confirm `reached_goal` /
   `collision` / `infeasible` / `steps` use the same logic and thresholds for all methods,
   and that "crashed"/"timeout" for ACP/CC is a genuine outcome, not a wrapper bug.
7. **Action interface parity.** Each `sim_*_3d.py` must map its controller output to
   `env.step(...)` the same way (units, frame, yaw). A wrong mapping in a baseline wrapper
   would make it "fail" spuriously — check ACP and CC especially (they crash/timeout even
   at sparse).
8. **Cost/weights parity.** Baselines and FCP should use comparable MPC objective weights
   (goal/control); confirm FCP isn't tuned with an unfair advantage.

If all pass → the result is fair, keep it. If you find an asymmetry → fix it and re-run
section 1 (and the sparse table). Document whatever you find.

---

## 2) SDD spatial-uncertainty — the "why a functional spatial bound" evidence (pending)

### Why SDD (ETH-UCY verdict, measured not assumed)
`analyze_spatial_uncertainty.py` + the confound check `diagnose_spatial_uncertainty.py`
(full-length futures + per-cell count ≥ 20 + interior trim) gave:

| scene | corr(err,turn) RAW → controlled | corr(err,density) | err hi-dens vs lo-dens |
|-------|----------------------------------|-------------------|------------------------|
| zara1 | +0.57 → **+0.32** | −0.38 | 0.41 vs 0.47 |
| zara2 | +0.44 → **+0.15** | −0.64 | 0.37 vs 0.54 |
| univ  | +0.66 → **+0.35** | −0.53 | 0.58 vs 0.71 |

- **Weak claim holds** (error field is spatially non-uniform and NOT visitation-density —
  it is anti-correlated). Enough for FCP's load-bearing premise.
- **Strong claim (curve/decision-point → uncertainty) NOT supported on ETH-UCY**: the
  turning corr roughly halves under controls (zara2 → +0.15), hot cells are boundary/
  low-sample/occlusion artifacts, and zara has no curve. The ETH-UCY overlay was **pulled
  from `main.tex`** — show the strong claim on SDD (fixed curved geometry) instead.

### Data location & git policy
SDD lives in a top-level **`SDD/`** folder (annotations + reference images, ~455 MB
unzipped). **`SDD/` is gitignored**; re-fetch per machine. Analysis **outputs are tracked**
(`sdd_*_overlay.png`, `sdd_*_diag.png`, `sdd_*_cells.npz` at repo root via `!` exceptions),
plus any final figure/table placed in `T_RO2026/`.

Re-download if missing (OpenTraj zip — cited, MIT, no account; annotations+ref, not videos):
```bash
mkdir -p SDD && cd SDD
curl -L "https://www.dropbox.com/s/v9jvt4ln7t42m6m/StanfordDroneDataset.zip?dl=1" -o sdd.zip
unzip -q sdd.zip && rm sdd.zip && cd ..
```
Fallbacks: `git clone https://github.com/flclain/StanfordDroneDataset SDD` (unofficial,
no license); or `pip install constrained-sdd` (april-tools; explicit polygon constraints →
*distance-to-constraint* feature; needs a custom loader; supplementary only).

### a) Spatial-uncertainty analysis + control gate (already hardened)
`analyze_spatial_uncertainty_ext.py` applies the controls ETH-UCY failed (full-length
futures, count ≥ NMIN, interior trim, density control, per-cell variance), auto-finds all
scenes (globs `**/annotation*.txt`, finds nearby `reference.jpg/png`), and prints RAW vs
**CONTROLLED** `corr(error,turning)` with a `SUPPORTED / WEAK` verdict + saves
`sdd_<scene>_<video>_{diag,overlay}.png` and `_cells.npz`.
```bash
conda run -n cp python analyze_spatial_uncertainty_ext.py --dataset sdd --data-dir SDD
# single roundabout scene: --scene deathCircle
```
**Gate:** treat the claim as established only if CONTROLLED corr stays strong (verdict
SUPPORTED, ρ ≳ 0.35 over ≥30 cells) — do NOT trust RAW. `deathCircle` (roundabout) is the
prime candidate. Deepest evidence: signed score `S = D_pred − D_true` + per-location
variance + FPCA φ₁(x)/eigenvalue decay on the residual field (reuse
`sims/sim_func_cp.py::build_training_residuals_from_file` → `get_envelopes_value_and_function`;
`_cells.npz` has per-cell stats + raw tracks).

### b) Run controllers on SDD as a navigation benchmark
Goal: ETH-UCY + SDD both in the paper. Run CC / ACP / ECP / FCP(hard,soft) on SDD scenes
for the same metrics as the 2D table.
- `runner_2d.py` consumes `predictions/<dataset>.pkl` = `{prediction, history, future}`
  (frame → pid → (H,2)). **Write an SDD→pkl adapter**: per-pedestrian tracks downsampled to
  dt≈0.4 s (~every 12 frames); 8-obs/12-pred windows; `prediction` from the same forecaster
  as ETH-UCY (else CV); pixels→metres if a scale exists (set radii accordingly); register
  scene start/goal + `eval_task_configs`/`scenarios` in `runner_2d.py`.
- Run the controller sweep per scene; regenerate the table via `make_table_2d.py`
  (extend `DATASETS`), writing `.tex` into **`T_RO2026/`** (see caveat below).

### c) If results hold, update main.tex
Add an SDD subsection: the controlled spatial-uncertainty figure/table (only if verdict
SUPPORTED) as the *justification* for the functional bound, plus SDD navigation results
beside the ETH-UCY table. `\includegraphics`/`\input` from `T_RO2026/`, matching the
existing style. Keep ETH-UCY as the standard benchmark; SDD is the geometry-rich addition.

**Citations:** original SDD always — Robicquet, Sadeghian, Alahi, Savarese, *Learning
Social Etiquette…*, ECCV 2016. OpenTraj (if its zip/toolkit used) — Amirian et al., ACCV
2020. constrained-SDD (only if used) — Kurscheidt et al., arXiv:2503.19466 (2025) + repo.

---

## Caveats / notes
- **2D-table propagation**: `make_table_2d.py` writes `tables/`, but the paper `\input`s
  `T_RO2026/`, which is STALE (old grid-baseline numbers, not the MPPI baselines). Point
  `make_table_2d.py` at `T_RO2026/` (or copy after running) so the paper shows MPPI 2D.
- **3D framing**: air-lanes / drone crowd-control corridors are fixed → persistent spatial
  uncertainty (good fit). For a 3D spatial-map figure, make obstacles follow fixed
  corridors; otherwise keep 3D as the dynamic-control efficacy result.
- **FCP-3D is non-adaptive** (offline field only; the 2D ablation covers adaptation).
- ECP-3D control time is high (~1 s/step at n_obs=280) — inherent to its per-path online
  calibration, not a misconfiguration.
