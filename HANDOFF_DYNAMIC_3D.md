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
   navigation benchmark → update `main.tex`. write the code and execute it and save the figure and/or tables to be used in the paper.
4. Fix the **2D-table propagation** caveat so the paper shows the MPPI 2D numbers.
5. **§5 (NEW)** Regenerate the 3D table with **mean±std over the 17 seeds** — see §5.
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

### The argument this experiment must support (state this in the paper, then back EACH link with data)
This is the justification for adopting the field-wise (functional) calibration formulation.
It is a **causal chain, not one claim** — extract evidence for every link:

> **(L1) uncertainty depends on FIXED spatial geometry**  ⟹  **(L2) the residual field is
> therefore (approximately) time-invariant / stationary**  ⟹  **(L3) so the ensemble of
> fields is low-rank / compressible**  ⟹  **(L4) so it can be calibrated OFFLINE once and
> evaluated cheaply ONLINE** (the method's compute advantage).

Key qualifier (must be shown, or the chain breaks): the dependence must be on **static
scene geometry** (curve/entrance that never moves), NOT on transient state (instantaneous
crowd config). That is exactly why SDD (fixed roads/buildings) is the right testbed, and
why L1's hotspots must sit on fixed features.

**What to extract per link (figures + numbers):**
- **L1 — spatial dependence on fixed geometry.** Controlled `corr(error, turning)` clearing
  the SUPPORTED gate (ρ≳0.35, ≥30 cells) + the overlay heatmap with top-uncertainty cells
  marked, and note they coincide with FIXED scene features (the roundabout ring / lane
  merges in `deathCircle`). Files: `sdd_<scene>_overlay.png`, `_diag.png`, the corr line.
  Already produced by `analyze_spatial_uncertainty_ext.py`.
- **L2 — time-invariance / stationarity (NEW; the linchpin).** Split the episodes (or the
  time axis) into two disjoint halves; build the per-cell error field on each half
  independently; show the two fields AGREE — scatter of cell means with a high Pearson r,
  plus the two heatmaps side by side. High agreement ⟹ the field is a property of the fixed
  scene, not of the moment. Doable from the per-window data in `sdd_*_cells.npz`
  (`pos/err/turn_w/full`); add a small `--split-halves` mode or a short script.
- **L3 — low-rank / learnable (NEW; the most direct justification).** FPCA on the residual
  field across episodes → eigenvalue/scree decay (cumulative variance explained by the
  first k components) + the leading eigenfunction φ₁(x) overlaid on the scene. Few
  components capturing most variance ⟹ compressible ⟹ offline-learnable. Reuse
  `sims/sim_func_cp.py::build_training_residuals_from_file` to get per-episode residual
  grids, then PCA/FPCA. Figures: scree plot + φ₁ overlay.
- **L4 — offline→online (already shown, no new run).** Cite the existing control-time
  column + `control_time_3d.png` scalability: FCP's online per-step cost stays low while
  per-path online calibration (ECP) blows up. This is the realized compute advantage that
  L1–L3 justify.

Mapping for the paper: L1 = intuitive/visual mechanism; **L2+L3 = the load-bearing
evidence** that the field is stationary and low-dimensional (this is what
`subsec:fcp-why`'s "S admits low-dimensional structure" actually needs); L4 = payoff.
Lead with L1's picture, clinch with L2/L3, point to L4 for the win.

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

---

## ✅ Completed overnight (2026-06-18) — commits `44c7bcf`, `231db4d`

Run on the desktop in conda env `cp` (built from scratch this session: Python 3.12 +
pinned `requirements.txt`, `gym-pybullet-drones` from the pinned commit, `setuptools` 75.8.0
for `pkg_resources`; **`rerun-sdk` is uninstallable here — glibc 2.27 < 2.31 — so a no-op
`rerun` shim sits in the env's site-packages**; rerun is viewer-logging only and irrelevant
to the tables/figures). Commits are local; **`git push origin main` still needed** (no
GitHub credentials on this machine).

### §1b — 3D fairness audit (read the code; two real asymmetries found + fixed)
The "only FCP reaches at dense" pattern was **partly an artifact**. Fixed and verified:
1. **Env RNG never reset across episodes.** `quad_env.reset()` re-inited obstacles from the
   *current* RNG without reseeding, so FCP's offline calibration (which steps the env and
   advances `self.rng`) made FCP fly a **different** obstacle realization than the baselines
   at the same seed. Fix: `reset()` reseeds `self.rng`/`oracle_rng` to the base seed →
   verified obstacle fields are now **byte-identical** across methods per seed.
2. **`goal_finish_dist` not pinned.** ACP/FCP used 0.8 but CC/ECP fell back to 0.3 (~2.7×
   closer to count as "reached"). Fix: `make_figs_3d.EXP_BASE` pins `goal_finish_dist=0.8`
   for all methods.
3. (Documented, not changed) ECP-3D ignores the static wall boxes CC/FCP enforce — this
   only **disadvantages** ECP, so it doesn't fake FCP's win.

### §1 — re-run under the fair env (the headline changed, honestly)
- **Sparse (N_obs=50):** CC-MPC now **reaches (1.00)** alongside FCP-hard/soft; ACP/ECP fail.
- **Dense (N_obs=280):** CC-MPC also **reaches (186.5 steps)**; ACP/ECP crash; **FCP-MPC
  (soft) is best** (84.8 steps, collision 0.032). Story is now "FCP is best," not "only FCP."
- Regenerated `table_3d_results.tex`, `table_3d_sparse.tex`, `traj_3d_seeds.png` (auto-seeds
  21/30/31), `control_time_3d.png`, and the env-dependent `Func_cp_3d_zoom.png` +
  `traj_3d_overlay.png` (added `goal_directed_frac=0.5` to their local `ENV_KWARGS`).

### §2 — SDD spatial-uncertainty evidence (see `SDD_FINDINGS.md`)
- **L1 (geometry/turning → uncertainty): NOT supported.** Full 60-video sweep: only 5
  low-sample videos clear the controlled gate; **deathCircle (prime roundabout) WEAK on all**.
  → do not put an SDD L1 overlay in the paper.
- **L2 (time-invariance): mostly holds.** `make_fig_stationarity_split.py` → split-halves
  cell agreement hotel +0.91, eth +0.73, univ +0.71 (zara1 +0.45, zara2 +0.04).
- **L3 (low-rank): holds.** `make_fig_fpca_lowrank.py` → 5–7 PCs for 90% variance on 4/5
  ETH-UCY datasets (univ 13), out of 1600 grid dims. **L2+L3 are the load-bearing
  justification** for offline calibration.
- **§2b nav benchmark: NOT run** (deliberate). L1 gate failed and a correct SDD→pkl adapter
  needs the pixel→metre scale, a forecaster choice, and per-scene nav tasks — guessing those
  would put wrong numbers in the paper. Blockers documented in `SDD_FINDINGS.md`.
- **§2c main.tex: deferred to review** (L1 didn't hold; a recommended L2/L3 snippet is in
  `SDD_FINDINGS.md`).

### §4 — 2D-table propagation: fixed
`make_table_2d.py` now writes the 2D tables into **both** `tables/` and `T_RO2026/`, so the
paper's `\input` can't desync. The current `T_RO2026/` tables already carry the MPPI numbers
(propagated earlier), so no number change.

### New files this session
`make_fig_fpca_lowrank.py` (L3), `make_fig_stationarity_split.py` (L2), `SDD_FINDINGS.md`,
`run_dynamic_3d.sh` / `run_extra_3d_figs.sh` (runners, held awake via `systemd-inhibit`),
and the tracked `sdd_*_{diag,overlay,cells}` outputs + `sdd_all_analysis.log`.

### Open items needing your input
- Push the two commits (`git push origin main`).
- SDD nav benchmark: confirm metre-scale source + CV-vs-trained forecaster, then it can run.
- main.tex: approve the L2/L3 figures/snippet from `SDD_FINDINGS.md`.

---

## 5) 3D table with mean±std over 17 seeds  (NEW — needs the desktop's full per-seed data)

**Why this is here:** the laptop only has a degenerate `metric_3d/results_3d.{csv,json}`
(seed 25, all-zero) + a few `table3d.csv` fcp seeds — **not** the real 17-seed × 5-method
run that produced the current `T_RO2026/table_3d_results.tex`. Std must be computed where
that data lives (this desktop).

**Code is already prepared.** `make_3d_results.py::write_latex_table` now emits
`mean$\pm$std` for **collision rate, infeasible rate, and steps-to-goal** (std shown only
when >1 seed; bolding still compares means). Ctrl-time stays a single pooled mean.

**Do:**
1. Re-run the full 17-seed dense run: `python make_3d_results.py` (env `cp`). This
   regenerates `T_RO2026/table_3d_results.tex` **with ±std** from the in-memory per-seed
   metrics, plus `metric_3d/results_3d.{csv,json}`.
2. Apply the **same ±std treatment to the sparse table** (`table_3d_sparse.tex`): whatever
   script writes it (`run_sparse_3d.py` / its table writer) should mirror the
   `_steps_cell` / `_fpm` changes in `make_3d_results.py` (collision, infeasible, steps as
   mean±std). If it shares `write_latex_table`, it's already done — just verify.
3. In `main.tex`, the two 3D table captions (`tab:pybullet_results`,
   `tab:results_sparse`) should say **"mean $\pm$ std over 17 seeds"** (currently
   "averaged over 17 seeds"). One-line caption edit each.
4. Rebuild `main.tex`, eyeball that `lcccc` still fits (it's a `table*`, full width —
   `0.044\pm0.021` etc. fit fine), commit.

**2D std:** intentionally NOT added — only n=3 scenes/dataset (univ n=1), so std is too
noisy to be meaningful; the coverage table already carries ±std (see `make_coverage_table.py`).

---

## 6) FCP envelope rewrite — re-run 3D with the corrected (LRW support-function) envelope

**Why (committed this session):** the offline envelope was a per-coordinate *box*
`mean + φ^T ξ̂ + ε` that ignored the sign of the FPCA basis (NOT a valid upper bound where
`φ_j(x)<0`). It is now the **LRW support function** (Lei–Rinaldo–Wasserman Eq. (8)–(9),
matching the paper Appendix):
```
U_i(x) = ε_i + max_k { μ_k^T φ_i(x) + r_k ( φ_i(x)^T Σ_k φ_i(x) )^{1/2} }.
```
Also: `CPConfig.split_alpha=False` (each conformal step at level α, not α/2 → less
conservative), lower-tail λ quantile (`Quantile_α`, was wrongly `1−α/2`), and **`p_base=5`**
everywhere (diagnostic: higher p ⇒ *more* conservative support function; ε is small and
≈p-independent, so 5 is a good middle and consistent with the L3 "5–7 PCs" story).

**Code changed (in this commit):**
- `cp/functional_cp.py`: `support_envelope_flat`; new `CPStepParameters` fields
  `{means,sigmas,radii,weights,lam}`; `split_alpha=False`; lower-tail λ.
- `controllers/func_cp_mpc.py` (2D): support-function grid + per-point eval; **online AFCP
  reworked to a scalar radius-multiplier `c` via ACI on the functional-violation indicator**
  (was a per-coordinate ξ̂ update).
- `controllers/func_3d_mpc.py` (3D): support-function grid + per-point eval. **3D is
  offline-only**, so its legacy per-coordinate online adapter is unused; if you ever enable
  3D online adaptation, port the scalar-`c` rule from `func_cp_mpc.py` (the current 3D adapter
  would be a no-op, since the grid is built from `{μ,Σ,r}`, not `coeff_upper`).
- `p_base=5`: `sim_func_3d.py` (L428, L760), `sims/sim_func_cp.py`, `multi_pedestrians_cp.py`.

**ACTION (desktop, full 17-seed):**
1. `git pull`; env `cp`.
2. **Invalidate stale envelope caches first** (old caches hold the box envelope):
   `rm -f predictions/*_p*_k*.pkl` and any cached CP envelope `*.pkl`. (2D caches at
   `sims/sim_func_cp.py` ~L310; 3D `sim_func_3d.py` does not appear to cache, but clear to be safe.)
3. Re-run `python make_3d_results.py` (dense, 17 seeds → `table_3d_results.tex` with the
   **new envelope + ±std**) and `run_sparse_3d.py` + the scalability sweep.
4. Numbers WILL shift (envelope is different and now valid; p=6/4→5). Sanity: FCP should still
   reach the goal; the support-function envelope is somewhat more conservative than the old
   (buggy) box, so infeasible may rise — that is expected/honest, and the soft variant + the
   online scalar-`c` adaptation are what claw it back. Report the new numbers.
5. Regenerate envelope-dependent 3D figures (`Func_cp_3d_zoom`, `traj_3d_seeds`,
   `control_time_3d`) and update the two table captions to "mean $\pm$ std over 17 seeds".

**Note:** ICS is removed (safety radius is plain `r_safe`). The commented-out Math-Setup
blocks in `main.tex` are NOT to be touched here (owner is reconciling them separately).
