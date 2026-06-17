# SDD spatial-uncertainty findings (overnight §2 run)

Run on the desktop in conda env `cp`. Scripts + figures are committed; SDD/ is gitignored.
This documents the measured verdicts so the paper claims stay honest.

## TL;DR
- **L1 (uncertainty depends on fixed scene geometry, via corr(error, turning)):** NOT
  supported on SDD. Of 60 videos, only **5 clear the CONTROLLED gate** (ρ≥0.35, ≥30 cells),
  and those 5 are scattered/low-sample (bookstore×1, coupa×1, nexus×3). **deathCircle (the
  prime roundabout candidate) is WEAK on all 5 videos** (controlled ρ ≈ −0.08…−0.17). This
  mirrors the ETH-UCY result where the strong claim halved under controls. → Do **not** put
  an SDD L1 "geometry→uncertainty" overlay in the paper as justification; the gate fails.
- **L2 (the residual field is time-invariant / stationary):** mostly supported. Split-halves
  (early vs late scenes, a genuine temporal split) per-cell agreement:
  hotel **+0.91**, eth **+0.73**, univ **+0.71**, zara1 +0.45, zara2 +0.04.
  3/5 strong, 1 moderate, 1 (zara2) weak.
- **L3 (the residual field is low-rank / compressible):** supported. # principal components
  for 90% of variance (grid = 1600 dims):
  eth **5**, hotel **6**, zara1 **7**, zara2 **7**, univ **13**. First 3 PCs already explain
  56–79%. Strongly low-rank ⇒ learnable offline.
- **L4 (offline→online compute win):** unchanged; shown by the existing control-time column
  + `control_time_3d.png` scalability (FCP stays cheap online; ECP's per-path online
  calibration blows up).

## What this means for the paper
The load-bearing justification for the functional (field-wise) offline calibration is
**L2 + L3** (stationary + low-dimensional residual field), measured on the ETH-UCY benchmark
the paper already uses — NOT the L1 geometry-turning mechanism, which does not survive
controls on either ETH-UCY or SDD. Lead `subsec:fcp-why`'s "S admits low-dimensional
structure" with L3 (scree + φ₁), support stationarity with L2, and point to L4 for the payoff.

Recommended (held for your review — not auto-applied to main.tex):
- Add `fcp_lowrank_fpca.png` (L3) and optionally `fcp_stationarity_split.png` (L2) to
  `subsec:fcp-why`.
- State the L1 limitation honestly (uncertainty is spatially structured but not explained by
  turning/visitation density; it is anti-correlated with density — see ETH-UCY table in the
  prior handoff). Keep ETH-UCY as the benchmark.

## Artifacts produced
- `make_fig_fpca_lowrank.py` → `T_RO2026/fcp_lowrank_fpca.png` (+ `fcp_lowrank_fpca.npz`)
- `make_fig_stationarity_split.py` → `T_RO2026/fcp_stationarity_split.png`
- `analyze_spatial_uncertainty_ext.py` (existing) → `sdd_<scene>_<video>_{diag,overlay}.png`,
  `_cells.npz`, full sweep log `sdd_all_analysis.log`.

## §2b SDD navigation benchmark — NOT run (deliberately)
The md gates this on "if results hold." The L1 motivation does not hold, and a correct
SDD→pkl adapter needs decisions that, if guessed, would put wrong numbers in the paper:
1. **pixel→metre scale** per scene (SDD annotations are in pixels; radii/collision metrics
   are meaningless without the right homography/scale).
2. **forecaster**: ETH-UCY uses a trained predictor; SDD would fall back to constant-velocity
   unless that predictor is wired up for SDD inputs.
3. **navigation tasks**: per-scene start/goal + `eval_task_configs`/`scenarios` must be
   defined; there is no canonical choice.
Rather than fabricate a benchmark from an unvalidated adapter, this is left as a scoped task.
If you want it, confirm (a) the metre scale source and (b) CV vs trained forecaster, and it
can be built and run.
