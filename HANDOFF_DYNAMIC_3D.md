# Handoff — re-run 3D on desktop (overnight OK)

Continue via `git pull` (env `cp`). The 2D experiments are re-run on the laptop;
**this handoff is only the 3D re-run** with the corrected FCP envelope.

## What changed (committed)
The FCP envelope was rewritten and is now shared by 2D and 3D:

- **LRW support-function envelope**
  `U_i(x) = ε_i + max_k { μ_k^T φ_i(x) + r_k (φ_i(x)^T Σ_k φ_i(x))^{1/2} }`
  — replaces the old per-coordinate box `mean + φ^T ξ̂ + ε`, which ignored the FPCA basis sign
  and was **not a valid upper bound**. (`cp/functional_cp.py`: `support_envelope_flat`; new
  `CPStepParameters` fields `{means,sigmas,radii,weights,lam}`; `split_alpha=False` so each of
  the two conformal steps uses level α; lower-tail λ quantile.)
- **Horizon-dependent clearance relaxation** re-added to the controllers (`func_cp_mpc.py`,
  `func_3d_mpc.py`; hard filter + soft penalty): the required clearance is relaxed by
  `Δ_t = ½·a_lat·(t·Δt)²` (a_lat: 2D = v_max·ω_max [unicycle centripetal]; **3D = g·tan(tilt_max)
  ≈ 5.66 m/s²** [drone translates laterally without yaw]). `Δ_0 = 0`, so the applied/1-step keeps
  full clearance (relaxation acts only for t≥1) and the **i=1 closed-loop guarantee is
  unaffected**. Rationale: rejecting a path because a probability-dependent bound flags it
  unsafe many steps ahead — a step that is re-planned and still evadable — is over-conservative.
  (Avoid the term "ICS", which usually denotes a tightening/avoid notion.)
- **Online AFCP** is now a scalar radius-multiplier `c` via ACI (2D only; 3D is offline, so its
  online adapter is unused).
- **Projection-residual ε kept** (`proj_residual=True`): ε = conformal sup-norm quantile of the
  truncation, so the envelope covers the FULL field (not just the projection). It is
  data-determined (~1.0) and not shrunk; any over-conservatism is handled by the clearance
  relaxation, not by lowering ε.
- **p_base = 5** everywhere (diagnostic: higher p ⇒ *more* conservative support function; ε is
  small and ≈p-independent; 5 is a good middle, consistent with the L3 "5–7 PCs" story).

## Action (desktop) — don't re-run the full 5×17 suite

Only two things changed: the **FCP envelope** (→ FCP only) and the **ECP per-horizon ACI**
(→ ECP only). CC and ACP are untouched, so reuse their existing cache. Use the decoupled
`run_subset_3d.py` (groups: `baseline`={ACP,CC,ECP}, `fcp`={FCP hard,soft}; 17 seeds each) +
`assemble_3d.py`.

1. `git pull`; env `cp`.
2. **Invalidate stale envelope caches first** (old caches hold the box envelope / old p):
   `rm -f sims/cp_cache/*.pkl` and any 3D CP cache `*.pkl`.
3. **Required — FCP only** (the actual contribution; envelope changed):
   `python run_subset_3d.py --which fcp`  → `fcp_3d_cache.pkl`.
   Then `python assemble_3d.py` to combine with the existing `baseline_3d_cache.pkl`
   (CC/ACP/ECP from before) → `T_RO2026/table_3d_results.tex` (mean±std), `metric_3d/results_3d.{csv,json}`.
   This alone is ~2/5 of the full suite.
4. **ECP fairness (optional for the 3D headline).** The ECP ACI was just fixed (per-path→global
   per-horizon, same bug/fix as 2D `ecp_mpc.py`). ECP is bundled in the `baseline` group, so to
   refresh it run `python run_subset_3d.py --which baseline` (re-runs ACP/CC/ECP; ACP/CC come out
   identical). **But:** in 3D, ECP already fails on **compute** (~306 ms ≫ the 100 ms Δt budget),
   and the ACI fix changes plan quality, **not** compute cost — so ECP's 3D headline (too slow /
   infeasible-by-budget) is unchanged. Re-run `baseline` only if you want the *fair*
   collision/infeasible numbers reported; the conclusion holds either way. (The ACI fix mainly
   matters for the **2D** comparison, where ECP runs within budget — already re-run on the laptop.)
5. Numbers will shift (valid + larger envelope, p→5, clearance relaxation). Sanity-check: FCP should
   reach the goal; the clearance relaxation should keep hard feasible (in 2D it dropped hard
   infeasible from ~0.85 to ~0.2–0.6).
6. Regenerate envelope-dependent 3D figures (`Func_cp_3d_zoom`, `traj_3d_seeds`,
   `control_time_3d`) and set the two 3D table captions to "mean $\pm$ std over 17 seeds".

## Notes
- **Do NOT touch the commented-out Math-Setup blocks in `main.tex`** (owner is reconciling the
  theory section separately).
- Coverage/reliability tables are **dropped** from the paper: with the valid (conservative)
  envelope, empirical coverage is ~100% everywhere → uninformative; safety is evidenced by the
  closed-loop collision rates instead.
- 2D (laptop): 2D experiment tables are regenerated with the same
  envelope; the paper's 2D numbers come from there.
