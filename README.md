# Planning Meets Functional Calibration: Function Conformal Prediction for Safe Motion Planning in Uncertain Environments

**T-RO 2026 (first draft, November 2025)**

This repository contains the experiment code for the paper. The method constructs a conformal upper envelope $U_i(x)$ over the distance-field prediction residual $S_{t+i|t}(x) = D_{t+i|t}(x) - D_{t+i}(x)$ and uses it to define a certified lower bound on the true distance field at every future step of an MPC rollout.

---

## Overview

### Key idea

Given an $i$-step-ahead distance-field prediction $D_{t+i|t}(x)$, the conformal upper envelope $U_i(x)$ satisfies

$$\mathbb{P}\!\left[D_{t+i}(x) \ge D_{t+i|t}(x) - U_i(x)\;\; \forall x \in \mathcal{X}\right] \ge 1 - \alpha$$

so the certified lower bound

$$\underline{D}_{t+i|t}(x) = \max\!\left\{D_{t+i|t}(x) - U_i(x),\; 0\right\}$$

can be used directly inside a motion planner.

### Pipeline

```
Offline calibration
  ├─ 1. Compute residuals  S_i(x) = D_{t+i|t}(x) - D_{t+i}(x)
  ├─ 2. FPCA decomposition → basis φ_i(x)  (cached as CPStepParameters)
  ├─ 3. Project residuals → coefficients ξ_{i,j}; fit GMM on coefficients
  └─ 4. Compute upper-quantile coefficients via CP calibration → U_i(x)

Online calibration  (ACP variant only)
  └─ Adaptive update of coefficient quantiles:
       ξ̂_j^{(t+1)} = ξ̂_j^{(t)} + γ · (𝟙[ξ_j^online > ξ̂_j^{(t)}] − α_target)

Control (MPC, 2-D pedestrian avoidance)
  ├─ 1. Sample N candidate control sequences; roll out unicycle dynamics
  ├─ 2. Filter / penalize paths based on certified lower bound:
  │       hard mode: reject paths where  D̲_{t+i|t}(p) < r_safe + ε
  │       soft mode: penalize  max(0, r_safe − D̲_{t+i|t}(p))² in MPC cost
  ├─ 3. ICS relaxation for large i:
  │       Δ(i) = ½ a_lat,max (i Δt)²
  │       effective r_safe(i) = max(0, r_safe − Δ(i))
  └─ 4. Score feasible paths; apply first control
```

---

## Repository structure

The experiments are split into two parallel families — `planar/` (2-D pedestrian
avoidance on ETH-UCY) and `quadrotor/` (3-D drone navigation) — over a shared core
(`utils.py`, `cp/`, `controllers/`, `viz_traj.py`) that lives at the repo root so both
families import the *same* conformal machinery and only the environment/planner differ.

```
# --- shared core (used by both planar/ and quadrotor/) ---
utils.py               # grid construction, distance fields, plotting helpers
viz_traj.py            # trajectory-image helpers (save_traj_image_2d / _3d)
cp/
  functional_cp.py     # FPCA + GMM offline calibration; CPStepParameters dataclass
controllers/
  func_cp_mpc.py       # FunctionalCPMPC — 2-D controller (all 4 variants)
  acp_mpc.py / ecp_mpc.py / cc.py                 # 2-D baselines
  func_3d_mpc.py / acp_3d_mpc.py / ecp_mpc_3d.py / cp_3d_mpc.py   # 3-D controllers
  utils.py             # path-sampling helpers shared by the 2-D and 3-D controllers

# --- 2-D pedestrian avoidance on ETH-UCY ---
planar/
  runner_2d.py         # Main entry point for 2-D experiments
  run.sh               # Full 2-D ablation sweep
  make_table_2d.py     # Aggregate metric/ -> latex / csv tables
  make_figs_2d.py      # Trajectory-overlay figure
  preprocess.py        # ETH-UCY raw-data loader
  prediction/cv.py     # Constant-velocity pedestrian prediction
  sims/                # Simulation loops (sim_func_cp / sim_acp_mpc / sim_ecp_mpc / sim_cc) + helpers
  predictions/         # Pre-computed prediction PKL files (one per dataset)
  assets/homographies/ # ETH-UCY scene homographies
  metric/  traj/       # Output JSON / NPY files (auto-created)

# --- 3-D quadrotor navigation ---
quadrotor/
  runner_3d.py         # Main entry point for 3-D experiments
  quad_env.py          # PyBullet quadrotor environment (+ repo-root sys.path bootstrap)
  sim_{func,cp,ecp,acp}_3d.py, make_*_3d.py, run_*_3d.sh, ...
```

Files in `planar/` and `quadrotor/` reach the shared root modules via a small
`sys.path` bootstrap at the top of each entry point (it inserts the repo root onto
`sys.path` so `import utils` / `from controllers... import ...` resolve from anywhere).

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.8–3.11.  Key dependencies:

| Package | Version |
|---|---|
| numpy | 2.0.2 |
| scipy | 1.13.1 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.9.4 |

---

## 2-D pedestrian-avoidance experiments

### Datasets

The 2-D experiments use the ETH/UCY pedestrian datasets (zara1, zara2, eth, univ). Pre-computed constant-velocity predictions are provided in `predictions/`.

### Controllers

| Key | Safety constraint | Online adaptation |
|---|---|---|
| `fcp-hard-adaptive`    | Hard filter (D̲ ≥ r_safe) | ACP update of coeff quantiles |
| `fcp-hard-nonadaptive` | Hard filter               | Fixed offline coefficients    |
| `fcp-soft-adaptive`    | Soft penalty in MPC cost  | ACP update of coeff quantiles |
| `fcp-soft-nonadaptive` | Soft penalty in MPC cost  | Fixed offline coefficients    |
| `acp-mpc`              | Scalar ACP (egocentric)   | ACP                           |
| `ecp-mpc`              | Ellipsoidal CP            | —                             |
| `cc`                   | None (baseline)           | —                             |

### Run a single experiment

```bash
python planar/runner_2d.py --dataset zara1 --controller fcp-hard-adaptive
```

Results are written to `planar/metric/<dataset>_<controller>.json` and `planar/traj/<dataset>_<controller>.npy`.

### Run the full ablation sweep

```bash
bash planar/run.sh
```

This evaluates all 7 controllers on all 4 datasets sequentially.

### Generate the results table

```bash
python planar/make_table_2d.py
```

---

## 3-D quadrotor experiments

```bash
bash run_3d.sh        # main 3-D sweep
bash run_3d_obs.sh    # varying number of obstacles
```

---

## Method details

### Offline calibration (`cp/functional_cp.py`)

**Class `PCAGMMResidualCP`**

1. Given a training set of residual fields $\{S_i^{(n)}\}_{n=1}^N$ for each horizon index $i$, fit a PCA basis $\phi_i$ with $p$ components (`p_base`).
2. Project each residual onto the basis to get coefficient vectors $\xi^{(n)} \in \mathbb{R}^p$.
3. Fit a $K$-component GMM on the training coefficients.
4. Find the $(1-\alpha)$-quantile of $\max_k \log(\pi_k \phi_k(\xi))$ on a held-out calibration split, then derive per-component ellipsoidal radii $r_k$.
5. The upper-quantile coefficient vector `coeff_upper` combines the GMM means and radii:
   $\hat{\xi}_j = \max_k \left(\mu_{k,j} + r_k \sigma_{k,j}\right)$

The result is a `CPStepParameters` object (one per horizon step) containing $\phi_i$, `coeff_upper`, and $\varepsilon_i$ (reconstruction slack).

### Online evaluation (`controllers/func_cp_mpc.py`)

**Class `CPOnlineAdapter`** — implements the ACP update rule

$$\hat{\xi}_j^{(t+1)} = \hat{\xi}_j^{(t)} + \gamma \cdot \left(\mathbb{1}\!\left[\xi_j^{\text{online}} > \hat{\xi}_j^{(t)}\right] - \alpha_{\text{target}}\right)$$

**Class `FunctionalCPMPC`** — Monte Carlo MPC

- `adaptive=True`  → enables `CPOnlineAdapter`; `adaptive=False` → fixed offline coefficients.
- `safety_mode="hard"` → filter candidate paths where $\underline{D}_{t+i|t}(p) < r_\text{safe}$ (with ICS relaxation for large $i$).
- `safety_mode="soft"` → skip the hard filter; add $\sum_i w_s \cdot \max(0,\, r_\text{safe}^{(i)} - \underline{D}_{t+i|t}(p))^2$ to the MPC cost.

### ICS relaxation

For large prediction horizons the robot cannot be blamed for a future collision it cannot avoid. The maximum lateral evasion distance is

$$\Delta(i) = \tfrac{1}{2} a_{\text{lat,max}} (i \Delta t)^2, \quad a_{\text{lat,max}} = v_{\text{max}} \omega_{\text{max}}$$

and the effective safety radius used at step $i$ is $r_\text{safe}^{(i)} = \max(0,\, r_\text{safe} - \Delta(i))$.

---

## Citation

```bibtex
@article{yoonseok2026fcp,
  title   = {Planning Meets Functional Calibration: Function Conformal Prediction
             for Safe Motion Planning in Uncertain Environments},
  author  = {},
  journal = {IEEE Transactions on Robotics},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT
