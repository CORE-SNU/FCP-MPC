# Planning Meets Functional Calibration: Function Conformal Prediction for Safe Motion Planning in Uncertain Environments

This repository contains the experiment code for the paper. The method constructs a conformal upper envelope $U_i(x)$ over the distance-field prediction residual $S_{t+i|t}(x) = D_{t+i|t}(x) - D_{t+i}(x)$ and uses it to define a *certified lower bound* on the true distance field at every future step of an MPC rollout. The same conformal machinery is evaluated in two settings: 2-D pedestrian avoidance (ETH-UCY) and 3-D quadrotor navigation (PyBullet).

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

Online calibration  (adaptive variant only)
  └─ Adaptive update of coefficient quantiles:
       ξ̂_j^{(t+1)} = ξ̂_j^{(t)} + γ · (𝟙[ξ_j^online > ξ̂_j^{(t)}] − α_target)

Control (sampling-based MPC)
  ├─ 1. Sample N candidate control sequences; roll out the dynamics
  ├─ 2. Filter / penalize paths based on the certified lower bound:
  │       hard mode: reject paths where  D̲_{t+i|t}(p) < r_safe + ε
  │       soft mode: penalize  max(0, r_safe − D̲_{t+i|t}(p))² in the MPC cost
  ├─ 3. ICS relaxation for large i:
  │       Δ(i) = ½ a_lat,max (i Δt)²
  │       effective r_safe(i) = max(0, r_safe − Δ(i))
  └─ 4. Score feasible paths; apply first control
```

---

## Repository structure

The experiments are split into two parallel families — `planar/` (2-D pedestrian avoidance on ETH-UCY) and `quadrotor/` (3-D drone navigation) — over a **shared core** (`utils.py`, `cp/`, `controllers/`, `viz_traj.py`) that lives at the repo root, so both families import the *same* conformal machinery and only the environment / planner differ.

```
.
├── utils.py                  # shared: grid construction, distance fields, plotting helpers
├── viz_traj.py               # shared: trajectory-image helpers (save_traj_image_2d / _3d)
├── cp/
│   └── functional_cp.py      # FPCA + GMM offline calibration (PCAGMMResidualCP, CPStepParameters)
├── controllers/              # shared conformal controllers (2-D and 3-D)
│   ├── func_cp_mpc.py        #   FunctionalCPMPC — 2-D FCP controller (all 4 variants)
│   ├── acp_mpc.py · ecp_mpc.py · cc.py                          #   2-D baselines
│   ├── func_3d_mpc.py · cp_3d_mpc.py · acp_3d_mpc.py · ecp_mpc_3d.py   #   3-D controllers
│   └── utils.py              #   path-sampling helpers shared by 2-D and 3-D
│
├── planar/                   # 2-D pedestrian avoidance on ETH-UCY
│   ├── runner_2d.py          #   main entry point
│   ├── run.sh                #   full 2-D ablation sweep
│   ├── make_table_2d.py      #   aggregate metric/ → LaTeX / CSV tables
│   ├── make_figs_2d.py       #   trajectory-overlay figure
│   ├── preprocess.py         #   ETH-UCY raw-data loader
│   ├── prediction/cv.py      #   constant-velocity pedestrian prediction
│   ├── sims/                 #   per-controller simulation loops + helpers
│   ├── predictions/          #   pre-computed prediction PKLs (one per dataset)
│   ├── assets/homographies/  #   ETH-UCY scene homographies
│   └── metric/  traj/        #   outputs (auto-created)
│
├── quadrotor/                # 3-D drone navigation (PyBullet)
│   ├── runner_3d.py          #   main entry point
│   ├── quad_env.py           #   PyBullet quadrotor environment (+ repo-root sys.path bootstrap)
│   ├── sim_{func,cp,ecp,acp}_3d.py   #   per-controller rollout wrappers
│   └── make_*_3d.py          #   tables / figures
│
├── run_3d.sh, run_3d_obs.sh, …   # 3-D sweep orchestration wrappers (call quadrotor/runner_3d.py)
├── docs/                     # handoff notes, experiment logs, planning docs
└── T_RO2026/  T_RO2026_v2/   # LaTeX manuscript (main.tex, figures, tables)
```

Files in `planar/` and `quadrotor/` reach the shared root modules via a small `sys.path` bootstrap at the top of each entry point — it inserts the repo root onto `sys.path` so `import utils` / `from controllers… import …` resolve no matter where the script is launched from.

---

## Installation

The pinned environment (conda env `cp`, Python 3.10) covers both the 2-D (ETH-UCY) and the 3-D (PyBullet) experiments:

```bash
conda create -n cp python=3.10
conda activate cp
pip install -r requirements.txt
```

`requirements.txt` installs `gym-pybullet-drones` directly from a pinned git commit (so the install needs `git` + network for that line). Key dependencies:

| Package | Version |
|---|---|
| numpy | 2.2.6 |
| scipy | 1.13.1 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.10.8 |
| pybullet | 3.2.5 |
| gymnasium | 1.2.3 |
| gym-pybullet-drones | `git@a8c238c` |

---

## 2-D pedestrian-avoidance experiments

### Datasets

The 2-D experiments use the ETH/UCY pedestrian datasets (`zara1`, `zara2`, `eth`, `univ`). Pre-computed constant-velocity predictions are provided in `planar/predictions/`.

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

### Usage

```bash
# single experiment
python planar/runner_2d.py --dataset zara1 --controller fcp-hard-adaptive

# full ablation sweep (all 7 controllers × all 4 datasets)
bash planar/run.sh

# aggregate metric/ into LaTeX / CSV tables
python planar/make_table_2d.py
```

Results are written to `planar/metric/<dataset>_<controller>.json` and `planar/traj/<dataset>_<controller>.npy`.

---

## 3-D quadrotor experiments

The 3-D drone-navigation experiments run in a PyBullet environment (`quadrotor/quad_env.py`). The entry point sweeps a set of methods over a range of random seeds:

```bash
# direct: all methods over seeds 20–39 with 280 dynamic obstacles
python quadrotor/runner_3d.py --methods nocp,cc,fcp,ecp --seed-from 20 --seed-to 39 --n-obs 280

# convenience wrappers (parse the same options, write to metric_3d/)
bash run_3d.sh        # main 3-D sweep
bash run_3d_obs.sh    # varying number of obstacles
```

Per-run metrics are written to `metric_3d/` (CSV + JSON). Aggregate them and build the paper figures with `quadrotor/make_table_3d.py` and `quadrotor/make_figs_3d.py`.

> The `run_*_3d.sh` wrappers at the repo root are batch-orchestration scripts; a few contain machine-specific absolute paths (`cd /home/…/cp_scratch`) that you may need to adjust for your setup.

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


## License

MIT
