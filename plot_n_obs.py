import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# CSV produced by run_3d_obs.sh; output figure goes straight into the paper folder.
CSV_PATH = os.environ.get("FCP_NOBS_CSV", "metric_3d/ablation_nobs.csv")
OUT_PATH = os.environ.get("FCP_NOBS_OUT", "T_RO2026/control_time_3d.png")

def main():
    df = pd.read_csv(CSV_PATH)

    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "0.3",
        "grid.color": "0.85",
    })
    plt.figure(figsize=(3.6, 2.8))

    # normalize legacy short names; the unified driver already uses display names
    short = {'acp': 'ACP-MPC', 'cc': 'CC-MPC', 'ecp': 'ECP-MPC',
             'fcp': 'FCP-MPC (ours)', 'nocp': 'Nominal MPC'}
    df['method_display'] = df['method'].map(lambda m: short.get(m, m))

    # colours/markers consistent with the trajectory figure (make_figs_3d); ours emphasized
    style = {
        "CC-MPC":         ("#7f7f7f", "-",  "o", 1.8, 5),
        "ECP-MPC":        ("#ff7f0e", "-",  "s", 1.8, 5),
        "ACP-MPC":        ("#9467bd", "-",  "^", 1.8, 5),
        "Nominal MPC":    ("#8c564b", "--", "D", 1.8, 5),
        "FCP-MPC (ours)": ("#1f77b4", "-",  "*", 2.8, 9),
    }
    # Nominal MPC is shown in the table but excluded from the scalability plot.
    order = [m for m in ["CC-MPC", "ECP-MPC", "ACP-MPC", "FCP-MPC (ours)"]
             if m in set(df['method_display'])]
    for m in order:
        sub = (df[df['method_display'] == m]
               .groupby('n_obs')['ctrl_mean_ms'].mean().reset_index().sort_values('n_obs'))
        c, ls, mk, lw, ms = style.get(m, ("0.5", "-", "o", 1.8, 5))
        plt.plot(sub['n_obs'], sub['ctrl_mean_ms'], color=c, linestyle=ls,
                 marker=mk, linewidth=lw, markersize=ms, label=m)

    plt.yscale('log')

    plt.xlabel(r"Number of dynamic obstacles $N_{\mathrm{obs}}$", fontsize=11)
    plt.ylabel("Mean control time [ms]", fontsize=11)

    plt.xticks([10, 50, 100, 150, 200, 280], fontsize=10)
    plt.yticks(fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(fontsize=9.5, loc='upper left', frameon=True, framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"[saved] scalability figure -> {OUT_PATH}")

if __name__ == "__main__":
    main()
