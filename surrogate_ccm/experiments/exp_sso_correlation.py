"""E1: SSO vs ΔAUROC Correlation Experiment.

Validates the Surrogate Spectral Overlap (SSO) theory by computing SSO
and ΔAUROC across all system × surrogate combinations, then testing
whether SSO predicts surrogate utility.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from ..evaluation.metrics import compute_sso
from ..generators import SYSTEM_CLASSES, create_system, generate_network
from ..surrogate import SURROGATE_METHODS, generate_surrogate
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map


# Default system configurations
DEFAULT_SYSTEMS = {
    "logistic": {"N": 5, "coupling": 0.3, "T": 2000, "sys_kwargs": {}},
    "henon": {"N": 5, "coupling": 0.3, "T": 2000, "sys_kwargs": {}},
    "lorenz": {"N": 5, "coupling": 0.5, "T": 2000, "sys_kwargs": {"dt": 0.01}},
    "rossler": {"N": 5, "coupling": 0.15, "T": 5000, "sys_kwargs": {"dt": 0.05}},
    "kuramoto": {"N": 5, "coupling": 0.5, "T": 2000, "sys_kwargs": {}},
    "hindmarsh_rose": {"N": 5, "coupling": 0.2, "T": 10000,
                       "sys_kwargs": {"subsample": 5}},
    "fitzhugh_nagumo": {"N": 5, "coupling": 0.3, "T": 2000, "sys_kwargs": {}},
    "van_der_pol": {"N": 5, "coupling": 0.3, "T": 5000, "sys_kwargs": {}},
}

DEFAULT_SURROGATES = [
    "fft", "aaft", "iaaft", "timeshift", "random_reorder",
    "cycle_shuffle", "twin", "phase", "small_shuffle",
    "truncated_fourier", "cycle_phase_A", "cycle_phase_B",
]


def _run_single_combo(args):
    """Worker: run one system × surrogate combination."""
    (system_name, sys_cfg, surr_method, n_surrogates, seed) = args

    try:
        N = sys_cfg["N"]
        coupling = sys_cfg["coupling"]
        T = sys_cfg["T"]
        sys_kwargs = sys_cfg.get("sys_kwargs", {})

        adj = generate_network("ER", N, seed=seed, p=0.5)
        system = create_system(system_name, adj, coupling, **sys_kwargs)
        data = system.generate(T, transient=1000, seed=seed)

        # Run SE-CCM with this surrogate method
        seccm = SECCM(
            surrogate_method=surr_method,
            n_surrogates=n_surrogates,
            alpha=0.05,
            fdr=True,
            seed=seed,
            verbose=False,
        )
        seccm.fit(data)
        metrics = seccm.score(adj)

        # Compute SSO for each cause variable j
        sso_values = []
        for j in range(N):
            surrogates_j = generate_surrogate(
                data[:, j], method=surr_method,
                n_surrogates=min(n_surrogates, 50),
                seed=seed + j + 1000,
            )
            sso_j = compute_sso(data[:, j], surrogates_j)
            sso_values.append(sso_j)
        mean_sso = float(np.mean(sso_values))

        return {
            "system": system_name,
            "surrogate": surr_method,
            "seed": seed,
            "SSO": mean_sso,
            "AUC_ROC_rho": metrics.get("AUC_ROC_rho", np.nan),
            "AUC_ROC_zscore": metrics.get("AUC_ROC_zscore", np.nan),
            "AUC_ROC_delta_zscore": metrics.get("AUC_ROC_delta_zscore", np.nan),
            "AUC_PR_rho": metrics.get("AUC_PR_rho", np.nan),
            "AUC_PR_zscore": metrics.get("AUC_PR_zscore", np.nan),
            "TPR": metrics.get("TPR", np.nan),
            "FPR": metrics.get("FPR", np.nan),
            "F1": metrics.get("F1", np.nan),
        }
    except Exception as e:
        return {
            "system": system_name, "surrogate": surr_method,
            "seed": seed, "error": str(e),
        }


def run_sso_correlation_experiment(config, output_dir="results/sso_correlation",
                                   n_jobs=-1):
    """Run the SSO correlation experiment (E1).

    Parameters
    ----------
    config : dict
        Configuration (can override defaults via 'sso_correlation' key).
    output_dir : str
        Output directory for results.
    n_jobs : int
        Number of parallel jobs.
    """
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("sso_correlation", {})
    n_surrogates = cfg.get("n_surrogates", 100)
    n_reps = cfg.get("n_reps", 10)
    systems = cfg.get("systems", DEFAULT_SYSTEMS)
    surrogates = cfg.get("surrogates", DEFAULT_SURROGATES)
    base_seed = cfg.get("seed", 42)

    # Build argument list
    args_list = []
    for sys_name, sys_cfg in systems.items():
        if sys_name not in SYSTEM_CLASSES:
            print(f"  Skipping unknown system: {sys_name}")
            continue
        for surr_method in surrogates:
            if surr_method.lower() not in SURROGATE_METHODS:
                print(f"  Skipping unknown surrogate: {surr_method}")
                continue
            for rep in range(n_reps):
                seed = base_seed + hash((sys_name, surr_method, rep)) % (2**31)
                args_list.append(
                    (sys_name, sys_cfg, surr_method, n_surrogates, seed)
                )

    print(f"  SSO Correlation: {len(args_list)} runs "
          f"({len(systems)} systems × {len(surrogates)} surrogates × {n_reps} reps)")

    # Run
    results = parallel_map(_run_single_combo, args_list, n_jobs=n_jobs,
                           desc="SSO Correlation")

    # Collect valid results
    rows = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"  {len(errors)} runs failed (first error: {errors[0].get('error')})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "sso_correlation_raw.csv"), index=False)

    # Aggregate: mean SSO and ΔAUROC per system × surrogate
    if len(df) > 0:
        agg = df.groupby(["system", "surrogate"]).agg({
            "SSO": "mean",
            "AUC_ROC_rho": "mean",
            "AUC_ROC_zscore": "mean",
            "AUC_ROC_delta_zscore": "mean",
        }).reset_index()
        agg.to_csv(os.path.join(output_dir, "sso_correlation_agg.csv"), index=False)

        # Compute Spearman correlation
        valid = agg.dropna(subset=["SSO", "AUC_ROC_delta_zscore"])
        if len(valid) > 5:
            spearman_r, spearman_p = stats.spearmanr(
                valid["SSO"], valid["AUC_ROC_delta_zscore"]
            )
            pearson_r, pearson_p = stats.pearsonr(
                valid["SSO"], valid["AUC_ROC_delta_zscore"]
            )
        else:
            spearman_r, spearman_p = np.nan, np.nan
            pearson_r, pearson_p = np.nan, np.nan

        correlation_stats = {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "n_datapoints": int(len(valid)),
            "success_criterion": "spearman_r > 0.5 and spearman_p < 0.01",
            "criterion_met": bool(spearman_r > 0.5 and spearman_p < 0.01),
        }
        with open(os.path.join(output_dir, "correlation_stats.json"), "w") as f:
            json.dump(correlation_stats, f, indent=2)

        print(f"\n  SSO vs ΔAUROC correlation:")
        print(f"    Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")
        print(f"    Pearson r  = {pearson_r:.4f} (p = {pearson_p:.2e})")
        print(f"    Criterion met: {correlation_stats['criterion_met']}")

        # Plot
        _plot_sso_scatter(agg, correlation_stats, output_dir)

    return df


def _plot_sso_scatter(agg, stats_dict, output_dir):
    """Generate SSO vs ΔAUROC scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    systems = agg["system"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    for i, sys_name in enumerate(sorted(systems)):
        mask = agg["system"] == sys_name
        ax.scatter(
            agg.loc[mask, "SSO"],
            agg.loc[mask, "AUC_ROC_delta_zscore"],
            c=[colors[i % len(colors)]],
            marker=markers[i % len(markers)],
            label=sys_name, s=60, alpha=0.8, edgecolors="black", linewidths=0.5,
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Surrogate Spectral Overlap (SSO = JSD)", fontsize=12)
    ax.set_ylabel("ΔAUROC (z-score − raw ρ)", fontsize=12)
    ax.set_title(
        f"SSO vs Surrogate Utility\n"
        f"Spearman ρ = {stats_dict['spearman_r']:.3f} "
        f"(p = {stats_dict['spearman_p']:.2e})",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sso_vs_auroc_scatter.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "sso_vs_auroc_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
