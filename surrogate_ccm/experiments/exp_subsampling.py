"""E_sub: Subsampling Experiment.

Tests the CausalDynamics (NeurIPS 2025) finding that lower sampling
frequency can improve causal discovery by reducing autocorrelation.

Key distinction from E5 (convergence): E5 varies T directly. This
experiment keeps the same underlying dynamics (T_base time steps)
but changes temporal resolution by subsampling every k-th step.
This tests autocorrelation reduction vs information loss.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..generators import SYSTEM_CLASSES, create_system, generate_network
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map
from ._config_helpers import collect_seccm_kwargs, stable_seed


DEFAULT_SYSTEMS = {
    "rossler":     {"N": 5, "coupling": 0.15, "T_base": 10000,
                    "sys_kwargs": {"dt": 0.05}},
    "van_der_pol": {"N": 5, "coupling": 0.3,  "T_base": 10000,
                    "sys_kwargs": {}},
    "kuramoto":    {"N": 5, "coupling": 0.5,  "T_base": 10000,
                    "sys_kwargs": {}},
}

DEFAULT_SURROGATES = ["iaaft", "cycle_phase_A"]

DEFAULT_SUBSAMPLE_FACTORS = [1, 2, 5, 10, 20]


def _worker(args):
    """Worker: one system × subsample factor × surrogate × replicate."""
    (system_name, sys_cfg, k, surr_method, n_surrogates, rep, graph_seed,
     data_seed, seccm_seed, extra_seccm_kwargs) = args

    try:
        N = sys_cfg["N"]
        T_base = sys_cfg["T_base"]
        coupling = sys_cfg["coupling"]
        sys_kwargs = sys_cfg.get("sys_kwargs", {})

        adj = generate_network("ER", N, seed=graph_seed, p=0.5)
        system = create_system(system_name, adj, coupling, **sys_kwargs)
        data = system.generate(T_base, transient=1000, seed=data_seed)

        # Subsample every k-th step
        data_sub = data[::k]
        T_eff = len(data_sub)

        # Compute autocorrelation at lag 1 (mean across nodes)
        acf1_vals = []
        for node in range(N):
            x = data_sub[:, node]
            x_centered = x - x.mean()
            var = np.var(x_centered)
            if var > 0:
                acf1 = np.correlate(x_centered[:-1], x_centered[1:]) / (var * (len(x) - 1))
                acf1_vals.append(float(acf1[0]))
        mean_acf1 = np.mean(acf1_vals) if acf1_vals else np.nan

        seccm = SECCM(
            surrogate_method=surr_method,
            n_surrogates=n_surrogates,
            alpha=0.05, fdr=True,
            seed=seccm_seed, verbose=False,
            **extra_seccm_kwargs,
        )
        seccm.fit(data_sub)
        metrics = seccm.score(adj)

        return {
            "system": system_name,
            "subsample_k": k,
            "T_base": T_base,
            "T_effective": T_eff,
            "mean_acf1": mean_acf1,
            "surrogate": surr_method,
            "rep": rep,
            "graph_seed": graph_seed,
            "data_seed": data_seed,
            "seccm_seed": seccm_seed,
            "AUC_ROC_rho":          metrics.get("AUC_ROC_rho", np.nan),
            "AUC_ROC_zscore":       metrics.get("AUC_ROC_zscore", np.nan),
            "AUC_ROC_delta_zscore": metrics.get("AUC_ROC_delta_zscore", np.nan),
            "AUC_PR_zscore":        metrics.get("AUC_PR_zscore", np.nan),
            "TPR":                  metrics.get("TPR", np.nan),
            "FPR":                  metrics.get("FPR", np.nan),
        }

    except Exception as e:
        return {
            "system": system_name, "subsample_k": k,
            "surrogate": surr_method,
            "rep": rep,
            "graph_seed": graph_seed,
            "data_seed": data_seed,
            "seccm_seed": seccm_seed,
            "error": str(e),
        }


def run_subsampling_experiment(config, output_dir="results/subsampling",
                               n_jobs=-1):
    """Run E_sub: Subsampling Experiment."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("subsampling", {})
    n_surrogates = cfg.get("n_surrogates", 100)
    n_reps       = cfg.get("n_reps", 10)
    base_seed    = cfg.get("seed", config.get("seed", 42))
    system_cfgs  = cfg.get("systems", DEFAULT_SYSTEMS)
    surrogates   = cfg.get("surrogates", DEFAULT_SURROGATES)
    k_values     = cfg.get("subsample_factors", DEFAULT_SUBSAMPLE_FACTORS)
    vary_graph_across_reps = cfg.get("vary_graph_across_reps", False)
    seccm_cfg = dict(config.get("surrogate", {}))
    seccm_cfg.update(cfg.get("seccm_kwargs", {}))
    extra_seccm_kwargs = collect_seccm_kwargs(seccm_cfg)

    valid_systems = {k: v for k, v in system_cfgs.items() if k in SYSTEM_CLASSES}

    args_list = []
    for sys_name, scfg in valid_systems.items():
        fixed_graph_seed = stable_seed(
            base_seed, "subsampling", "graph", sys_name, scfg["N"], scfg["coupling"],
        )
        for k in k_values:
            for surr in surrogates:
                for rep in range(n_reps):
                    args_list.append(
                        (
                            sys_name,
                            scfg,
                            k,
                            surr,
                            n_surrogates,
                            rep,
                            stable_seed(
                                base_seed,
                                "subsampling",
                                "graph",
                                sys_name,
                                scfg["N"],
                                scfg["coupling"],
                                rep,
                            ) if vary_graph_across_reps else fixed_graph_seed,
                            stable_seed(
                                base_seed,
                                "subsampling",
                                "data",
                                sys_name,
                                scfg["N"],
                                scfg["coupling"],
                                rep,
                            ),
                            stable_seed(
                                base_seed,
                                "subsampling",
                                "seccm",
                                sys_name,
                                k,
                                surr,
                                rep,
                            ),
                            extra_seccm_kwargs,
                        )
                    )

    n_total = len(args_list)
    print(f"\n  E_sub: {n_total} runs "
          f"({len(valid_systems)} systems × {len(k_values)} k-values "
          f"× {len(surrogates)} surrogates × {n_reps} reps)")

    results = parallel_map(_worker, args_list, n_jobs=n_jobs,
                           desc="E_sub Subsampling")

    rows   = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if errors:
        from collections import Counter
        print(f"  {len(errors)} failures: "
              f"{dict(Counter(e.get('system','?') for e in errors))}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "subsampling_raw.csv"), index=False)

    if len(df) == 0:
        return df

    # Aggregation
    agg = df.groupby(["system", "subsample_k", "surrogate"]).agg(
        T_effective=("T_effective", "first"),
        mean_acf1_mean=("mean_acf1", "mean"),
        AUC_ROC_rho_mean=("AUC_ROC_rho", "mean"),
        AUC_ROC_rho_std=("AUC_ROC_rho", "std"),
        AUC_ROC_rho_sem=("AUC_ROC_rho", "sem"),
        AUC_ROC_zscore_mean=("AUC_ROC_zscore", "mean"),
        AUC_ROC_zscore_std=("AUC_ROC_zscore", "std"),
        AUC_ROC_zscore_sem=("AUC_ROC_zscore", "sem"),
        delta_auroc_mean=("AUC_ROC_delta_zscore", "mean"),
        delta_auroc_std=("AUC_ROC_delta_zscore", "std"),
        delta_auroc_sem=("AUC_ROC_delta_zscore", "sem"),
        count=("rep", "count"),
    ).reset_index()
    agg.to_csv(os.path.join(output_dir, "subsampling_agg.csv"), index=False)
    print(f"  Saved {len(df)} raw rows, {len(agg)} aggregated rows")

    _plot_subsampling(df, output_dir)

    return df


def _plot_subsampling(df, output_dir):
    """Plot AUROC vs subsample factor and autocorrelation."""
    systems = sorted(df["system"].unique())
    surrogates = sorted(df["surrogate"].unique())
    surr_colors = {"iaaft": "#3498db", "cycle_phase_A": "#e74c3c",
                   "cycle_phase_B": "#e67e22", "fft": "#2ecc71"}

    # --- AUROC vs subsample factor k ---
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]

    for ax, sys_name in zip(axes, systems):
        for surr in surrogates:
            sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("subsample_k")["AUC_ROC_zscore"].agg(["mean", "sem"])
            color = surr_colors.get(surr, "#7f8c8d")
            ax.errorbar(agg.index, agg["mean"], yerr=agg["sem"],
                        marker="o", label=f"z-score ({surr})", color=color,
                        capsize=3, linewidth=1.5)

        # Raw rho
        sub_any = df[df["system"] == sys_name]
        if len(sub_any) > 0:
            agg_raw = sub_any.groupby("subsample_k")["AUC_ROC_rho"].mean()
            ax.plot(agg_raw.index, agg_raw.values, "k--", label="raw ρ",
                    linewidth=1, alpha=0.7)

        ax.set_xlabel("Subsample factor k")
        ax.set_ylabel("AUROC")
        ax.set_title(f"{sys_name}\n(T_eff = T_base / k)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Add effective T on top axis
        ax2 = ax.twiny()
        k_ticks = sorted(sub_any["subsample_k"].unique()) if len(sub_any) > 0 else []
        if k_ticks:
            T_base = sub_any["T_base"].iloc[0]
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(k_ticks)
            ax2.set_xticklabels([str(int(T_base / k)) for k in k_ticks],
                                fontsize=7)
            ax2.set_xlabel("Effective T", fontsize=8)

    fig.suptitle("E_sub: AUROC vs Subsampling Factor", fontsize=13, y=1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auroc_vs_subsample.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "auroc_vs_subsample.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Autocorrelation vs subsample factor ---
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]

    for ax, sys_name in zip(axes, systems):
        sub = df[df["system"] == sys_name]
        if len(sub) == 0:
            continue
        agg = sub.groupby("subsample_k")["mean_acf1"].agg(["mean", "sem"])
        ax.errorbar(agg.index, agg["mean"], yerr=agg["sem"],
                    marker="s", color="#2c3e50", capsize=3, linewidth=1.5)
        ax.set_xlabel("Subsample factor k")
        ax.set_ylabel("Mean ACF(1)")
        ax.set_title(sys_name)
        ax.grid(True, alpha=0.3)

    fig.suptitle("E_sub: Autocorrelation Reduction via Subsampling",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "acf_vs_subsample.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "acf_vs_subsample.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- ΔAUROC vs subsample factor ---
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]

    for ax, sys_name in zip(axes, systems):
        for surr in surrogates:
            sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("subsample_k")["AUC_ROC_delta_zscore"].agg(
                ["mean", "sem"])
            color = surr_colors.get(surr, "#7f8c8d")
            ax.errorbar(agg.index, agg["mean"], yerr=agg["sem"],
                        marker="o", label=surr, color=color,
                        capsize=3, linewidth=1.5)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Subsample factor k")
        ax.set_ylabel("ΔAUROC")
        ax.set_title(sys_name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("E_sub: ΔAUROC vs Subsampling Factor", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "delta_auroc_vs_subsample.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "delta_auroc_vs_subsample.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
