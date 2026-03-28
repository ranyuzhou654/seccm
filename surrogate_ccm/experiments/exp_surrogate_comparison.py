"""Experiment: compare surrogate methods across systems and surrogate counts.

Answers: which surrogate method helps most for which system?
         How many surrogates are needed for stable p-value resolution?
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ..generators import create_system, generate_network
from ._config_helpers import collect_seccm_kwargs, get_system_kwargs
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map


# ── Metrics of interest ──────────────────────────────────────────────
AUC_KEYS = [
    "AUC_ROC_rho",
    "AUC_ROC_surrogate",
    "AUC_ROC_zscore",
    "AUC_ROC_delta",
    "AUC_ROC_delta_zscore",
]
COLLECT_KEYS = AUC_KEYS + ["TPR", "FPR"]


# ── Worker ───────────────────────────────────────────────────────────
def _run_single_rep(args):
    """Run one repetition for a (system, method, n_surr) combination.

    Same seed → same network & data across all (method, n_surr) combos,
    enabling paired comparison.
    """
    (system_name, topology, N, eps, method, n_surr,
     ts_cfg, seed, net_kwargs, fdr, extra_seccm_kwargs,
     sys_kwargs) = args

    T = ts_cfg.get("T", 3000)
    transient = ts_cfg.get("transient", 1000)

    try:
        adj = generate_network(topology, N, seed=seed, **net_kwargs)
        system = create_system(system_name, adj, eps, **sys_kwargs)
        data = system.generate(T, transient=transient, seed=seed)

        seccm = SECCM(
            surrogate_method=method,
            n_surrogates=n_surr,
            alpha=0.05,
            fdr=fdr,
            seed=seed,
            verbose=False,
            **extra_seccm_kwargs,
        )
        seccm.fit(data)
        metrics = seccm.score(adj)
        return metrics
    except Exception:
        return None


# ── Orchestrator ─────────────────────────────────────────────────────
def run_surrogate_comparison_experiment(config, output_dir, n_jobs=-1):
    """Compare surrogate methods across systems and surrogate counts."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("surrogate_comparison", {})
    systems = cfg.get("systems", ["logistic", "lorenz", "henon"])
    methods = cfg.get("methods", ["fft", "aaft", "iaaft", "timeshift", "random_reorder"])
    n_surr_values = cfg.get("n_surrogates_values", [19, 49, 99, 199, 499])
    N = cfg.get("N", 10)
    topology = cfg.get("topology", "ER")
    coupling_map = cfg.get("coupling", {"logistic": 0.1, "lorenz": 1.0, "henon": 0.05})
    n_reps = cfg.get("n_reps", 20)
    er_p = cfg.get("er_p", 0.3)
    # Use uncorrected p-values by default for binary detection, since
    # BH-FDR with rank-based p-values needs very high n_surrogates to
    # achieve the tiny p-values required.
    fdr = cfg.get("fdr", False)
    seed_base = config.get("seed", 42)
    ts_cfg = config.get("time_series", {})

    # Extra SECCM kwargs from config
    seccm_cfg = cfg.get("seccm_kwargs", {})
    extra_seccm_kwargs = collect_seccm_kwargs(seccm_cfg)
    sys_kwargs_map = {sys: get_system_kwargs(config, sys) for sys in systems}

    net_kwargs = {"p": er_p}

    # Build all combos
    combos = [
        (sys, method, n_surr)
        for sys in systems
        for method in methods
        for n_surr in n_surr_values
    ]

    rows = []
    combo_pbar = tqdm(combos, desc="Surrogate comparison")

    for sys_name, method, n_surr in combo_pbar:
        eps = coupling_map.get(sys_name, 0.1)
        combo_pbar.set_postfix_str(f"{sys_name}/{method}/n={n_surr}")

        args_list = [
            (sys_name, topology, N, eps, method, n_surr,
             ts_cfg, seed_base + rep, net_kwargs, fdr, extra_seccm_kwargs,
             sys_kwargs_map.get(sys_name, {}))
            for rep in range(n_reps)
        ]

        results = parallel_map(
            _run_single_rep, args_list,
            n_jobs=n_jobs,
            desc=f"  reps {sys_name}/{method}/n={n_surr}",
        )

        valid = [r for r in results if r is not None]
        for r in valid:
            row = {
                "system": sys_name,
                "method": method,
                "n_surrogates": n_surr,
            }
            for k in COLLECT_KEYS:
                row[k] = r.get(k, np.nan)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "surrogate_comparison.csv"), index=False)
    print(f"  Results saved ({len(df)} rows)")

    _plot_all(df, output_dir)
    return df


# ── Visualization ────────────────────────────────────────────────────
def _plot_all(df, output_dir):
    """Generate all comparison plots."""
    if df.empty:
        return

    systems = df["system"].unique()
    methods = df["method"].unique()

    # (a) Boxplots — AUC delta per method, one subplot per system
    for metric in ["AUC_ROC_delta", "AUC_ROC_delta_zscore"]:
        fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 5),
                                 sharey=True, squeeze=False)
        axes = axes.ravel()
        for i, sys in enumerate(systems):
            sub = df[df["system"] == sys]
            sns.boxplot(data=sub, x="method", y=metric, ax=axes[i],
                        order=methods, palette="Set2")
            axes[i].set_title(sys)
            axes[i].axhline(0, ls="--", color="grey", lw=0.8)
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=45)
        axes[0].set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"boxplot_{metric}.png"), dpi=150)
        plt.close(fig)

    # (b) Line charts — AUC vs n_surrogates per method
    for metric in AUC_KEYS:
        fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                                 sharey=True, squeeze=False)
        axes = axes.ravel()
        for i, sys in enumerate(systems):
            sub = df[df["system"] == sys]
            # Baseline: raw rho (constant across methods/n_surr)
            if metric != "AUC_ROC_rho":
                baseline = sub.groupby("n_surrogates")["AUC_ROC_rho"].mean()
                axes[i].plot(baseline.index, baseline.values,
                             "k--", lw=1.5, label="raw ρ", zorder=0)
            for method in methods:
                msub = sub[sub["method"] == method]
                grouped = msub.groupby("n_surrogates")[metric]
                mean = grouped.mean()
                sem = grouped.sem()
                axes[i].errorbar(mean.index, mean.values, yerr=sem.values,
                                 marker="o", ms=4, capsize=3, label=method)
            axes[i].set_title(sys)
            axes[i].set_xlabel("n_surrogates")
            axes[i].set_xscale("log")
        axes[0].set_ylabel(metric)
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"line_{metric}_vs_nsurr.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # (c) Heatmaps — system × method, color = mean AUC delta
    for metric in ["AUC_ROC_delta", "AUC_ROC_delta_zscore"]:
        pivot = df.groupby(["system", "method"])[metric].mean().unstack("method")
        pivot = pivot.reindex(columns=methods, index=systems)

        fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.2),
                                        max(3, len(systems) * 1.0)))
        vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 0.01)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r",
                    center=0, vmin=-vmax, vmax=vmax, ax=ax,
                    linewidths=0.5)
        ax.set_title(f"Mean {metric}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"heatmap_{metric}.png"), dpi=150)
        plt.close(fig)

    # (d) TPR vs n_surrogates — shows p-value resolution effect
    if "TPR" in df.columns and df["TPR"].sum() > 0:
        fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                                 sharey=True, squeeze=False)
        axes = axes.ravel()
        for i, sys in enumerate(systems):
            sub = df[df["system"] == sys]
            for method in methods:
                msub = sub[sub["method"] == method]
                grouped = msub.groupby("n_surrogates")["TPR"]
                mean = grouped.mean()
                axes[i].plot(mean.index, mean.values, marker="o", ms=4, label=method)
            axes[i].set_title(sys)
            axes[i].set_xlabel("n_surrogates")
            axes[i].set_xscale("log")
        axes[0].set_ylabel("TPR")
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "line_TPR_vs_nsurr.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Plots saved to {output_dir}/")
