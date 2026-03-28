"""E_edge: Edge Density Sweep.

Sweeps ER edge probability p to study how network density affects
CCM and surrogate-based causal discovery performance.
Also tests BA and WS topologies as comparison points.
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


DEFAULT_SYSTEMS = {
    "logistic":       {"N": 5, "coupling": 0.3,  "T": 2000, "sys_kwargs": {}},
    "rossler":        {"N": 5, "coupling": 0.15, "T": 5000,
                       "sys_kwargs": {"dt": 0.05}},
    "hindmarsh_rose": {"N": 5, "coupling": 0.2,  "T": 10000,
                       "sys_kwargs": {}, "subsample": 5},
}

DEFAULT_SURROGATES = ["iaaft", "cycle_phase_A"]

DEFAULT_ER_PROBS = [0.2, 0.3, 0.5, 0.7, 0.9]

# Additional topologies to compare against ER sweep
DEFAULT_EXTRA_TOPOS = [
    ("BA", {"m": 2}),
    ("WS", {"k": 4, "p": 0.3}),
]


def _worker(args):
    """Worker: one system × topology × surrogate × replicate."""
    (system_name, sys_cfg, topology, topo_kwargs, surr_method,
     n_surrogates, seed) = args

    try:
        N = sys_cfg["N"]
        T = sys_cfg["T"]
        coupling = sys_cfg["coupling"]
        sys_kwargs = sys_cfg.get("sys_kwargs", {})
        subsample = sys_cfg.get("subsample", 1)

        adj = generate_network(topology, N, seed=seed, **topo_kwargs)
        system = create_system(system_name, adj, coupling, **sys_kwargs)
        data = system.generate(T, transient=1000, seed=seed)

        if subsample > 1:
            data = data[::subsample]

        # Compute edge density of generated network
        mask = ~np.eye(N, dtype=bool)
        n_possible = mask.sum()
        n_edges = adj[mask].sum()
        edge_density = n_edges / n_possible if n_possible > 0 else 0.0

        seccm = SECCM(
            surrogate_method=surr_method,
            n_surrogates=n_surrogates,
            alpha=0.05, fdr=True,
            seed=seed, verbose=False,
        )
        seccm.fit(data)
        metrics = seccm.score(adj)

        # Surrogate diagnostics
        rho_mat = seccm.ccm_matrix_
        z_mat = seccm.zscore_matrix_
        surr_means = seccm.surrogate_mean_matrix_ if hasattr(seccm, 'surrogate_mean_matrix_') else None

        rho_gap = np.nan
        null_overlap = np.nan
        if surr_means is not None:
            y_true = adj[mask].ravel().astype(int)
            rho_vals = rho_mat[mask].ravel()
            surr_vals = surr_means[mask].ravel()
            gaps = rho_vals - surr_vals
            if y_true.sum() > 0 and (1 - y_true).sum() > 0:
                rho_gap = float(np.mean(gaps[y_true == 1]) - np.mean(gaps[y_true == 0]))

        return {
            "system": system_name,
            "topology": topology,
            "topo_param": str(topo_kwargs),
            "edge_density": edge_density,
            "surrogate": surr_method,
            "seed": seed,
            "AUC_ROC_rho":          metrics.get("AUC_ROC_rho", np.nan),
            "AUC_ROC_zscore":       metrics.get("AUC_ROC_zscore", np.nan),
            "AUC_ROC_delta_zscore": metrics.get("AUC_ROC_delta_zscore", np.nan),
            "AUC_PR_zscore":        metrics.get("AUC_PR_zscore", np.nan),
            "rho_gap":              rho_gap,
            "n_edges":              int(n_edges),
        }

    except Exception as e:
        return {
            "system": system_name, "topology": topology,
            "topo_param": str(topo_kwargs),
            "surrogate": surr_method, "seed": seed,
            "error": str(e),
        }


def run_edge_density_experiment(config, output_dir="results/edge_density",
                                n_jobs=-1):
    """Run E_edge: Edge Density Sweep."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("edge_density", {})
    n_surrogates = cfg.get("n_surrogates", 100)
    n_reps       = cfg.get("n_reps", 10)
    base_seed    = cfg.get("seed", 42)
    system_cfgs  = cfg.get("systems", DEFAULT_SYSTEMS)
    surrogates   = cfg.get("surrogates", DEFAULT_SURROGATES)
    er_probs     = cfg.get("er_probs", DEFAULT_ER_PROBS)
    extra_topos  = cfg.get("extra_topos", DEFAULT_EXTRA_TOPOS)

    valid_systems = {k: v for k, v in system_cfgs.items() if k in SYSTEM_CLASSES}

    args_list = []
    for sys_name, scfg in valid_systems.items():
        # ER sweep
        for p in er_probs:
            for surr in surrogates:
                for rep in range(n_reps):
                    seed = base_seed + hash(
                        ("e_edge", sys_name, "ER", p, surr, rep)
                    ) % (2**31)
                    args_list.append(
                        (sys_name, scfg, "ER", {"p": p}, surr,
                         n_surrogates, seed)
                    )

        # Extra topologies
        for topo_name, topo_kw in extra_topos:
            for surr in surrogates:
                for rep in range(n_reps):
                    seed = base_seed + hash(
                        ("e_edge", sys_name, topo_name, str(topo_kw), surr, rep)
                    ) % (2**31)
                    args_list.append(
                        (sys_name, scfg, topo_name, topo_kw, surr,
                         n_surrogates, seed)
                    )

    n_total = len(args_list)
    print(f"\n  E_edge: {n_total} runs "
          f"({len(valid_systems)} systems × densities × "
          f"{len(surrogates)} surrogates × {n_reps} reps)")

    results = parallel_map(_worker, args_list, n_jobs=n_jobs,
                           desc="E_edge Edge Density")

    rows   = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if errors:
        from collections import Counter
        print(f"  {len(errors)} failures: "
              f"{dict(Counter(e.get('system','?') for e in errors))}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "edge_density_raw.csv"), index=False)

    if len(df) == 0:
        return df

    # Aggregation
    agg = df.groupby(["system", "topology", "topo_param", "surrogate"]).agg(
        edge_density_mean=("edge_density", "mean"),
        AUC_ROC_rho_mean=("AUC_ROC_rho", "mean"),
        AUC_ROC_rho_std=("AUC_ROC_rho", "std"),
        AUC_ROC_zscore_mean=("AUC_ROC_zscore", "mean"),
        AUC_ROC_zscore_std=("AUC_ROC_zscore", "std"),
        delta_auroc_mean=("AUC_ROC_delta_zscore", "mean"),
        delta_auroc_std=("AUC_ROC_delta_zscore", "std"),
        rho_gap_mean=("rho_gap", "mean"),
        n_edges_mean=("n_edges", "mean"),
        count=("seed", "count"),
    ).reset_index()
    agg.to_csv(os.path.join(output_dir, "edge_density_agg.csv"), index=False)
    print(f"  Saved {len(df)} raw rows, {len(agg)} aggregated rows")

    _plot_edge_density(df, output_dir)

    return df


def _plot_edge_density(df, output_dir):
    """Plot AUROC vs edge density for each system."""
    systems = sorted(df["system"].unique())
    surrogates = sorted(df["surrogate"].unique())
    surr_colors = {"iaaft": "#3498db", "cycle_phase_A": "#e74c3c",
                   "cycle_phase_B": "#e67e22", "fft": "#2ecc71"}

    # --- AUROC(z-score) vs edge density (ER only) ---
    df_er = df[df["topology"] == "ER"]
    if len(df_er) == 0:
        return

    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]

    for ax, sys_name in zip(axes, systems):
        for surr in surrogates:
            sub = df_er[(df_er["system"] == sys_name) & (df_er["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("edge_density")["AUC_ROC_zscore"].agg(["mean", "std"])
            color = surr_colors.get(surr, "#7f8c8d")
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                        marker="o", label=f"z-score ({surr})", color=color,
                        capsize=3, linewidth=1.5)

        # Raw rho AUROC
        sub_any = df_er[df_er["system"] == sys_name]
        if len(sub_any) > 0:
            agg_raw = sub_any.groupby("edge_density")["AUC_ROC_rho"].mean()
            ax.plot(agg_raw.index, agg_raw.values, "k--", label="raw ρ",
                    linewidth=1, alpha=0.7)

        # Mark BA and WS points
        for topo in ["BA", "WS"]:
            sub_t = df[(df["system"] == sys_name) & (df["topology"] == topo)]
            if len(sub_t) > 0:
                for surr in surrogates:
                    sub_ts = sub_t[sub_t["surrogate"] == surr]
                    if len(sub_ts) > 0:
                        color = surr_colors.get(surr, "#7f8c8d")
                        ax.scatter(
                            sub_ts["edge_density"].mean(),
                            sub_ts["AUC_ROC_zscore"].mean(),
                            marker="^" if topo == "BA" else "s",
                            color=color, s=80, zorder=5,
                            edgecolors="black", linewidths=0.8,
                            label=f"{topo} ({surr})",
                        )

        ax.set_xlabel("Edge Density")
        ax.set_ylabel("AUROC")
        ax.set_title(sys_name)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("E_edge: AUROC vs Network Edge Density", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auroc_vs_density.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "auroc_vs_density.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- ΔAUROC vs edge density ---
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]

    for ax, sys_name in zip(axes, systems):
        for surr in surrogates:
            sub = df_er[(df_er["system"] == sys_name) & (df_er["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("edge_density")["AUC_ROC_delta_zscore"].agg(
                ["mean", "std"])
            color = surr_colors.get(surr, "#7f8c8d")
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                        marker="o", label=surr, color=color,
                        capsize=3, linewidth=1.5)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Edge Density")
        ax.set_ylabel("ΔAUROC")
        ax.set_title(sys_name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("E_edge: ΔAUROC vs Network Edge Density", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "delta_auroc_vs_density.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "delta_auroc_vs_density.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
