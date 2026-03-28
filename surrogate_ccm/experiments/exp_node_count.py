"""E_node: Node-count sweep experiment.

Studies how the number of nodes in the generated system affects
SE-CCM detection quality and runtime while keeping the dynamical
system family, topology type, and surrogate method fixed.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..generators import SYSTEM_CLASSES, create_system, generate_network
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map
from ._config_helpers import collect_seccm_kwargs, get_system_kwargs


DEFAULT_SYSTEMS = ["logistic", "lorenz", "henon"]
DEFAULT_SURROGATES = ["iaaft"]
DEFAULT_N_VALUES = [5, 10, 20, 30]


def _resolve_topology_kwargs(topology, cfg, N):
    """Build topology kwargs for a given network size."""
    topology = topology.upper()

    if topology == "ER":
        return {"p": cfg.get("er_p", 0.3)}
    if topology == "WS":
        k = min(cfg.get("ws_k", 4), max(N - 1, 1))
        # Watts-Strogatz requires even k
        if k % 2 == 1:
            k = max(k - 1, 0)
        k = max(k, 2) if N >= 3 else 1
        return {"k": k, "p": cfg.get("ws_p", 0.3)}
    if topology == "BA":
        return {"m": min(cfg.get("ba_m", 2), max(N - 1, 1))}
    if topology == "RING":
        return {"k": min(cfg.get("ring_k", 1), max((N - 1) // 2, 1))}
    raise ValueError(f"Unsupported topology for node_count experiment: {topology}")


def _worker(args):
    """Worker: one system × N × surrogate × replicate."""
    (system_name, N, topology, topo_kwargs, coupling, T, transient,
     surr_method, n_surrogates, seed, fdr, sys_kwargs, extra_seccm_kwargs) = args

    try:
        adj = generate_network(topology, N, seed=seed, **topo_kwargs)
        system = create_system(system_name, adj, coupling, **sys_kwargs)
        data = system.generate(T, transient=transient, seed=seed)

        fit_start = time.perf_counter()
        seccm = SECCM(
            surrogate_method=surr_method,
            n_surrogates=n_surrogates,
            alpha=0.05,
            fdr=fdr,
            seed=seed,
            verbose=False,
            **extra_seccm_kwargs,
        )
        seccm.fit(data)
        fit_time_sec = time.perf_counter() - fit_start
        metrics = seccm.score(adj)

        mask = ~np.eye(N, dtype=bool)
        n_edges = int(adj[mask].sum())
        n_possible = int(mask.sum())
        edge_density = n_edges / n_possible if n_possible > 0 else 0.0

        return {
            "system": system_name,
            "N": N,
            "topology": topology,
            "surrogate": surr_method,
            "seed": seed,
            "n_edges": n_edges,
            "edge_density": edge_density,
            "fit_time_sec": fit_time_sec,
            "AUC_ROC_rho": metrics.get("AUC_ROC_rho", np.nan),
            "AUC_ROC_zscore": metrics.get("AUC_ROC_zscore", np.nan),
            "AUC_ROC_delta_zscore": metrics.get("AUC_ROC_delta_zscore", np.nan),
            "AUC_PR_zscore": metrics.get("AUC_PR_zscore", np.nan),
            "TPR": metrics.get("TPR", np.nan),
            "FPR": metrics.get("FPR", np.nan),
        }
    except Exception as e:
        return {
            "system": system_name,
            "N": N,
            "topology": topology,
            "surrogate": surr_method,
            "seed": seed,
            "error": str(e),
        }


def run_node_count_experiment(config, output_dir="results/node_count", n_jobs=-1):
    """Run E_node: sweep node count N for selected systems."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("node_count", {})
    systems = cfg.get("systems", DEFAULT_SYSTEMS)
    surrogates = cfg.get("surrogates", DEFAULT_SURROGATES)
    N_values = cfg.get("N_values", DEFAULT_N_VALUES)
    topology = cfg.get("topology", "ER")
    coupling_cfg = cfg.get("coupling", {})
    T_cfg = cfg.get("T", config.get("time_series", {}).get("T", 3000))
    transient = cfg.get("transient", config.get("time_series", {}).get("transient", 1000))
    n_surrogates = cfg.get("n_surrogates", 99)
    n_reps = cfg.get("n_reps", 10)
    fdr = cfg.get("fdr", False)
    base_seed = cfg.get("seed", config.get("seed", 42))

    seccm_cfg = cfg.get("seccm_kwargs", {})
    extra_seccm_kwargs = collect_seccm_kwargs(seccm_cfg)

    valid_systems = [s for s in systems if s in SYSTEM_CLASSES]
    skipped_systems = sorted(set(systems) - set(valid_systems))
    if skipped_systems:
        print(f"  Skipping unknown systems: {skipped_systems}")

    args_list = []
    for system_name in valid_systems:
        coupling = coupling_cfg.get(system_name, 0.1) if isinstance(coupling_cfg, dict) else coupling_cfg
        T = T_cfg.get(system_name, 3000) if isinstance(T_cfg, dict) else T_cfg
        sys_kwargs = get_system_kwargs(config, system_name)

        for N in N_values:
            topo_kwargs = _resolve_topology_kwargs(topology, cfg, N)
            for surr in surrogates:
                for rep in range(n_reps):
                    seed = base_seed + hash(
                        ("e_node", system_name, N, topology, surr, rep)
                    ) % (2**31)
                    args_list.append((
                        system_name, N, topology, topo_kwargs, coupling, T, transient,
                        surr, n_surrogates, seed, fdr, sys_kwargs, extra_seccm_kwargs,
                    ))

    n_total = len(args_list)
    print(f"\n  E_node: {n_total} runs "
          f"({len(valid_systems)} systems × {len(N_values)} N-values × "
          f"{len(surrogates)} surrogates × {n_reps} reps)")

    results = parallel_map(_worker, args_list, n_jobs=n_jobs,
                           desc="E_node Node Count")

    rows = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if errors:
        from collections import Counter
        print(f"  {len(errors)} failures: "
              f"{dict(Counter(e.get('system', '?') for e in errors))}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "node_count_raw.csv"), index=False)

    if len(df) == 0:
        return df

    agg = df.groupby(["system", "N", "surrogate"]).agg(
        edge_density_mean=("edge_density", "mean"),
        n_edges_mean=("n_edges", "mean"),
        fit_time_sec_mean=("fit_time_sec", "mean"),
        fit_time_sec_std=("fit_time_sec", "std"),
        AUC_ROC_rho_mean=("AUC_ROC_rho", "mean"),
        AUC_ROC_rho_std=("AUC_ROC_rho", "std"),
        AUC_ROC_zscore_mean=("AUC_ROC_zscore", "mean"),
        AUC_ROC_zscore_std=("AUC_ROC_zscore", "std"),
        delta_auroc_mean=("AUC_ROC_delta_zscore", "mean"),
        delta_auroc_std=("AUC_ROC_delta_zscore", "std"),
        TPR_mean=("TPR", "mean"),
        FPR_mean=("FPR", "mean"),
        count=("seed", "count"),
    ).reset_index()
    agg.to_csv(os.path.join(output_dir, "node_count_agg.csv"), index=False)
    print(f"  Saved {len(df)} raw rows, {len(agg)} aggregated rows")

    _plot_node_count(df, output_dir)
    return df


def _plot_node_count(df, output_dir):
    """Plot AUROC and runtime vs node count."""
    systems = sorted(df["system"].unique())
    surrogates = sorted(df["surrogate"].unique())
    surr_colors = {
        "iaaft": "#3498db",
        "cycle_phase_A": "#e74c3c",
        "cycle_phase_B": "#e67e22",
        "fft": "#2ecc71",
        "auto": "#8e44ad",
    }

    # AUROC vs N
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]
    for ax, sys_name in zip(axes, systems):
        for surr in surrogates:
            sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("N")["AUC_ROC_zscore"].agg(["mean", "std"])
            color = surr_colors.get(surr, "#7f8c8d")
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                        marker="o", label=f"z-score ({surr})",
                        color=color, capsize=3, linewidth=1.5)

        sub_any = df[df["system"] == sys_name]
        if len(sub_any) > 0:
            agg_raw = sub_any.groupby("N")["AUC_ROC_rho"].mean()
            ax.plot(agg_raw.index, agg_raw.values, "k--", label="raw ρ",
                    linewidth=1, alpha=0.7)

        ax.set_xlabel("Node count N")
        ax.set_ylabel("AUROC")
        ax.set_title(sys_name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("E_node: AUROC vs Node Count", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auroc_vs_node_count.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "auroc_vs_node_count.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Delta-AUROC vs N
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]
    for ax, sys_name in zip(axes, systems):
        for surr in surrogates:
            sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("N")["AUC_ROC_delta_zscore"].agg(["mean", "std"])
            color = surr_colors.get(surr, "#7f8c8d")
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                        marker="o", color=color, capsize=3, linewidth=1.5,
                        label=surr)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Node count N")
        ax.set_ylabel("ΔAUROC (z-score - raw ρ)")
        ax.set_title(sys_name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("E_node: ΔAUROC vs Node Count", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "delta_auroc_vs_node_count.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "delta_auroc_vs_node_count.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Runtime vs N
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]
    for ax, sys_name in zip(axes, systems):
        for surr in surrogates:
            sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("N")["fit_time_sec"].agg(["mean", "std"])
            color = surr_colors.get(surr, "#7f8c8d")
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                        marker="o", color=color, capsize=3, linewidth=1.5,
                        label=surr)

        ax.set_xlabel("Node count N")
        ax.set_ylabel("SECCM fit time (s)")
        ax.set_title(sys_name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("E_node: Runtime vs Node Count", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "runtime_vs_node_count.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "runtime_vs_node_count.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
