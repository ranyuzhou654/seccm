"""D2: Regime Boundary Analysis.

Sweeps coupling strength continuously to map where surrogate utility
transitions between regimes (helps → neutral → hurts → impermeable).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..generators import SYSTEM_CLASSES, create_system, generate_network
from ..surrogate import generate_surrogate
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map
from ._config_helpers import collect_seccm_kwargs, stable_seed


DEFAULT_CONFIGS = {
    "rossler": {
        "couplings": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
        "T": 5000, "sys_kwargs": {"dt": 0.05},
    },
    "kuramoto": {
        "couplings": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        "T": 2000, "sys_kwargs": {},
    },
    "van_der_pol": {
        "couplings": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
        "T": 5000, "sys_kwargs": {},
    },
}

DEFAULT_SURROGATES = ["iaaft", "cycle_phase_A"]


def _worker(args):
    """Worker: one system × coupling × surrogate × replicate."""
    (system_name, coupling, T, surr_method, n_surrogates, N, rep, graph_seed,
     data_seed, seccm_seed, sys_kwargs, extra_seccm_kwargs) = args

    try:
        adj = generate_network("ER", N, seed=graph_seed, p=0.5)
        system = create_system(system_name, adj, coupling, **sys_kwargs)
        data = system.generate(T, transient=1000, seed=data_seed)

        seccm = SECCM(
            surrogate_method=surr_method,
            n_surrogates=n_surrogates,
            alpha=0.05, fdr=True,
            seed=seccm_seed, verbose=False,
            **extra_seccm_kwargs,
        )
        seccm.fit(data)
        metrics = seccm.score(adj)

        result = {
            "system": system_name,
            "coupling": coupling,
            "surrogate": surr_method,
            "rep": rep,
            "graph_seed": graph_seed,
            "data_seed": data_seed,
            "seccm_seed": seccm_seed,
            "AUC_ROC_rho":          metrics.get("AUC_ROC_rho", np.nan),
            "AUC_ROC_zscore":       metrics.get("AUC_ROC_zscore", np.nan),
            "AUC_ROC_delta_zscore": metrics.get("AUC_ROC_delta_zscore", np.nan),
        }

        # Diagnostics
        mask = ~np.eye(N, dtype=bool)
        if seccm.surrogate_distributions_:
            rho_obs_vals, rho_surr_means = [], []
            overlap_fracs = []
            for (i, j), rho_surr in seccm.surrogate_distributions_.items():
                obs = seccm.ccm_matrix_[i, j]
                rho_obs_vals.append(obs)
                rho_surr_means.append(np.mean(rho_surr))
                overlap_fracs.append(np.mean(rho_surr >= obs))
            result["rho_gap"]      = float(np.mean(rho_obs_vals) -
                                           np.mean(rho_surr_means))
            result["null_overlap"] = float(np.mean(overlap_fracs))

        if hasattr(seccm, "convergence_matrix_") and seccm.convergence_matrix_ is not None:
            conv_off = seccm.convergence_matrix_[mask]
            result["frac_converging"] = float(np.mean(conv_off > 0))

        return result

    except Exception as e:
        return {
            "system": system_name, "coupling": coupling,
            "surrogate": surr_method,
            "rep": rep,
            "graph_seed": graph_seed,
            "data_seed": data_seed,
            "seccm_seed": seccm_seed,
            "error": str(e),
        }


def run_regime_boundary_experiment(config, output_dir="results/regime_boundaries",
                                    n_jobs=-1):
    """Run D2: Regime Boundary Analysis."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("regime_boundaries", {})
    n_surrogates = cfg.get("n_surrogates", 100)
    n_reps       = cfg.get("n_reps", 10)
    N            = cfg.get("N", 5)
    base_seed    = cfg.get("seed", config.get("seed", 42))
    system_cfgs  = cfg.get("systems", DEFAULT_CONFIGS)
    surrogates   = cfg.get("surrogates", DEFAULT_SURROGATES)
    vary_graph_across_reps = cfg.get("vary_graph_across_reps", False)
    seccm_cfg = dict(config.get("surrogate", {}))
    seccm_cfg.update(cfg.get("seccm_kwargs", {}))
    extra_seccm_kwargs = collect_seccm_kwargs(seccm_cfg)

    # Filter to available systems
    valid_systems = {k: v for k, v in system_cfgs.items() if k in SYSTEM_CLASSES}

    args_list = []
    for sys_name, scfg in valid_systems.items():
        T = scfg["T"]
        sys_kwargs = scfg.get("sys_kwargs", {})
        fixed_graph_seed = stable_seed(
            base_seed, "regime_boundaries", "graph", sys_name, N,
        )
        for coupling in scfg["couplings"]:
            for surr in surrogates:
                for rep in range(n_reps):
                    args_list.append(
                        (
                            sys_name,
                            coupling,
                            T,
                            surr,
                            n_surrogates,
                            N,
                            rep,
                            stable_seed(
                                base_seed,
                                "regime_boundaries",
                                "graph",
                                sys_name,
                                N,
                                rep,
                            ) if vary_graph_across_reps else fixed_graph_seed,
                            stable_seed(
                                base_seed,
                                "regime_boundaries",
                                "data",
                                sys_name,
                                N,
                                rep,
                            ),
                            stable_seed(
                                base_seed,
                                "regime_boundaries",
                                "seccm",
                                sys_name,
                                coupling,
                                surr,
                                rep,
                            ),
                            sys_kwargs,
                            extra_seccm_kwargs,
                        )
                    )

    n_total = len(args_list)
    print(f"\n  D2 Regime Boundaries: {n_total} runs "
          f"({len(valid_systems)} systems × varying couplings "
          f"× {len(surrogates)} surrogates × {n_reps} reps)")

    results = parallel_map(_worker, args_list, n_jobs=n_jobs,
                           desc="D2 Regime")

    rows   = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if errors:
        from collections import Counter
        err_sys = [e.get("system", "?") for e in errors]
        print(f"  {len(errors)} failures: {dict(Counter(err_sys))}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "regime_boundaries_raw.csv"), index=False)
    print(f"  Saved {len(df)} rows")

    if len(df) == 0:
        return df

    # Aggregate
    agg_cols = ["AUC_ROC_rho", "AUC_ROC_zscore", "AUC_ROC_delta_zscore",
                "rho_gap", "null_overlap", "frac_converging"]
    existing = [c for c in agg_cols if c in df.columns]
    agg = df.groupby(["system", "coupling", "surrogate"])[existing].agg(
        ["mean", "std", "sem"]
    ).reset_index()
    # Flatten multi-level columns
    agg.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0] for c in agg.columns
    ]
    agg.to_csv(os.path.join(output_dir, "regime_boundaries_agg.csv"), index=False)

    # Plot
    _plot_regime_boundaries(df, output_dir)

    return df


def _plot_regime_boundaries(df, output_dir):
    """Multi-panel: diagnostics vs coupling for each system."""
    systems = sorted(df["system"].unique())
    surrogates = sorted(df["surrogate"].unique())
    diag_metrics = ["AUC_ROC_delta_zscore", "rho_gap", "null_overlap",
                    "frac_converging"]
    metric_labels = ["ΔAUROC", "rho_gap", "null_overlap", "frac_converging"]

    available_metrics = [m for m in diag_metrics if m in df.columns]
    available_labels  = [metric_labels[i] for i, m in enumerate(diag_metrics)
                         if m in df.columns]

    n_metrics = len(available_metrics)
    if n_metrics == 0:
        return

    surr_colors = {"iaaft": "#3498db", "cycle_phase_A": "#e74c3c",
                   "cycle_phase_B": "#e67e22", "fft": "#2ecc71"}

    for sys_name in systems:
        sub = df[df["system"] == sys_name]
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4),
                                 squeeze=False)
        axes = axes[0]

        for ax, metric, label in zip(axes, available_metrics, available_labels):
            for surr in surrogates:
                ss = sub[sub["surrogate"] == surr]
                if len(ss) == 0:
                    continue
                agg = ss.groupby("coupling")[metric].agg(["mean", "sem"])
                color = surr_colors.get(surr, "#7f8c8d")
                ax.errorbar(agg.index, agg["mean"], yerr=agg["sem"],
                            marker="o", markersize=4, capsize=3,
                            label=surr, color=color, linewidth=1.5)

            ax.set_xlabel("Coupling strength", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            if label == "ΔAUROC":
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.legend(fontsize=7)

        fig.suptitle(f"D2: Regime Boundaries — {sys_name}", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"boundaries_{sys_name}.pdf"),
                    dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"boundaries_{sys_name}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Combined phase diagram: coupling vs ΔAUROC for all systems
    fig, ax = plt.subplots(figsize=(10, 6))
    sys_colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))
    markers_list = ["o", "s", "^", "D", "v", "<", ">"]

    for si, sys_name in enumerate(systems):
        # Use iaaft as the reference surrogate
        sub = df[(df["system"] == sys_name) & (df["surrogate"] == "iaaft")]
        if len(sub) == 0:
            sub = df[(df["system"] == sys_name) &
                     (df["surrogate"] == surrogates[0])]
        if len(sub) == 0:
            continue
        agg = sub.groupby("coupling")["AUC_ROC_delta_zscore"].agg(["mean", "sem"])
        ax.errorbar(agg.index, agg["mean"], yerr=agg["sem"],
                    marker=markers_list[si % len(markers_list)],
                    label=sys_name, color=sys_colors[si],
                    linewidth=1.5, markersize=6, capsize=3)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Coupling strength", fontsize=12)
    ax.set_ylabel("ΔAUROC (iaaft)", fontsize=12)
    ax.set_title("D2: Surrogate Utility vs Coupling Strength", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "phase_diagram.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "phase_diagram.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
