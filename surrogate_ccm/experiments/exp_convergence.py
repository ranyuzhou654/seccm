"""E5: Convergence Behavior.

Verifies that cycle-phase z-scores improve monotonically with sample size T.
Plots ρ(L) convergence curves and AUROC vs T.
"""

import json
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
    "rossler":     {"coupling": 0.15, "T_values": [500, 1000, 2000, 5000, 10000],
                    "sys_kwargs": {"dt": 0.05}},
    "kuramoto":    {"coupling": 0.5,  "T_values": [500, 1000, 2000, 5000, 10000],
                    "sys_kwargs": {}},
    "van_der_pol": {"coupling": 0.3,  "T_values": [500, 1000, 2000, 5000, 10000],
                    "sys_kwargs": {}},
}

DEFAULT_SURROGATES = ["iaaft", "cycle_phase_A"]


def _worker(args):
    """Worker: one system × T × surrogate × replicate."""
    (system_name, coupling, T, surr_method, n_surrogates,
     N, seed, sys_kwargs) = args

    try:
        adj = generate_network("ER", N, seed=seed, p=0.5)
        system = create_system(system_name, adj, coupling, **sys_kwargs)
        data = system.generate(T, transient=1000, seed=seed)

        seccm = SECCM(
            surrogate_method=surr_method,
            n_surrogates=n_surrogates,
            alpha=0.05, fdr=True,
            seed=seed, verbose=False,
        )
        seccm.fit(data)
        metrics = seccm.score(adj)

        result = {
            "system": system_name,
            "coupling": coupling,
            "T": T,
            "surrogate": surr_method,
            "seed": seed,
            "AUC_ROC_rho":          metrics.get("AUC_ROC_rho", np.nan),
            "AUC_ROC_zscore":       metrics.get("AUC_ROC_zscore", np.nan),
            "AUC_ROC_delta_zscore": metrics.get("AUC_ROC_delta_zscore", np.nan),
            "AUC_PR_zscore":        metrics.get("AUC_PR_zscore", np.nan),
            "TPR":                  metrics.get("TPR", np.nan),
            "FPR":                  metrics.get("FPR", np.nan),
        }

        # Mean z-score for true and false pairs
        mask = ~np.eye(N, dtype=bool)
        y_true = adj[mask].ravel().astype(int)
        z_all  = seccm.zscore_matrix_[mask].ravel()
        z_all  = np.nan_to_num(z_all, nan=0.0)

        if y_true.sum() > 0:
            result["mean_z_true"]  = float(np.mean(z_all[y_true == 1]))
        if (1 - y_true).sum() > 0:
            result["mean_z_false"] = float(np.mean(z_all[y_true == 0]))

        return result

    except Exception as e:
        return {
            "system": system_name, "coupling": coupling,
            "T": T, "surrogate": surr_method,
            "seed": seed, "error": str(e),
        }


def run_convergence_experiment(config, output_dir="results/convergence",
                                n_jobs=-1):
    """Run E5: Convergence Behavior."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("convergence", {})
    n_surrogates = cfg.get("n_surrogates", 100)
    n_reps       = cfg.get("n_reps", 10)
    N            = cfg.get("N", 5)
    base_seed    = cfg.get("seed", 42)
    system_cfgs  = cfg.get("systems", DEFAULT_SYSTEMS)
    surrogates   = cfg.get("surrogates", DEFAULT_SURROGATES)

    valid_systems = {k: v for k, v in system_cfgs.items() if k in SYSTEM_CLASSES}

    args_list = []
    for sys_name, scfg in valid_systems.items():
        coupling   = scfg["coupling"]
        sys_kwargs = scfg.get("sys_kwargs", {})
        for T in scfg["T_values"]:
            for surr in surrogates:
                for rep in range(n_reps):
                    seed = base_seed + hash(
                        ("e5", sys_name, T, surr, rep)
                    ) % (2**31)
                    args_list.append(
                        (sys_name, coupling, T, surr, n_surrogates,
                         N, seed, sys_kwargs)
                    )

    n_total = len(args_list)
    print(f"\n  E5 Convergence: {n_total} runs "
          f"({len(valid_systems)} systems × T-values "
          f"× {len(surrogates)} surrogates × {n_reps} reps)")

    results = parallel_map(_worker, args_list, n_jobs=n_jobs,
                           desc="E5 Convergence")

    rows   = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if errors:
        from collections import Counter
        print(f"  {len(errors)} failures: "
              f"{dict(Counter(e.get('system','?') for e in errors))}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "convergence_raw.csv"), index=False)
    print(f"  Saved {len(df)} rows")

    if len(df) == 0:
        return df

    # Check monotonic improvement
    _check_convergence(df, output_dir)

    # Plot
    _plot_convergence(df, output_dir)

    return df


def _check_convergence(df, output_dir):
    """Check if AUROC improves monotonically with T."""
    results = {}
    for sys_name in sorted(df["system"].unique()):
        results[sys_name] = {}
        for surr in sorted(df["surrogate"].unique()):
            sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
            agg = sub.groupby("T")["AUC_ROC_zscore"].mean().sort_index()
            if len(agg) < 3:
                continue
            # Spearman correlation between T and AUROC
            r, p = (np.nan, np.nan)
            try:
                from scipy.stats import spearmanr
                r, p = spearmanr(agg.index, agg.values)
            except Exception:
                pass

            is_monotonic = bool(all(
                agg.values[i] <= agg.values[i+1]
                for i in range(len(agg)-1)
            ))

            results[sys_name][surr] = {
                "spearman_r": float(r) if np.isfinite(r) else None,
                "spearman_p": float(p) if np.isfinite(p) else None,
                "is_monotonic": is_monotonic,
                "auroc_values": {int(t): float(v) for t, v in agg.items()},
            }

    print("\n  Convergence check (AUROC vs T):")
    for sys_name, surr_dict in results.items():
        for surr, info in surr_dict.items():
            mono = "MONO" if info["is_monotonic"] else "non-mono"
            r = info["spearman_r"]
            r_str = f"{r:+.3f}" if r is not None else "N/A"
            print(f"    {sys_name:15s} × {surr:15s}: "
                  f"ρ(T,AUROC) = {r_str}  [{mono}]")

    with open(os.path.join(output_dir, "convergence_check.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


def _plot_convergence(df, output_dir):
    """Plot AUROC and ΔAUROC vs T for each system."""
    systems    = sorted(df["system"].unique())
    surrogates = sorted(df["surrogate"].unique())
    surr_colors = {"iaaft": "#3498db", "cycle_phase_A": "#e74c3c",
                   "cycle_phase_B": "#e67e22", "fft": "#2ecc71"}

    # --- AUROC(z-score) vs T ---
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]

    for ax, sys_name in zip(axes, systems):
        for surr in surrogates:
            sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("T")["AUC_ROC_zscore"].agg(["mean", "std"])
            color = surr_colors.get(surr, "#7f8c8d")
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                        marker="o", label=surr, color=color,
                        capsize=3, linewidth=1.5)

        # Also plot raw AUROC (no surrogates)
        sub_any = df[df["system"] == sys_name]
        agg_raw = sub_any.groupby("T")["AUC_ROC_rho"].mean()
        ax.plot(agg_raw.index, agg_raw.values, "k--", label="raw ρ",
                linewidth=1, alpha=0.7)

        ax.set_xlabel("Sample size T")
        ax.set_ylabel("AUROC")
        ax.set_title(sys_name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    fig.suptitle("E5: AUROC Convergence with Sample Size", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auroc_vs_T.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "auroc_vs_T.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- ΔAUROC vs T ---
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]

    for ax, sys_name in zip(axes, systems):
        for surr in surrogates:
            sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("T")["AUC_ROC_delta_zscore"].agg(["mean", "std"])
            color = surr_colors.get(surr, "#7f8c8d")
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                        marker="o", label=surr, color=color,
                        capsize=3, linewidth=1.5)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Sample size T")
        ax.set_ylabel("ΔAUROC")
        ax.set_title(sys_name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    fig.suptitle("E5: ΔAUROC Convergence with Sample Size", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "delta_auroc_vs_T.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "delta_auroc_vs_T.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Mean z-score (true vs false pairs) vs T ---
    if "mean_z_true" in df.columns and "mean_z_false" in df.columns:
        fig, axes = plt.subplots(1, len(systems),
                                 figsize=(5 * len(systems), 4), squeeze=False)
        axes = axes[0]
        for ax, sys_name in zip(axes, systems):
            for surr in surrogates:
                sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
                if len(sub) == 0:
                    continue
                agg_true  = sub.groupby("T")["mean_z_true"].mean()
                agg_false = sub.groupby("T")["mean_z_false"].mean()
                color = surr_colors.get(surr, "#7f8c8d")
                ax.plot(agg_true.index, agg_true.values,
                        marker="o", color=color, label=f"{surr} (true)",
                        linewidth=1.5)
                ax.plot(agg_false.index, agg_false.values,
                        marker="x", color=color, label=f"{surr} (false)",
                        linewidth=1, linestyle="--", alpha=0.7)

            ax.set_xlabel("Sample size T")
            ax.set_ylabel("Mean z-score")
            ax.set_title(sys_name)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

        fig.suptitle("E5: Z-score Separation vs Sample Size", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "zscore_separation_vs_T.pdf"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
