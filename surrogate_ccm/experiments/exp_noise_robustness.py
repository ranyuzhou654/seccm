"""E7: Noise Robustness.

Tests cycle detection quality and ΔAUROC under observation noise.
Also monitors number of cycles detected and cycle length variance.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..generators import SYSTEM_CLASSES, create_system, generate_network
from ..surrogate import generate_surrogate
from ..surrogate.cycle_phase_surrogate import (
    _identify_cycles_hilbert, _identify_cycles_peaks,
)
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map


DEFAULT_SYSTEMS = {
    "van_der_pol": {"coupling": 0.3, "T": 5000, "sys_kwargs": {}},
    "kuramoto":    {"coupling": 0.5, "T": 2000, "sys_kwargs": {}},
    "rossler":     {"coupling": 0.15, "T": 5000, "sys_kwargs": {"dt": 0.05}},
}

DEFAULT_NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
DEFAULT_SURROGATES   = ["iaaft", "cycle_phase_A"]


def _worker(args):
    """Worker: one system × noise_sigma × surrogate × replicate."""
    (system_name, coupling, T, noise_sigma, surr_method,
     n_surrogates, N, seed, sys_kwargs) = args

    try:
        adj = generate_network("ER", N, seed=seed, p=0.5)
        system = create_system(system_name, adj, coupling, **sys_kwargs)
        data_clean = system.generate(T, transient=1000, seed=seed)

        # Add observation noise
        rng = np.random.default_rng(seed + 9999)
        data_std = data_clean.std(axis=0, keepdims=True) + 1e-12
        noise = rng.normal(0, noise_sigma, size=data_clean.shape) * data_std
        data = data_clean + noise

        # SE-CCM
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
            "noise_sigma": noise_sigma,
            "surrogate": surr_method,
            "seed": seed,
            "AUC_ROC_rho":          metrics.get("AUC_ROC_rho", np.nan),
            "AUC_ROC_zscore":       metrics.get("AUC_ROC_zscore", np.nan),
            "AUC_ROC_delta_zscore": metrics.get("AUC_ROC_delta_zscore", np.nan),
            "AUC_PR_zscore":        metrics.get("AUC_PR_zscore", np.nan),
            "TPR":                  metrics.get("TPR", np.nan),
            "FPR":                  metrics.get("FPR", np.nan),
        }

        # Cycle detection quality on noisy data
        n_cycles_list = []
        cycle_len_cv_list = []  # coefficient of variation
        for col in range(N):
            x = data[:, col]
            # Try Hilbert
            boundaries = _identify_cycles_hilbert(x)
            if boundaries is None or len(boundaries) < 3:
                boundaries = _identify_cycles_peaks(x)
            if boundaries is None:
                boundaries = np.array([])

            n_cyc = max(0, len(boundaries) - 1)
            n_cycles_list.append(n_cyc)

            if n_cyc >= 2:
                lens = np.diff(boundaries)
                cv = float(np.std(lens) / (np.mean(lens) + 1e-12))
                cycle_len_cv_list.append(cv)

        result["mean_n_cycles"]    = float(np.mean(n_cycles_list))
        result["min_n_cycles"]     = int(np.min(n_cycles_list))
        result["cycle_len_cv"]     = (float(np.mean(cycle_len_cv_list))
                                      if cycle_len_cv_list else np.nan)
        result["fallback_frac"]    = float(np.mean(
            [1 if nc < 3 else 0 for nc in n_cycles_list]
        ))

        return result

    except Exception as e:
        return {
            "system": system_name, "noise_sigma": noise_sigma,
            "surrogate": surr_method, "seed": seed, "error": str(e),
        }


def run_noise_robustness_experiment(config, output_dir="results/noise_robustness",
                                     n_jobs=-1):
    """Run E7: Noise Robustness."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("noise_robustness", {})
    n_surrogates  = cfg.get("n_surrogates", 100)
    n_reps        = cfg.get("n_reps", 10)
    N             = cfg.get("N", 5)
    base_seed     = cfg.get("seed", 42)
    system_cfgs   = cfg.get("systems", DEFAULT_SYSTEMS)
    noise_levels  = cfg.get("noise_levels", DEFAULT_NOISE_LEVELS)
    surrogates    = cfg.get("surrogates", DEFAULT_SURROGATES)

    valid_systems = {k: v for k, v in system_cfgs.items() if k in SYSTEM_CLASSES}

    args_list = []
    for sys_name, scfg in valid_systems.items():
        coupling   = scfg["coupling"]
        T          = scfg["T"]
        sys_kwargs = scfg.get("sys_kwargs", {})
        for sigma in noise_levels:
            for surr in surrogates:
                for rep in range(n_reps):
                    seed = base_seed + hash(
                        ("e7", sys_name, sigma, surr, rep)
                    ) % (2**31)
                    args_list.append(
                        (sys_name, coupling, T, sigma, surr,
                         n_surrogates, N, seed, sys_kwargs)
                    )

    n_total = len(args_list)
    print(f"\n  E7 Noise Robustness: {n_total} runs "
          f"({len(valid_systems)} systems × {len(noise_levels)} noise levels "
          f"× {len(surrogates)} surrogates × {n_reps} reps)")

    results = parallel_map(_worker, args_list, n_jobs=n_jobs,
                           desc="E7 Noise")

    rows   = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if errors:
        from collections import Counter
        print(f"  {len(errors)} failures: "
              f"{dict(Counter(e.get('system','?') for e in errors))}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "noise_robustness_raw.csv"), index=False)
    print(f"  Saved {len(df)} rows")

    if len(df) == 0:
        return df

    # Summary
    _summarize_noise(df, output_dir)

    # Plots
    _plot_noise(df, output_dir)

    return df


def _summarize_noise(df, output_dir):
    """Summarise noise robustness results."""
    summary = {}
    for sys_name in sorted(df["system"].unique()):
        summary[sys_name] = {}
        for surr in sorted(df["surrogate"].unique()):
            sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("noise_sigma").agg({
                "AUC_ROC_delta_zscore": "mean",
                "mean_n_cycles": "mean",
                "fallback_frac": "mean",
                "cycle_len_cv": "mean",
            })
            summary[sys_name][surr] = {
                str(sigma): {
                    "delta_auroc":   float(row["AUC_ROC_delta_zscore"]),
                    "mean_n_cycles": float(row["mean_n_cycles"]),
                    "fallback_frac": float(row["fallback_frac"]),
                    "cycle_len_cv":  float(row["cycle_len_cv"])
                                     if np.isfinite(row["cycle_len_cv"]) else None,
                }
                for sigma, row in agg.iterrows()
            }

    with open(os.path.join(output_dir, "noise_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print key results
    print("\n  Noise robustness summary:")
    for sys_name in sorted(df["system"].unique()):
        sub = df[(df["system"] == sys_name) & (df["surrogate"] == "cycle_phase_A")]
        if len(sub) == 0:
            continue
        agg = sub.groupby("noise_sigma")[
            ["AUC_ROC_delta_zscore", "mean_n_cycles", "fallback_frac"]
        ].mean()
        print(f"\n  {sys_name} (cycle_phase_A):")
        for sigma, row in agg.iterrows():
            delta = row["AUC_ROC_delta_zscore"]
            n_cyc = row["mean_n_cycles"]
            fb    = row["fallback_frac"]
            print(f"    σ={sigma:.2f}: ΔAUROC={delta:+.4f}  "
                  f"cycles={n_cyc:.1f}  fallback={fb:.1%}")


def _plot_noise(df, output_dir):
    """Plot noise robustness curves."""
    systems    = sorted(df["system"].unique())
    surrogates = sorted(df["surrogate"].unique())
    surr_colors = {"iaaft": "#3498db", "cycle_phase_A": "#e74c3c",
                   "cycle_phase_B": "#e67e22", "fft": "#2ecc71"}

    # --- ΔAUROC vs noise ---
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4),
                             squeeze=False)
    axes = axes[0]

    for ax, sys_name in zip(axes, systems):
        for surr in surrogates:
            sub = df[(df["system"] == sys_name) & (df["surrogate"] == surr)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("noise_sigma")["AUC_ROC_delta_zscore"].agg(
                ["mean", "std"]
            )
            color = surr_colors.get(surr, "#7f8c8d")
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                        marker="o", label=surr, color=color,
                        capsize=3, linewidth=1.5)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Noise σ (relative to signal std)", fontsize=10)
        ax.set_ylabel("ΔAUROC", fontsize=10)
        ax.set_title(sys_name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("E7: ΔAUROC vs Observation Noise", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auroc_vs_noise.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "auroc_vs_noise.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Cycle detection quality vs noise ---
    if "mean_n_cycles" in df.columns:
        fig, axes = plt.subplots(1, len(systems),
                                 figsize=(5 * len(systems), 4), squeeze=False)
        axes = axes[0]

        for ax, sys_name in zip(axes, systems):
            # Only plot for cycle_phase_A
            sub = df[(df["system"] == sys_name) &
                     (df["surrogate"] == "cycle_phase_A")]
            if len(sub) == 0:
                continue
            agg = sub.groupby("noise_sigma").agg({
                "mean_n_cycles": ["mean", "std"],
                "fallback_frac": "mean",
            })

            ax.errorbar(
                agg.index,
                agg[("mean_n_cycles", "mean")],
                yerr=agg[("mean_n_cycles", "std")],
                marker="o", color="#e74c3c", capsize=3, linewidth=1.5,
                label="n_cycles",
            )

            ax2 = ax.twinx()
            ax2.plot(agg.index, agg[("fallback_frac", "mean")],
                     "s--", color="#f39c12", label="fallback frac",
                     linewidth=1.5)
            ax2.set_ylabel("Fallback fraction", fontsize=10, color="#f39c12")
            ax2.set_ylim(-0.05, 1.05)

            ax.set_xlabel("Noise σ", fontsize=10)
            ax.set_ylabel("Mean cycles detected", fontsize=10, color="#e74c3c")
            ax.set_title(sys_name)
            ax.grid(True, alpha=0.3)

            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

        fig.suptitle("E7: Cycle Detection Quality vs Noise", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "cycle_detection_quality.pdf"),
                    dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "cycle_detection_quality.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
