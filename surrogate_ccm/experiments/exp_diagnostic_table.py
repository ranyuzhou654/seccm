"""D1: Comprehensive Diagnostic Table.

Main result for the diagnostic framework paper.
For every system × surrogate combination, computes:
  - Raw AUROC (no surrogates), AUROC(z-score), ΔAUROC
  - AUPRC
  - rho_gap, surr_std, null_overlap
  - convergence_fraction
  - SSO, spectral_conc, acf_decay, perm_entropy
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
from ..surrogate.adaptive import spectral_concentration, autocorrelation_decay_time
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map


DEFAULT_SYSTEMS = {
    "logistic":         {"N": 5, "coupling": 0.3,  "T": 2000, "sys_kwargs": {}},
    "henon":            {"N": 5, "coupling": 0.2,  "T": 2000, "sys_kwargs": {}},
    "lorenz":           {"N": 5, "coupling": 0.5,  "T": 2000, "sys_kwargs": {"dt": 0.01}},
    "rossler":          {"N": 5, "coupling": 0.15, "T": 5000, "sys_kwargs": {"dt": 0.05}},
    "kuramoto":         {"N": 5, "coupling": 0.5,  "T": 2000, "sys_kwargs": {}},
    "van_der_pol":      {"N": 5, "coupling": 0.3,  "T": 5000, "sys_kwargs": {}},
    "fitzhugh_nagumo":  {"N": 5, "coupling": 0.3,  "T": 2000, "sys_kwargs": {}},
    "hindmarsh_rose":   {"N": 5, "coupling": 0.2,  "T": 10000,
                         "sys_kwargs": {"subsample": 5}},
}

DEFAULT_SURROGATES = [
    "fft", "aaft", "iaaft", "timeshift", "random_reorder",
    "cycle_shuffle", "twin", "phase", "small_shuffle",
    "truncated_fourier", "cycle_phase_A", "cycle_phase_B",
]


def _permutation_entropy(x, order=3, delay=1):
    """Bandt & Pompe permutation entropy, normalised to [0, 1]."""
    x = np.asarray(x, dtype=float).ravel()
    T = len(x)
    from math import factorial
    n_perms = factorial(order)
    counts = {}
    for i in range(T - (order - 1) * delay):
        pat = tuple(np.argsort(x[i:i + order * delay:delay]))
        counts[pat] = counts.get(pat, 0) + 1
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    return float(-np.sum(probs * np.log2(probs)) / np.log2(n_perms))


def _worker(args):
    """Worker: one system × surrogate × replicate."""
    (system_name, sys_cfg, surr_method, n_surrogates, seed) = args

    try:
        N = sys_cfg["N"]
        coupling = sys_cfg["coupling"]
        T = sys_cfg["T"]
        sys_kwargs = sys_cfg.get("sys_kwargs", {})

        adj = generate_network("ER", N, seed=seed, p=0.5)
        system = create_system(system_name, adj, coupling, **sys_kwargs)
        data = system.generate(T, transient=1000, seed=seed)

        # --- SE-CCM ---
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
            "surrogate": surr_method,
            "seed": seed,
            "AUC_ROC_rho":          metrics.get("AUC_ROC_rho", np.nan),
            "AUC_ROC_zscore":       metrics.get("AUC_ROC_zscore", np.nan),
            "AUC_ROC_delta_zscore": metrics.get("AUC_ROC_delta_zscore", np.nan),
            "AUC_PR_rho":           metrics.get("AUC_PR_rho", np.nan),
            "AUC_PR_zscore":        metrics.get("AUC_PR_zscore", np.nan),
            "TPR":                  metrics.get("TPR", np.nan),
            "FPR":                  metrics.get("FPR", np.nan),
            "F1":                   metrics.get("F1", np.nan),
        }

        # --- Surrogate distribution diagnostics ---
        mask = ~np.eye(N, dtype=bool)
        if seccm.surrogate_distributions_:
            rho_obs_vals, rho_surr_means, rho_surr_stds = [], [], []
            overlap_fracs = []
            for (i, j), rho_surr in seccm.surrogate_distributions_.items():
                obs = seccm.ccm_matrix_[i, j]
                rho_obs_vals.append(obs)
                rho_surr_means.append(np.mean(rho_surr))
                rho_surr_stds.append(np.std(rho_surr))
                overlap_fracs.append(np.mean(rho_surr >= obs))

            result["mean_rho_obs"]     = float(np.mean(rho_obs_vals))
            result["mean_rho_surr"]    = float(np.mean(rho_surr_means))
            result["rho_gap"]          = float(np.mean(rho_obs_vals) -
                                               np.mean(rho_surr_means))
            result["surr_std"]         = float(np.mean(rho_surr_stds))
            result["null_overlap"]     = float(np.mean(overlap_fracs))

        # --- Convergence ---
        if hasattr(seccm, "convergence_matrix_") and seccm.convergence_matrix_ is not None:
            conv_off = seccm.convergence_matrix_[mask]
            result["mean_convergence"]   = float(np.nanmean(conv_off))
            result["frac_converging"]    = float(np.mean(conv_off > 0))

        # --- SSO ---
        sso_vals = []
        for j in range(N):
            surrs_j = generate_surrogate(
                data[:, j], method=surr_method,
                n_surrogates=min(n_surrogates, 30),
                seed=seed + j + 2000,
            )
            sso_vals.append(compute_sso(data[:, j], surrs_j))
        result["SSO"] = float(np.mean(sso_vals))

        # --- Signal properties (system-level, same across surrogates) ---
        result["spectral_conc"] = float(np.mean(
            [spectral_concentration(data[:, i]) for i in range(N)]))
        result["acf_decay"] = float(np.mean(
            [autocorrelation_decay_time(data[:, i]) for i in range(N)]))
        result["perm_entropy"] = float(np.mean(
            [_permutation_entropy(data[:, i]) for i in range(N)]))

        return result

    except Exception as e:
        return {
            "system": system_name, "surrogate": surr_method,
            "seed": seed, "error": str(e),
        }


def run_diagnostic_table_experiment(config, output_dir="results/diagnostic_table",
                                     n_jobs=-1):
    """Run D1: Comprehensive Diagnostic Table."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("diagnostic_table", {})
    n_surrogates = cfg.get("n_surrogates", 100)
    n_reps       = cfg.get("n_reps", 20)
    systems      = cfg.get("systems", DEFAULT_SYSTEMS)
    surrogates   = cfg.get("surrogates", DEFAULT_SURROGATES)
    base_seed    = cfg.get("seed", 42)

    # Filter to available systems / surrogates
    valid_systems = {k: v for k, v in systems.items() if k in SYSTEM_CLASSES}
    valid_surrogates = [s for s in surrogates if s.lower() in SURROGATE_METHODS]
    skipped_sys  = set(systems) - set(valid_systems)
    skipped_surr = set(surrogates) - set(valid_surrogates)
    if skipped_sys:
        print(f"  Skipping unknown systems: {skipped_sys}")
    if skipped_surr:
        print(f"  Skipping unknown surrogates: {skipped_surr}")

    # Build arg list
    args_list = []
    for sys_name, sys_cfg in valid_systems.items():
        for surr in valid_surrogates:
            for rep in range(n_reps):
                seed = base_seed + hash((sys_name, surr, rep)) % (2**31)
                args_list.append((sys_name, sys_cfg, surr, n_surrogates, seed))

    n_total = len(args_list)
    print(f"\n  D1 Diagnostic Table: {n_total} runs "
          f"({len(valid_systems)} systems × {len(valid_surrogates)} surrogates "
          f"× {n_reps} reps)")

    results = parallel_map(_worker, args_list, n_jobs=n_jobs,
                           desc="D1 Diagnostic")

    rows   = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if errors:
        err_systems = [e.get("system", "?") for e in errors]
        from collections import Counter
        print(f"  {len(errors)} failures: {dict(Counter(err_systems))}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "full_diagnostics.csv"), index=False)
    print(f"  Saved {len(df)} rows to full_diagnostics.csv")

    if len(df) == 0:
        return df

    # ---- Aggregated table ----
    agg_cols = [
        "AUC_ROC_rho", "AUC_ROC_zscore", "AUC_ROC_delta_zscore",
        "AUC_PR_rho", "AUC_PR_zscore",
        "rho_gap", "surr_std", "null_overlap",
        "mean_convergence", "frac_converging",
        "SSO", "spectral_conc", "acf_decay", "perm_entropy",
    ]
    existing_cols = [c for c in agg_cols if c in df.columns]
    agg = df.groupby(["system", "surrogate"])[existing_cols].mean().reset_index()
    agg.to_csv(os.path.join(output_dir, "diagnostics_agg.csv"), index=False)

    # ---- Summary: per-system diagnostics (averaged across surrogates) ----
    sys_agg = df.groupby("system")[existing_cols].mean()
    print("\n  Per-system diagnostics (averaged across surrogates):")
    print(sys_agg.to_string(float_format="%.4f"))

    # ---- Regime classification ----
    _classify_regimes(agg, output_dir)

    # ---- Plots ----
    _plot_heatmap(agg, output_dir)
    _plot_regime_scatter(agg, output_dir)

    return df


def _classify_regimes(agg, output_dir):
    """Classify each system × surrogate combo into a regime."""
    regime_rows = []
    for _, row in agg.iterrows():
        rho_gap   = row.get("rho_gap", np.nan)
        overlap   = row.get("null_overlap", np.nan)
        conv      = row.get("frac_converging", np.nan)
        delta     = row.get("AUC_ROC_delta_zscore", np.nan)
        raw_auroc = row.get("AUC_ROC_rho", np.nan)

        if conv < 0.5:
            regime = "ccm_unreliable"
        elif overlap < 0.02 and delta < 0:
            regime = "surrogate_impermeable"
        elif delta > 0.02:
            regime = "surrogate_helps"
        elif delta < -0.02:
            regime = "surrogate_hurts"
        else:
            regime = "neutral"

        regime_rows.append({
            "system": row["system"],
            "surrogate": row["surrogate"],
            "regime": regime,
            "delta_auroc": delta,
            "rho_gap": rho_gap,
            "null_overlap": overlap,
            "frac_converging": conv,
        })

    df_regime = pd.DataFrame(regime_rows)
    df_regime.to_csv(os.path.join(output_dir, "regime_classification.csv"),
                     index=False)

    # Summary
    regime_counts = df_regime.groupby(["system", "regime"]).size().unstack(fill_value=0)
    print("\n  Regime classification counts (system × regime):")
    print(regime_counts.to_string())

    # Per-system dominant regime
    dominant = df_regime.groupby("system")["regime"].agg(
        lambda x: x.value_counts().index[0]
    )
    print("\n  Dominant regime per system:")
    for sys_name, regime in dominant.items():
        print(f"    {sys_name:20s}: {regime}")

    # Classification accuracy: can diagnostics predict ΔAUROC sign?
    df_regime["delta_sign"] = (df_regime["delta_auroc"] > 0).astype(int)
    df_regime["predicted_positive"] = df_regime["regime"].isin(
        ["surrogate_helps"]
    ).astype(int)
    valid = df_regime.dropna(subset=["delta_auroc"])
    if len(valid) > 0:
        accuracy = float(np.mean(
            valid["delta_sign"] == valid["predicted_positive"]
        ))
        print(f"\n  Regime classifier accuracy (predict ΔAUROC > 0): {accuracy:.3f}")

        summary = {
            "n_combos": int(len(valid)),
            "regime_counts": regime_counts.to_dict(),
            "dominant_regime": dominant.to_dict(),
            "classifier_accuracy": accuracy,
        }
        with open(os.path.join(output_dir, "regime_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)


def _plot_heatmap(agg, output_dir):
    """Heatmap: ΔAUROC (system × surrogate)."""
    pivot = agg.pivot_table(
        values="AUC_ROC_delta_zscore",
        index="system", columns="surrogate", aggfunc="mean",
    )

    # Order: systems by mean ΔAUROC, surrogates by mean ΔAUROC
    sys_order  = pivot.mean(axis=1).sort_values().index
    surr_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot.loc[sys_order, surr_order]

    fig, ax = plt.subplots(figsize=(14, 7))
    vabs = max(abs(pivot.values[np.isfinite(pivot.values)].min()),
               abs(pivot.values[np.isfinite(pivot.values)].max()), 0.15)
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vabs, vmax=vabs)

    ax.set_xticks(range(len(surr_order)))
    ax.set_xticklabels(surr_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(sys_order)))
    ax.set_yticklabels(sys_order, fontsize=10)

    # Annotate cells
    for i in range(len(sys_order)):
        for j in range(len(surr_order)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                color = "white" if abs(val) > vabs * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_xlabel("Surrogate Method", fontsize=12)
    ax.set_ylabel("Dynamical System", fontsize=12)
    ax.set_title("D1: ΔAUROC (z-score − raw ρ) across System × Surrogate",
                 fontsize=13)
    plt.colorbar(im, ax=ax, label="ΔAUROC", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heatmap_delta_auroc.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "heatmap_delta_auroc.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_regime_scatter(agg, output_dir):
    """Scatter: rho_gap vs null_overlap, coloured by ΔAUROC sign."""
    if "rho_gap" not in agg.columns or "null_overlap" not in agg.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    systems = sorted(agg["system"].unique())
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    delta = agg["AUC_ROC_delta_zscore"].values
    colors = np.where(delta > 0, "#2ecc71", "#e74c3c")
    sizes  = 30 + 200 * np.abs(delta)

    for i, sys_name in enumerate(systems):
        mask = agg["system"] == sys_name
        ax.scatter(
            agg.loc[mask, "rho_gap"],
            agg.loc[mask, "null_overlap"],
            c=colors[mask.values],
            marker=markers[i % len(markers)],
            s=sizes[mask.values],
            label=sys_name,
            alpha=0.7, edgecolors="black", linewidths=0.5,
        )

    ax.set_xlabel("rho_gap (mean ρ_obs − mean ρ_surr)", fontsize=12)
    ax.set_ylabel("null_overlap (frac surr ρ ≥ obs ρ)", fontsize=12)
    ax.set_title("Regime Scatter: Green = surrogate helps, Red = surrogate hurts",
                 fontsize=12)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add regime region annotations
    ax.axhline(y=0.02, color="gray", linestyle=":", alpha=0.5)
    ax.text(0.6, 0.01, "Surrogate-impermeable zone", fontsize=8,
            color="gray", style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "regime_scatter.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "regime_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
