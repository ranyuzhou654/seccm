"""Trivariate causal structure experiment.

Tests whether surrogate-enhanced CCM can distinguish three classic
3-node causal structures when testing the X→Y direction:

- **chain**   : X→Z→Y  (indirect cause)
- **direct**  : X→Y, Z independent  (direct cause)
- **confound**: Z→X, Z→Y  (common cause / confounding)

Adjacency convention: A[i,j]=1 means j→i  (X=0, Y=1, Z=2).
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..generators import create_system
from ._config_helpers import get_system_kwargs
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map

# ── Causal structures (A[i,j]=1 ⇔ j→i) ────────────────────────
STRUCTURES = {
    "chain": np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]),  # X→Z→Y
    "direct": np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]),  # X→Y, Z independent
    "confound": np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0],
    ]),  # Z→X, Z→Y
}

# ── Display helpers ─────────────────────────────────────────────
SYSTEM_DISPLAY = {
    "logistic": "Logistic",
    "lorenz": "Lorenz",
    "henon": "Hénon",
    "rossler": "Rössler",
    "hindmarsh_rose": "Hindmarsh–Rose",
    "fitzhugh_nagumo": "FitzHugh–Nagumo",
    "kuramoto": "Kuramoto",
}
METHOD_DISPLAY = {
    "fft": "FFT",
    "aaft": "AAFT",
    "iaaft": "iAAFT",
    "timeshift": "Timeshift",
    "random_reorder": "Rand. reorder",
    "cycle_shuffle": "Cycle shuffle",
    "twin": "Twin",
    "phase": "Phase",
    "small_shuffle": "Small shuffle",
    "truncated_fourier": "Trunc. Fourier",
    "auto": "Adaptive",
}
STRUCTURE_DISPLAY = {
    "chain": "Chain (X→Z→Y)",
    "direct": "Direct (X→Y)",
    "confound": "Confound (Z→X,Y)",
}


def _pub_rcparams():
    return {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    }


# ── Worker ──────────────────────────────────────────────────────
def _run_single(args):
    """Run one rep for a (structure, system, method) combination."""
    (structure_name, system_name, method, coupling, T, transient,
     n_surrogates, seed, sys_kwargs) = args
    try:
        adj = STRUCTURES[structure_name]
        system = create_system(system_name, adj, coupling, **sys_kwargs)
        data = system.generate(T, transient=transient, seed=seed)

        if not np.all(np.isfinite(data)):
            raise RuntimeError("Non-finite data")

        seccm = SECCM(
            surrogate_method=method,
            n_surrogates=n_surrogates,
            alpha=0.05,
            fdr=False,
            seed=seed,
            verbose=False,
        )
        seccm.fit(data)

        # X→Y direction: effect=Y(1), cause=X(0) → matrix[1,0]
        # Y→X direction: effect=X(0), cause=Y(1) → matrix[0,1]
        return {
            "structure": structure_name,
            "system": system_name,
            "method": method,
            "seed": seed,
            "rho_xy": float(seccm.ccm_matrix_[1, 0]),
            "pvalue_xy": float(seccm.pvalue_matrix_[1, 0]),
            "zscore_xy": float(seccm.zscore_matrix_[1, 0]),
            "detected_xy": int(seccm.detected_[1, 0]),
            "rho_yx": float(seccm.ccm_matrix_[0, 1]),
            "pvalue_yx": float(seccm.pvalue_matrix_[0, 1]),
            "zscore_yx": float(seccm.zscore_matrix_[0, 1]),
            "detected_yx": int(seccm.detected_[0, 1]),
        }
    except Exception as exc:
        warnings.warn(f"Failed: {structure_name}/{system_name}/{method} "
                      f"seed={seed}: {exc}")
        return None


# ── Main entry point ────────────────────────────────────────────
def run_trivariate_experiment(config, output_dir="results/trivariate",
                              n_jobs=-1):
    """Run the trivariate causal structure experiment."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("trivariate", {})
    ts_cfg = config.get("time_series", {})
    seed_base = config.get("seed", 42)

    systems = cfg.get("systems", ["logistic", "lorenz", "henon", "rossler"])
    methods = cfg.get("methods", ["fft", "iaaft", "cycle_shuffle", "twin",
                                   "phase"])
    n_surrogates = cfg.get("n_surrogates", 99)
    n_reps = cfg.get("n_reps", 20)
    coupling_map = cfg.get("coupling", {})
    T = ts_cfg.get("T", 3000)
    transient = ts_cfg.get("transient", 1000)

    structures = list(STRUCTURES.keys())

    # Build args list
    args_list = []
    sys_kwargs_map = {sys: get_system_kwargs(config, sys) for sys in systems}
    for struct in structures:
        for sys_name in systems:
            coupling = coupling_map.get(sys_name, 0.1)
            for method in methods:
                for rep in range(n_reps):
                    seed = seed_base + rep
                    args_list.append((
                        struct, sys_name, method, coupling,
                        T, transient, n_surrogates, seed,
                        sys_kwargs_map.get(sys_name, {}),
                    ))

    total = len(args_list)
    print(f"  Trivariate: {len(structures)} structures × {len(systems)} "
          f"systems × {len(methods)} methods × {n_reps} reps = {total} runs")

    results = parallel_map(_run_single, args_list, n_jobs=n_jobs,
                           desc="Trivariate experiment")

    # Collect valid results
    rows = [r for r in results if r is not None]
    n_failed = total - len(rows)
    if n_failed:
        print(f"  Warning: {n_failed}/{total} runs failed")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "trivariate_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} rows → {csv_path}")

    # ── Plots ───────────────────────────────────────────────────
    if len(df) > 0:
        _plot_bar_charts(df, output_dir)
        _plot_detection_heatmap(df, output_dir)

    return df


# ── Plotting ────────────────────────────────────────────────────
def _plot_bar_charts(df, output_dir):
    """Grouped bar chart: X→Y rho and zscore by structure, one panel per system."""
    with plt.rc_context(_pub_rcparams()):
        for metric, label in [("rho_xy", "CCM ρ (X→Y)"),
                              ("zscore_xy", "Z-score (X→Y)")]:
            systems = df["system"].unique()
            n_sys = len(systems)
            fig, axes = plt.subplots(1, n_sys, figsize=(4 * n_sys, 3.5),
                                     sharey=True, squeeze=False)
            axes = axes.ravel()

            for ax, sys_name in zip(axes, systems):
                sub = df[df["system"] == sys_name]
                sns.barplot(
                    data=sub, x="structure", y=metric, hue="method",
                    ax=ax, errorbar="se", order=["direct", "chain", "confound"],
                )
                ax.set_title(SYSTEM_DISPLAY.get(sys_name, sys_name))
                ax.set_xlabel("")
                ax.set_ylabel(label if ax == axes[0] else "")
                ax.set_xticklabels([STRUCTURE_DISPLAY.get(s, s)
                                    for s in ["direct", "chain", "confound"]],
                                   rotation=20, ha="right")
                if ax != axes[-1]:
                    ax.get_legend().remove()
                else:
                    ax.legend(title="Method", bbox_to_anchor=(1.02, 1),
                              loc="upper left", fontsize=7)

            fig.tight_layout()
            stem = f"bar_{metric}"
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(output_dir, f"{stem}.{ext}"),
                            dpi=300, bbox_inches="tight")
            plt.close(fig)


def _plot_detection_heatmap(df, output_dir):
    """Detection rate heatmap: structure × method, one panel per system."""
    with plt.rc_context(_pub_rcparams()):
        systems = df["system"].unique()
        n_sys = len(systems)
        fig, axes = plt.subplots(1, n_sys, figsize=(4.5 * n_sys, 3.5),
                                 squeeze=False)
        axes = axes.ravel()

        struct_order = ["direct", "chain", "confound"]
        for ax, sys_name in zip(axes, systems):
            sub = df[df["system"] == sys_name]
            pivot = sub.pivot_table(
                values="detected_xy", index="structure", columns="method",
                aggfunc="mean",
            )
            # Reorder rows
            pivot = pivot.reindex(
                [s for s in struct_order if s in pivot.index])
            # Rename for display
            pivot.index = [STRUCTURE_DISPLAY.get(s, s) for s in pivot.index]
            pivot.columns = [METHOD_DISPLAY.get(m, m) for m in pivot.columns]

            sns.heatmap(
                pivot, ax=ax, vmin=0, vmax=1, cmap="RdYlGn",
                annot=True, fmt=".2f", linewidths=0.5,
                cbar_kws={"label": "Detection rate"},
            )
            ax.set_title(SYSTEM_DISPLAY.get(sys_name, sys_name))
            ax.set_ylabel("" if ax != axes[0] else "Structure")
            ax.set_xlabel("Method")

        fig.tight_layout()
        stem = "heatmap_detected_xy"
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(output_dir, f"{stem}.{ext}"),
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
