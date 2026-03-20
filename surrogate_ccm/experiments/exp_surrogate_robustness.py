"""Experiment: surrogate robustness across T, coupling, obs-noise, dyn-noise.

Four sub-experiments sweep one factor at a time, measuring how each affects
the benefit (or harm) that surrogate-based significance testing adds over
raw CCM cross-map correlation.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ..generators import create_system, generate_network
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map


# ── Metrics ──────────────────────────────────────────────────────────
AUC_KEYS = [
    "AUC_ROC_rho",
    "AUC_ROC_surrogate",
    "AUC_ROC_zscore",
    "AUC_ROC_delta",
    "AUC_ROC_delta_zscore",
]
COLLECT_KEYS = AUC_KEYS + ["TPR", "FPR"]

# ── Display names ────────────────────────────────────────────────────
SYSTEM_DISPLAY = {
    "logistic": "Logistic",
    "lorenz": "Lorenz",
    "henon": "H\u00e9non",
    "rossler": "R\u00f6ssler",
    "hindmarsh_rose": "Hindmarsh\u2013Rose",
    "fitzhugh_nagumo": "FitzHugh\u2013Nagumo",
    "kuramoto": "Kuramoto",
}

METHOD_DISPLAY = {
    "fft": "FFT",
    "aaft": "AAFT",
    "timeshift": "Time-shift",
    "iaaft": "iAAFT",
    "random_reorder": "Random reorder",
    "cycle_shuffle": "Cycle shuffle",
    "twin": "Twin",
    "phase": "Phase",
    "small_shuffle": "Small shuffle",
    "truncated_fourier": "Trunc. Fourier",
    "auto": "Adaptive",
}

SWEEP_LABEL = {
    "T": r"Time series length $T$",
    "coupling": r"Coupling strength $\varepsilon$",
    "noise_std": r"Observation noise $\sigma_{\mathrm{obs}}$",
    "dyn_noise_std": r"Dynamical noise $\sigma_{\mathrm{dyn}}$",
}

METRIC_LABEL = {
    "AUC_ROC_rho": r"AUROC (raw $\rho$)",
    "AUC_ROC_surrogate": r"AUROC (surrogate $p$-value)",
    "AUC_ROC_zscore": r"AUROC ($z$-score)",
    "AUC_ROC_delta": r"$\Delta$AUROC (surrogate $-$ raw)",
    "AUC_ROC_delta_zscore": r"$\Delta$AUROC ($z$-score $-$ raw)",
}

# Color-blind safe palette (Okabe & Ito + greys)
METHOD_COLORS = {
    "fft": "#0072B2",       # blue
    "aaft": "#D55E00",      # vermillion
    "timeshift": "#009E73",  # bluish green
    "iaaft": "#CC79A7",     # reddish purple
    "random_reorder": "#F0E442",  # yellow
    "cycle_shuffle": "#56B4E9",   # sky blue
    "twin": "#E69F00",            # orange
    "phase": "#000000",           # black
    "small_shuffle": "#8C564B",   # brown
    "truncated_fourier": "#17BECF",  # cyan
    "auto": "#7F7F7F",           # grey
}
BASELINE_COLOR = "#555555"

METHOD_MARKERS = {
    "fft": "o",
    "aaft": "s",
    "timeshift": "^",
    "iaaft": "D",
    "random_reorder": "v",
    "cycle_shuffle": "P",
    "twin": "X",
    "phase": "*",
    "small_shuffle": "h",
    "truncated_fourier": "p",
    "auto": "d",
}


# ── Publication rcParams ─────────────────────────────────────────────
def _pub_rcparams():
    """Return matplotlib rcParams dict for publication-quality figures."""
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.7",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.3,
        "lines.markersize": 5,
        "errorbar.capsize": 2.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


# ── Worker ───────────────────────────────────────────────────────────
def _run_single_rep(args):
    """Run one repetition for a given parameter configuration."""
    (system_name, topology, N, eps, method, n_surr,
     T, transient, noise_std, dyn_noise_std, seed, net_kwargs, fdr,
     extra_seccm_kwargs) = args

    try:
        adj = generate_network(topology, N, seed=seed, **net_kwargs)
        system = create_system(system_name, adj, eps)
        data = system.generate(T, transient=transient, seed=seed,
                               noise_std=noise_std, dyn_noise_std=dyn_noise_std)

        if not np.all(np.isfinite(data)):
            raise RuntimeError(f"{system_name} produced non-finite data.")

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
        return seccm.score(adj)
    except Exception:
        return None


# ── Generic sweep runner ─────────────────────────────────────────────
def _run_sweep(sweep_name, sweep_param, sweep_values_map, systems, methods,
               fixed_params, config, n_jobs):
    """Run a parameter sweep and return a DataFrame of results.

    Parameters
    ----------
    sweep_name : str
        Name for progress bars and logging.
    sweep_param : str
        Column name for the swept parameter in the output DataFrame.
    sweep_values_map : dict or list
        If dict: {system_name: [values]}. If list: shared across all systems.
    systems : list of str
    methods : list of str
    fixed_params : dict
        Keys: N, topology, coupling (dict), n_surr, T, transient,
              noise_std, dyn_noise_std, er_p, n_reps, fdr, seed_base.
    config : dict
        Full config (unused here but available for extension).
    n_jobs : int
    """
    N = fixed_params["N"]
    topology = fixed_params["topology"]
    coupling_map = fixed_params["coupling"]
    n_surr = fixed_params["n_surr"]
    T = fixed_params["T"]
    transient = fixed_params["transient"]
    noise_std = fixed_params["noise_std"]
    dyn_noise_std = fixed_params["dyn_noise_std"]
    er_p = fixed_params["er_p"]
    n_reps = fixed_params["n_reps"]
    fdr = fixed_params["fdr"]
    seed_base = fixed_params["seed_base"]
    extra_seccm_kwargs = fixed_params.get("extra_seccm_kwargs", {})

    net_kwargs = {"p": er_p}

    # Build combos
    combos = []
    for sys_name in systems:
        if isinstance(sweep_values_map, dict):
            values = sweep_values_map.get(sys_name, [])
        else:
            values = sweep_values_map
        for val in values:
            for method in methods:
                combos.append((sys_name, method, val))

    rows = []
    pbar = tqdm(combos, desc=sweep_name)

    for sys_name, method, sweep_val in pbar:
        pbar.set_postfix_str(f"{sys_name}/{method}/{sweep_param}={sweep_val}")

        # Determine per-run parameters based on which parameter is swept
        eps = coupling_map.get(sys_name, 0.1)
        run_T = T
        run_noise = noise_std
        run_dyn_noise = dyn_noise_std

        if sweep_param == "T":
            run_T = sweep_val
        elif sweep_param == "coupling":
            eps = sweep_val
        elif sweep_param == "noise_std":
            run_noise = sweep_val
        elif sweep_param == "dyn_noise_std":
            run_dyn_noise = sweep_val

        # Fixed seed list — no retry/resampling to avoid survivorship bias
        args_list = [
            (sys_name, topology, N, eps, method, n_surr,
             run_T, transient, run_noise, run_dyn_noise,
             seed_base + i, net_kwargs, fdr, extra_seccm_kwargs)
            for i in range(n_reps)
        ]

        results = parallel_map(
            _run_single_rep, args_list,
            n_jobs=n_jobs,
            desc=f"  reps {sys_name}/{method}/{sweep_param}={sweep_val}",
        )

        valid = [r for r in results if r is not None]
        n_failed = n_reps - len(valid)

        if n_failed > 0:
            print(f"    WARNING: {n_failed}/{n_reps} reps failed for "
                  f"{sys_name}/{method}/{sweep_param}={sweep_val}")

        for r in valid:
            row = {
                "system": sys_name,
                "method": method,
                sweep_param: sweep_val,
                "n_total_reps": n_reps,
                "n_failed_reps": n_failed,
                "success_rate": len(valid) / n_reps,
            }
            for k in COLLECT_KEYS:
                row[k] = r.get(k, np.nan)
            rows.append(row)

    return pd.DataFrame(rows)


# ── Visualization ────────────────────────────────────────────────────

def _plot_auroc_lines(df, sweep_param, output_dir):
    """Publication-quality AUROC line charts with error bars (mean +/- SEM).

    Generates one figure per AUROC metric (surrogate, zscore, rho, delta,
    delta_zscore).  Each figure has one subplot per system.

    Output files
    ------------
    line_{metric}_vs_{sweep_param}.pdf / .png
    """
    systems = df["system"].unique()
    methods = df["method"].unique()
    n_sys = len(systems)
    xlabel = SWEEP_LABEL.get(sweep_param, sweep_param)

    for metric in AUC_KEYS:
        n_cols = min(n_sys, 4)
        n_rows = int(np.ceil(n_sys / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.3 * n_cols, 2.8 * n_rows),
            sharey=True, squeeze=False,
        )
        axes_flat = axes.ravel()

        is_delta = "delta" in metric

        for i, sys in enumerate(systems):
            ax = axes_flat[i]
            sub = df[df["system"] == sys]

            # Raw ρ baseline (with error band) — skip for delta metrics
            if not is_delta:
                bl = sub.groupby(sweep_param)["AUC_ROC_rho"]
                bl_mean = bl.mean()
                bl_sem = bl.sem()
                ax.fill_between(
                    bl_mean.index, bl_mean - bl_sem, bl_mean + bl_sem,
                    color=BASELINE_COLOR, alpha=0.15, zorder=0,
                )
                ax.plot(
                    bl_mean.index, bl_mean.values,
                    color=BASELINE_COLOR, ls="--", lw=1.2,
                    label=r"Raw $\rho$", zorder=1,
                )
            else:
                ax.axhline(0, color=BASELINE_COLOR, ls="--", lw=0.8, zorder=0)

            # Per-method curves
            for method in methods:
                msub = sub[sub["method"] == method]
                grp = msub.groupby(sweep_param)[metric]
                m = grp.mean()
                s = grp.sem()
                color = METHOD_COLORS.get(method, None)
                marker = METHOD_MARKERS.get(method, "o")
                label = METHOD_DISPLAY.get(method, method)
                ax.errorbar(
                    m.index, m.values, yerr=s.values,
                    marker=marker, color=color, label=label,
                    capsize=2.5, capthick=0.8, elinewidth=0.8,
                    markeredgewidth=0.6, zorder=2,
                )

            ax.set_title(SYSTEM_DISPLAY.get(sys, sys), fontweight="medium")
            ax.set_xlabel(xlabel)
            if i % n_cols == 0:
                ax.set_ylabel(METRIC_LABEL.get(metric, metric))

            # Integer ticks for T
            if sweep_param == "T":
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Hide unused axes
        for j in range(n_sys, len(axes_flat)):
            axes_flat[j].set_visible(False)

        # Single shared legend
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center", ncol=len(handles),
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
        )

        fig.tight_layout(rect=[0, 0.05, 1, 1])
        for ext in ("pdf", "png"):
            fig.savefig(
                os.path.join(output_dir, f"line_{metric}_vs_{sweep_param}.{ext}"),
            )
        plt.close(fig)


def _plot_delta_heatmaps(df, sweep_param, output_dir):
    """Publication-quality heatmaps (system x sweep value, color = ΔAUROC).

    (a) Averaged over all methods.
    (b) One heatmap per method.

    Output files
    ------------
    heatmap_{metric}_vs_{sweep_param}.pdf / .png
    heatmap_{metric}_{method}_vs_{sweep_param}.pdf / .png
    """
    systems = df["system"].unique()
    methods = df["method"].unique()
    xlabel = SWEEP_LABEL.get(sweep_param, sweep_param)

    for metric in ["AUC_ROC_delta", "AUC_ROC_delta_zscore"]:
        metric_lbl = METRIC_LABEL.get(metric, metric)

        # ── (a) All-method average ────────────────────────────────
        pivot = (df.groupby(["system", sweep_param])[metric]
                   .mean().unstack(sweep_param))
        pivot = pivot.reindex(index=systems)
        pivot = pivot.rename(index=SYSTEM_DISPLAY)

        fig, ax = plt.subplots(figsize=(
            max(5, len(pivot.columns) * 0.95 + 1.5),
            max(2.5, len(systems) * 0.55 + 0.8),
        ))
        vals = pivot.values[np.isfinite(pivot.values)]
        if len(vals) == 0:
            plt.close(fig)
            continue
        vmax = max(abs(vals.min()), abs(vals.max()), 0.01)
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, vmin=-vmax, vmax=vmax, ax=ax,
            linewidths=0.6, linecolor="white",
            cbar_kws={"label": metric_lbl, "shrink": 0.85},
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(
                output_dir, f"heatmap_{metric}_vs_{sweep_param}.{ext}"))
        plt.close(fig)

        # ── (b) Per-method heatmaps ──────────────────────────────
        for method in methods:
            msub = df[df["method"] == method]
            pivot_m = (msub.groupby(["system", sweep_param])[metric]
                           .mean().unstack(sweep_param))
            pivot_m = pivot_m.reindex(index=systems).rename(index=SYSTEM_DISPLAY)

            fig, ax = plt.subplots(figsize=(
                max(5, len(pivot_m.columns) * 0.95 + 1.5),
                max(2.5, len(systems) * 0.55 + 0.8),
            ))
            vals = pivot_m.values[np.isfinite(pivot_m.values)]
            if len(vals) == 0:
                plt.close(fig)
                continue
            vm = max(abs(vals.min()), abs(vals.max()), 0.01)
            method_lbl = METHOD_DISPLAY.get(method, method)
            sns.heatmap(
                pivot_m, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=-vm, vmax=vm, ax=ax,
                linewidths=0.6, linecolor="white",
                cbar_kws={"label": metric_lbl, "shrink": 0.85},
            )
            ax.set_title(method_lbl, fontweight="medium")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("")
            ax.tick_params(axis="y", rotation=0)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(
                    output_dir,
                    f"heatmap_{metric}_{method}_vs_{sweep_param}.{ext}",
                ))
            plt.close(fig)


def _plot_sweep(df, sweep_param, output_dir):
    """Generate all plots for one sub-experiment."""
    if df.empty:
        return

    with plt.rc_context(_pub_rcparams()):
        _plot_auroc_lines(df, sweep_param, output_dir)
        _plot_delta_heatmaps(df, sweep_param, output_dir)

    print(f"  Plots saved to {output_dir}/")


# ── Combined overview figure ─────────────────────────────────────────

def _plot_combined_overview(all_dfs, output_dir):
    """4-panel figure: one panel per sweep factor, showing ΔAUROC (surrogate).

    Each panel: x = sweep values, y = ΔAUROC averaged over systems, one curve
    per method, with error bars.  Provides a single-glance summary.

    Output: combined_delta_overview.pdf / .png
    """
    sweep_order = [
        ("T_sweep", "T"),
        ("coupling_sweep", "coupling"),
        ("obs_noise_sweep", "noise_std"),
        ("dyn_noise_sweep", "dyn_noise_std"),
    ]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    metric = "AUC_ROC_delta"

    available = [(name, param) for name, param in sweep_order
                 if name in all_dfs and not all_dfs[name].empty]
    if not available:
        return

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.0),
                             squeeze=False)
    axes = axes.ravel()

    for idx, (df_name, sweep_param) in enumerate(available):
        ax = axes[idx]
        df = all_dfs[df_name]
        methods = df["method"].unique()

        ax.axhline(0, color=BASELINE_COLOR, ls="--", lw=0.8, zorder=0)

        for method in methods:
            msub = df[df["method"] == method]
            # Average over all systems
            grp = msub.groupby(sweep_param)[metric]
            m = grp.mean()
            s = grp.sem()
            color = METHOD_COLORS.get(method, None)
            marker = METHOD_MARKERS.get(method, "o")
            label = METHOD_DISPLAY.get(method, method)
            ax.errorbar(
                m.index, m.values, yerr=s.values,
                marker=marker, color=color, label=label,
                capsize=2.5, capthick=0.8, elinewidth=0.8,
                markeredgewidth=0.6, zorder=2,
            )

        xlabel = SWEEP_LABEL.get(sweep_param, sweep_param)
        ax.set_xlabel(xlabel)
        if idx == 0:
            ax.set_ylabel(METRIC_LABEL.get(metric, metric))
        ax.set_title(panel_labels[idx], loc="left", fontweight="bold",
                     fontsize=11)

        if sweep_param == "T":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(handles),
        bbox_to_anchor=(0.5, -0.04), frameon=True,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"combined_delta_overview.{ext}"))
    plt.close(fig)
    print(f"  Combined overview saved to {output_dir}/")


# ── Ablation table ───────────────────────────────────────────────────

def _generate_ablation_table(all_dfs, output_dir):
    """Generate comprehensive ablation summary tables.

    Table 1 — Detailed: for each (sub-experiment, system, method, sweep_value)
               report AUROC_surrogate and ΔAUROC as mean +/- SEM.
    Table 2 — Summary: for each (sub-experiment, system), report best method
               and overall ΔAUROC range.

    Output
    ------
    ablation_detailed.csv
    ablation_summary.csv
    ablation_summary.tex
    """
    sweep_info = {
        "T_sweep": "T",
        "coupling_sweep": "coupling",
        "obs_noise_sweep": "noise_std",
        "dyn_noise_sweep": "dyn_noise_std",
    }

    # ── Table 1: Detailed ────────────────────────────────────────
    detail_rows = []
    for df_name, sweep_param in sweep_info.items():
        if df_name not in all_dfs or all_dfs[df_name].empty:
            continue
        df = all_dfs[df_name]
        for (sys, method, val), grp in df.groupby(
                ["system", "method", sweep_param]):
            rho_mean = grp["AUC_ROC_rho"].mean()
            rho_sem = grp["AUC_ROC_rho"].sem()
            surr_mean = grp["AUC_ROC_surrogate"].mean()
            surr_sem = grp["AUC_ROC_surrogate"].sem()
            z_mean = grp["AUC_ROC_zscore"].mean()
            z_sem = grp["AUC_ROC_zscore"].sem()
            delta_mean = grp["AUC_ROC_delta"].mean()
            delta_sem = grp["AUC_ROC_delta"].sem()
            n = len(grp)
            detail_rows.append({
                "sub_experiment": df_name.replace("_sweep", "").replace("_", " "),
                "sweep_param": sweep_param,
                "sweep_value": val,
                "system": SYSTEM_DISPLAY.get(sys, sys),
                "method": METHOD_DISPLAY.get(method, method),
                "n_reps": n,
                "AUROC_rho_mean": round(rho_mean, 4),
                "AUROC_rho_sem": round(rho_sem, 4),
                "AUROC_surr_mean": round(surr_mean, 4),
                "AUROC_surr_sem": round(surr_sem, 4),
                "AUROC_z_mean": round(z_mean, 4),
                "AUROC_z_sem": round(z_sem, 4),
                "delta_AUROC_mean": round(delta_mean, 4),
                "delta_AUROC_sem": round(delta_sem, 4),
            })

    df_detail = pd.DataFrame(detail_rows)
    if not df_detail.empty:
        df_detail.to_csv(os.path.join(output_dir, "ablation_detailed.csv"),
                         index=False)

    # ── Table 2: Summary — one row per (sub-experiment, system) ──
    summary_rows = []
    for df_name, sweep_param in sweep_info.items():
        if df_name not in all_dfs or all_dfs[df_name].empty:
            continue
        df = all_dfs[df_name]
        sweep_label = SWEEP_LABEL.get(sweep_param, sweep_param)

        for sys in df["system"].unique():
            sub = df[df["system"] == sys]
            sys_display = SYSTEM_DISPLAY.get(sys, sys)

            # Per-method mean ΔAUROC (averaged over sweep values)
            method_deltas = (
                sub.groupby("method")["AUC_ROC_delta"].mean()
            )
            best_method_key = method_deltas.idxmax()
            best_method = METHOD_DISPLAY.get(best_method_key, best_method_key)
            best_delta = method_deltas.max()

            # Sweep-value dependence (averaged over methods)
            val_deltas = sub.groupby(sweep_param)["AUC_ROC_delta"].mean()
            delta_min = val_deltas.min()
            delta_max = val_deltas.max()
            best_val = val_deltas.idxmax()

            # Overall AUROC range
            rho_range = (sub.groupby(sweep_param)["AUC_ROC_rho"].mean()
                           .agg(["min", "max"]))
            surr_range = (sub.groupby(sweep_param)["AUC_ROC_surrogate"].mean()
                            .agg(["min", "max"]))

            summary_rows.append({
                "Factor": sweep_label,
                "System": sys_display,
                "AUROC_rho_range": f"{rho_range['min']:.3f}\u2013{rho_range['max']:.3f}",
                "AUROC_surr_range": f"{surr_range['min']:.3f}\u2013{surr_range['max']:.3f}",
                "best_method": best_method,
                "best_method_delta": f"{best_delta:+.3f}",
                "delta_range": f"{delta_min:+.3f} to {delta_max:+.3f}",
                "best_sweep_value": best_val,
            })

    df_summary = pd.DataFrame(summary_rows)
    if df_summary.empty:
        return

    df_summary.to_csv(os.path.join(output_dir, "ablation_summary.csv"),
                      index=False)

    # ── LaTeX output ─────────────────────────────────────────────
    _write_latex_table(df_summary, df_detail, all_dfs, output_dir)

    print(f"  Ablation tables saved to {output_dir}/")


def _write_latex_table(df_summary, df_detail, all_dfs, output_dir):
    """Write a publication-ready LaTeX ablation table.

    Format: rows = systems, column groups = sweep factors.
    Cell content: best-method ΔAUROC (mean ± SEM).
    """
    sweep_info = {
        "T_sweep": ("T", r"$T$"),
        "coupling_sweep": ("coupling", r"$\varepsilon$"),
        "obs_noise_sweep": ("noise_std", r"$\sigma_{\mathrm{obs}}$"),
        "dyn_noise_sweep": ("dyn_noise_std", r"$\sigma_{\mathrm{dyn}}$"),
    }

    systems_order = [
        "logistic", "lorenz", "henon", "rossler",
        "hindmarsh_rose", "fitzhugh_nagumo", "kuramoto",
    ]

    # Collect data: for each (factor, system), take best method's
    # ΔAUROC averaged over all sweep values
    table_data = {}  # (system, factor_latex) -> "mean ± sem"
    for df_name, (sweep_param, factor_latex) in sweep_info.items():
        if df_name not in all_dfs or all_dfs[df_name].empty:
            continue
        df = all_dfs[df_name]
        for sys in systems_order:
            sub = df[df["system"] == sys]
            if sub.empty:
                continue
            # Best method by mean ΔAUROC
            method_agg = sub.groupby("method")["AUC_ROC_delta"]
            best_method = method_agg.mean().idxmax()
            best_sub = sub[sub["method"] == best_method]
            m = best_sub["AUC_ROC_delta"].mean()
            s = best_sub["AUC_ROC_delta"].sem()
            cell = f"${m:+.3f} \\pm {s:.3f}$"
            table_data[(sys, factor_latex)] = cell

    # Build LaTeX
    factors_available = [
        latex for _, (_, latex) in sweep_info.items()
        if any((sys, latex) in table_data for sys in systems_order)
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study: best-method $\Delta$AUROC "
                 r"(surrogate $-$ raw $\rho$) across robustness factors. "
                 r"Values: mean $\pm$ SEM over repetitions and sweep values.}")
    lines.append(r"\label{tab:ablation}")
    col_spec = "l" + "c" * len(factors_available)
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    header = "System & " + " & ".join(factors_available) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for sys in systems_order:
        sys_display = SYSTEM_DISPLAY.get(sys, sys)
        cells = []
        for factor_latex in factors_available:
            cells.append(table_data.get((sys, factor_latex), "--"))
        lines.append(f"{sys_display} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(os.path.join(output_dir, "ablation_summary.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Experiment parameter table ────────────────────────────────────────

def _generate_params_table(cfg, config, output_dir):
    """Generate a table documenting all experimental parameters.

    For each sub-experiment, lists the swept parameter and its values,
    and all other parameters held fixed.

    Output
    ------
    experiment_parameters.csv
    experiment_parameters.tex
    """
    ts_cfg = config.get("time_series", {})
    default_T = ts_cfg.get("T", 3000)
    default_transient = ts_cfg.get("transient", 1000)

    systems = cfg.get("systems", [])
    methods = cfg.get("methods", [])
    n_surr = cfg.get("n_surrogates", 99)
    N = cfg.get("N", 10)
    topology = cfg.get("topology", "ER")
    er_p = cfg.get("er_p", 0.3)
    n_reps = cfg.get("n_reps", 10)
    fdr = cfg.get("fdr", False)
    seed_base = config.get("seed", 42)
    coupling_map = cfg.get("coupling", {})

    t_cfg = cfg.get("T_sweep", {})
    T_values = t_cfg.get("values", [500, 1000, 2000, 3000, 5000])
    T_transient = t_cfg.get("transient", default_transient)

    coupling_sweep = cfg.get("coupling_sweep", {})
    obs_cfg = cfg.get("obs_noise_sweep", {})
    obs_values = obs_cfg.get("values", [0.0, 0.01, 0.05, 0.1, 0.2])
    dyn_cfg = cfg.get("dyn_noise_sweep", {})
    dyn_values = dyn_cfg.get("values", [0.0, 0.001, 0.005, 0.01, 0.05])

    # Format coupling map as string
    coupling_str = ", ".join(f"{SYSTEM_DISPLAY.get(k,k)}: {v}"
                             for k, v in coupling_map.items())

    # Format coupling sweep ranges
    coupling_sweep_strs = {
        SYSTEM_DISPLAY.get(k, k): f"[{min(v)}, {max(v)}] ({len(v)} values)"
        for k, v in coupling_sweep.items() if v
    }
    coupling_sweep_str = "; ".join(f"{k}: {v}"
                                   for k, v in coupling_sweep_strs.items())

    systems_str = ", ".join(SYSTEM_DISPLAY.get(s, s) for s in systems)
    methods_str = ", ".join(METHOD_DISPLAY.get(m, m) for m in methods)

    # Build rows: each parameter appears once, with its value per sub-experiment
    # Columns: Parameter | Sub-A (T) | Sub-B (eps) | Sub-C (sigma_obs) | Sub-D (sigma_dyn)
    params = [
        ("Systems",
         systems_str, systems_str, systems_str, systems_str),
        ("Surrogate methods",
         methods_str, methods_str, methods_str, methods_str),
        (r"$N$ (network size)",
         str(N), str(N), str(N), str(N)),
        ("Topology",
         f"{topology} (p={er_p})", f"{topology} (p={er_p})",
         f"{topology} (p={er_p})", f"{topology} (p={er_p})"),
        (r"$n_{\mathrm{surr}}$",
         str(n_surr), str(n_surr), str(n_surr), str(n_surr)),
        (r"$n_{\mathrm{reps}}$",
         str(n_reps), str(n_reps), str(n_reps), str(n_reps)),
        ("FDR correction",
         str(fdr), str(fdr), str(fdr), str(fdr)),
        ("Seed base",
         str(seed_base), str(seed_base), str(seed_base), str(seed_base)),
        (r"$T$ (time series length)",
         f"SWEPT: {T_values}", str(default_T), str(default_T), str(default_T)),
        ("Transient",
         str(T_transient), str(default_transient),
         str(default_transient), str(default_transient)),
        (r"$\varepsilon$ (coupling)",
         coupling_str, f"SWEPT: {coupling_sweep_str}",
         coupling_str, coupling_str),
        (r"$\sigma_{\mathrm{obs}}$ (obs. noise)",
         "0.0", "0.0", f"SWEPT: {obs_values}", "0.0"),
        (r"$\sigma_{\mathrm{dyn}}$ (dyn. noise)",
         "0.0", "0.0", "0.0", f"SWEPT: {dyn_values}"),
    ]

    # ── CSV ───────────────────────────────────────────────────────
    col_names = [
        "Parameter",
        "Sub-A: T sweep",
        "Sub-B: Coupling sweep",
        "Sub-C: Obs. noise sweep",
        "Sub-D: Dyn. noise sweep",
    ]
    df_params = pd.DataFrame(params, columns=col_names)
    df_params.to_csv(os.path.join(output_dir, "experiment_parameters.csv"),
                     index=False)

    # ── LaTeX ─────────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Experimental parameters for the four robustness "
                 r"sub-experiments. ``SWEPT'' indicates the variable under "
                 r"investigation; all other parameters are held fixed.}")
    lines.append(r"\label{tab:exp_params}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Parameter & Sub-A ($T$) & Sub-B ($\varepsilon$) "
                 r"& Sub-C ($\sigma_{\mathrm{obs}}$) "
                 r"& Sub-D ($\sigma_{\mathrm{dyn}}$) \\")
    lines.append(r"\midrule")

    # Simplified LaTeX rows (avoid overly long cells)
    tex_params = [
        ("Systems", systems_str, systems_str, systems_str, systems_str),
        ("Methods", methods_str, methods_str, methods_str, methods_str),
        ("$N$", str(N), str(N), str(N), str(N)),
        ("Topology", f"{topology} ($p$={er_p})", f"{topology} ($p$={er_p})",
         f"{topology} ($p$={er_p})", f"{topology} ($p$={er_p})"),
        ("$n_{\\mathrm{surr}}$",
         str(n_surr), str(n_surr), str(n_surr), str(n_surr)),
        ("$n_{\\mathrm{reps}}$",
         str(n_reps), str(n_reps), str(n_reps), str(n_reps)),
        ("$T$", "\\textbf{swept}", str(default_T),
         str(default_T), str(default_T)),
        ("Transient", str(T_transient), str(default_transient),
         str(default_transient), str(default_transient)),
        ("$\\varepsilon$", "per-system default", "\\textbf{swept}",
         "per-system default", "per-system default"),
        ("$\\sigma_{\\mathrm{obs}}$", "0", "0", "\\textbf{swept}", "0"),
        ("$\\sigma_{\\mathrm{dyn}}$", "0", "0", "0", "\\textbf{swept}"),
    ]

    for row in tex_params:
        cells = " & ".join(row)
        lines.append(cells + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(os.path.join(output_dir, "experiment_parameters.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ── Also write per-system default coupling as a small side table ──
    coup_rows = []
    for sys in systems:
        coup_rows.append({
            "system": SYSTEM_DISPLAY.get(sys, sys),
            "default_coupling": coupling_map.get(sys, "N/A"),
        })
        if sys in coupling_sweep:
            coup_rows[-1]["coupling_sweep_values"] = str(coupling_sweep[sys])
        else:
            coup_rows[-1]["coupling_sweep_values"] = "N/A"

    df_coup = pd.DataFrame(coup_rows)
    df_coup.to_csv(os.path.join(output_dir, "coupling_parameters.csv"),
                   index=False)

    print(f"  Parameter tables saved to {output_dir}/")


# ── Orchestrator ─────────────────────────────────────────────────────
def run_surrogate_robustness_experiment(config, output_dir, n_jobs=-1):
    """Run all four robustness sub-experiments."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("surrogate_robustness", {})
    systems = cfg.get("systems", [
        "logistic", "lorenz", "henon", "rossler",
        "hindmarsh_rose", "fitzhugh_nagumo", "kuramoto",
    ])
    methods = cfg.get("methods", ["fft", "aaft", "timeshift"])
    n_surr = cfg.get("n_surrogates", 99)
    N = cfg.get("N", 10)
    topology = cfg.get("topology", "ER")
    coupling_map = cfg.get("coupling", {
        "logistic": 0.1, "lorenz": 1.0, "henon": 0.05,
        "rossler": 0.2, "hindmarsh_rose": 0.1,
        "fitzhugh_nagumo": 0.05, "kuramoto": 0.1,
    })
    n_reps = cfg.get("n_reps", 10)
    er_p = cfg.get("er_p", 0.3)
    fdr = cfg.get("fdr", False)
    seed_base = config.get("seed", 42)
    ts_cfg = config.get("time_series", {})
    default_T = ts_cfg.get("T", 3000)
    default_transient = ts_cfg.get("transient", 1000)

    # Extra SECCM kwargs from config (theiler_w, adaptive_rho, E_method, etc.)
    seccm_cfg = cfg.get("seccm_kwargs", {})
    extra_seccm_kwargs = {}
    for key in ("theiler_w", "adaptive_rho", "E_method",
                "convergence_filter", "convergence_threshold",
                "min_rho", "adaptive_rho_quantile", "iaaft_max_iter"):
        if key in seccm_cfg:
            extra_seccm_kwargs[key] = seccm_cfg[key]

    # Common fixed params (overridden per sub-experiment as needed)
    base_fixed = dict(
        N=N, topology=topology, coupling=coupling_map,
        n_surr=n_surr, T=default_T, transient=default_transient,
        noise_std=0.0, dyn_noise_std=0.0,
        er_p=er_p, n_reps=n_reps, fdr=fdr, seed_base=seed_base,
        extra_seccm_kwargs=extra_seccm_kwargs,
    )

    all_dfs = {}

    # ── Sub-A: T sweep ───────────────────────────────────────────
    t_cfg = cfg.get("T_sweep", {})
    T_values = t_cfg.get("values", [500, 1000, 2000, 3000, 5000])
    T_transient = t_cfg.get("transient", default_transient)

    sub_dir = os.path.join(output_dir, "T_sweep")
    os.makedirs(sub_dir, exist_ok=True)

    fixed_a = {**base_fixed, "transient": T_transient}
    print("\n--- Sub-A: Time series length sweep ---")
    df_a = _run_sweep("T_sweep", "T", T_values, systems, methods,
                      fixed_a, config, n_jobs)
    df_a.to_csv(os.path.join(sub_dir, "T_sweep.csv"), index=False)
    _plot_sweep(df_a, "T", sub_dir)
    all_dfs["T_sweep"] = df_a

    # ── Sub-B: Coupling sweep ────────────────────────────────────
    coupling_sweep = cfg.get("coupling_sweep", {})

    sub_dir = os.path.join(output_dir, "coupling_sweep")
    os.makedirs(sub_dir, exist_ok=True)

    print("\n--- Sub-B: Coupling strength sweep ---")
    df_b = _run_sweep("coupling_sweep", "coupling", coupling_sweep,
                      systems, methods, base_fixed, config, n_jobs)
    df_b.to_csv(os.path.join(sub_dir, "coupling_sweep.csv"), index=False)
    _plot_sweep(df_b, "coupling", sub_dir)
    all_dfs["coupling_sweep"] = df_b

    # ── Sub-C: Observation noise sweep ───────────────────────────
    obs_cfg = cfg.get("obs_noise_sweep", {})
    obs_values = obs_cfg.get("values", [0.0, 0.01, 0.05, 0.1, 0.2])

    sub_dir = os.path.join(output_dir, "obs_noise_sweep")
    os.makedirs(sub_dir, exist_ok=True)

    print("\n--- Sub-C: Observation noise sweep ---")
    df_c = _run_sweep("obs_noise_sweep", "noise_std", obs_values,
                      systems, methods, base_fixed, config, n_jobs)
    df_c.to_csv(os.path.join(sub_dir, "obs_noise_sweep.csv"), index=False)
    _plot_sweep(df_c, "noise_std", sub_dir)
    all_dfs["obs_noise_sweep"] = df_c

    # ── Sub-D: Dynamical noise sweep ─────────────────────────────
    dyn_cfg = cfg.get("dyn_noise_sweep", {})
    dyn_values = dyn_cfg.get("values", [0.0, 0.001, 0.005, 0.01, 0.05])

    sub_dir = os.path.join(output_dir, "dyn_noise_sweep")
    os.makedirs(sub_dir, exist_ok=True)

    print("\n--- Sub-D: Dynamical noise sweep ---")
    df_d = _run_sweep("dyn_noise_sweep", "dyn_noise_std", dyn_values,
                      systems, methods, base_fixed, config, n_jobs)
    df_d.to_csv(os.path.join(sub_dir, "dyn_noise_sweep.csv"), index=False)
    _plot_sweep(df_d, "dyn_noise_std", sub_dir)
    all_dfs["dyn_noise_sweep"] = df_d

    # ── Cross-sweep outputs ──────────────────────────────────────
    with plt.rc_context(_pub_rcparams()):
        _plot_combined_overview(all_dfs, output_dir)
    _generate_ablation_table(all_dfs, output_dir)
    _generate_params_table(cfg, config, output_dir)

    print(f"\nAll robustness sub-experiments complete. Results in {output_dir}/")
    return all_dfs
