"""E2-E4: Cycle-Phase Surrogate Experiments.

E2: Cycle-phase vs. all surrogates on Rossler (primary test)
E3: Generalization to other oscillatory systems
E4: Safety check on broadband chaotic systems
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..generators import SYSTEM_CLASSES, create_system, generate_network
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map


def _run_single_rep(args):
    """Worker: run one system × surrogate × config replicate."""
    (system_name, coupling, T, surr_method, n_surrogates,
     N, seed, sys_kwargs) = args

    try:
        adj = generate_network("ER", N, seed=seed, p=0.5)
        system = create_system(system_name, adj, coupling, **sys_kwargs)
        data = system.generate(T, transient=1000, seed=seed)

        seccm = SECCM(
            surrogate_method=surr_method,
            n_surrogates=n_surrogates,
            alpha=0.05,
            fdr=True,
            seed=seed,
            verbose=False,
        )
        seccm.fit(data)
        metrics = seccm.score(adj)

        return {
            "system": system_name,
            "coupling": coupling,
            "T": T,
            "surrogate": surr_method,
            "seed": seed,
            "AUC_ROC_rho": metrics.get("AUC_ROC_rho", np.nan),
            "AUC_ROC_zscore": metrics.get("AUC_ROC_zscore", np.nan),
            "AUC_ROC_delta_zscore": metrics.get("AUC_ROC_delta_zscore", np.nan),
            "AUC_PR_rho": metrics.get("AUC_PR_rho", np.nan),
            "AUC_PR_zscore": metrics.get("AUC_PR_zscore", np.nan),
            "TPR": metrics.get("TPR", np.nan),
            "FPR": metrics.get("FPR", np.nan),
            "F1": metrics.get("F1", np.nan),
        }
    except Exception as e:
        return {
            "system": system_name, "surrogate": surr_method,
            "coupling": coupling, "T": T, "seed": seed,
            "error": str(e),
        }


def run_cycle_phase_experiment(config, output_dir="results/cycle_phase",
                               n_jobs=-1):
    """Run E2-E4 cycle-phase experiments.

    Parameters
    ----------
    config : dict
        Configuration dict. Override via 'cycle_phase' key.
    output_dir : str
        Output directory.
    n_jobs : int
        Number of parallel jobs.
    """
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("cycle_phase", {})
    n_surrogates = cfg.get("n_surrogates", 200)
    n_reps = cfg.get("n_reps", 20)
    N = cfg.get("N", 5)
    base_seed = cfg.get("seed", 42)

    # ---- E2: Rossler sweep ----
    e2_dir = os.path.join(output_dir, "E2_rossler")
    os.makedirs(e2_dir, exist_ok=True)

    e2_couplings = cfg.get("e2_couplings", [0.05, 0.1, 0.15, 0.2, 0.3])
    e2_T_values = cfg.get("e2_T_values", [1000, 2000, 5000, 10000])
    e2_surrogates = cfg.get("e2_surrogates",
                            ["iaaft", "fft", "cycle_shuffle",
                             "cycle_phase_A", "cycle_phase_B"])

    args_e2 = []
    for coupling in e2_couplings:
        for T in e2_T_values:
            for surr in e2_surrogates:
                for rep in range(n_reps):
                    seed = base_seed + hash(("e2", coupling, T, surr, rep)) % (2**31)
                    args_e2.append(
                        ("rossler", coupling, T, surr, n_surrogates,
                         N, seed, {"dt": 0.05})
                    )

    print(f"  E2 (Rossler): {len(args_e2)} runs")
    results_e2 = parallel_map(_run_single_rep, args_e2, n_jobs=n_jobs,
                              desc="E2 Rossler")
    df_e2 = pd.DataFrame([r for r in results_e2 if "error" not in r])
    df_e2.to_csv(os.path.join(e2_dir, "rossler_results.csv"), index=False)

    errors_e2 = [r for r in results_e2 if "error" in r]
    if errors_e2:
        print(f"  E2: {len(errors_e2)} failures")

    if len(df_e2) > 0:
        _plot_e2_results(df_e2, e2_dir)
        _check_e2_success(df_e2, e2_dir)

    # ---- E3: Oscillatory generalization ----
    e3_dir = os.path.join(output_dir, "E3_oscillatory")
    os.makedirs(e3_dir, exist_ok=True)

    e3_systems = {
        "rossler": {"coupling": 0.15, "T": 5000, "sys_kwargs": {"dt": 0.05}},
        "kuramoto": {"coupling": 0.5, "T": 2000, "sys_kwargs": {}},
        "fitzhugh_nagumo": {"coupling": 0.3, "T": 2000, "sys_kwargs": {}},
    }
    # Add van_der_pol if available
    if "van_der_pol" in SYSTEM_CLASSES:
        e3_systems["van_der_pol"] = {
            "coupling": 0.3, "T": 5000, "sys_kwargs": {},
        }

    e3_surrogates = ["iaaft", "cycle_phase_A", "cycle_phase_B"]

    args_e3 = []
    for sys_name, sys_cfg in e3_systems.items():
        for surr in e3_surrogates:
            for rep in range(n_reps):
                seed = base_seed + hash(("e3", sys_name, surr, rep)) % (2**31)
                args_e3.append(
                    (sys_name, sys_cfg["coupling"], sys_cfg["T"],
                     surr, n_surrogates, N, seed, sys_cfg.get("sys_kwargs", {}))
                )

    print(f"  E3 (Oscillatory): {len(args_e3)} runs")
    results_e3 = parallel_map(_run_single_rep, args_e3, n_jobs=n_jobs,
                              desc="E3 Oscillatory")
    df_e3 = pd.DataFrame([r for r in results_e3 if "error" not in r])
    df_e3.to_csv(os.path.join(e3_dir, "oscillatory_results.csv"), index=False)

    if len(df_e3) > 0:
        _plot_e3_results(df_e3, e3_dir)

    # ---- E4: Chaotic safety check ----
    e4_dir = os.path.join(output_dir, "E4_chaotic")
    os.makedirs(e4_dir, exist_ok=True)

    e4_systems = {
        "logistic": {"coupling": 0.3, "T": 2000, "sys_kwargs": {}},
        "henon": {"coupling": 0.3, "T": 2000, "sys_kwargs": {}},
        "lorenz": {"coupling": 0.5, "T": 2000, "sys_kwargs": {"dt": 0.01}},
    }
    e4_surrogates = ["iaaft", "fft", "cycle_phase_A", "cycle_phase_B"]

    args_e4 = []
    for sys_name, sys_cfg in e4_systems.items():
        for surr in e4_surrogates:
            for rep in range(n_reps):
                seed = base_seed + hash(("e4", sys_name, surr, rep)) % (2**31)
                args_e4.append(
                    (sys_name, sys_cfg["coupling"], sys_cfg["T"],
                     surr, n_surrogates, N, seed, sys_cfg.get("sys_kwargs", {}))
                )

    print(f"  E4 (Chaotic safety): {len(args_e4)} runs")
    results_e4 = parallel_map(_run_single_rep, args_e4, n_jobs=n_jobs,
                              desc="E4 Chaotic")
    df_e4 = pd.DataFrame([r for r in results_e4 if "error" not in r])
    df_e4.to_csv(os.path.join(e4_dir, "chaotic_results.csv"), index=False)

    if len(df_e4) > 0:
        _plot_e4_results(df_e4, e4_dir)

    print(f"\n  Cycle-phase experiments complete. Results in: {output_dir}/")
    return df_e2, df_e3, df_e4


def _check_e2_success(df, output_dir):
    """Check E2 success criteria and save verdict."""
    agg = df.groupby("surrogate")["AUC_ROC_delta_zscore"].mean()

    cp_a = agg.get("cycle_phase_A", np.nan)
    cp_b = agg.get("cycle_phase_B", np.nan)
    iaaft = agg.get("iaaft", np.nan)

    best_cp = max(cp_a, cp_b) if not (np.isnan(cp_a) and np.isnan(cp_b)) else np.nan

    verdict = {
        "cycle_phase_A_delta": float(cp_a) if not np.isnan(cp_a) else None,
        "cycle_phase_B_delta": float(cp_b) if not np.isnan(cp_b) else None,
        "iaaft_delta": float(iaaft) if not np.isnan(iaaft) else None,
        "criterion_1_surrogates_help": bool(best_cp > 0) if not np.isnan(best_cp) else False,
        "criterion_2_beats_iaaft": bool(best_cp > iaaft + 0.03)
            if not (np.isnan(best_cp) or np.isnan(iaaft)) else False,
    }
    verdict["overall_success"] = (
        verdict["criterion_1_surrogates_help"] and
        verdict["criterion_2_beats_iaaft"]
    )

    with open(os.path.join(output_dir, "e2_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)

    status = "PASS" if verdict["overall_success"] else "FAIL"
    print(f"\n  E2 Verdict: {status}")
    print(f"    cycle_phase_A ΔAUROC: {cp_a:.4f}")
    print(f"    cycle_phase_B ΔAUROC: {cp_b:.4f}")
    print(f"    iaaft ΔAUROC: {iaaft:.4f}")


def _plot_e2_results(df, output_dir):
    """Plot E2 Rossler comparison results."""
    # Bar chart: mean ΔAUROC per surrogate
    agg = df.groupby("surrogate")["AUC_ROC_delta_zscore"].agg(["mean", "std"])
    agg = agg.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if "cycle_phase" in idx else "#3498db" for idx in agg.index]
    bars = ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"],
                  color=colors, alpha=0.8, capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg.index, rotation=45, ha="right")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("ΔAUROC (z-score − raw ρ)", fontsize=11)
    ax.set_title("E2: Surrogate Utility on Rössler System", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rossler_deltaAUROC_bar.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "rossler_deltaAUROC_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Heatmap: ΔAUROC by coupling × T for cycle_phase_A
    cp_a = df[df["surrogate"] == "cycle_phase_A"]
    if len(cp_a) > 0:
        pivot = cp_a.pivot_table(
            values="AUC_ROC_delta_zscore",
            index="coupling", columns="T", aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                       vmin=-0.1, vmax=0.1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{c:.2f}" for c in pivot.index])
        ax.set_xlabel("Time series length T")
        ax.set_ylabel("Coupling strength ε")
        ax.set_title("E2: Cycle-Phase A ΔAUROC on Rössler (coupling × T)")
        plt.colorbar(im, ax=ax, label="ΔAUROC")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "rossler_comparison_heatmap.pdf"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_e3_results(df, output_dir):
    """Plot E3 oscillatory generalization results."""
    agg = df.groupby(["system", "surrogate"])["AUC_ROC_delta_zscore"].agg(
        ["mean", "std"]
    ).reset_index()

    systems = sorted(df["system"].unique())
    surrogates = sorted(df["surrogate"].unique())
    n_sys = len(systems)
    n_surr = len(surrogates)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_sys)
    width = 0.8 / n_surr

    for i, surr in enumerate(surrogates):
        surr_data = agg[agg["surrogate"] == surr]
        means = [surr_data[surr_data["system"] == s]["mean"].values[0]
                 if len(surr_data[surr_data["system"] == s]) > 0 else 0
                 for s in systems]
        stds = [surr_data[surr_data["system"] == s]["std"].values[0]
                if len(surr_data[surr_data["system"] == s]) > 0 else 0
                for s in systems]
        color = "#e74c3c" if "cycle_phase" in surr else "#3498db"
        ax.bar(x + i * width, means, width, yerr=stds,
               label=surr, alpha=0.8, capsize=2, color=color)

    ax.set_xticks(x + width * (n_surr - 1) / 2)
    ax.set_xticklabels(systems, rotation=30, ha="right")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("ΔAUROC (z-score − raw ρ)")
    ax.set_title("E3: Cycle-Phase Generalization Across Oscillatory Systems")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "oscillatory_comparison.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_e4_results(df, output_dir):
    """Plot E4 chaotic safety check results."""
    agg = df.groupby(["system", "surrogate"])["AUC_ROC_delta_zscore"].agg(
        ["mean", "std"]
    ).reset_index()

    systems = sorted(df["system"].unique())
    surrogates = sorted(df["surrogate"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(systems))
    width = 0.8 / len(surrogates)

    for i, surr in enumerate(surrogates):
        surr_data = agg[agg["surrogate"] == surr]
        means = [surr_data[surr_data["system"] == s]["mean"].values[0]
                 if len(surr_data[surr_data["system"] == s]) > 0 else 0
                 for s in systems]
        stds = [surr_data[surr_data["system"] == s]["std"].values[0]
                if len(surr_data[surr_data["system"] == s]) > 0 else 0
                for s in systems]
        color = "#e74c3c" if "cycle_phase" in surr else "#3498db"
        ax.bar(x + i * width, means, width, yerr=stds,
               label=surr, alpha=0.8, capsize=2, color=color)

    ax.set_xticks(x + width * (len(surrogates) - 1) / 2)
    ax.set_xticklabels(systems)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=-0.02, color="red", linestyle=":", alpha=0.5, label="harm threshold")
    ax.set_ylabel("ΔAUROC (z-score − raw ρ)")
    ax.set_title("E4: Cycle-Phase Safety Check on Broadband Chaotic Systems")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "chaotic_safety_check.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
