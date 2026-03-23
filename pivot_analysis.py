#!/usr/bin/env python
"""Three-pivot diagnostic analysis for SE-CCM."""

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

from surrogate_ccm.generators import create_system, generate_network, SYSTEM_CLASSES
from surrogate_ccm.testing.se_ccm import SECCM
from surrogate_ccm.surrogate import generate_surrogate, SURROGATE_METHODS
from surrogate_ccm.surrogate.adaptive import spectral_concentration, autocorrelation_decay_time
from surrogate_ccm.evaluation.metrics import compute_sso
from surrogate_ccm.utils.parallel import parallel_map

OUT = "results/pivot_analysis"
os.makedirs(OUT, exist_ok=True)
N_JOBS = 5

# ============================================================
# Pivot 1: cycle_phase advantage on phase-coupled oscillators
# ============================================================
def pivot1_worker(args):
    sys_name, coupling, T, surr, N, seed, sys_kw = args
    try:
        adj = generate_network("ER", N, seed=seed, p=0.5)
        system = create_system(sys_name, adj, coupling, **sys_kw)
        data = system.generate(T, transient=1000, seed=seed)
        seccm = SECCM(surrogate_method=surr, n_surrogates=50,
                       alpha=0.05, fdr=True, seed=seed, verbose=False)
        seccm.fit(data)
        m = seccm.score(adj)
        return {"system": sys_name, "coupling": coupling, "T": T,
                "surrogate": surr, "seed": seed,
                "AUC_ROC_rho": m.get("AUC_ROC_rho", np.nan),
                "AUC_ROC_zscore": m.get("AUC_ROC_zscore", np.nan),
                "AUC_ROC_delta_zscore": m.get("AUC_ROC_delta_zscore", np.nan)}
    except Exception as e:
        return {"system": sys_name, "surrogate": surr, "error": str(e)}


def run_pivot1():
    print("\n" + "="*60)
    print("PIVOT 1: Cycle-phase advantage on phase-coupled oscillators")
    print("="*60)

    configs = {
        "kuramoto": {"T": 2000, "couplings": [0.1, 0.3, 0.5, 0.8, 1.0],
                     "sys_kw": {}},
        "van_der_pol": {"T": 5000, "couplings": [0.1, 0.2, 0.3, 0.5, 0.8],
                        "sys_kw": {}},
        "rossler": {"T": 5000, "couplings": [0.05, 0.1, 0.15, 0.2, 0.3],
                    "sys_kw": {"dt": 0.05}},
    }
    surrogates = ["iaaft", "fft", "cycle_phase_A", "cycle_phase_B"]
    N, n_reps = 5, 5

    args_list = []
    for sys_name, cfg in configs.items():
        for coup in cfg["couplings"]:
            for surr in surrogates:
                for rep in range(n_reps):
                    seed = 100 + hash((sys_name, coup, surr, rep)) % (2**31)
                    args_list.append((sys_name, coup, cfg["T"], surr,
                                     N, seed, cfg["sys_kw"]))

    print(f"  {len(args_list)} runs")
    results = parallel_map(pivot1_worker, args_list, n_jobs=N_JOBS,
                           desc="Pivot 1")
    rows = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    if errs:
        print(f"  {len(errs)} failures")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "pivot1_raw.csv"), index=False)

    # Analyze: cycle_phase_A vs iaaft by system × coupling
    for sys_name in configs:
        sub = df[df["system"] == sys_name]
        if len(sub) == 0:
            continue
        pivot = sub.groupby(["coupling", "surrogate"])["AUC_ROC_delta_zscore"].mean().unstack()
        print(f"\n  {sys_name} — ΔAUROC by coupling × surrogate:")
        print(pivot.to_string(float_format="%.4f"))

        # Where does cycle_phase_A beat iaaft?
        if "cycle_phase_A" in pivot.columns and "iaaft" in pivot.columns:
            diff = pivot["cycle_phase_A"] - pivot["iaaft"]
            wins = diff[diff > 0]
            print(f"  → cycle_phase_A beats iaaft at {len(wins)}/{len(diff)} "
                  f"coupling values (mean advantage: {diff.mean():.4f})")

    return df


# ============================================================
# Pivot 2: Diagnose why ALL surrogates fail on Rossler
# ============================================================
def pivot2_worker(args):
    sys_name, coupling, T, N, seed, sys_kw = args
    try:
        adj = generate_network("ER", N, seed=seed, p=0.5)
        system = create_system(sys_name, adj, coupling, **sys_kw)
        data = system.generate(T, transient=1000, seed=seed)

        results = {"system": sys_name, "coupling": coupling, "seed": seed}

        # Raw CCM (no surrogates) — just AUROC
        seccm_raw = SECCM(surrogate_method="iaaft", n_surrogates=50,
                          alpha=0.05, fdr=True, seed=seed, verbose=False)
        seccm_raw.fit(data)
        m_raw = seccm_raw.score(adj)
        results["AUC_ROC_rho"] = m_raw.get("AUC_ROC_rho", np.nan)
        results["AUC_ROC_zscore_iaaft"] = m_raw.get("AUC_ROC_zscore", np.nan)
        results["delta_iaaft"] = m_raw.get("AUC_ROC_delta_zscore", np.nan)

        # Surrogate distribution diagnostics
        mask = ~np.eye(N, dtype=bool)
        if seccm_raw.surrogate_distributions_:
            # Mean surrogate rho vs observed rho
            rho_obs_vals = []
            rho_surr_means = []
            rho_surr_stds = []
            for (i, j), rho_surr in seccm_raw.surrogate_distributions_.items():
                rho_obs_vals.append(seccm_raw.ccm_matrix_[i, j])
                rho_surr_means.append(np.mean(rho_surr))
                rho_surr_stds.append(np.std(rho_surr))
            results["mean_rho_obs"] = float(np.mean(rho_obs_vals))
            results["mean_rho_surr"] = float(np.mean(rho_surr_means))
            results["mean_rho_gap"] = float(np.mean(rho_obs_vals) -
                                            np.mean(rho_surr_means))
            results["mean_surr_std"] = float(np.mean(rho_surr_stds))
            # Null-alt overlap: fraction of surr rho >= obs rho
            overlap_fracs = []
            for (i, j), rho_surr in seccm_raw.surrogate_distributions_.items():
                obs = seccm_raw.ccm_matrix_[i, j]
                overlap_fracs.append(np.mean(rho_surr >= obs))
            results["mean_null_alt_overlap"] = float(np.mean(overlap_fracs))

        # Convergence score
        if hasattr(seccm_raw, 'convergence_matrix_'):
            conv = seccm_raw.convergence_matrix_
            conv_off = conv[mask]
            results["mean_convergence"] = float(np.nanmean(conv_off))
            results["frac_converging"] = float(np.mean(conv_off > 0))

        # Signal properties
        sc_vals = [spectral_concentration(data[:, i]) for i in range(N)]
        acf_vals = [autocorrelation_decay_time(data[:, i]) for i in range(N)]
        results["mean_spectral_conc"] = float(np.mean(sc_vals))
        results["mean_acf_decay"] = float(np.mean(acf_vals))

        return results
    except Exception as e:
        return {"system": sys_name, "error": str(e)}


def run_pivot2():
    print("\n" + "="*60)
    print("PIVOT 2: Diagnose why ALL surrogates fail on Rossler")
    print("="*60)

    configs = {
        "logistic": {"coupling": 0.3, "T": 2000, "sys_kw": {}},
        "henon": {"coupling": 0.2, "T": 2000, "sys_kw": {}},  # lower coupling
        "lorenz": {"coupling": 0.5, "T": 2000, "sys_kw": {"dt": 0.01}},
        "rossler": {"coupling": 0.15, "T": 5000, "sys_kw": {"dt": 0.05}},
        "kuramoto": {"coupling": 0.5, "T": 2000, "sys_kw": {}},
        "van_der_pol": {"coupling": 0.3, "T": 5000, "sys_kw": {}},
        "fitzhugh_nagumo": {"coupling": 0.3, "T": 2000, "sys_kw": {}},
    }
    N, n_reps = 5, 5

    args_list = []
    for sys_name, cfg in configs.items():
        for rep in range(n_reps):
            seed = 200 + hash((sys_name, rep)) % (2**31)
            args_list.append((sys_name, cfg["coupling"], cfg["T"],
                              N, seed, cfg["sys_kw"]))

    print(f"  {len(args_list)} runs")
    results = parallel_map(pivot2_worker, args_list, n_jobs=N_JOBS,
                           desc="Pivot 2")
    rows = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    if errs:
        err_sys = [e.get("system", "?") for e in errs]
        print(f"  {len(errs)} failures: {err_sys}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "pivot2_diagnostics.csv"), index=False)

    # Aggregate
    diag_cols = ["AUC_ROC_rho", "delta_iaaft", "mean_rho_obs",
                 "mean_rho_surr", "mean_rho_gap", "mean_surr_std",
                 "mean_null_alt_overlap", "mean_convergence",
                 "frac_converging", "mean_spectral_conc", "mean_acf_decay"]
    agg = df.groupby("system")[diag_cols].mean()
    print("\n  Diagnostic summary across systems:")
    print(agg.to_string(float_format="%.4f"))

    # Key insight: compare rho_gap and null_alt_overlap
    print("\n  Interpretation:")
    for sys_name in agg.index:
        row = agg.loc[sys_name]
        gap = row["mean_rho_gap"]
        overlap = row["mean_null_alt_overlap"]
        auroc = row["AUC_ROC_rho"]
        delta = row["delta_iaaft"]
        conv = row["mean_convergence"]
        print(f"  {sys_name:20s}: raw_AUROC={auroc:.3f}  Δ_iaaft={delta:+.3f}  "
              f"rho_gap={gap:.3f}  null_overlap={overlap:.3f}  conv={conv:.3f}")

    return df


# ============================================================
# Pivot 3: Multi-feature predictor
# ============================================================
def permutation_entropy(x, order=3, delay=1):
    """Compute permutation entropy (Bandt & Pompe 2002)."""
    x = np.asarray(x, dtype=float).ravel()
    T = len(x)
    from math import factorial
    n_perms = factorial(order)
    perm_counts = {}
    for i in range(T - (order - 1) * delay):
        pattern = tuple(np.argsort(x[i:i + order * delay:delay]))
        perm_counts[pattern] = perm_counts.get(pattern, 0) + 1
    total = sum(perm_counts.values())
    probs = np.array(list(perm_counts.values())) / total
    return -np.sum(probs * np.log2(probs)) / np.log2(n_perms)  # normalized


def recurrence_rate(x, threshold_pct=10, max_points=1000):
    """Estimate recurrence rate from time series."""
    x = np.asarray(x, dtype=float).ravel()
    if len(x) > max_points:
        x = x[::len(x) // max_points]
    diffs = np.abs(x[:, None] - x[None, :])
    threshold = np.percentile(diffs, threshold_pct)
    return float(np.mean(diffs < threshold))


def lyapunov_proxy(x, lag=1):
    """Simple proxy for largest Lyapunov exponent: mean log divergence rate."""
    x = np.asarray(x, dtype=float).ravel()
    dx = np.abs(np.diff(x))
    dx = dx[dx > 1e-12]
    if len(dx) < 10:
        return 0.0
    return float(np.mean(np.log(dx + 1e-12)))


def pivot3_worker(args):
    sys_name, coupling, T, surr, N, seed, sys_kw = args
    try:
        adj = generate_network("ER", N, seed=seed, p=0.5)
        system = create_system(sys_name, adj, coupling, **sys_kw)
        data = system.generate(T, transient=1000, seed=seed)

        # Run SE-CCM
        seccm = SECCM(surrogate_method=surr, n_surrogates=50,
                       alpha=0.05, fdr=True, seed=seed, verbose=False)
        seccm.fit(data)
        m = seccm.score(adj)

        # Compute SSO
        sso_vals = []
        for j in range(N):
            surrs_j = generate_surrogate(data[:, j], method=surr,
                                         n_surrogates=20, seed=seed+j+500)
            sso_vals.append(compute_sso(data[:, j], surrs_j))
        mean_sso = float(np.mean(sso_vals))

        # Compute signal features (averaged across nodes)
        sc = float(np.mean([spectral_concentration(data[:, i]) for i in range(N)]))
        acf = float(np.mean([autocorrelation_decay_time(data[:, i]) for i in range(N)]))
        pe = float(np.mean([permutation_entropy(data[:, i]) for i in range(N)]))
        rr = float(np.mean([recurrence_rate(data[:, i]) for i in range(N)]))
        lyap = float(np.mean([lyapunov_proxy(data[:, i]) for i in range(N)]))

        return {
            "system": sys_name, "surrogate": surr, "coupling": coupling,
            "seed": seed,
            "SSO": mean_sso,
            "spectral_conc": sc, "acf_decay": acf,
            "perm_entropy": pe, "recurrence_rate": rr,
            "lyapunov_proxy": lyap,
            "AUC_ROC_rho": m.get("AUC_ROC_rho", np.nan),
            "AUC_ROC_delta_zscore": m.get("AUC_ROC_delta_zscore", np.nan),
        }
    except Exception as e:
        return {"system": sys_name, "surrogate": surr, "error": str(e)}


def run_pivot3():
    print("\n" + "="*60)
    print("PIVOT 3: Multi-feature predictor of surrogate utility")
    print("="*60)

    systems = {
        "logistic": {"coupling": 0.3, "T": 2000, "sys_kw": {}},
        "lorenz": {"coupling": 0.5, "T": 2000, "sys_kw": {"dt": 0.01}},
        "rossler": {"coupling": 0.15, "T": 5000, "sys_kw": {"dt": 0.05}},
        "kuramoto": {"coupling": 0.5, "T": 2000, "sys_kw": {}},
        "van_der_pol": {"coupling": 0.3, "T": 5000, "sys_kw": {}},
        "fitzhugh_nagumo": {"coupling": 0.3, "T": 2000, "sys_kw": {}},
    }
    surrogates = ["fft", "iaaft", "random_reorder", "cycle_shuffle",
                  "cycle_phase_A", "cycle_phase_B"]
    N, n_reps = 5, 3

    args_list = []
    for sys_name, cfg in systems.items():
        for surr in surrogates:
            for rep in range(n_reps):
                seed = 300 + hash((sys_name, surr, rep)) % (2**31)
                args_list.append((sys_name, cfg["coupling"], cfg["T"],
                                  surr, N, seed, cfg["sys_kw"]))

    print(f"  {len(args_list)} runs")
    results = parallel_map(pivot3_worker, args_list, n_jobs=N_JOBS,
                           desc="Pivot 3")
    rows = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    if errs:
        print(f"  {len(errs)} failures")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "pivot3_features.csv"), index=False)

    # Aggregate by system × surrogate
    agg = df.groupby(["system", "surrogate"]).agg({
        "SSO": "mean", "spectral_conc": "mean", "acf_decay": "mean",
        "perm_entropy": "mean", "recurrence_rate": "mean",
        "lyapunov_proxy": "mean",
        "AUC_ROC_rho": "mean", "AUC_ROC_delta_zscore": "mean",
    }).reset_index()

    # Test each feature as predictor of ΔAUROC
    features = ["SSO", "spectral_conc", "acf_decay", "perm_entropy",
                "recurrence_rate", "lyapunov_proxy"]
    valid = agg.dropna(subset=["AUC_ROC_delta_zscore"])
    target = valid["AUC_ROC_delta_zscore"]

    print(f"\n  Feature correlations with ΔAUROC ({len(valid)} datapoints):")
    corr_results = {}
    for feat in features:
        if feat in valid.columns:
            r, p = stats.spearmanr(valid[feat], target)
            corr_results[feat] = {"spearman_r": r, "p_value": p}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {feat:20s}: ρ = {r:+.4f}  (p = {p:.3e}) {sig}")

    # Multi-feature: simple linear combination
    from numpy.linalg import lstsq
    feat_matrix = valid[features].values
    # Standardize
    feat_mean = feat_matrix.mean(axis=0)
    feat_std = feat_matrix.std(axis=0) + 1e-12
    feat_norm = (feat_matrix - feat_mean) / feat_std
    # Add intercept
    X = np.column_stack([feat_norm, np.ones(len(feat_norm))])
    y = target.values
    coeffs, residuals, _, _ = lstsq(X, y, rcond=None)

    y_pred = X @ coeffs
    r_multi, p_multi = stats.spearmanr(y, y_pred)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

    print(f"\n  Multi-feature linear model:")
    print(f"    R² = {r2:.4f}")
    print(f"    Spearman ρ = {r_multi:.4f}  (p = {p_multi:.3e})")
    print(f"    Coefficients (standardized):")
    for feat, coeff in zip(features, coeffs[:-1]):
        print(f"      {feat:20s}: {coeff:+.4f}")

    with open(os.path.join(OUT, "pivot3_correlations.json"), "w") as f:
        json.dump({
            "single_feature": {k: {kk: float(vv) for kk, vv in v.items()}
                               for k, v in corr_results.items()},
            "multi_feature": {
                "R2": float(r2), "spearman_r": float(r_multi),
                "spearman_p": float(p_multi),
                "coefficients": {feat: float(c)
                                 for feat, c in zip(features, coeffs[:-1])},
            }
        }, f, indent=2)

    return df


if __name__ == "__main__":
    df1 = run_pivot1()
    df2 = run_pivot2()
    df3 = run_pivot3()

    print("\n" + "="*60)
    print("ALL PIVOTS COMPLETE")
    print(f"Results saved to: {OUT}/")
    print("="*60)
