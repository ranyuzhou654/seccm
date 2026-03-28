"""Bivariate (2-node) sanity check experiment."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..generators import create_system, generate_network
from ._config_helpers import get_system_kwargs
from ..testing.se_ccm import SECCM
from ..visualization.convergence_plot import plot_convergence
from ..visualization.heatmap import plot_comparison_heatmaps
from ..visualization.network_plot import plot_surrogate_distribution


def run_bivariate_experiment(config, output_dir="results/bivariate"):
    """Run 2-node unidirectional sanity check for each system type.

    Tests that SE-CCM correctly detects X0 -> X1 but not X1 -> X0
    in a simple unidirectional coupling setup.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    output_dir : str
        Output directory for figures and results.
    """
    os.makedirs(output_dir, exist_ok=True)
    biv_cfg = config.get("bivariate", {})
    ts_cfg = config.get("time_series", {})
    surr_cfg = config.get("surrogate", {})

    systems = biv_cfg.get("systems", ["logistic", "lorenz", "henon"])
    coupling_cfg = biv_cfg.get("coupling_strengths", 0.1)
    n_reps = biv_cfg.get("n_reps", 10)
    T = ts_cfg.get("T", 3000)
    transient = ts_cfg.get("transient", 1000)
    n_surrogates = surr_cfg.get("n_surrogates", 100)
    seed_base = config.get("seed", 42)

    # Support per-system coupling: dict or scalar
    default_coupling = {"logistic": 0.1, "lorenz": 0.5, "henon": 0.1}
    if isinstance(coupling_cfg, dict):
        coupling_map = {k: v for k, v in coupling_cfg.items()}
    elif isinstance(coupling_cfg, (list, tuple)):
        coupling_map = {s: coupling_cfg[0] for s in systems}
    else:
        coupling_map = {s: float(coupling_cfg) for s in systems}

    # 2-node unidirectional: node 0 -> node 1
    adj = np.array([[0, 0], [1, 0]])  # A[1,0]=1 means 0->1

    all_results = {}

    for system_name in systems:
        eps = coupling_map.get(system_name, default_coupling.get(system_name, 0.1))
        sys_kwargs = get_system_kwargs(config, system_name)
        print(f"\n=== Bivariate test: {system_name} (ε={eps}) ===")
        system_results = {"TPR": [], "FPR": []}

        pbar = tqdm(total=n_reps, desc=f"{system_name}")

        for rep in range(n_reps):
            seed = seed_base + rep

            for _eps in [eps]:
                try:
                    system = create_system(system_name, adj, eps, **sys_kwargs)
                    data = system.generate(T, transient=transient, seed=seed)
                except RuntimeError as e:
                    tqdm.write(f"  Rep {rep}: {e}")
                    pbar.update(1)
                    continue

                seccm = SECCM(
                    surrogate_method="iaaft",
                    n_surrogates=n_surrogates,
                    alpha=0.05,
                    fdr=False,  # Only 2 tests, skip FDR
                    seed=seed,
                    verbose=False,
                )
                seccm.fit(data)
                metrics = seccm.score(adj)

                system_results["TPR"].append(metrics["TPR"])
                system_results["FPR"].append(metrics["FPR"])

                pbar.set_postfix(
                    TPR=f"{metrics['TPR']:.2f}",
                    FPR=f"{metrics['FPR']:.2f}",
                )
                pbar.update(1)

                # Save plots for first rep
                if rep == 0:
                    sys_dir = os.path.join(output_dir, system_name)
                    os.makedirs(sys_dir, exist_ok=True)

                    plot_comparison_heatmaps(
                        seccm.ccm_matrix_,
                        seccm.pvalue_matrix_,
                        seccm.detected_,
                        adj,
                        save_path=os.path.join(sys_dir, "heatmaps.png"),
                    )

                    # Surrogate distribution for the true edge (0->1)
                    if (1, 0) in seccm.surrogate_distributions_:
                        plot_surrogate_distribution(
                            seccm.ccm_matrix_[1, 0],
                            seccm.surrogate_distributions_[(1, 0)],
                            title=f"{system_name}: X0→X1 surrogate dist",
                            save_path=os.path.join(sys_dir, "surr_dist_0to1.png"),
                        )

        pbar.close()
        tpr_mean = np.mean(system_results["TPR"]) if system_results["TPR"] else 0
        fpr_mean = np.mean(system_results["FPR"]) if system_results["FPR"] else 0
        print(f"  Mean TPR: {tpr_mean:.3f}, Mean FPR: {fpr_mean:.3f}")
        all_results[system_name] = system_results

    return all_results
