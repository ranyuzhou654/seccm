"""Coupling strength sweep experiment."""

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
from tqdm import tqdm

from ..generators import create_system, generate_network
from ..testing.se_ccm import SECCM
from ..utils.io import save_results
from ..utils.parallel import parallel_map
from ..visualization.network_plot import plot_performance_curves


def _run_single_rep(args):
    """Run a single repetition for a given configuration."""
    system_name, topology, N, eps, surr_cfg, ts_cfg, seed, net_kwargs = args

    T = ts_cfg.get("T", 3000)
    transient = ts_cfg.get("transient", 1000)
    n_surrogates = surr_cfg.get("n_surrogates", 100)

    adj = generate_network(topology, N, seed=seed, **net_kwargs)

    try:
        system = create_system(system_name, adj, eps)
        data = system.generate(T, transient=transient, seed=seed)

        seccm = SECCM(
            surrogate_method="iaaft",
            n_surrogates=n_surrogates,
            alpha=0.05,
            seed=seed,
            verbose=False,
        )
        seccm.fit(data)
        return seccm.score(adj)
    except Exception:
        return None


def run_coupling_strength_experiment(config, output_dir="results/coupling_strength", n_jobs=-1):
    """Sweep coupling strength across systems and topologies.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    output_dir : str
        Output directory.
    n_jobs : int
        Number of parallel jobs.
    """
    os.makedirs(output_dir, exist_ok=True)
    cs_cfg = config.get("coupling_strength", {})
    ts_cfg = config.get("time_series", {})
    surr_cfg = config.get("surrogate", {})

    systems = cs_cfg.get("systems", ["logistic"])
    topologies = cs_cfg.get("topologies", ["ER"])
    N = cs_cfg.get("N", 10)
    coupling_values_cfg = cs_cfg.get("coupling_values", [0.01, 0.05, 0.1, 0.2, 0.5])
    n_reps = cs_cfg.get("n_reps", 20)
    seed_base = config.get("seed", 42)

    # Support per-system coupling values (dict) or shared list
    default_coupling_values = [0.01, 0.05, 0.1, 0.2, 0.5]

    net_kwargs = {
        "er_p": cs_cfg.get("er_p", 0.3),
        "ws_k": cs_cfg.get("ws_k", 4),
        "ws_p": cs_cfg.get("ws_p", 0.3),
    }

    all_results = {}
    total_combos = len(systems) * len(topologies)
    combo_pbar = tqdm(total=total_combos, desc="Coupling sweep (configs)", position=0)

    for system_name in systems:
        # Resolve per-system coupling values
        if isinstance(coupling_values_cfg, dict):
            coupling_values = coupling_values_cfg.get(system_name, default_coupling_values)
        else:
            coupling_values = coupling_values_cfg

        for topology in topologies:
            key = f"{system_name}_{topology}"
            combo_pbar.set_description(f"Coupling sweep: {key}")

            tpr_by_eps = []
            fpr_by_eps = []

            eps_pbar = tqdm(coupling_values, desc="  ε values", position=1, leave=False)
            for eps in eps_pbar:
                eps_pbar.set_description(f"  ε={eps:.3f}")

                args_list = [
                    (system_name, topology, N, eps, surr_cfg, ts_cfg, seed_base + rep, net_kwargs)
                    for rep in range(n_reps)
                ]

                results = parallel_map(
                    _run_single_rep, args_list,
                    n_jobs=n_jobs, desc=f"    reps (ε={eps:.3f})",
                )

                valid = [r for r in results if r is not None]
                if valid:
                    tpr_by_eps.append([r["TPR"] for r in valid])
                    fpr_by_eps.append([r["FPR"] for r in valid])
                else:
                    tpr_by_eps.append([0.0])
                    fpr_by_eps.append([0.0])

                tpr_mean = np.mean(tpr_by_eps[-1])
                fpr_mean = np.mean(fpr_by_eps[-1])
                eps_pbar.set_postfix(TPR=f"{tpr_mean:.3f}", FPR=f"{fpr_mean:.3f}")

            eps_pbar.close()
            combo_pbar.update(1)

            # Plot
            metrics_plot = {
                "TPR": np.array([np.mean(t) for t in tpr_by_eps]),
                "FPR": np.array([np.mean(f) for f in fpr_by_eps]),
            }
            plot_performance_curves(
                coupling_values,
                metrics_plot,
                x_label="Coupling strength ε",
                title=f"Detection Performance: {key}",
                save_path=os.path.join(output_dir, f"{key}_performance.png"),
            )

            all_results[key] = {
                "coupling_values": coupling_values,
                "TPR": tpr_by_eps,
                "FPR": fpr_by_eps,
            }

    combo_pbar.close()
    return all_results
