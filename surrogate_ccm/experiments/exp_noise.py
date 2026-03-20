"""Noise level sweep experiment."""

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
from tqdm import tqdm

from ..generators import create_system, generate_network
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map
from ..visualization.network_plot import plot_performance_curves


def _run_single_rep(args):
    """Run a single repetition for a given noise level."""
    system_name, topology, N, eps, noise_std, surr_cfg, ts_cfg, seed, net_kwargs = args

    T = ts_cfg.get("T", 3000)
    transient = ts_cfg.get("transient", 1000)
    n_surrogates = surr_cfg.get("n_surrogates", 100)

    adj = generate_network(topology, N, seed=seed, **net_kwargs)

    try:
        system = create_system(system_name, adj, eps)
        data = system.generate(T, transient=transient, seed=seed, noise_std=noise_std)

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


def run_noise_experiment(config, output_dir="results/noise", n_jobs=-1):
    """Sweep observation noise levels at fixed coupling.

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
    noise_cfg = config.get("noise", {})
    ts_cfg = config.get("time_series", {})
    surr_cfg = config.get("surrogate", {})

    systems = noise_cfg.get("systems", ["logistic"])
    topologies = noise_cfg.get("topologies", ["ER"])
    N = noise_cfg.get("N", 10)
    coupling_cfg = noise_cfg.get("coupling", 0.1)
    noise_levels = noise_cfg.get("noise_levels", [0.0, 0.01, 0.05, 0.1])
    n_reps = noise_cfg.get("n_reps", 20)
    seed_base = config.get("seed", 42)

    # Default per-system coupling
    default_coupling = {"logistic": 0.1, "lorenz": 1.0, "henon": 0.05}

    net_kwargs = {"er_p": noise_cfg.get("er_p", 0.3)}

    all_results = {}
    total_combos = len(systems) * len(topologies)
    combo_pbar = tqdm(total=total_combos, desc="Noise sweep (configs)", position=0)

    for system_name in systems:
        # Resolve per-system coupling
        if isinstance(coupling_cfg, dict):
            coupling = coupling_cfg.get(system_name, default_coupling.get(system_name, 0.1))
        else:
            coupling = float(coupling_cfg)

        for topology in topologies:
            key = f"{system_name}_{topology}"
            combo_pbar.set_description(f"Noise sweep: {key}")

            tpr_by_noise = []
            fpr_by_noise = []

            noise_pbar = tqdm(noise_levels, desc="  σ values", position=1, leave=False)
            for noise_std in noise_pbar:
                noise_pbar.set_description(f"  σ={noise_std:.3f}")

                args_list = [
                    (system_name, topology, N, coupling, noise_std,
                     surr_cfg, ts_cfg, seed_base + rep, net_kwargs)
                    for rep in range(n_reps)
                ]

                results = parallel_map(
                    _run_single_rep, args_list,
                    n_jobs=n_jobs, desc=f"    reps (σ={noise_std:.3f})",
                )

                valid = [r for r in results if r is not None]
                if valid:
                    tpr_by_noise.append([r["TPR"] for r in valid])
                    fpr_by_noise.append([r["FPR"] for r in valid])
                else:
                    tpr_by_noise.append([0.0])
                    fpr_by_noise.append([0.0])

                noise_pbar.set_postfix(
                    TPR=f"{np.mean(tpr_by_noise[-1]):.3f}",
                    FPR=f"{np.mean(fpr_by_noise[-1]):.3f}",
                )

            noise_pbar.close()
            combo_pbar.update(1)

            metrics_plot = {
                "TPR": np.array([np.mean(t) for t in tpr_by_noise]),
                "FPR": np.array([np.mean(f) for f in fpr_by_noise]),
            }
            plot_performance_curves(
                noise_levels,
                metrics_plot,
                x_label="Noise σ",
                title=f"Noise Robustness: {key}",
                save_path=os.path.join(output_dir, f"{key}_noise.png"),
            )

            all_results[key] = {
                "noise_levels": noise_levels,
                "TPR": tpr_by_noise,
                "FPR": fpr_by_noise,
            }

    combo_pbar.close()
    return all_results
