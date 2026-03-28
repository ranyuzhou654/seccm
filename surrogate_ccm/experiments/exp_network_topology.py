"""Network topology comparison experiment."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..generators import create_system, generate_network
from ._config_helpers import get_system_kwargs
from ..testing.se_ccm import SECCM
from ..utils.parallel import parallel_map
from ..visualization.network_plot import plot_performance_curves


def _run_single_rep(args):
    """Run a single repetition for a given topology and network size."""
    system_name, topology, N, eps, surr_cfg, ts_cfg, seed, net_kwargs, sys_kwargs = args

    T = ts_cfg.get("T", 3000)
    transient = ts_cfg.get("transient", 1000)
    n_surrogates = surr_cfg.get("n_surrogates", 100)

    adj = generate_network(topology, N, seed=seed, **net_kwargs)

    try:
        system = create_system(system_name, adj, eps, **sys_kwargs)
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


def run_network_topology_experiment(config, output_dir="results/topology", n_jobs=-1):
    """Compare ER vs WS vs Ring at different network sizes.

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
    topo_cfg = config.get("network_topology", {})
    ts_cfg = config.get("time_series", {})
    surr_cfg = config.get("surrogate", {})

    systems = topo_cfg.get("systems", ["logistic"])
    topologies = topo_cfg.get("topologies", ["ER", "WS", "ring"])
    N_values = topo_cfg.get("N_values", [10, 20, 30])
    coupling_cfg = topo_cfg.get("coupling", 0.1)
    n_reps = topo_cfg.get("n_reps", 20)
    seed_base = config.get("seed", 42)

    # Default per-system coupling
    default_coupling = {"logistic": 0.1, "lorenz": 1.0, "henon": 0.05}

    net_kwargs = {
        "er_p": topo_cfg.get("er_p", 0.3),
        "ws_k": topo_cfg.get("ws_k", 4),
        "ws_p": topo_cfg.get("ws_p", 0.3),
    }

    all_results = {}
    total_combos = len(systems) * len(topologies)
    combo_pbar = tqdm(total=total_combos, desc="Topology experiment", position=0)
    system_kwargs_map = {sys: get_system_kwargs(config, sys) for sys in systems}

    for system_name in systems:
        # Resolve per-system coupling
        if isinstance(coupling_cfg, dict):
            coupling = coupling_cfg.get(system_name, default_coupling.get(system_name, 0.1))
        else:
            coupling = float(coupling_cfg)

        for topology in topologies:
            combo_pbar.set_description(f"Topology: {system_name}/{topology}")

            tpr_by_N = []
            fpr_by_N = []

            n_pbar = tqdm(N_values, desc=f"  N values ({topology})", position=1, leave=False)
            for N in n_pbar:
                n_pbar.set_description(f"  {topology} N={N}")

                args_list = [
                    (system_name, topology, N, coupling,
                     surr_cfg, ts_cfg, seed_base + rep, net_kwargs,
                     system_kwargs_map.get(system_name, {}))
                    for rep in range(n_reps)
                ]

                results = parallel_map(
                    _run_single_rep, args_list,
                    n_jobs=n_jobs, desc=f"    reps ({topology} N={N})",
                )

                valid = [r for r in results if r is not None]
                if valid:
                    tpr_by_N.append([r["TPR"] for r in valid])
                    fpr_by_N.append([r["FPR"] for r in valid])
                else:
                    tpr_by_N.append([0.0])
                    fpr_by_N.append([0.0])

                n_pbar.set_postfix(
                    TPR=f"{np.mean(tpr_by_N[-1]):.3f}",
                    FPR=f"{np.mean(fpr_by_N[-1]):.3f}",
                )

            n_pbar.close()
            combo_pbar.update(1)

            all_results[f"{system_name}_{topology}"] = {
                "N_values": N_values,
                "TPR": tpr_by_N,
                "FPR": fpr_by_N,
            }

        # Summary plot: TPR by topology and N
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for topology in topologies:
            key = f"{system_name}_{topology}"
            r = all_results[key]
            tpr_means = [np.mean(t) for t in r["TPR"]]
            fpr_means = [np.mean(f) for f in r["FPR"]]
            axes[0].plot(N_values, tpr_means, "-o", label=topology)
            axes[1].plot(N_values, fpr_means, "-o", label=topology)

        axes[0].set_xlabel("Network size N")
        axes[0].set_ylabel("TPR")
        axes[0].set_title(f"{system_name}: TPR by Topology")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Network size N")
        axes[1].set_ylabel("FPR")
        axes[1].set_title(f"{system_name}: FPR by Topology")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{system_name}_topology_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    combo_pbar.close()
    return all_results
