#!/usr/bin/env python
"""Main entry point for SE-CCM experiments."""

import argparse
import os
import sys
import time

from surrogate_ccm.experiments import (
    run_bivariate_experiment,
    run_coupling_strength_experiment,
    run_network_topology_experiment,
    run_noise_experiment,
    run_surrogate_comparison_experiment,
    run_surrogate_robustness_experiment,
    run_trivariate_experiment,
)
from surrogate_ccm.utils.io import load_config


EXPERIMENTS = {
    "bivariate": run_bivariate_experiment,
    "coupling": run_coupling_strength_experiment,
    "noise": run_noise_experiment,
    "topology": run_network_topology_experiment,
    "surrogate": run_surrogate_comparison_experiment,
    "robustness": run_surrogate_robustness_experiment,
    "trivariate": run_trivariate_experiment,
}


def main():
    parser = argparse.ArgumentParser(
        description="SE-CCM: Surrogate-Enhanced CCM Experiments"
    )
    parser.add_argument(
        "--experiment",
        choices=["all"] + list(EXPERIMENTS.keys()),
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.n_jobs is not None:
        config["n_jobs"] = args.n_jobs

    output_base = args.output_dir or config.get("output_dir", "results")

    # Determine experiments to run
    if args.experiment == "all":
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = [args.experiment]

    n_jobs = config.get("n_jobs", -1)

    for exp_name in experiments_to_run:
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*60}")

        exp_dir = os.path.join(output_base, exp_name)
        start = time.time()

        func = EXPERIMENTS[exp_name]

        if exp_name == "bivariate":
            func(config, output_dir=exp_dir)
        else:
            func(config, output_dir=exp_dir, n_jobs=n_jobs)

        elapsed = time.time() - start
        print(f"\n  Completed {exp_name} in {elapsed:.1f}s")

    print(f"\nAll experiments complete. Results saved to: {output_base}/")


if __name__ == "__main__":
    main()
