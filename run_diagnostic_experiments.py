#!/usr/bin/env python
"""Run all diagnostic framework experiments (D1, D2, E5, E7).

Usage:
    # Run all with default settings (n_jobs=8)
    python run_diagnostic_experiments.py

    # Feasibility mode (small scale, fast)
    python run_diagnostic_experiments.py --feasibility --n-jobs 5

    # Run specific experiment(s)
    python run_diagnostic_experiments.py --only D1 D2

    # Custom parallelism
    python run_diagnostic_experiments.py --n-jobs 4

    # Custom reps (reduce for testing)
    python run_diagnostic_experiments.py --n-reps 3 --n-surrogates 30
"""

import argparse
import os
import sys
import time

from surrogate_ccm.experiments import (
    run_diagnostic_table_experiment,
    run_regime_boundary_experiment,
    run_convergence_experiment,
    run_noise_robustness_experiment,
)


def build_config(args):
    """Build experiment config from CLI args."""

    if args.feasibility:
        # Small-scale feasibility run
        n_reps = 3
        n_surrogates = 30
    else:
        n_reps = args.n_reps
        n_surrogates = args.n_surrogates

    config = {
        "diagnostic_table": {
            "n_surrogates": n_surrogates,
            "n_reps": n_reps,
            "seed": 42,
            # Use defaults from exp_diagnostic_table.py
        },
        "regime_boundaries": {
            "n_surrogates": n_surrogates,
            "n_reps": n_reps,
            "N": 5,
            "seed": 42,
        },
        "convergence": {
            "n_surrogates": n_surrogates,
            "n_reps": n_reps,
            "N": 5,
            "seed": 42,
        },
        "noise_robustness": {
            "n_surrogates": n_surrogates,
            "n_reps": n_reps,
            "N": 5,
            "seed": 42,
        },
    }

    if args.feasibility:
        # Reduce system and surrogate count for feasibility
        config["diagnostic_table"]["systems"] = {
            "logistic":    {"N": 5, "coupling": 0.3,  "T": 2000, "sys_kwargs": {}},
            "rossler":     {"N": 5, "coupling": 0.15, "T": 5000,
                            "sys_kwargs": {"dt": 0.05}},
            "kuramoto":    {"N": 5, "coupling": 0.5,  "T": 2000, "sys_kwargs": {}},
            "van_der_pol": {"N": 5, "coupling": 0.3,  "T": 5000, "sys_kwargs": {}},
        }
        config["diagnostic_table"]["surrogates"] = [
            "iaaft", "fft", "cycle_phase_A", "random_reorder",
        ]

        config["regime_boundaries"]["systems"] = {
            "rossler":     {"couplings": [0.05, 0.15, 0.3],
                            "T": 5000, "sys_kwargs": {"dt": 0.05}},
            "van_der_pol": {"couplings": [0.1, 0.3, 0.8],
                            "T": 5000, "sys_kwargs": {}},
        }

        config["convergence"]["systems"] = {
            "van_der_pol": {"coupling": 0.3,
                            "T_values": [500, 2000, 5000],
                            "sys_kwargs": {}},
        }

        config["noise_robustness"]["systems"] = {
            "van_der_pol": {"coupling": 0.3, "T": 5000, "sys_kwargs": {}},
        }
        config["noise_robustness"]["noise_levels"] = [0.0, 0.1, 0.3]

    return config


EXPERIMENTS = {
    "D1": ("diagnostic_table",  run_diagnostic_table_experiment),
    "D2": ("regime_boundaries", run_regime_boundary_experiment),
    "E5": ("convergence",       run_convergence_experiment),
    "E7": ("noise_robustness",  run_noise_robustness_experiment),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run diagnostic framework experiments (D1, D2, E5, E7)"
    )
    parser.add_argument(
        "--only", nargs="+", choices=list(EXPERIMENTS.keys()),
        default=None,
        help="Run only specific experiments (e.g. --only D1 E5)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=8,
        help="Number of parallel jobs (default: 8)",
    )
    parser.add_argument(
        "--n-reps", type=int, default=20,
        help="Number of replicates per condition (default: 20)",
    )
    parser.add_argument(
        "--n-surrogates", type=int, default=100,
        help="Number of surrogates per test (default: 100)",
    )
    parser.add_argument(
        "--feasibility", action="store_true",
        help="Run small-scale feasibility test first",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Base output directory (default: results)",
    )
    args = parser.parse_args()

    config = build_config(args)
    to_run = args.only or list(EXPERIMENTS.keys())
    output_base = args.output_dir

    mode_label = "FEASIBILITY" if args.feasibility else "FULL"
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC FRAMEWORK EXPERIMENTS ({mode_label})")
    print(f"{'='*60}")
    print(f"  Experiments: {to_run}")
    print(f"  n_jobs: {args.n_jobs}")
    if not args.feasibility:
        print(f"  n_reps: {args.n_reps}")
        print(f"  n_surrogates: {args.n_surrogates}")
    print(f"  Output: {output_base}/")
    print()

    total_start = time.time()
    results = {}

    for exp_id in to_run:
        dir_name, func = EXPERIMENTS[exp_id]
        exp_dir = os.path.join(output_base, dir_name)

        print(f"\n{'='*60}")
        print(f"  [{exp_id}] Starting: {dir_name}")
        print(f"{'='*60}")

        start = time.time()
        try:
            df = func(config, output_dir=exp_dir, n_jobs=args.n_jobs)
            elapsed = time.time() - start
            n_rows = len(df) if df is not None else 0
            results[exp_id] = {"status": "OK", "rows": n_rows,
                               "time": elapsed}
            print(f"\n  [{exp_id}] Done: {n_rows} results in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start
            results[exp_id] = {"status": "FAILED", "error": str(e),
                               "time": elapsed}
            print(f"\n  [{exp_id}] FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE ({total_elapsed:.0f}s total)")
    print(f"{'='*60}")
    for exp_id, info in results.items():
        status = info["status"]
        t = info["time"]
        if status == "OK":
            print(f"  [{exp_id}] {status} — {info['rows']} rows in {t:.0f}s")
        else:
            print(f"  [{exp_id}] {status} — {info.get('error', '?')}")
    print(f"\n  Results saved to: {output_base}/")


if __name__ == "__main__":
    main()
