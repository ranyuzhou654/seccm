from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surrogate_ccm.experiments.exp_diagnostic_table import (
    _build_run_args,
    _stable_seed,
)


def test_stable_seed_is_deterministic():
    seed_a = _stable_seed(42, "diagnostic_table", "realization", "lorenz", 0)
    seed_b = _stable_seed(42, "diagnostic_table", "realization", "lorenz", 0)
    seed_c = _stable_seed(42, "diagnostic_table", "realization", "lorenz", 1)

    assert seed_a == seed_b
    assert seed_a != seed_c


def test_diagnostic_args_pair_surrogates_on_same_realization():
    valid_systems = {
        "lorenz": {"N": 5, "coupling": 0.5, "T": 2000, "sys_kwargs": {}},
        "henon": {"N": 5, "coupling": 0.1, "T": 2000, "sys_kwargs": {}},
    }
    valid_surrogates = ["fft", "iaaft", "twin"]

    args_list = _build_run_args(
        valid_systems=valid_systems,
        valid_surrogates=valid_surrogates,
        n_reps=2,
        n_surrogates=19,
        base_seed=42,
    )

    grouped = {}
    for (
        system_name,
        _sys_cfg,
        surrogate,
        _n_surrogates,
        rep,
        realization_seed,
        analysis_seed,
    ) in args_list:
        key = (system_name, rep)
        grouped.setdefault(key, {"realization": set(), "analysis": {}})
        grouped[key]["realization"].add(realization_seed)
        grouped[key]["analysis"][surrogate] = analysis_seed

    assert len(args_list) == len(valid_systems) * len(valid_surrogates) * 2

    for key, seeds in grouped.items():
        assert len(seeds["realization"]) == 1, key
        assert set(seeds["analysis"]) == set(valid_surrogates), key
        assert len(set(seeds["analysis"].values())) == len(valid_surrogates), key
