"""Surrogate data generation methods with factory dispatch."""

import numpy as np

from .aaft_surrogate import aaft_surrogate
from .fft_surrogate import fft_surrogate
from .iaaft_surrogate import iaaft_surrogate
from .random_reorder import random_reorder_surrogate
from .timeshift_surrogate import timeshift_surrogate
from .cycle_shuffle_surrogate import cycle_shuffle_surrogate
from .twin_surrogate import twin_surrogate, _precompute_twins

SURROGATE_METHODS = {
    "fft": fft_surrogate,
    "aaft": aaft_surrogate,
    "iaaft": iaaft_surrogate,
    "timeshift": timeshift_surrogate,
    "random_reorder": random_reorder_surrogate,
    "cycle_shuffle": cycle_shuffle_surrogate,
    "twin": twin_surrogate,
}


def generate_surrogate(x, method="iaaft", n_surrogates=100, seed=None, **kwargs):
    """Generate multiple surrogate time series.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    method : str
        Surrogate method name: 'fft', 'aaft', 'iaaft', 'timeshift', 'random_reorder'.
    n_surrogates : int
        Number of surrogates to generate.
    seed : int, optional
        Base random seed.
    **kwargs
        Additional keyword arguments passed to the surrogate function.

    Returns
    -------
    surrogates : ndarray, shape (n_surrogates, T)
        Array of surrogate time series.
    """
    func = SURROGATE_METHODS.get(method.lower())
    if func is None:
        raise ValueError(
            f"Unknown surrogate method: {method}. "
            f"Choose from {list(SURROGATE_METHODS)}"
        )

    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, len(x)))

    # For twin surrogates, pre-compute the expensive embedding/neighbour/twin
    # structure once and reuse across all n_surrogates calls.
    if method.lower() == "twin":
        from ..ccm.embedding import select_parameters

        x_arr = np.asarray(x, dtype=float).ravel()
        E = kwargs.get("E")
        tau = kwargs.get("tau")
        if E is None or tau is None:
            E_auto, tau_auto = select_parameters(x_arr)
            if E is None:
                E = E_auto
            if tau is None:
                tau = tau_auto
        cache = _precompute_twins(
            x_arr, E, tau,
            epsilon=kwargs.get("epsilon"),
            target_rr=kwargs.get("target_rr", 0.05),
            min_dist=kwargs.get("min_dist", 7),
            rng=rng,
        )
        for i in range(n_surrogates):
            surrogates[i] = func(x, rng=rng, _twin_cache=cache, **kwargs)
    else:
        for i in range(n_surrogates):
            surrogates[i] = func(x, rng=rng, **kwargs)

    return surrogates
