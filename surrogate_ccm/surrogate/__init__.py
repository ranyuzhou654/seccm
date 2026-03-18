"""Surrogate data generation methods with factory dispatch."""

import numpy as np

from .aaft_surrogate import aaft_surrogate
from .fft_surrogate import fft_surrogate
from .iaaft_surrogate import iaaft_surrogate
from .random_reorder import random_reorder_surrogate
from .timeshift_surrogate import timeshift_surrogate
from .cycle_shuffle_surrogate import cycle_shuffle_surrogate

SURROGATE_METHODS = {
    "fft": fft_surrogate,
    "aaft": aaft_surrogate,
    "iaaft": iaaft_surrogate,
    "timeshift": timeshift_surrogate,
    "random_reorder": random_reorder_surrogate,
    "cycle_shuffle": cycle_shuffle_surrogate,
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

    for i in range(n_surrogates):
        surrogates[i] = func(x, rng=rng, **kwargs)

    return surrogates
