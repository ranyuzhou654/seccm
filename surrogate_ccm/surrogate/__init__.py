"""Surrogate data generation methods with factory dispatch."""

import numpy as np

from .aaft_surrogate import aaft_surrogate
from .fft_surrogate import fft_surrogate
from .iaaft_surrogate import iaaft_surrogate
from .random_reorder import random_reorder_surrogate
from .timeshift_surrogate import timeshift_surrogate
from .cycle_shuffle_surrogate import cycle_shuffle_surrogate
from .twin_surrogate import twin_surrogate, _precompute_twins
from .phase_surrogate import phase_surrogate
from .small_shuffle_surrogate import small_shuffle_surrogate
from .truncated_fourier_surrogate import truncated_fourier_surrogate
from .cycle_phase_surrogate import (
    cycle_phase_surrogate_A,
    cycle_phase_surrogate_B,
)
from .adaptive import select_surrogate_method, signal_profile
from .multivariate_surrogate import (
    multivariate_fft_surrogate,
    multivariate_iaaft_surrogate,
)

SURROGATE_METHODS = {
    "fft": fft_surrogate,
    "aaft": aaft_surrogate,
    "iaaft": iaaft_surrogate,
    "timeshift": timeshift_surrogate,
    "random_reorder": random_reorder_surrogate,
    "cycle_shuffle": cycle_shuffle_surrogate,
    "twin": twin_surrogate,
    "phase": phase_surrogate,
    "small_shuffle": small_shuffle_surrogate,
    "truncated_fourier": truncated_fourier_surrogate,
    "cycle_phase_a": cycle_phase_surrogate_A,
    "cycle_phase_b": cycle_phase_surrogate_B,
}


def generate_surrogate(x, method="iaaft", n_surrogates=100, seed=None, **kwargs):
    """Generate multiple surrogate time series.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    method : str
        Surrogate method name: 'fft', 'aaft', 'iaaft', 'timeshift',
        'random_reorder', 'cycle_shuffle', 'twin', 'phase',
        'small_shuffle', 'truncated_fourier'.
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
        # Pass E/tau explicitly to avoid redundant select_parameters calls
        twin_kwargs = {**kwargs, "E": E, "tau": tau}
        for i in range(n_surrogates):
            surrogates[i] = func(x, rng=rng, _twin_cache=cache, **twin_kwargs)
    else:
        for i in range(n_surrogates):
            surrogates[i] = func(x, rng=rng, **kwargs)

    return surrogates


MULTIVARIATE_METHODS = {
    "multivariate_fft": multivariate_fft_surrogate,
    "multivariate_iaaft": multivariate_iaaft_surrogate,
}


def generate_multivariate_surrogate(X, method="multivariate_fft",
                                     n_surrogates=100, seed=None, **kwargs):
    """Generate multivariate surrogates preserving cross-correlations.

    Unlike univariate surrogates, these operate on the full (T, N) matrix
    and preserve the linear cross-spectral structure between variables.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Multivariate time series.
    method : str
        'multivariate_fft' or 'multivariate_iaaft'.
    n_surrogates : int
        Number of surrogates to generate.
    seed : int, optional
        Base random seed.

    Returns
    -------
    surrogates : list of ndarray, each shape (T, N)
        List of multivariate surrogate datasets.
    """
    func = MULTIVARIATE_METHODS.get(method.lower())
    if func is None:
        raise ValueError(
            f"Unknown multivariate surrogate method: {method}. "
            f"Choose from {list(MULTIVARIATE_METHODS)}"
        )

    rng = np.random.default_rng(seed)
    surrogates = []
    for _ in range(n_surrogates):
        surrogates.append(func(X, rng=rng, **kwargs))
    return surrogates
