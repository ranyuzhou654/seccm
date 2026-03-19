"""Small-shuffle surrogate for testing fine-scale temporal structure.

Perturbs time indices by small random amounts, preserving large-scale
trends and slow dynamics while destroying fine temporal alignment.
Provides a middle ground between time-shift (too conservative) and
random-reorder (too aggressive).

Reference
---------
Nakamura, T. & Small, M. (2005). Small-shuffle surrogate data: testing
for dynamics in fluctuating data with trends. Physical Review E, 72(5),
056216.
"""

import numpy as np


def small_shuffle_surrogate(x, rng=None, delta=None):
    """Generate a small-shuffle surrogate.

    Algorithm:
    1. For each time index t, compute a perturbed index t' = t + U(-δ, δ)
    2. Sort by perturbed indices to get new ordering
    3. Surrogate = x[new_ordering]

    This preserves the large-scale shape of the time series (trends,
    slow oscillations) while destroying the fine-grained temporal order
    within windows of size ~2δ.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : numpy.random.Generator, optional
        Random number generator.
    delta : int, optional
        Half-width of the shuffle window. Each index is perturbed by
        a uniform random value in [-delta, delta]. Default: T // 20.

    Returns
    -------
    surrogate : ndarray, shape (T,)
        Small-shuffle surrogate time series.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    T = len(x)

    if delta is None:
        delta = max(T // 20, 1)

    # Perturb indices
    indices = np.arange(T, dtype=float)
    perturbed = indices + rng.uniform(-delta, delta, size=T)

    # Sort by perturbed indices
    new_order = np.argsort(perturbed)

    return x[new_order]
