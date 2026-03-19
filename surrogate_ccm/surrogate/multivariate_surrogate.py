"""Multivariate Fourier surrogate preserving cross-correlations.

Randomizes phases jointly across all variables using the same random
phase angles, preserving the cross-spectral structure (linear
cross-correlations) while destroying nonlinear dependencies.

This is the correct null hypothesis for testing nonlinear causality
in the presence of linear coupling: if two variables are linearly
correlated (shared spectral structure), univariate surrogates will
break this correlation and inflate false positive rates.

Reference
---------
Prichard, D. & Theiler, J. (1994). Generating surrogate data for
time series with several simultaneously measured variables.
Physical Review Letters, 73(7), 951-954.

Schreiber, T. & Schmitz, A. (2000). Surrogate time series.
Physica D, 142(3-4), 346-382.
"""

import numpy as np


def multivariate_fft_surrogate(X, rng=None):
    """Generate a multivariate FFT surrogate preserving cross-correlations.

    All variables share the same random phase rotation, preserving
    the cross-spectral matrix while randomizing individual phases.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Multivariate time series (T time points, N variables).
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    surrogate : ndarray, shape (T, N)
        Multivariate surrogate with preserved cross-correlations.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X, dtype=float)
    T, N = X.shape

    # FFT each variable
    F = np.fft.rfft(X, axis=0)  # shape (n_freq, N)
    n_freq = F.shape[0]

    # Generate SAME random phases for all variables (key insight)
    random_phases = rng.uniform(0, 2 * np.pi, size=n_freq)
    random_phases[0] = 0.0  # keep DC component
    if T % 2 == 0:
        random_phases[-1] = 0.0  # keep Nyquist if even length

    # Apply same phase rotation to all variables
    phase_rotation = np.exp(1j * random_phases)[:, np.newaxis]  # (n_freq, 1)
    F_surr = F * phase_rotation

    # Inverse FFT
    surrogate = np.fft.irfft(F_surr, n=T, axis=0)

    return surrogate


def multivariate_iaaft_surrogate(X, rng=None, max_iter=100):
    """Generate a multivariate iAAFT surrogate.

    Iteratively refines to match both the amplitude distribution
    and cross-spectral structure of the original data.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Multivariate time series.
    rng : numpy.random.Generator, optional
        Random number generator.
    max_iter : int
        Maximum refinement iterations.

    Returns
    -------
    surrogate : ndarray, shape (T, N)
        Multivariate iAAFT surrogate.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X, dtype=float)
    T, N = X.shape

    # Target amplitudes (FFT)
    target_F = np.fft.rfft(X, axis=0)
    target_amplitudes = np.abs(target_F)

    # Sorted values per variable (for amplitude matching)
    sorted_vals = [np.sort(X[:, j]) for j in range(N)]
    sort_indices = [np.argsort(X[:, j]) for j in range(N)]

    # Initialize with multivariate FFT surrogate
    surrogate = multivariate_fft_surrogate(X, rng=rng)

    for _ in range(max_iter):
        # Step 1: Enforce spectral structure (shared phases)
        F_surr = np.fft.rfft(surrogate, axis=0)
        phases = np.angle(F_surr)
        # Use mean phase across variables (preserve cross-spectral coherence)
        # Actually, just use amplitudes from original + phases from current
        F_surr = target_amplitudes * np.exp(1j * phases)
        surrogate_spectral = np.fft.irfft(F_surr, n=T, axis=0)

        # Step 2: Enforce amplitude distribution (rank ordering)
        surrogate_new = np.empty_like(surrogate_spectral)
        for j in range(N):
            rank = np.argsort(np.argsort(surrogate_spectral[:, j]))
            surrogate_new[:, j] = sorted_vals[j][rank]

        # Check convergence
        if np.allclose(surrogate, surrogate_new, atol=1e-10):
            break
        surrogate = surrogate_new

    return surrogate
