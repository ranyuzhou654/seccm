"""Iterative Amplitude-Adjusted Fourier Transform (iAAFT) surrogate."""

import numpy as np

from ..utils.backend import get_array_module, to_numpy


def iaaft_surrogate(x, rng=None, max_iter=200, tol=1e-8):
    """Generate an iAAFT surrogate.

    Iteratively adjusts both the amplitude distribution and power spectrum
    to match the original series.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : np.random.Generator, optional
        Random number generator.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on spectral difference.

    Returns
    -------
    surr : ndarray, shape (T,)
        iAAFT surrogate time series.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(x)
    sorted_x = np.sort(x)
    target_amplitudes = np.abs(np.fft.rfft(x))

    # Initialize with a random shuffle
    surr = x.copy()
    rng.shuffle(surr)

    # Normalization factor for relative spectral error
    target_power = np.mean(target_amplitudes ** 2)
    if target_power < 1e-30:
        target_power = 1.0

    # Pre-allocate rank-order output to avoid repeated allocation
    ranked = np.empty(T, dtype=x.dtype)

    for _ in range(max_iter):
        # Step 1: Match power spectrum
        surr_fft = np.fft.rfft(surr)
        surr_phases = surr_fft
        surr_phases /= np.abs(surr_fft) + 1e-30  # normalise to unit phases
        surr_phases *= target_amplitudes
        surr = np.fft.irfft(surr_phases, n=T)

        # Step 2: Match amplitude distribution (rank-order mapping)
        # Single argsort + scatter is ~6× faster than double argsort
        order = np.argsort(surr)
        ranked[order] = sorted_x
        surr = ranked.copy()

        # Check convergence: distance to target spectrum (not just stationarity)
        current_spectrum = np.abs(np.fft.rfft(surr))
        spectral_error = np.mean((current_spectrum - target_amplitudes) ** 2) / target_power
        if spectral_error < tol:
            break

    return surr


def iaaft_surrogate_batch(x, n_surrogates, rng=None, max_iter=200, tol=1e-8,
                          use_gpu=False):
    """Generate multiple iAAFT surrogates in a single batch.

    All surrogates are processed in parallel using batch FFT and argsort,
    yielding significant speedup over sequential generation.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    n_surrogates : int
        Number of surrogates to generate.
    rng : np.random.Generator, optional
        Random number generator.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on mean spectral error across batch.
    use_gpu : bool
        If True, use CuPy for GPU-accelerated FFT and argsort.

    Returns
    -------
    surrogates : ndarray, shape (n_surrogates, T)
        Batch of iAAFT surrogate time series.
    """
    if rng is None:
        rng = np.random.default_rng()

    xp = get_array_module(use_gpu)

    T = len(x)
    sorted_x = xp.asarray(np.sort(x))                    # (T,)
    target_amp = xp.asarray(np.abs(np.fft.rfft(x)))       # (T//2+1,)

    target_power = float(np.mean(np.abs(np.fft.rfft(x)) ** 2))
    if target_power < 1e-30:
        target_power = 1.0

    # Initialize each surrogate as an independent random shuffle
    surr_batch_np = np.empty((n_surrogates, T), dtype=x.dtype)
    for i in range(n_surrogates):
        surr_batch_np[i] = x.copy()
        rng.shuffle(surr_batch_np[i])
    surr_batch = xp.asarray(surr_batch_np)                # (n_surr, T)

    rows = xp.arange(n_surrogates)[:, None]               # (n_surr, 1) for indexing

    for _ in range(max_iter):
        # Step 1: Match power spectrum (batch FFT)
        surr_fft = xp.fft.rfft(surr_batch, axis=1)        # (n_surr, T//2+1)
        phases = surr_fft / (xp.abs(surr_fft) + 1e-30)
        surr_batch = xp.fft.irfft(phases * target_amp[None, :], n=T, axis=1)

        # Step 2: Match amplitude distribution (batch rank-order mapping)
        order = xp.argsort(surr_batch, axis=1)             # (n_surr, T)
        ranked = xp.empty_like(surr_batch)
        ranked[rows, order] = sorted_x[None, :]
        surr_batch = ranked

        # Convergence: mean spectral error across all surrogates
        current_spec = xp.abs(xp.fft.rfft(surr_batch, axis=1))
        err = float(xp.mean((current_spec - target_amp[None, :]) ** 2)) / target_power
        if err < tol:
            break

    return to_numpy(surr_batch)
