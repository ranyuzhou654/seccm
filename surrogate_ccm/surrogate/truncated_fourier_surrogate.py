"""Truncated Fourier surrogate with band-selective phase randomization.

Randomizes phases only within a specified frequency band, preserving
structure outside that band. This enables fine-grained null hypothesis
control:
  - Randomize low frequencies: preserve fast dynamics, test slow causality
  - Randomize high frequencies: preserve slow trends, test fast coupling
  - Randomize mid-band: target specific timescales

Reference
---------
Lancaster, G. et al. (2018). Surrogate data for hypothesis testing of
physical systems. Physics Reports, 748, 1-60.
Keylock, C.J. (2006). Constrained surrogate time series with preservation
of the mean and variance and analogy with the Fourier transform. Physical
Review E, 73(3), 036707.
"""

import numpy as np


def truncated_fourier_surrogate(x, rng=None, f_low=None, f_high=None,
                                 frac_low=0.0, frac_high=1.0):
    """Generate a truncated Fourier surrogate with band-selective phase randomization.

    Only phases in the frequency range [f_low, f_high] are randomized.
    Phases outside this range are preserved.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : numpy.random.Generator, optional
        Random number generator.
    f_low : float, optional
        Lower frequency bound (in normalized frequency, 0 to 0.5).
        Default: determined by frac_low.
    f_high : float, optional
        Upper frequency bound (in normalized frequency, 0 to 0.5).
        Default: determined by frac_high.
    frac_low : float
        Lower bound as fraction of Nyquist (0.0 = DC, 1.0 = Nyquist).
        Ignored if f_low is set.
    frac_high : float
        Upper bound as fraction of Nyquist (0.0 = DC, 1.0 = Nyquist).
        Ignored if f_high is set.

    Returns
    -------
    surrogate : ndarray, shape (T,)
        Truncated Fourier surrogate.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    T = len(x)

    # Compute FFT
    X = np.fft.rfft(x)
    n_freq = len(X)

    # Determine frequency bounds
    if f_low is None:
        f_low = frac_low * 0.5  # frac of Nyquist
    if f_high is None:
        f_high = frac_high * 0.5

    # Convert to index bounds
    freqs = np.fft.rfftfreq(T)
    idx_low = max(1, np.searchsorted(freqs, f_low))  # skip DC
    idx_high = min(n_freq, np.searchsorted(freqs, f_high))

    # Randomize phases only within the band
    phases = np.angle(X)
    amplitudes = np.abs(X)

    random_phases = rng.uniform(0, 2 * np.pi, size=n_freq)
    phases[idx_low:idx_high] = random_phases[idx_low:idx_high]

    # Reconstruct
    X_surr = amplitudes * np.exp(1j * phases)
    surrogate = np.fft.irfft(X_surr, n=T)

    return surrogate
