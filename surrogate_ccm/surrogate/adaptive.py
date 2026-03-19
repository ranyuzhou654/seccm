"""Adaptive surrogate method selection based on signal characteristics.

Analyzes time series properties and selects the most appropriate
surrogate method for hypothesis testing.
"""

import numpy as np


def spectral_concentration(x, top_k=3):
    """Fraction of total spectral power in the top-k frequency components.

    High concentration (> 0.5) indicates narrowband oscillatory dynamics.
    Low concentration (< 0.2) indicates broadband/chaotic dynamics.

    Parameters
    ----------
    x : ndarray
        Input time series.
    top_k : int
        Number of top frequency components to consider.

    Returns
    -------
    concentration : float
        Fraction of power in top-k components (0 to 1).
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - np.mean(x)
    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2
    # Exclude DC component
    power = power[1:]
    if power.sum() < 1e-15:
        return 0.0
    top_k = min(top_k, len(power))
    sorted_power = np.sort(power)[::-1]
    return float(sorted_power[:top_k].sum() / power.sum())


def autocorrelation_decay_time(x, threshold=1.0 / np.e, max_lag=200):
    """Time for autocorrelation to drop below threshold.

    Fast decay (< 5) → discrete map or broadband chaos.
    Slow decay (> 20) → oscillatory or flow dynamics.

    Parameters
    ----------
    x : ndarray
        Input time series.
    threshold : float
        ACF threshold (default: 1/e).
    max_lag : int
        Maximum lag to check.

    Returns
    -------
    decay_time : int
        Lag at which ACF drops below threshold.
    """
    x = np.asarray(x, dtype=float).ravel()
    x_c = x - np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return 0
    T = len(x)
    for lag in range(1, min(max_lag + 1, T // 2)):
        acf = np.mean(x_c[:T - lag] * x_c[lag:]) / var
        if acf < threshold:
            return lag
    return max_lag


def signal_profile(x, top_k=3):
    """Compute a diagnostic profile of signal characteristics.

    Parameters
    ----------
    x : ndarray
        Input time series.
    top_k : int
        Number of top spectral components for concentration.

    Returns
    -------
    profile : dict
        Signal characteristics:
        - 'spectral_concentration': float (0-1)
        - 'acf_decay': int (lag)
        - 'recommended_method': str
        - 'signal_type': str ('broadband_chaotic', 'narrowband_oscillatory',
          'mixed')
    """
    sc = spectral_concentration(x, top_k)
    decay = autocorrelation_decay_time(x)

    # Classification logic
    if sc > 0.8 and decay > 15:
        # Near-sinusoidal: cycles are too similar for cycle_shuffle
        # Phase surrogate breaks phase coupling while preserving envelope
        signal_type = "phase_coupled_oscillatory"
        method = "phase"
    elif sc > 0.5 and decay > 15:
        # Narrowband oscillatory with distinct cycle shapes
        signal_type = "narrowband_oscillatory"
        method = "cycle_shuffle"
    elif sc < 0.2 and decay < 5:
        signal_type = "broadband_chaotic"
        method = "fft"
    elif sc > 0.3:
        signal_type = "mixed"
        method = "iaaft"
    else:
        signal_type = "broadband_chaotic"
        method = "fft"

    return {
        "spectral_concentration": sc,
        "acf_decay": decay,
        "signal_type": signal_type,
        "recommended_method": method,
    }


def select_surrogate_method(x, available_methods=None):
    """Automatically select the best surrogate method for a time series.

    Parameters
    ----------
    x : ndarray
        Input time series.
    available_methods : list of str, optional
        Available methods to choose from. Default: all methods.

    Returns
    -------
    method : str
        Recommended surrogate method name.
    profile : dict
        Signal diagnostic profile.
    """
    if available_methods is None:
        available_methods = ["fft", "aaft", "iaaft", "timeshift",
                             "cycle_shuffle", "twin", "phase",
                             "small_shuffle"]

    profile = signal_profile(x)
    method = profile["recommended_method"]

    # Ensure recommended method is available
    if method.lower() not in [m.lower() for m in available_methods]:
        # Fallback hierarchy
        for fallback in ["iaaft", "fft", "aaft"]:
            if fallback in available_methods:
                method = fallback
                break

    return method.lower(), profile
