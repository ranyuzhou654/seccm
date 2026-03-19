"""0-1 Test for Chaos (Gottwald & Melbourne, 2004, 2009).

Determines whether a deterministic dynamical system is chaotic (K ≈ 1)
or regular/periodic (K ≈ 0) from a scalar time series.

Reference
---------
Gottwald, G.A. & Melbourne, I. (2009). On the implementation of the 0-1
test for chaos. SIAM J. Appl. Dyn. Syst., 8(1), 129-145.
"""

import numpy as np


def _estimate_subsample_step(x, target_acf=0.5, max_step=50):
    """Estimate subsampling step so consecutive points are ~decorrelated.

    For oversampled ODE flows (dt << characteristic time), consecutive
    points are nearly identical, causing the 0-1 test to incorrectly
    report K ≈ 0. Subsampling to the decorrelation time fixes this.

    Parameters
    ----------
    x : ndarray
        Input time series.
    target_acf : float
        Target autocorrelation level (subsample until ACF drops below this).
    max_step : int
        Maximum subsampling step to try.

    Returns
    -------
    step : int
        Recommended subsampling step (1 = no subsampling).
    """
    x = np.asarray(x, dtype=float)
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return 1

    T = len(x)
    for lag in range(1, min(max_step + 1, T // 2)):
        acf = np.mean(x_centered[:T - lag] * x_centered[lag:]) / var
        if acf < target_acf:
            return max(1, lag)

    return max_step


def test_01_chaos(x, n_c=100, c_range=(np.pi / 5, 4 * np.pi / 5),
                  n_cut=None, seed=None, auto_subsample=True):
    """Run the modified 0-1 test for chaos on a scalar time series.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    n_c : int
        Number of random test frequencies c to sample.
    c_range : tuple of float
        Range (c_min, c_max) for sampling c. Avoid c near 0 or pi to
        prevent resonances. Default (π/5, 4π/5) follows Gottwald & Melbourne.
    n_cut : int, optional
        Use only the first n_cut points for the MSD computation.
        Defaults to T // 10.
    seed : int, optional
        Random seed.
    auto_subsample : bool
        If True, automatically subsample oversampled continuous flows.
        Essential for ODE systems with small dt.

    Returns
    -------
    K_median : float
        Median K statistic over all test frequencies.
        K ≈ 1 → chaotic, K ≈ 0 → regular (periodic/quasi-periodic).
    K_values : ndarray, shape (n_c,)
        Individual K values for each test frequency.
    """
    x = np.asarray(x, dtype=float).ravel()

    # Auto-subsample for oversampled ODE flows
    subsample_step = 1
    if auto_subsample:
        subsample_step = _estimate_subsample_step(x)
        if subsample_step > 1:
            x = x[::subsample_step]

    T = len(x)
    if T < 100:
        import warnings
        warnings.warn(
            f"Only {T} effective points after subsampling (step={subsample_step}). "
            f"Results may be unreliable. Use T >= {100 * subsample_step} for robust detection.",
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)

    if n_cut is None:
        n_cut = T // 10
    n_cut = max(n_cut, 10)  # ensure at least 10 points

    # Sample random test frequencies
    c_values = rng.uniform(c_range[0], c_range[1], size=n_c)

    K_values = np.empty(n_c)

    for idx, c in enumerate(c_values):
        # Compute translation variables p(n), q(n)
        j = np.arange(1, T + 1)
        cos_cj = np.cos(c * j)
        sin_cj = np.sin(c * j)

        p = np.cumsum(x * cos_cj)
        q = np.cumsum(x * sin_cj)

        # Compute modified mean-square displacement M_c(n)
        # Using the modified version that subtracts the oscillatory term
        # to improve convergence (Gottwald & Melbourne, 2009, Eq. 8)
        x_mean = np.mean(x)
        V_osc = x_mean ** 2 * (1 - np.cos(c * np.arange(1, n_cut + 1))) / (1 - np.cos(c))

        M = np.empty(n_cut)
        for n in range(1, n_cut + 1):
            dp = p[n:T] - p[:T - n]
            dq = q[n:T] - q[:T - n]
            M[n - 1] = np.mean(dp ** 2 + dq ** 2) - V_osc[n - 1]

        # K = correlation between M(n) and n
        n_vec = np.arange(1, n_cut + 1, dtype=float)
        K_values[idx] = _correlation(n_vec, M)

    K_median = float(np.median(K_values))
    return K_median, K_values


def _correlation(x, y):
    """Compute correlation coefficient, clipped to [0, 1]."""
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(np.clip(r, 0.0, 1.0))


def is_chaotic(x, threshold=0.5, **kwargs):
    """Quick check: is the time series likely chaotic?

    Parameters
    ----------
    x : ndarray
        Input time series.
    threshold : float
        K threshold. K > threshold → chaotic.
    **kwargs
        Passed to test_01_chaos.

    Returns
    -------
    chaotic : bool
        True if K > threshold.
    K : float
        The median K statistic.
    """
    K, _ = test_01_chaos(x, **kwargs)
    return K > threshold, K
