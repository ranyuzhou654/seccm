"""Cycle-phase surrogate for oscillatory dynamical systems.

Designed to disrupt inter-cycle phase coupling while preserving
intra-cycle waveform shape. Targets the failure mode where
standard surrogates (FFT, IAAFT) preserve too much spectral
structure for oscillatory systems, causing null distribution collapse.

Two modes:
- Mode A (phase-only): circular-shift within each cycle, keep cycle order
- Mode B (phase+order): circular-shift within cycles AND permute cycle order
"""

import numpy as np
from scipy.signal import hilbert
from .timeshift_surrogate import timeshift_surrogate


def _identify_cycles_hilbert(x, min_cycles=3):
    """Identify oscillatory cycle boundaries using Hilbert transform phase.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    min_cycles : int
        Minimum number of complete cycles required.

    Returns
    -------
    boundaries : ndarray of int, or None
        Indices of cycle boundaries (phase wraps from ~2π → ~0).
        Returns None if fewer than min_cycles+1 boundaries found.
    """
    # Center and normalize
    x_centered = x - np.mean(x)
    std = np.std(x_centered)
    if std < 1e-12:
        return None
    x_norm = x_centered / std

    # Compute analytic signal via Hilbert transform
    analytic = hilbert(x_norm)
    phase = np.angle(analytic)  # in [-π, π]

    # Unwrap to find 2π crossings (cycle completions)
    # A cycle boundary occurs when the phase wraps from positive to negative
    # (i.e., crosses from ~π to ~-π going forward)
    phase_diff = np.diff(phase)
    # Phase wraps show as large negative jumps (from ~π to ~-π)
    wrap_threshold = -np.pi  # phase jump < -π indicates a wrap
    boundaries = np.where(phase_diff < wrap_threshold)[0] + 1

    if len(boundaries) < min_cycles + 1:
        return None

    return boundaries


def _identify_cycles_peaks(x, min_cycles=3):
    """Fallback cycle identification using zero-crossings (upward).

    Parameters
    ----------
    x : ndarray, shape (T,)
    min_cycles : int

    Returns
    -------
    boundaries : ndarray of int, or None
    """
    centered = x - np.mean(x)
    sign = np.sign(centered)
    sign[sign == 0] = 1
    # Upward zero crossings
    crossings = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1

    if len(crossings) < min_cycles + 1:
        return None

    return crossings


def cycle_phase_surrogate(x, rng=None, mode="A", min_cycles=3):
    """Generate a cycle-phase surrogate.

    Disrupts inter-cycle phase coupling while preserving intra-cycle
    waveform shape. For oscillatory systems where standard surrogates
    fail due to spectral overlap (SSO → 0).

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : np.random.Generator, optional
        Random number generator.
    mode : str, 'A' or 'B'
        - 'A' (phase-only): circular-shift within cycles, keep cycle order.
          Preserves slow amplitude modulation.
        - 'B' (phase+order): circular-shift AND permute cycle order.
          More aggressive; destroys amplitude ordering.
    min_cycles : int
        Minimum cycles required. Falls back to timeshift if fewer detected.

    Returns
    -------
    surr : ndarray, shape (T,)
        Surrogate time series.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(x)

    # Try Hilbert-based cycle identification first, fall back to zero-crossings
    boundaries = _identify_cycles_hilbert(x, min_cycles=min_cycles)
    if boundaries is None:
        boundaries = _identify_cycles_peaks(x, min_cycles=min_cycles)
    if boundaries is None:
        # Not enough cycles detected → fall back to timeshift
        return timeshift_surrogate(x, rng=rng)

    # Extract cycle segments
    # Head: before first boundary, Tail: after last boundary
    head = x[:boundaries[0]]
    tail = x[boundaries[-1]:]
    cycles = [x[boundaries[i]:boundaries[i + 1]]
              for i in range(len(boundaries) - 1)]

    # Apply circular shift within each cycle
    shifted_cycles = []
    for c in cycles:
        if len(c) > 1:
            delta = rng.integers(0, len(c))
            shifted_cycles.append(np.roll(c, delta))
        else:
            shifted_cycles.append(c.copy())

    # Mode B: also permute cycle order
    if mode.upper() == "B":
        order = rng.permutation(len(shifted_cycles))
        shifted_cycles = [shifted_cycles[i] for i in order]

    # Concatenate
    surr = np.concatenate([head] + shifted_cycles + [tail])

    # Ensure exact length T
    if len(surr) == T:
        return surr
    elif len(surr) > T:
        return surr[:T]
    else:
        # Pad with original tail if shorter (shouldn't happen normally)
        return np.concatenate([surr, x[len(surr):]])


def cycle_phase_surrogate_A(x, rng=None, min_cycles=3):
    """Cycle-phase surrogate Mode A (phase-only, conservative)."""
    return cycle_phase_surrogate(x, rng=rng, mode="A", min_cycles=min_cycles)


def cycle_phase_surrogate_B(x, rng=None, min_cycles=3):
    """Cycle-phase surrogate Mode B (phase+order, aggressive)."""
    return cycle_phase_surrogate(x, rng=rng, mode="B", min_cycles=min_cycles)
