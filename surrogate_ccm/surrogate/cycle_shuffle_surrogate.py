"""Cycle-shuffle surrogate for narrowband oscillatory systems."""

import numpy as np

from .timeshift_surrogate import timeshift_surrogate


def cycle_shuffle_surrogate(x, rng=None, min_cycles=3):
    """Generate a surrogate by shuffling complete oscillation cycles.

    Detects oscillation cycles via mean-crossing (rising edges), then
    randomly permutes the order of complete cycles. This destroys
    inter-cycle phase coupling (the carrier of causal information in
    narrowband oscillatory systems) while preserving intra-cycle
    waveforms and amplitude distribution.

    Falls back to timeshift surrogate if fewer than *min_cycles*
    complete cycles are detected.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : np.random.Generator, optional
        Random number generator.
    min_cycles : int
        Minimum number of complete cycles required. If fewer are found,
        falls back to timeshift surrogate.

    Returns
    -------
    surr : ndarray, shape (T,)
        Cycle-shuffled surrogate of the same length as *x*.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(x)
    centered = x - np.mean(x)

    # Detect rising zero-crossings (mean-crossings)
    sign = np.sign(centered)
    # Replace exact zeros with +1 to avoid ambiguity
    sign[sign == 0] = 1
    crossings = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1

    if len(crossings) < min_cycles + 1:
        # Not enough cycles detected — fall back to timeshift
        return timeshift_surrogate(x, rng=rng)

    # Split into: head (before first crossing), complete cycles, tail
    head = x[:crossings[0]]
    tail = x[crossings[-1]:]
    cycles = [x[crossings[i]:crossings[i + 1]] for i in range(len(crossings) - 1)]

    # Shuffle complete cycles
    order = rng.permutation(len(cycles))
    shuffled_cycles = [cycles[i] for i in order]

    # Reassemble: head + shuffled cycles + tail
    surr = np.concatenate([head] + shuffled_cycles + [tail])

    # Ensure output length matches input (should already match)
    if len(surr) == T:
        return surr
    elif len(surr) > T:
        return surr[:T]
    else:
        # Pad with tail values if somehow shorter
        return np.concatenate([surr, x[len(surr):]])
