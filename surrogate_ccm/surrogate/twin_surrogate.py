"""Twin surrogate (Thiel et al. 2006, Europhysics Letters 75(4), 535-541).

Generates surrogates that preserve the recurrence structure of the
original attractor.  Works by reconstructing phase space via time-delay
embedding, identifying "twin" states (identical rows in the recurrence
matrix), then constructing a new trajectory that randomly switches
between twin successors at each step.

This method is particularly suited for testing synchronisation / causal
coupling because it preserves the *topology* of the attractor while
destroying inter-system phase relationships.

Performance notes
-----------------
- Uses KDTree (Chebyshev metric) + sparse neighbour sets instead of a
  dense N×N recurrence matrix → memory O(N × avg_neighbors) instead of
  O(N²).
- Row-hash acceleration for twin detection: only states whose neighbour
  sets hash identically are compared → expected ~O(N) instead of O(N³).
- ``_precompute_twins`` caches the heavy work; ``twin_surrogate`` reuses
  it across multiple calls via the ``_twin_cache`` parameter.
"""

import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree

from ..ccm.embedding import delay_embed, select_parameters


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_epsilon(X, target_rr=0.05, rr_range=(0.01, 0.10), n_sample=2000,
                    rng=None):
    """Choose recurrence threshold to achieve a target recurrence rate.

    Samples pairwise Chebyshev distances and picks the quantile that
    yields a recurrence rate closest to *target_rr*.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = X.shape[0]
    n_sample = min(n_sample, N * (N - 1) // 2)
    idx_i = rng.integers(0, N, size=n_sample)
    idx_j = rng.integers(0, N, size=n_sample)
    same = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % N

    dists = np.max(np.abs(X[idx_i] - X[idx_j]), axis=1)

    epsilon = np.quantile(dists, target_rr)
    if epsilon < 1e-12:
        epsilon = np.quantile(dists, rr_range[1])
    return float(epsilon)


def _build_neighbor_sets(X, epsilon):
    """Build neighbour sets using KDTree with Chebyshev (L∞) metric.

    Returns
    -------
    neighbors : list[frozenset[int]]
        ``neighbors[j]`` is the set of indices within epsilon of j
        (including j itself).
    """
    tree = KDTree(X)
    raw = tree.query_ball_point(X, r=epsilon, p=np.inf)
    return [frozenset(nb) for nb in raw]


def _find_twins_hashed(neighbors, min_dist=7):
    """Identify twins via row-hash grouping.

    Two states j, k are twins iff they have identical neighbour sets
    (after ensuring temporal separation and non-isolation).

    Strategy: hash each frozenset → group by hash → only compare within
    groups → verify exact equality (collision guard).

    Returns
    -------
    twins : list[list[int]]
    """
    N = len(neighbors)
    twins = [[] for _ in range(N)]

    # Group indices by (neighbor_count, hash_of_neighbor_set)
    buckets = defaultdict(list)
    for j in range(N):
        nb = neighbors[j]
        if len(nb) <= 1:
            continue
        key = (len(nb), hash(nb))
        buckets[key].append(j)

    # Within each bucket, verify exact equality and apply min_dist
    for group in buckets.values():
        if len(group) < 2:
            continue

        # Sub-group by exact neighbor set (handles hash collisions)
        exact_groups = defaultdict(list)
        for j in group:
            exact_groups[neighbors[j]].append(j)

        for indices in exact_groups.values():
            if len(indices) < 2:
                continue
            # All pairs in `indices` are true twins; apply min_dist filter
            for i_pos, j in enumerate(indices):
                for k in indices[i_pos + 1:]:
                    if abs(j - k) >= min_dist:
                        twins[j].append(k)
                        twins[k].append(j)

    return twins


def _construct_trajectory(x, twins, N, rng):
    """Walk the attractor, randomly switching at twin points."""
    surr = np.empty(N)
    k = rng.integers(0, N)

    for j in range(N):
        surr[j] = x[k]
        tw = twins[k]
        q = len(tw)
        if q == 0:
            k = k + 1
        else:
            r = rng.integers(0, q + 1)
            if r == q:
                k = k + 1
            else:
                k = tw[r] + 1
        if k >= N:
            k = rng.integers(0, N)

    return surr


def _precompute_twins(x, E, tau, epsilon, target_rr, min_dist, rng):
    """Heavy pre-computation: embedding → neighbours → twins.

    Returns a cache dict that ``twin_surrogate`` can reuse.
    """
    X = delay_embed(x, E, tau)
    N = X.shape[0]

    if epsilon is None:
        epsilon = _select_epsilon(X, target_rr=target_rr, rng=rng)

    neighbors = _build_neighbor_sets(X, epsilon)
    twins = _find_twins_hashed(neighbors, min_dist=min_dist)

    return {"N": N, "twins": twins, "E": E, "tau": tau, "epsilon": epsilon}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def twin_surrogate(x, rng=None, E=None, tau=None, epsilon=None,
                   target_rr=0.05, min_dist=7, _twin_cache=None):
    """Generate a twin surrogate (Thiel et al. 2006).

    Preserves the recurrence structure (attractor topology) of the
    original time series while destroying inter-system coupling.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : np.random.Generator, optional
        Random number generator.
    E : int, optional
        Embedding dimension.  Auto-selected if None.
    tau : int, optional
        Embedding delay.  Auto-selected if None.
    epsilon : float, optional
        Recurrence threshold (supremum norm).  Auto-selected from a
        target recurrence rate if None.
    target_rr : float
        Target recurrence rate for automatic epsilon selection.
        Typical range 0.01–0.10; default 0.05.
    min_dist : int
        Minimum temporal separation for twin candidates.
    _twin_cache : dict, optional
        Pre-computed cache from ``_precompute_twins``.  When generating
        multiple surrogates from the same signal, pass this to skip the
        expensive embedding / neighbour / twin computation on subsequent
        calls.

    Returns
    -------
    surr : ndarray, shape (T,)
        Twin surrogate of the same length as *x*.

    References
    ----------
    Thiel, M., Romano, M.C., Kurths, J., Rolfs, M. & Kliegl, R. (2006).
    Twin surrogates to test for complex synchronisation.
    Europhysics Letters 75(4), 535–541.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    T = len(x)

    # --- Auto-select embedding parameters if needed ---
    if E is None or tau is None:
        E_auto, tau_auto = select_parameters(x)
        if E is None:
            E = E_auto
        if tau is None:
            tau = tau_auto

    # --- Reuse or compute twin structure ---
    if _twin_cache is not None:
        cache = _twin_cache
    else:
        cache = _precompute_twins(x, E, tau, epsilon, target_rr, min_dist, rng)

    N = cache["N"]
    twins = cache["twins"]

    # --- Construct surrogate trajectory ---
    surr_core = _construct_trajectory(x, twins, N, rng)

    # --- Restore original length ---
    offset = (cache["E"] - 1) * cache["tau"]
    if offset > 0:
        surr = np.concatenate([x[:offset], surr_core])
    else:
        surr = surr_core

    if len(surr) > T:
        surr = surr[:T]
    elif len(surr) < T:
        surr = np.concatenate([surr, x[len(surr):]])

    return surr
