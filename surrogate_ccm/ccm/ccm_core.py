"""Core CCM algorithm implementation.

Convention: ccm(x, y, ...) cross-maps from M_x to predict y, testing "y causes x".
"""

import numpy as np
from scipy.spatial import KDTree

from .embedding import delay_embed


def _find_neighbors_theiler(M, k, theiler_w=0):
    """Find k nearest neighbors with optional Theiler window exclusion.

    Parameters
    ----------
    M : ndarray, shape (L, E)
        Embedded library points.
    k : int
        Number of neighbors to find.
    theiler_w : int
        Theiler window: exclude neighbors within ±theiler_w time steps.
        0 means only exclude self (standard behavior).

    Returns
    -------
    dists : ndarray, shape (L, k)
        Distances to k nearest neighbors.
    idxs : ndarray, shape (L, k)
        Indices of k nearest neighbors.
    """
    L = len(M)

    if theiler_w <= 0:
        # Fast path: use FAISS GPU kNN if available, else KDTree
        try:
            from ..utils.knn import knn_query
            return knn_query(M, k, use_gpu=True)
        except ImportError:
            tree = KDTree(M)
            dists, idxs = tree.query(M, k=k + 1)
            return dists[:, 1:], idxs[:, 1:]

    # With Theiler window: query more neighbors, filter out temporally close
    tree = KDTree(M)
    # Query enough extra to compensate for exclusion zone
    k_query = min(k + 2 * theiler_w + 1, L)
    dists_all, idxs_all = tree.query(M, k=k_query)

    dists_out = np.empty((L, k))
    idxs_out = np.empty((L, k), dtype=int)

    for i in range(L):
        # Exclude points within ±theiler_w of point i
        valid = np.abs(idxs_all[i] - i) > theiler_w
        valid_d = dists_all[i][valid]
        valid_i = idxs_all[i][valid]
        n_valid = min(k, len(valid_d))
        dists_out[i, :n_valid] = valid_d[:n_valid]
        idxs_out[i, :n_valid] = valid_i[:n_valid]
        # If not enough valid neighbors, repeat last
        if n_valid < k:
            dists_out[i, n_valid:] = valid_d[n_valid - 1] if n_valid > 0 else 1.0
            idxs_out[i, n_valid:] = valid_i[n_valid - 1] if n_valid > 0 else 0

    return dists_out, idxs_out


def ccm(x, y, E, tau, L=None, theiler_w=0):
    """Convergent Cross Mapping from M_x to predict y.

    Tests the hypothesis "y causes x" by checking if the attractor
    reconstructed from x contains information about y.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Target series (effect candidate).
    y : ndarray, shape (T,)
        Source series (cause candidate) to predict.
    E : int
        Embedding dimension.
    tau : int
        Time delay.
    L : int, optional
        Library size. If None, uses full length.
    theiler_w : int
        Theiler window: exclude neighbors within ±theiler_w time steps
        when finding nearest neighbors. Prevents temporal autocorrelation
        from inflating ρ. Default 0 (only exclude self).

    Returns
    -------
    rho : float
        Pearson correlation between predicted and actual y.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Create shadow manifold from x
    M_x = delay_embed(x, E, tau)
    T_eff = len(M_x)

    # Align y with embedded x (use the last time index of each embedding vector)
    offset = (E - 1) * tau
    y_aligned = y[offset : offset + T_eff]

    if L is None:
        L = T_eff

    L = min(L, T_eff)

    # Use first L points as library
    lib = M_x[:L]
    y_lib = y_aligned[:L]

    # Build KDTree on library and find neighbors
    k = E + 1
    dists, idxs = _find_neighbors_theiler(lib, k, theiler_w)

    # Exponential weights
    eps = 1e-12
    w = np.exp(-dists / (dists[:, 0:1] + eps))
    w = w / (w.sum(axis=1, keepdims=True) + eps)

    # Predict y
    y_pred = np.sum(w * y_lib[idxs], axis=1)

    # Pearson correlation
    rho = np.corrcoef(y_lib, y_pred)[0, 1]
    if np.isnan(rho):
        rho = 0.0

    return rho


def _ccm_xval(M_x, y_aligned, E, lib_idx, pred_idx, theiler_w=0):
    """Cross-validated CCM: library and prediction sets are disjoint.

    Parameters
    ----------
    M_x : ndarray, shape (T_eff, E)
        Full embedded manifold.
    y_aligned : ndarray, shape (T_eff,)
        Aligned target series.
    E : int
        Embedding dimension.
    lib_idx : ndarray
        Indices for the library set.
    pred_idx : ndarray
        Indices for the prediction set.
    theiler_w : int
        Theiler window.

    Returns
    -------
    rho : float
        Cross-validated CCM correlation.
    """
    lib = M_x[lib_idx]
    y_lib = y_aligned[lib_idx]
    pred = M_x[pred_idx]
    y_pred_true = y_aligned[pred_idx]

    if len(lib) < E + 1 or len(pred) < 2:
        return 0.0

    k = E + 1
    tree = KDTree(lib)

    if theiler_w > 0:
        # Query more neighbors and filter by Theiler window
        k_query = min(k + 2 * theiler_w + 1, len(lib))
        dists_all, idxs_all = tree.query(pred, k=k_query)
        dists_out = np.empty((len(pred), k))
        idxs_out = np.empty((len(pred), k), dtype=int)
        for p in range(len(pred)):
            # Theiler window applies to the original time indices
            orig_t = pred_idx[p]
            valid = np.abs(lib_idx[idxs_all[p]] - orig_t) > theiler_w
            valid_d = dists_all[p][valid]
            valid_i = idxs_all[p][valid]
            n_valid = min(k, len(valid_d))
            dists_out[p, :n_valid] = valid_d[:n_valid]
            idxs_out[p, :n_valid] = valid_i[:n_valid]
            if n_valid < k:
                dists_out[p, n_valid:] = valid_d[n_valid - 1] if n_valid > 0 else 1.0
                idxs_out[p, n_valid:] = valid_i[n_valid - 1] if n_valid > 0 else 0
        dists, idxs = dists_out, idxs_out
    else:
        dists, idxs = tree.query(pred, k=k)

    eps = 1e-12
    w = np.exp(-dists / (dists[:, 0:1] + eps))
    w = w / (w.sum(axis=1, keepdims=True) + eps)

    y_pred = np.sum(w * y_lib[idxs], axis=1)
    rho = np.corrcoef(y_pred_true, y_pred)[0, 1]
    return 0.0 if np.isnan(rho) else float(rho)


def ccm_convergence(x, y, E, tau, n_points=20, theiler_w=0,
                    cross_validate=False, n_reps=5, seed=None):
    """Compute CCM correlation across library sizes (convergence check).

    Following Sugihara et al. (2012): randomly sample library subsets of
    increasing size and use the remaining points for prediction.

    Parameters
    ----------
    x, y : ndarray
        Time series.
    E : int
        Embedding dimension.
    tau : int
        Time delay.
    n_points : int
        Number of library sizes to test.
    theiler_w : int
        Theiler window for neighbor exclusion.
    cross_validate : bool
        If True, use disjoint library and prediction sets (out-of-sample).
        If False, use in-sample evaluation (original behavior).
    n_reps : int
        Number of random library samples per L value (only for cross_validate).
    seed : int, optional
        Random seed for library sampling.

    Returns
    -------
    L_values : ndarray
        Library sizes tested.
    rho_values : ndarray
        CCM correlation at each library size (mean over reps if cross-validated).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    T_eff = len(x) - (E - 1) * tau
    L_min = E + 2
    L_max = T_eff

    if not cross_validate:
        if L_min >= L_max:
            return np.array([L_max]), np.array([ccm(x, y, E, tau, L_max, theiler_w)])
        L_values = np.unique(np.linspace(L_min, L_max, n_points, dtype=int))
        rho_values = np.array([ccm(x, y, E, tau, L, theiler_w) for L in L_values])
        return L_values, rho_values

    # Cross-validated convergence
    M_x = delay_embed(x, E, tau)
    offset = (E - 1) * tau
    y_aligned = y[offset:offset + T_eff]
    all_idx = np.arange(T_eff)

    rng = np.random.default_rng(seed)

    if L_min >= L_max:
        return np.array([L_max]), np.array([ccm(x, y, E, tau, L_max, theiler_w)])

    L_values = np.unique(np.linspace(L_min, L_max, n_points, dtype=int))
    # Exclude L = L_max (no prediction points left)
    L_values = L_values[L_values < T_eff]

    rho_values = np.empty(len(L_values))
    for li, L in enumerate(L_values):
        rhos = np.empty(n_reps)
        for r in range(n_reps):
            lib_idx = rng.choice(T_eff, size=L, replace=False)
            lib_idx.sort()
            pred_mask = np.ones(T_eff, dtype=bool)
            pred_mask[lib_idx] = False
            pred_idx = all_idx[pred_mask]
            rhos[r] = _ccm_xval(M_x, y_aligned, E, lib_idx, pred_idx,
                                theiler_w)
        rho_values[li] = np.mean(rhos)

    return L_values, rho_values


def convergence_score(x, y, E, tau, n_points=20, theiler_w=0,
                      cross_validate=True, n_reps=5, seed=None):
    """Compute a convergence score for CCM.

    True causal relationships show monotonically increasing ρ(L).
    The score is the Kendall rank correlation between L and ρ(L),
    measuring the strength and consistency of convergence.

    Parameters
    ----------
    x, y : ndarray
        Time series.
    E : int
        Embedding dimension.
    tau : int
        Time delay.
    n_points : int
        Number of library sizes to test.
    theiler_w : int
        Theiler window for neighbor exclusion.
    cross_validate : bool
        If True (default), use out-of-sample evaluation for proper
        convergence detection. In-sample evaluation shows artificial
        ρ decrease with L due to overfitting.
    n_reps : int
        Repetitions per library size (cross-validate only).
    seed : int, optional
        Random seed.

    Returns
    -------
    score : float
        Kendall τ of ρ(L) vs L. Range [-1, 1].
        +1 = perfect monotonic increase (strong convergence).
        0 = no trend (no convergence).
        -1 = monotonic decrease.
    rho_final : float
        ρ at the maximum library size.
    """
    from scipy.stats import kendalltau

    L_values, rho_values = ccm_convergence(
        x, y, E, tau, n_points, theiler_w,
        cross_validate=cross_validate, n_reps=n_reps, seed=seed,
    )

    if len(L_values) < 3:
        return 0.0, rho_values[-1]

    tau_stat, _ = kendalltau(L_values, rho_values)
    if np.isnan(tau_stat):
        tau_stat = 0.0

    return float(tau_stat), float(rho_values[-1])
