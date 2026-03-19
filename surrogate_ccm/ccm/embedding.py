"""Delay embedding and automatic parameter selection."""

import numpy as np
from scipy.spatial import KDTree


def delay_embed(x, E, tau):
    """Create a time-delay embedding matrix.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    E : int
        Embedding dimension.
    tau : int
        Time delay.

    Returns
    -------
    embedded : ndarray, shape (T_eff, E)
        Delay-embedded matrix where T_eff = T - (E-1)*tau.
    """
    x = np.asarray(x).ravel()
    T = len(x)
    T_eff = T - (E - 1) * tau
    if T_eff <= 0:
        raise ValueError(f"Time series too short (T={T}) for E={E}, tau={tau}")

    indices = np.arange(T_eff)[:, None] + np.arange(E)[None, :] * tau
    return x[indices]


def _mutual_information(x, tau, n_bins=64):
    """Compute auto mutual information at a given lag."""
    T = len(x)
    x1 = x[: T - tau]
    x2 = x[tau:]

    c_xy, xedges, yedges = np.histogram2d(x1, x2, bins=n_bins)
    c_xy = c_xy / c_xy.sum()
    c_x = c_xy.sum(axis=1)
    c_y = c_xy.sum(axis=0)

    mask = c_xy > 0
    mi = np.sum(
        c_xy[mask] * np.log(c_xy[mask] / (c_x[:, None] * c_y[None, :])[mask])
    )
    return mi


def _autocorrelation_tau(x, tau_max=50):
    """Find tau where autocorrelation first drops below 1/e.

    This is a robust fallback when MI has no local minimum (common for
    chaotic maps where MI decays monotonically).
    """
    x_centered = x - np.mean(x)
    var = np.var(x_centered)
    if var < 1e-12:
        return 1

    threshold = 1.0 / np.e
    T = len(x_centered)

    for t in range(1, min(tau_max + 1, T)):
        acf = np.mean(x_centered[: T - t] * x_centered[t:]) / var
        if acf < threshold:
            return max(t, 1)

    return 1


def select_tau(x, tau_max=50):
    """Select delay tau for embedding.

    Strategy:
    - Compute autocorrelation 1/e decay time (tau_acf).
    - If tau_acf <= 2 (fast decorrelation, typical for maps): return tau_acf.
    - Otherwise (slow decorrelation, typical for flows): search for MI
      first local minimum up to 2*tau_acf. If found, use it; else use tau_acf.

    Parameters
    ----------
    x : ndarray
        Input time series.
    tau_max : int
        Maximum lag to search.

    Returns
    -------
    tau : int
        Selected time delay (minimum 1).
    """
    x = np.asarray(x).ravel()

    # Step 1: autocorrelation 1/e decay time
    tau_acf = _autocorrelation_tau(x, tau_max)

    # For maps (fast decorrelation), autocorrelation is sufficient
    if tau_acf <= 2:
        return tau_acf

    # Step 2: for continuous systems, MI first minimum is more precise
    search_limit = min(tau_acf * 2, tau_max)
    mi_values = np.array(
        [_mutual_information(x, t) for t in range(1, search_limit + 1)]
    )

    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
            return i + 1

    return tau_acf


def _simplex_predict_rho(x, E, tau):
    """Compute simplex prediction accuracy with tp=tau prediction horizon.

    For maps (tau=1), this is one-step-ahead. For continuous flows (tau>1),
    predicting tau steps ahead is harder and forces proper attractor
    reconstruction, allowing meaningful E discrimination.
    """
    tp = tau  # prediction horizon = tau (standard in rEDM)
    emb = delay_embed(x, E, tau)
    T_eff = len(emb)

    # Target: x value tp steps after the last index of each embedding vector
    offset = (E - 1) * tau
    target_indices = np.arange(T_eff) + offset + tp
    valid = target_indices < len(x)
    emb = emb[valid]
    target_indices = target_indices[valid]
    T_eff = len(emb)

    if T_eff < 2 * (E + 2):
        return -1.0

    targets = x[target_indices]

    k = E + 1
    tree = KDTree(emb)
    dists, idxs = tree.query(emb, k=k + 1)
    dists = dists[:, 1:]  # remove self
    idxs = idxs[:, 1:]

    eps = 1e-12
    w = np.exp(-dists / (dists[:, 0:1] + eps))
    w = w / (w.sum(axis=1, keepdims=True) + eps)

    y_pred = np.sum(w * targets[idxs], axis=1)
    rho = np.corrcoef(targets, y_pred)[0, 1]
    return rho if not np.isnan(rho) else -1.0


def select_E(x, tau, E_max=10):
    """Select embedding dimension E using simplex projection.

    Tests E=2,...,E_max and picks the E that maximizes one-step-ahead
    prediction accuracy. This is the standard method used with CCM
    (Sugihara et al. 2012).

    Parameters
    ----------
    x : ndarray
        Input time series.
    tau : int
        Time delay.
    E_max : int
        Maximum dimension to test.

    Returns
    -------
    E : int
        Selected embedding dimension (minimum 2).
    """
    x = np.asarray(x).ravel()
    best_E = 2
    best_rho = -np.inf

    for E in range(2, E_max + 1):
        if len(x) - (E - 1) * tau < 2 * (E + 2):
            break
        rho = _simplex_predict_rho(x, E, tau)
        if rho > best_rho:
            best_rho = rho
            best_E = E

    return best_E


def select_E_fnn(x, tau, E_max=10, rtol=15.0, atol=2.0, threshold=0.01):
    """Select embedding dimension using False Nearest Neighbors (FNN).

    At insufficient E, some nearest neighbors are "false" — close only
    due to projection, not true proximity. Increasing E unfolds the
    attractor and eliminates false neighbors.

    Parameters
    ----------
    x : ndarray
        Input time series.
    tau : int
        Time delay.
    E_max : int
        Maximum embedding dimension.
    rtol : float
        Relative distance threshold for FNN criterion 1.
    atol : float
        Absolute threshold for FNN criterion 2 (ratio to attractor size).
    threshold : float
        FNN fraction below which embedding is considered sufficient.

    Returns
    -------
    E : int
        Selected embedding dimension (minimum 2).

    Reference
    ---------
    Kennel, M.B., Brown, R. & Abarbanel, H.D.I. (1992). Determining
    embedding dimension for phase-space reconstruction using a geometrical
    construction on the attractor. Phys. Rev. A, 45(6), 3403.
    """
    x = np.asarray(x, dtype=float).ravel()
    R_A = np.std(x)  # attractor size estimate
    if R_A < 1e-15:
        return 2

    best_E = 2
    for E in range(2, E_max + 1):
        T_eff_next = len(x) - E * tau
        if T_eff_next < E + 2:
            break

        M = delay_embed(x, E, tau)
        # Trim to points that have an (E+1)-th coordinate available
        n_valid = len(x) - E * tau
        if n_valid < E + 2:
            break
        M = M[:n_valid]
        T_eff = len(M)

        tree = KDTree(M)
        dists, idxs = tree.query(M, k=2)
        nn_dists = dists[:, 1]
        nn_idxs = idxs[:, 1]

        # Extra coordinate at dimension E+1
        base_indices = np.arange(T_eff) + (E - 1) * tau
        extra = x[base_indices + tau]
        nn_extra = x[nn_idxs + (E - 1) * tau + tau]

        # Criterion 1: relative distance increase
        delta = np.abs(extra - nn_extra)
        safe_dists = np.where(nn_dists > 1e-15, nn_dists, 1e-15)
        crit1 = delta / safe_dists > rtol

        # Criterion 2: absolute size (distance too large relative to attractor)
        new_dist = np.sqrt(nn_dists ** 2 + delta ** 2)
        crit2 = new_dist / R_A > atol

        fnn_frac = np.mean(crit1 | crit2)

        if fnn_frac < threshold:
            best_E = E
            break
        best_E = E

    return best_E


def select_E_cao(x, tau, E_max=10):
    """Select embedding dimension using Cao's method.

    Computes the E1 statistic: the ratio of nearest-neighbor distances
    at successive embedding dimensions. E1 saturates at E_opt.
    Also computes E2 to distinguish deterministic from stochastic data
    (E2 ≈ 1 for random, E2 ≠ 1 for deterministic).

    Parameters
    ----------
    x : ndarray
        Input time series.
    tau : int
        Time delay.
    E_max : int
        Maximum embedding dimension.

    Returns
    -------
    E : int
        Selected embedding dimension (minimum 2).

    Reference
    ---------
    Cao, L. (1997). Practical method for determining the minimum embedding
    dimension of a scalar time series. Physica D, 110(1-2), 43-50.
    """
    x = np.asarray(x, dtype=float).ravel()

    a_values = []  # a(E) for E = 1, 2, ..., E_max
    for E in range(1, E_max + 1):
        # Need E+1 embedding to compute a(E)
        T_eff_next = len(x) - E * tau
        if T_eff_next < E + 2:
            break

        M = delay_embed(x, E, tau)
        M_next = delay_embed(x, E + 1, tau)
        # Align: M_next is shorter; trim M to match
        n = min(len(M), len(M_next))
        M = M[:n]
        M_next = M_next[:n]

        tree = KDTree(M)
        _, nn_idx = tree.query(M, k=2)
        nn_idx = nn_idx[:, 1]

        # a(i, E) = ||M_next[i] - M_next[nn[i]]||_inf / ||M[i] - M[nn[i]]||_inf
        d_E = np.max(np.abs(M - M[nn_idx]), axis=1)
        d_E1 = np.max(np.abs(M_next - M_next[nn_idx]), axis=1)

        safe_d = np.where(d_E > 1e-15, d_E, 1e-15)
        a_values.append(np.mean(d_E1 / safe_d))

    if len(a_values) < 3:
        return 2

    # E1(E) = a(E+1) / a(E)
    E1 = np.array([a_values[i + 1] / max(a_values[i], 1e-15)
                    for i in range(len(a_values) - 1)])

    # Find where E1 saturates (stops changing significantly)
    for i in range(1, len(E1)):
        if abs(E1[i] - 1.0) < 0.1:
            return max(i + 1, 2)  # E = i+1 (1-indexed)

    # If no clear saturation, take the argmin of |E1 - 1|
    return max(int(np.argmin(np.abs(E1 - 1.0))) + 1, 2)


def delay_embed_nonuniform(x, delays):
    """Create a non-uniform delay embedding matrix.

    Unlike standard delay_embed which uses evenly-spaced delays
    [0, tau, 2*tau, ...], this uses arbitrary delays [d1, d2, ...].

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    delays : list of int
        Delay values for each embedding dimension. The first delay
        is typically 0 (current time). E.g., [0, 5, 20, 50].

    Returns
    -------
    embedded : ndarray, shape (T_eff, E)
        Non-uniformly embedded matrix where E = len(delays) and
        T_eff = T - max(delays).
    """
    x = np.asarray(x).ravel()
    delays = np.asarray(delays, dtype=int)
    T = len(x)
    max_delay = int(np.max(delays))
    T_eff = T - max_delay

    if T_eff <= 0:
        raise ValueError(f"Time series too short (T={T}) for delays {delays}")

    indices = np.arange(T_eff)[:, None] + delays[None, :]
    return x[indices]


def select_delays_nonuniform(x, E_max=5, tau_max=50, n_candidates=20):
    """Select non-uniform delays via greedy prediction maximization.

    Iteratively adds delays that maximize simplex prediction accuracy,
    covering multiple timescales for systems with fast-slow dynamics.

    Parameters
    ----------
    x : ndarray
        Input time series.
    E_max : int
        Maximum number of delays (dimensions).
    tau_max : int
        Maximum delay to consider.
    n_candidates : int
        Number of candidate delays to evaluate at each step.

    Returns
    -------
    delays : list of int
        Selected delays, e.g., [0, 5, 23, 48].

    Reference
    ---------
    Pecora, L.M. et al. (2007). A unified approach to attractor
    reconstruction. Chaos, 17(1), 013110.
    """
    x = np.asarray(x, dtype=float).ravel()
    T = len(x)

    # Start with delay 0 (current time)
    selected = [0]

    # Candidate delays: combine linear and log-spaced
    candidates = np.unique(np.concatenate([
        np.arange(1, min(tau_max + 1, n_candidates + 1)),
        np.geomspace(1, tau_max, n_candidates).astype(int),
    ]))
    candidates = candidates[candidates <= tau_max]

    for _ in range(E_max - 1):
        best_tau = None
        best_score = -np.inf

        for tau_c in candidates:
            if tau_c in selected:
                continue

            trial_delays = sorted(selected + [tau_c])
            max_d = max(trial_delays)
            E_trial = len(trial_delays)

            # Need enough points for prediction
            T_eff = T - max_d
            if T_eff < 2 * (E_trial + 2) + 1:
                continue

            try:
                M = delay_embed_nonuniform(x, trial_delays)
                # Trim last point to leave room for prediction target
                M_pred = M[:-1]
                T_pred = len(M_pred)

                if T_pred < 2 * (E_trial + 2):
                    continue

                # Targets: x value one step after the "current" time
                # Row i of M corresponds to time index (max_d + i)
                # so target is x[max_d + i + 1]
                targets = x[max_d + 1: max_d + 1 + T_pred]

                k = min(E_trial + 1, T_pred - 1)
                tree = KDTree(M_pred)
                dists, idxs = tree.query(M_pred, k=k + 1)
                dists = dists[:, 1:]
                idxs = idxs[:, 1:]

                eps = 1e-12
                w = np.exp(-dists / (dists[:, 0:1] + eps))
                w = w / (w.sum(axis=1, keepdims=True) + eps)

                y_pred = np.sum(w * targets[idxs], axis=1)
                rho = np.corrcoef(targets, y_pred)[0, 1]
                if np.isnan(rho):
                    rho = -1.0

                if rho > best_score:
                    best_score = rho
                    best_tau = tau_c
            except (ValueError, IndexError):
                continue

        if best_tau is None:
            break
        selected.append(best_tau)

    return sorted(selected)


def select_parameters(x, tau_max=50, E_max=10, E_method="simplex"):
    """Automatically select embedding parameters (E, tau).

    Parameters
    ----------
    x : ndarray
        Input time series.
    tau_max : int
        Maximum lag for MI / autocorrelation.
    E_max : int
        Maximum embedding dimension.
    E_method : str
        Method for selecting E: 'simplex' (default), 'fnn', or 'cao'.

    Returns
    -------
    E : int
        Embedding dimension.
    tau : int
        Time delay.
    """
    tau = select_tau(x, tau_max)

    if E_method == "fnn":
        E = select_E_fnn(x, tau, E_max)
    elif E_method == "cao":
        E = select_E_cao(x, tau, E_max)
    else:
        E = select_E(x, tau, E_max)

    return E, tau
