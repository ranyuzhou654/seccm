"""Pairwise CCM computation over a network."""

import numpy as np

from .ccm_core import ccm
from .embedding import select_parameters


def compute_pairwise_ccm(data, E=None, tau=None, params_per_node=None,
                         theiler_w=0):
    """Compute pairwise CCM matrix for all node pairs.

    For each pair (i, j), computes ccm(x_i, x_j) which tests "j causes i".
    Result[i, j] = correlation for the hypothesis j -> i.

    Parameters
    ----------
    data : ndarray, shape (T, N)
        Time series data, one column per node.
    E : int, optional
        Global embedding dimension (overrides per-node).
    tau : int, optional
        Global time delay (overrides per-node).
    params_per_node : list of (E, tau), optional
        Per-node embedding parameters. If None and E/tau are None,
        parameters are auto-selected.
    theiler_w : int
        Theiler window for neighbor exclusion.

    Returns
    -------
    ccm_matrix : ndarray, shape (N, N)
        CCM correlation matrix. ccm_matrix[i,j] tests j->i.
    params : list of (E, tau)
        Embedding parameters used for each node.
    """
    N = data.shape[1]
    ccm_matrix = np.zeros((N, N))

    # Determine embedding parameters
    if params_per_node is not None:
        params = params_per_node
    elif E is not None and tau is not None:
        params = [(E, tau)] * N
    else:
        params = [select_parameters(data[:, i]) for i in range(N)]

    for i in range(N):
        E_i, tau_i = params[i]
        for j in range(N):
            if i == j:
                continue
            ccm_matrix[i, j] = ccm(data[:, i], data[:, j], E_i, tau_i,
                                   theiler_w=theiler_w)

    return ccm_matrix, params
