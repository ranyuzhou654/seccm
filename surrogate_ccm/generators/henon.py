"""Coupled Henon Map network generator."""

import numpy as np


class HenonNetwork:
    """Coupled Henon map network.

    X_i(t+1) = 1 - a*X_i(t)^2 + Y_i(t) + C * sum_j A[i,j] * (X_j(t) - X_i(t))
    Y_i(t+1) = b * X_i(t)

    Convention: A[i,j]=1 means j drives i. Observes X component only.

    Parameters
    ----------
    adj : ndarray, shape (N, N)
        Adjacency matrix.
    coupling : float
        Coupling strength C.
    a, b : float
        Henon parameters.
    """

    def __init__(self, adj, coupling, a=1.2, b=0.3, init_scale=0.1,
                 max_retries=10):
        self.adj = np.asarray(adj, dtype=float)
        self.coupling = coupling
        self.a = a
        self.b = b
        self.init_scale = init_scale
        self.max_retries = max_retries
        self.N = adj.shape[0]

    def generate(self, T, transient=1000, seed=None, noise_std=0.0,
                 dyn_noise_std=0.0):
        """Generate coupled Henon map time series.

        Parameters
        ----------
        T : int
            Number of time steps to return (after transient).
        transient : int
            Steps to discard.
        seed : int, optional
            Random seed.
        noise_std : float
            Observation noise standard deviation.
        dyn_noise_std : float
            Dynamical noise standard deviation (added at each iteration).

        Returns
        -------
        data : ndarray, shape (T, N)
            X-component time series for each node.
        """
        rng = np.random.default_rng(seed)
        total = T + transient
        a, b, C = self.a, self.b, self.coupling

        divergence_threshold = 1e10
        k_in = self.adj.sum(axis=1)
        k_in_safe = np.where(k_in > 0, k_in, 1.0)

        last_error = None
        for _ in range(self.max_retries):
            X = rng.uniform(-self.init_scale, self.init_scale, size=self.N)
            Y = rng.uniform(-self.init_scale, self.init_scale, size=self.N)
            data = np.empty((total, self.N))

            try:
                for t in range(total):
                    data[t] = X
                    # Diffusive coupling: sum_j A[i,j] * (X_j - X_i)
                    coupled = (self.adj @ X - k_in * X) / k_in_safe
                    X_new = 1 - a * X**2 + Y + C * coupled
                    if dyn_noise_std > 0:
                        X_new += rng.normal(0, dyn_noise_std, size=self.N)
                    Y_new = b * X

                    if np.any(np.abs(X_new) > divergence_threshold):
                        raise RuntimeError(
                            f"Henon map diverged at step {t}. "
                            f"Try reducing coupling={C} or parameter a={a}."
                        )

                    X = X_new
                    Y = Y_new
                break
            except RuntimeError as exc:
                last_error = exc
        else:
            raise RuntimeError(
                f"Henon map diverged after {self.max_retries} retries. "
                f"Last error: {last_error}"
            )

        data = data[transient:]

        if noise_std > 0:
            data = data + rng.normal(0, noise_std, size=data.shape)

        return data
