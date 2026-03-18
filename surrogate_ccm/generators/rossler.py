"""Coupled Rössler system network generator."""

import numpy as np
from scipy.integrate import solve_ivp


class RosslerNetwork:
    """Coupled Rössler oscillator network.

    dx_i/dt = -y_i - z_i + eps * sum_j A[i,j] * (x_j - x_i) / k_in_i
    dy_i/dt = x_i + a * y_i
    dz_i/dt = b + z_i * (x_i - c_i)

    Convention: A[i,j]=1 means j drives i. Observes x component.

    Parameters
    ----------
    adj : ndarray, shape (N, N)
        Adjacency matrix.
    coupling : float
        Coupling strength epsilon.
    a, b, c : float
        Rössler parameters.
    dt : float
        Integration / sampling time step.
    hetero_sigma : float
        Standard deviation for heterogeneous c parameter across nodes.
    """

    def __init__(self, adj, coupling, a=0.2, b=0.2, c=5.7, dt=0.05,
                 hetero_sigma=0.0):
        self.adj = np.asarray(adj, dtype=float)
        self.coupling = coupling
        self.a = a
        self.b = b
        self.c = c
        self.dt = dt
        self.hetero_sigma = hetero_sigma
        self.N = adj.shape[0]

    def _deriv(self, t, state, A, eps, c_vec):
        N = self.N
        X = state[0:N]
        Y = state[N:2*N]
        Z = state[2*N:3*N]

        # Diffusive coupling on x
        k_in = A.sum(axis=1)
        k_in_safe = np.where(k_in > 0, k_in, 1.0)
        coupling_term = (A @ X - k_in * X) / k_in_safe

        dX = -Y - Z + eps * coupling_term
        dY = X + self.a * Y
        dZ = self.b + Z * (X - c_vec)

        return np.concatenate([dX, dY, dZ])

    def generate(self, T, transient=1000, seed=None, noise_std=0.0,
                 dyn_noise_std=0.0):
        """Generate coupled Rössler time series.

        Parameters
        ----------
        dyn_noise_std : float
            Dynamical noise std (Euler-Maruyama SDE on x-components).

        Returns
        -------
        data : ndarray, shape (T, N)
            x-component time series for each node.
        """
        rng = np.random.default_rng(seed)
        N = self.N
        total = T + transient

        c_vec = self.c + rng.normal(0.0, self.hetero_sigma, size=N)
        state0 = rng.uniform(-5.0, 5.0, size=3 * N)

        if dyn_noise_std > 0:
            deriv_fn = lambda t, s: self._deriv(t, s, self.adj, self.coupling, c_vec)
            state = state0.copy()
            data_all = np.empty((total, 3 * N))
            sqrt_dt = np.sqrt(self.dt)
            for t in range(total):
                data_all[t] = state
                d = deriv_fn(t * self.dt, state)
                noise = rng.normal(0, dyn_noise_std, size=N)
                state[0:N] += d[0:N] * self.dt + noise * sqrt_dt
                state[N:] += d[N:] * self.dt
                if not np.all(np.isfinite(state)):
                    raise RuntimeError(f"Rössler SDE diverged at step {t}.")
            data = data_all[transient:, 0:N]
        else:
            t_span = (0, total * self.dt)
            t_eval = np.linspace(0, total * self.dt, total)

            sol = solve_ivp(
                lambda t, s: self._deriv(t, s, self.adj, self.coupling, c_vec),
                t_span, state0,
                method="RK45", t_eval=t_eval,
                rtol=1e-8, atol=1e-10, max_step=self.dt,
            )

            if sol.status != 0:
                raise RuntimeError(f"Rössler ODE integration failed: {sol.message}")

            data = sol.y[0:N, transient:].T

        if not np.all(np.isfinite(data)):
            raise RuntimeError("Rössler produced non-finite values.")

        if noise_std > 0:
            data = data + rng.normal(0, noise_std, size=data.shape)

        return data
