"""Coupled Lorenz system network generator."""

import numpy as np
from scipy.integrate import solve_ivp


class LorenzNetwork:
    """Coupled Lorenz oscillator network.

    dx_i/dt = sigma*(y_i - x_i) + eps * sum_j A[i,j]*(x_j - x_i)
    dy_i/dt = x_i*(rho - z_i) - y_i
    dz_i/dt = x_i*y_i - beta*z_i

    Convention: A[i,j]=1 means j drives i.

    Parameters
    ----------
    adj : ndarray, shape (N, N)
        Adjacency matrix.
    coupling : float
        Coupling strength epsilon.
    sigma, rho, beta : float
        Lorenz parameters.
    dt : float
        Sampling interval for output.
    """

    def __init__(self, adj, coupling, sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01):
        self.adj = np.asarray(adj, dtype=float)
        self.coupling = coupling
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.N = adj.shape[0]

    def _deriv(self, t, state):
        N = self.N
        X = state[0:N]
        Y = state[N : 2 * N]
        Z = state[2 * N : 3 * N]

        # Coupling term: sum_j A[i,j] * (x_j - x_i)
        coupling_term = self.adj @ X - self.adj.sum(axis=1) * X

        dX = self.sigma * (Y - X) + self.coupling * coupling_term
        dY = X * (self.rho - Z) - Y
        dZ = X * Y - self.beta * Z

        return np.concatenate([dX, dY, dZ])

    def generate(self, T, transient=1000, seed=None, noise_std=0.0,
                 dyn_noise_std=0.0):
        """Generate coupled Lorenz time series.

        Parameters
        ----------
        T : int
            Number of output time steps (after transient).
        transient : int
            Transient steps to discard.
        seed : int, optional
            Random seed for initial conditions.
        noise_std : float
            Observation noise standard deviation.
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

        # Random initial conditions near the attractor
        state0 = rng.normal(0, 1, size=3 * N)
        state0[0:N] *= 5
        state0[N : 2 * N] *= 5
        state0[2 * N : 3 * N] = np.abs(state0[2 * N : 3 * N]) * 10 + 10

        if dyn_noise_std > 0:
            # Euler-Maruyama SDE integration
            state = state0.copy()
            data_all = np.empty((total, 3 * N))
            sqrt_dt = np.sqrt(self.dt)
            for t in range(total):
                data_all[t] = state
                deriv = self._deriv(t * self.dt, state)
                noise = rng.normal(0, dyn_noise_std, size=N)
                state[0:N] += deriv[0:N] * self.dt + noise * sqrt_dt
                state[N:] += deriv[N:] * self.dt
                if not np.all(np.isfinite(state)):
                    raise RuntimeError(f"Lorenz SDE diverged at step {t}.")
            data = data_all[transient:, 0:N]
        else:
            t_span = (0, total * self.dt)
            t_eval = np.linspace(0, total * self.dt, total)

            sol = solve_ivp(
                self._deriv,
                t_span,
                state0,
                method="RK45",
                t_eval=t_eval,
                rtol=1e-8,
                atol=1e-10,
                max_step=self.dt,
            )

            if sol.status != 0:
                raise RuntimeError(f"ODE integration failed: {sol.message}")

            data = sol.y[0:N, transient:].T  # shape (T, N)

        if not np.all(np.isfinite(data)):
            raise RuntimeError("Lorenz produced non-finite values.")

        if noise_std > 0:
            data = data + rng.normal(0, noise_std, size=data.shape)

        return data
