"""Coupled van der Pol oscillator network generator.

The van der Pol oscillator is a classic nonlinear oscillator with limit cycle
behavior. With coupling, it produces oscillatory dynamics useful for testing
cycle-phase surrogates on a different oscillatory system than Rossler.

Convention: A[i,j] = 1 means j drives i. Observes x component.
"""

import numpy as np
from scipy.integrate import solve_ivp


class VanDerPolNetwork:
    """Coupled van der Pol oscillator network.

    Each node's dynamics:
        dx_i/dt = y_i
        dy_i/dt = mu * (1 - x_i^2) * y_i - x_i + eps * sum_j A[i,j] * (x_j - x_i)

    Parameters
    ----------
    adj : ndarray, shape (N, N)
        Adjacency matrix. A[i,j]=1 means j→i.
    coupling : float
        Coupling strength epsilon.
    mu : float
        Nonlinearity parameter. Default 1.0. Larger mu = more relaxation.
    dt : float
        Integration/sampling time step.
    hetero_sigma : float
        Std for heterogeneous mu across nodes.
    max_retries : int
        Retries on divergence.
    """

    def __init__(self, adj, coupling, mu=1.0, dt=0.01,
                 hetero_sigma=0.0, max_retries=10):
        self.adj = np.asarray(adj, dtype=float)
        self.N = self.adj.shape[0]
        self.coupling = coupling
        self.mu = mu
        self.dt = dt
        self.hetero_sigma = hetero_sigma
        self.max_retries = max_retries

    def _deriv(self, t, state, A, eps, mu_vec):
        N = self.N
        x = state[:N]
        y = state[N:]
        dxdt = y.copy()
        dydt = mu_vec * (1.0 - x**2) * y - x

        # Coupling: diffusive on x
        for i in range(N):
            coupling_sum = 0.0
            k_in = A[i].sum()
            if k_in > 0:
                for j in range(N):
                    if A[i, j] > 0:
                        coupling_sum += A[i, j] * (x[j] - x[i])
                dydt[i] += eps * coupling_sum / k_in

        return np.concatenate([dxdt, dydt])

    def generate(self, T, transient=1000, seed=None, noise_std=0.0,
                 dyn_noise_std=0.0):
        """Generate coupled van der Pol time series.

        Parameters
        ----------
        T : int
            Number of output time steps (after transient).
        transient : int
            Steps to discard.
        seed : int, optional
            Random seed.
        noise_std : float
            Observation noise std.
        dyn_noise_std : float
            Dynamical noise std.

        Returns
        -------
        data : ndarray, shape (T, N)
            x-component per node.
        """
        rng = np.random.default_rng(seed)

        # Heterogeneous mu
        mu_vec = np.full(self.N, self.mu)
        if self.hetero_sigma > 0:
            mu_vec += rng.normal(0, self.hetero_sigma, self.N)
            mu_vec = np.clip(mu_vec, 0.1, None)

        total_steps = T + transient
        t_eval = np.arange(total_steps) * self.dt
        t_span = (0, t_eval[-1])

        for attempt in range(self.max_retries):
            # Random initial conditions
            x0 = rng.uniform(-2, 2, self.N)
            y0 = rng.uniform(-2, 2, self.N)
            state0 = np.concatenate([x0, y0])

            if dyn_noise_std > 0:
                # Euler-Maruyama for SDE
                data = self._euler_maruyama(
                    state0, total_steps, mu_vec, dyn_noise_std, rng
                )
            else:
                try:
                    sol = solve_ivp(
                        self._deriv, t_span, state0,
                        args=(self.adj, self.coupling, mu_vec),
                        t_eval=t_eval, method="RK45",
                        max_step=self.dt * 2,
                    )
                    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
                        continue
                    data = sol.y.T  # shape (total_steps, 2*N)
                except Exception:
                    continue

            # Check for divergence
            if not np.all(np.isfinite(data)) or np.max(np.abs(data)) > 1e4:
                continue

            # Extract x components, discard transient
            x_data = data[transient:, :self.N]

            # Add observation noise
            if noise_std > 0:
                x_data = x_data + rng.normal(0, noise_std, x_data.shape)

            return x_data

        raise RuntimeError(
            f"VanDerPolNetwork diverged after {self.max_retries} attempts"
        )

    def _euler_maruyama(self, state0, total_steps, mu_vec, dyn_noise_std, rng):
        """Euler-Maruyama integration for SDE case."""
        N = self.N
        state = state0.copy()
        data = np.empty((total_steps, 2 * N))
        data[0] = state
        sqrt_dt = np.sqrt(self.dt)

        for t in range(1, total_steps):
            deriv = self._deriv(t * self.dt, state, self.adj,
                                self.coupling, mu_vec)
            noise = np.zeros(2 * N)
            noise[:N] = rng.normal(0, dyn_noise_std, N) * sqrt_dt
            state = state + deriv * self.dt + noise
            data[t] = state

            if np.max(np.abs(state)) > 1e4:
                raise RuntimeError("Euler-Maruyama diverged")

        return data
