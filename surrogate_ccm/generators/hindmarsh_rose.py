"""Coupled Hindmarsh-Rose neuronal network generator."""

import numpy as np
from scipy.integrate import solve_ivp


class HindmarshRoseNetwork:
    """Coupled Hindmarsh-Rose neuronal network.

    dx_i/dt = y_i - a*x_i^3 + b*x_i^2 - z_i + I + eps * coupling(x)
    dy_i/dt = c - d*x_i^2 - y_i
    dz_i/dt = r * (s * (x_i - x_R) - z_i)

    Convention: A[i,j]=1 means j drives i. Observes x (membrane potential).

    Parameters
    ----------
    adj : ndarray, shape (N, N)
        Adjacency matrix.
    coupling : float
        Coupling strength epsilon.
    """

    def __init__(self, adj, coupling, a=1.0, b=3.0, c=1.0, d=5.0,
                 r=0.01, s=4.0, x_R=-1.6, I_ext=3.5, dt=0.05):
        self.adj = np.asarray(adj, dtype=float)
        self.coupling = coupling
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.r = r
        self.s = s
        self.x_R = x_R
        self.I_ext = I_ext
        self.dt = dt
        self.N = adj.shape[0]

    def _deriv(self, t, state):
        N = self.N
        X = state[0:N]
        Y = state[N:2*N]
        Z = state[2*N:3*N]

        k_in = self.adj.sum(axis=1)
        k_in_safe = np.where(k_in > 0, k_in, 1.0)
        coupling_term = (self.adj @ X - k_in * X) / k_in_safe

        dX = Y - self.a * X**3 + self.b * X**2 - Z + self.I_ext + self.coupling * coupling_term
        dY = self.c - self.d * X**2 - Y
        dZ = self.r * (self.s * (X - self.x_R) - Z)

        return np.concatenate([dX, dY, dZ])

    def generate(self, T, transient=1000, seed=None, noise_std=0.0,
                 dyn_noise_std=0.0):
        """Generate coupled Hindmarsh-Rose time series.

        Parameters
        ----------
        dyn_noise_std : float
            Dynamical noise std (Euler-Maruyama SDE on x-components).

        Returns
        -------
        data : ndarray, shape (T, N)
            x-component (membrane potential) for each node.
        """
        rng = np.random.default_rng(seed)
        N = self.N
        total = T + transient

        state0 = rng.uniform(-2.0, 2.0, size=3 * N)

        if dyn_noise_std > 0:
            state = state0.copy()
            data_all = np.empty((total, 3 * N))
            sqrt_dt = np.sqrt(self.dt)
            for t in range(total):
                data_all[t] = state
                d = self._deriv(t * self.dt, state)
                noise = rng.normal(0, dyn_noise_std, size=N)
                state[0:N] += d[0:N] * self.dt + noise * sqrt_dt
                state[N:] += d[N:] * self.dt
                if not np.all(np.isfinite(state)):
                    raise RuntimeError(f"HindmarshRose SDE diverged at step {t}.")
            data = data_all[transient:, 0:N]
        else:
            t_span = (0, total * self.dt)
            t_eval = np.linspace(0, total * self.dt, total)

            sol = solve_ivp(
                self._deriv, t_span, state0,
                method="RK45", t_eval=t_eval,
                rtol=1e-8, atol=1e-10, max_step=self.dt,
            )

            if sol.status != 0:
                raise RuntimeError(f"HindmarshRose ODE integration failed: {sol.message}")

            data = sol.y[0:N, transient:].T

        if not np.all(np.isfinite(data)):
            raise RuntimeError("HindmarshRose produced non-finite values.")

        if noise_std > 0:
            data = data + rng.normal(0, noise_std, size=data.shape)

        return data
