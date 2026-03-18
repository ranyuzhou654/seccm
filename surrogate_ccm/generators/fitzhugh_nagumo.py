"""Coupled FitzHugh-Nagumo neuronal network generator."""

import numpy as np
from scipy.integrate import solve_ivp


class FitzHughNagumoNetwork:
    """Coupled FitzHugh-Nagumo neuronal network (2D per node).

    dv_i/dt = v_i - v_i^3/3 - w_i + I + eps * coupling(v)
    dw_i/dt = (v_i + a - b*w_i) / tau

    Convention: A[i,j]=1 means j drives i. Observes v (membrane potential).

    Parameters
    ----------
    adj : ndarray, shape (N, N)
        Adjacency matrix.
    coupling : float
        Coupling strength epsilon.
    """

    def __init__(self, adj, coupling, a=0.7, b=0.8, tau=12.5, I_ext=0.5,
                 dt=0.05):
        self.adj = np.asarray(adj, dtype=float)
        self.coupling = coupling
        self.a = a
        self.b = b
        self.tau = tau
        self.I_ext = I_ext
        self.dt = dt
        self.N = adj.shape[0]

    def _deriv(self, t, state):
        N = self.N
        V = state[0:N]
        W = state[N:2*N]

        k_in = self.adj.sum(axis=1)
        k_in_safe = np.where(k_in > 0, k_in, 1.0)
        coupling_term = (self.adj @ V - k_in * V) / k_in_safe

        dV = V - (V**3) / 3.0 - W + self.I_ext + self.coupling * coupling_term
        dW = (V + self.a - self.b * W) / self.tau

        return np.concatenate([dV, dW])

    def generate(self, T, transient=1000, seed=None, noise_std=0.0,
                 dyn_noise_std=0.0):
        """Generate coupled FitzHugh-Nagumo time series.

        Parameters
        ----------
        dyn_noise_std : float
            Dynamical noise std (Euler-Maruyama SDE on v-components).

        Returns
        -------
        data : ndarray, shape (T, N)
            v-component (membrane potential) for each node.
        """
        rng = np.random.default_rng(seed)
        N = self.N
        total = T + transient

        state0 = rng.uniform(-2.0, 2.0, size=2 * N)

        if dyn_noise_std > 0:
            state = state0.copy()
            data_all = np.empty((total, 2 * N))
            sqrt_dt = np.sqrt(self.dt)
            for t in range(total):
                data_all[t] = state
                d = self._deriv(t * self.dt, state)
                noise = rng.normal(0, dyn_noise_std, size=N)
                state[0:N] += d[0:N] * self.dt + noise * sqrt_dt
                state[N:] += d[N:] * self.dt
                if not np.all(np.isfinite(state)):
                    raise RuntimeError(f"FitzHughNagumo SDE diverged at step {t}.")
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
                raise RuntimeError(f"FitzHughNagumo ODE integration failed: {sol.message}")

            data = sol.y[0:N, transient:].T

        if not np.all(np.isfinite(data)):
            raise RuntimeError("FitzHughNagumo produced non-finite values.")

        if noise_std > 0:
            data = data + rng.normal(0, noise_std, size=data.shape)

        return data
