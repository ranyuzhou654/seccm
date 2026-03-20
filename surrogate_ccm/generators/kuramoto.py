"""Coupled Kuramoto oscillator network generator."""

import numpy as np
from scipy.integrate import solve_ivp


class KuramotoNetwork:
    """Coupled Kuramoto oscillator network.

    dθ_i/dt = ω_i + eps * sum_j A[i,j] * sin(θ_j - θ_i) / k_in_i

    Convention: A[i,j]=1 means j drives i. Observes sin(θ) by default.

    Parameters
    ----------
    adj : ndarray, shape (N, N)
        Adjacency matrix.
    coupling : float
        Coupling strength epsilon.
    omega_mean : float
        Mean natural frequency.
    omega_sigma : float
        Std of natural frequency dispersion.
    observable : str
        'sin' (default) to observe sin(theta), 'phase' for raw phase.
    dt : float
        Integration / sampling time step.
    """

    def __init__(self, adj, coupling, omega_mean=1.0, omega_sigma=0.2,
                 observable="sin", dt=0.05):
        self.adj = np.asarray(adj, dtype=float)
        self.coupling = coupling
        self.omega_mean = omega_mean
        self.omega_sigma = omega_sigma
        self.observable = observable
        self.dt = dt
        self.N = adj.shape[0]

    def _deriv(self, t, theta, omega):
        N = self.N
        # sin(theta_j - theta_i) for all pairs
        diff = theta[:, None] - theta[None, :]  # diff[j,i] = theta_j - theta_i
        coupling_mat = self.adj * np.sin(diff.T)  # A[i,j] * sin(theta_j - theta_i)
        sum_in = coupling_mat.sum(axis=1)
        k_in = self.adj.sum(axis=1)
        k_in_safe = np.where(k_in > 0, k_in, 1.0)
        coupling_term = sum_in / k_in_safe

        return omega + self.coupling * coupling_term

    def generate(self, T, transient=1000, seed=None, noise_std=0.0,
                 dyn_noise_std=0.0):
        """Generate coupled Kuramoto time series.

        Parameters
        ----------
        dyn_noise_std : float
            Dynamical noise std (Euler-Maruyama SDE on theta).

        Returns
        -------
        data : ndarray, shape (T, N)
            Observable (sin(theta) or raw phase) for each node.
        """
        rng = np.random.default_rng(seed)
        N = self.N
        total = T + transient

        omega = rng.normal(self.omega_mean, self.omega_sigma, size=N)
        theta0 = rng.uniform(-np.pi, np.pi, size=N)

        if dyn_noise_std > 0:
            theta = theta0.copy()
            theta_all = np.empty((N, total))
            sqrt_dt = np.sqrt(self.dt)
            for t in range(total):
                theta_all[:, t] = theta
                d = self._deriv(t * self.dt, theta, omega)
                noise = rng.normal(0, dyn_noise_std, size=N)
                theta += d * self.dt + noise * sqrt_dt
                if not np.all(np.isfinite(theta)):
                    raise RuntimeError(f"Kuramoto SDE diverged at step {t}.")
            theta_all = theta_all[:, transient:]
        else:
            t_eval = np.arange(total) * self.dt
            t_span = (0, t_eval[-1])

            sol = solve_ivp(
                lambda t, s: self._deriv(t, s, omega),
                t_span, theta0,
                method="RK45", t_eval=t_eval,
                rtol=1e-8, atol=1e-10, max_step=self.dt,
            )

            if sol.status != 0:
                raise RuntimeError(f"Kuramoto ODE integration failed: {sol.message}")

            theta_all = sol.y[:, transient:]  # (N, T)

        if self.observable == "phase":
            data = theta_all.T
        else:
            data = np.sin(theta_all).T

        if not np.all(np.isfinite(data)):
            raise RuntimeError("Kuramoto produced non-finite values.")

        if noise_std > 0:
            data = data + rng.normal(0, noise_std, size=data.shape)

        return data
