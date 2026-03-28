"""Dysts-based coupled chaotic network generators.

Wraps single-node chaotic systems from the dysts package into our
diffusive coupling framework:

    dx_i/dt = f_dysts(x_i) + eps * sum_j A[i,j]*(x_j - x_i) / k_in_i

Convention: A[i,j]=1 means j drives i.
"""

import numpy as np
from scipy.integrate import solve_ivp

try:
    import dysts.flows as flows
except ImportError:
    flows = None


# Map of our system names -> (dysts class name, recommended coupling, dt)
DYSTS_REGISTRY = {
    "chen":         ("Chen",        0.3,  0.01),
    "thomas":       ("Thomas",      0.2,  0.05),
    "halvorsen":    ("Halvorsen",   0.2,  0.01),
    "rucklidge":    ("Rucklidge",   0.2,  0.01),
    "sprott_b":     ("SprottB",     0.3,  0.05),
    "bouali":       ("Bouali2",     0.1,  0.01),
    "nose_hoover":  ("NoseHoover",  0.3,  0.05),
}


def _get_dysts_system(class_name):
    """Instantiate a dysts flow by class name and extract parameters."""
    if flows is None:
        raise ImportError(
            "dysts package is required. Install with: pip install dysts"
        )
    cls = getattr(flows, class_name, None)
    if cls is None:
        raise ValueError(f"dysts has no flow named '{class_name}'")
    system = cls()
    # Get canonical parameters and initial conditions
    params = np.array(system.default_params) if hasattr(system, 'default_params') else None
    ic = np.array(system.ic)
    return system, params, ic


class DystsNetwork:
    """Coupled network of identical dysts chaotic oscillators.

    Each node runs the same dysts system with diffusive coupling on the
    first state variable (x-component).

    Parameters
    ----------
    adj : ndarray, shape (N, N)
        Adjacency matrix. A[i,j]=1 means j drives i.
    coupling : float
        Coupling strength epsilon.
    dysts_name : str
        Name of the dysts flow class (e.g., "Chen", "Thomas").
    dt : float
        Integration/sampling timestep.
    """

    def __init__(self, adj, coupling, dysts_name="Chen", dt=0.01):
        self.adj = np.asarray(adj, dtype=float)
        self.coupling = coupling
        self.dt = dt
        self.N = adj.shape[0]
        self.dysts_name = dysts_name

        # Get the dysts system to extract its RHS and IC
        self._dysts_sys, self._params, self._ic0 = _get_dysts_system(dysts_name)
        self.dim = len(self._ic0)  # state dimension per node (typically 3)

    def _single_node_rhs(self, state_1d):
        """Evaluate RHS for a single node (dim-dimensional state)."""
        # dysts systems have a .rhs method: rhs(state, t) -> derivatives
        return np.array(self._dysts_sys.rhs(state_1d, 0))

    def _deriv(self, t, state):
        N = self.N
        dim = self.dim

        # Reshape: (N, dim)
        S = state.reshape(N, dim)

        # Compute per-node dynamics
        dS = np.zeros_like(S)
        for i in range(N):
            dS[i] = self._single_node_rhs(S[i])

        # Diffusive coupling on x-component (index 0)
        X = S[:, 0]
        k_in = self.adj.sum(axis=1)
        k_in_safe = np.where(k_in > 0, k_in, 1.0)
        coupling_term = (self.adj @ X - k_in * X) / k_in_safe
        dS[:, 0] += self.coupling * coupling_term

        return dS.ravel()

    def generate(self, T, transient=1000, seed=None, noise_std=0.0,
                 dyn_noise_std=0.0):
        """Generate coupled dysts time series.

        Parameters
        ----------
        T : int
            Number of output time steps (after transient).
        transient : int
            Transient steps to discard.
        seed : int, optional
            Random seed.
        noise_std : float
            Observation noise standard deviation.
        dyn_noise_std : float
            Dynamical noise std (unused for ODE, kept for API compat).

        Returns
        -------
        data : ndarray, shape (T, N)
            x-component (first variable) time series for each node.
        """
        rng = np.random.default_rng(seed)
        N = self.N
        dim = self.dim
        total = T + transient

        # Initial conditions: perturb canonical IC for each node
        ic = self._ic0
        state0 = np.zeros(N * dim)
        for i in range(N):
            perturbation = rng.normal(0, 0.01, size=dim)
            state0[i * dim:(i + 1) * dim] = ic + perturbation * np.abs(ic + 1e-6)

        t_eval = np.arange(total) * self.dt
        t_span = (0, t_eval[-1])

        sol = solve_ivp(
            self._deriv,
            t_span,
            state0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            max_step=self.dt * 2,
        )

        if sol.status != 0:
            raise RuntimeError(
                f"ODE integration failed for {self.dysts_name}: {sol.message}"
            )

        # Extract x-component (index 0) for each node
        # sol.y shape: (N*dim, total_steps)
        all_states = sol.y  # (N*dim, n_steps)
        data = np.zeros((total, N))
        for i in range(N):
            data[:, i] = all_states[i * dim, :]

        # Discard transient
        data = data[transient:]

        if not np.all(np.isfinite(data)):
            raise RuntimeError(
                f"{self.dysts_name} network produced non-finite values."
            )

        if noise_std > 0:
            data = data + rng.normal(0, noise_std, size=data.shape)

        return data


def make_dysts_network_class(registry_name, dysts_class_name, default_dt=0.01):
    """Factory: create a specialized DystsNetwork subclass for registration."""

    class _DystsSubclass(DystsNetwork):
        __doc__ = f"Coupled {dysts_class_name} network (via dysts)."

        def __init__(self, adj, coupling, dt=default_dt, **kwargs):
            super().__init__(adj, coupling, dysts_name=dysts_class_name, dt=dt)

    _DystsSubclass.__name__ = f"{dysts_class_name}Network"
    _DystsSubclass.__qualname__ = f"{dysts_class_name}Network"
    return _DystsSubclass


# Pre-build classes for each registered dysts system
DYSTS_SYSTEM_CLASSES = {}
for _name, (_cls_name, _coupling, _dt) in DYSTS_REGISTRY.items():
    DYSTS_SYSTEM_CLASSES[_name] = make_dysts_network_class(_name, _cls_name, _dt)
