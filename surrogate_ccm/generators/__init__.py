"""System generators for coupled chaotic networks."""

from .fitzhugh_nagumo import FitzHughNagumoNetwork
from .henon import HenonNetwork
from .hindmarsh_rose import HindmarshRoseNetwork
from .kuramoto import KuramotoNetwork
from .logistic import LogisticNetwork
from .lorenz import LorenzNetwork
from .network import generate_network
from .rossler import RosslerNetwork
from .van_der_pol import VanDerPolNetwork

SYSTEM_CLASSES = {
    "logistic": LogisticNetwork,
    "lorenz": LorenzNetwork,
    "henon": HenonNetwork,
    "rossler": RosslerNetwork,
    "hindmarsh_rose": HindmarshRoseNetwork,
    "fitzhugh_nagumo": FitzHughNagumoNetwork,
    "kuramoto": KuramotoNetwork,
    "van_der_pol": VanDerPolNetwork,
}


def create_system(system_name, adj, coupling, **kwargs):
    """Factory to create a system generator by name."""
    cls = SYSTEM_CLASSES.get(system_name.lower())
    if cls is None:
        raise ValueError(f"Unknown system: {system_name}. Choose from {list(SYSTEM_CLASSES)}")
    return cls(adj, coupling, **kwargs)
