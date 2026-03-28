"""Network topology generation (ER, WS, Ring).

Convention: A[i,j] = 1 means j -> i (j drives i).
"""

import networkx as nx
import numpy as np


def generate_network(topology, N, seed=None, **kwargs):
    """Generate a directed adjacency matrix.

    Parameters
    ----------
    topology : str
        One of 'ER', 'WS', 'ring'.
    N : int
        Number of nodes.
    seed : int, optional
        Random seed.
    **kwargs
        Topology-specific parameters:
        - ER: p (edge probability, default 0.3)
        - WS: k (nearest neighbors, default 4), p (rewiring prob, default 0.3)
        - BA: m (edges per new node, default 2)
        - ring: k (nearest neighbors per side, default 1)

    Returns
    -------
    adj : ndarray, shape (N, N)
        Adjacency matrix where A[i,j]=1 means j->i.
    """
    topology = topology.upper()

    rng = np.random.default_rng(seed)

    if topology == "ER":
        p = kwargs.get("p", kwargs.get("er_p", 0.3))
        G = nx.erdos_renyi_graph(N, p, directed=True, seed=seed)

    elif topology == "WS":
        k = kwargs.get("k", kwargs.get("ws_k", 4))
        p = kwargs.get("p", kwargs.get("ws_p", 0.3))
        G_undirected = nx.watts_strogatz_graph(N, k, p, seed=seed)
        # Randomly orient each undirected edge to avoid full reciprocity
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        for u, v in G_undirected.edges():
            if rng.random() < 0.5:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)

    elif topology == "BA":
        m = kwargs.get("m", kwargs.get("ba_m", 2))
        G_undirected = nx.barabasi_albert_graph(N, m, seed=seed)
        # Randomly orient each undirected edge (same pattern as WS)
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        for u, v in G_undirected.edges():
            if rng.random() < 0.5:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)

    elif topology == "RING":
        k = kwargs.get("k", 1)
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for offset in range(1, k + 1):
                j = (i + offset) % N
                # Randomly orient each neighbor pair
                if rng.random() < 0.5:
                    G.add_edge(j, i)
                else:
                    G.add_edge(i, j)

    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # nx.to_numpy_array returns adj[i,j]=1 meaning i->j (row=source).
    # Our convention is A[i,j]=1 meaning j->i (row=receiver), so transpose.
    adj = nx.to_numpy_array(G, dtype=int).T
    return adj
