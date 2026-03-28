"""Network graph visualization and performance curve plots."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_network_comparison(
    detected, ground_truth, title="Network Comparison", ax=None, save_path=None
):
    """Plot directed graph with TP (green), FP (red), FN (dashed gray).

    Parameters
    ----------
    detected : ndarray, shape (N, N)
        Detected adjacency matrix.
    ground_truth : ndarray, shape (N, N)
        Ground truth adjacency matrix.
    title : str
    ax : matplotlib.axes.Axes, optional
    save_path : str, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    N = detected.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    pos = nx.circular_layout(G)

    tp_edges, fp_edges, fn_edges = [], [], []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            is_true = ground_truth[i, j] > 0
            is_detected = detected[i, j] > 0
            # Edge j -> i in networkx
            if is_true and is_detected:
                tp_edges.append((j, i))
            elif not is_true and is_detected:
                fp_edges.append((j, i))
            elif is_true and not is_detected:
                fn_edges.append((j, i))

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    if tp_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=tp_edges, edge_color="green", width=2,
            alpha=0.8, ax=ax, connectionstyle="arc3,rad=0.1",
        )
    if fp_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=fp_edges, edge_color="red", width=2,
            alpha=0.8, ax=ax, connectionstyle="arc3,rad=0.1",
        )
    if fn_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=fn_edges, edge_color="gray", width=1.5,
            style="dashed", alpha=0.6, ax=ax, connectionstyle="arc3,rad=0.1",
        )

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="green", linewidth=2, label=f"TP ({len(tp_edges)})"),
        Line2D([0], [0], color="red", linewidth=2, label=f"FP ({len(fp_edges)})"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label=f"FN ({len(fn_edges)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return ax


def plot_surrogate_distribution(
    rho_obs, rho_surr, title="Surrogate Distribution", ax=None, save_path=None
):
    """Plot histogram of surrogate CCM values with observed value line.

    Parameters
    ----------
    rho_obs : float
        Observed CCM correlation.
    rho_surr : ndarray
        Surrogate CCM correlations.
    title : str
    ax : matplotlib.axes.Axes, optional
    save_path : str, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.hist(rho_surr, bins=30, alpha=0.7, color="steelblue", edgecolor="white", density=True)
    ax.axvline(rho_obs, color="red", linewidth=2, linestyle="--", label=f"Observed ρ={rho_obs:.3f}")
    ax.set_xlabel("CCM ρ")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return ax


def plot_performance_curves(
    x_values,
    metrics_dict,
    x_label="Coupling strength ε",
    title="Detection Performance",
    save_path=None,
):
    """Plot performance metrics vs experimental parameter.

    Parameters
    ----------
    x_values : array-like
        Parameter values (e.g., coupling strengths).
    metrics_dict : dict
        Keys are metric names, values are arrays of shape (len(x_values),)
        or (len(x_values), n_reps) for error bars.
    x_label : str
    title : str
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set1(np.linspace(0, 1, len(metrics_dict)))

    for (name, values), color in zip(metrics_dict.items(), colors):
        values = np.asarray(values, dtype=object)
        if values.ndim == 1 and len(values) > 0 and not np.isscalar(values[0]):
            max_reps = max(len(np.asarray(row).ravel()) for row in values)
            padded = np.full((len(values), max_reps), np.nan, dtype=float)
            for i, row in enumerate(values):
                row_arr = np.asarray(row, dtype=float).ravel()
                padded[i, :len(row_arr)] = row_arr
            values = padded
        else:
            values = np.asarray(values, dtype=float)

        if values.ndim == 2:
            mean = np.nanmean(values, axis=1)
            counts = np.sum(np.isfinite(values), axis=1)
            std = np.nanstd(values, axis=1, ddof=1)
            sem = np.divide(
                std,
                np.sqrt(np.maximum(counts, 1)),
                out=np.zeros_like(std),
                where=counts > 1,
            )
            ax.fill_between(x_values, mean - sem, mean + sem, alpha=0.2, color=color)
            ax.plot(x_values, mean, "-o", color=color, label=name, markersize=5)
        else:
            ax.plot(x_values, values, "-o", color=color, label=name, markersize=5)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_method_comparison(
    results_by_method, metric="TPR", title="Surrogate Method Comparison", save_path=None
):
    """Boxplot comparing surrogate methods on a given metric.

    Parameters
    ----------
    results_by_method : dict
        Keys are method names, values are arrays of metric values across reps.
    metric : str
        Name of the metric (for axis label).
    title : str
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results_by_method.keys())
    data = [np.asarray(results_by_method[m]) for m in methods]

    bp = ax.boxplot(data, labels=methods, patch_artist=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
