"""Optional FAISS-accelerated k-nearest-neighbor search."""

import numpy as np


def _faiss_available():
    """Check if faiss (CPU or GPU) is importable."""
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def _faiss_gpu_available():
    """Check if faiss-gpu resources are available."""
    try:
        import faiss
        res = faiss.StandardGpuResources()  # noqa: F841
        return True
    except Exception:
        return False


def knn_query(M, k, use_gpu=True):
    """Find k nearest neighbors (excluding self) using FAISS or KDTree.

    Parameters
    ----------
    M : ndarray, shape (L, E)
        Query/database points (same set for self-query).
    k : int
        Number of neighbors to return.
    use_gpu : bool
        If True and faiss-gpu is available, use GPU acceleration.

    Returns
    -------
    dists : ndarray, shape (L, k)
        Euclidean distances to k nearest neighbors.
    idxs : ndarray, shape (L, k)
        Indices of k nearest neighbors.
    """
    L, E = M.shape

    if _faiss_available() and L > 500:
        return _knn_faiss(M, k, use_gpu=use_gpu)
    return _knn_kdtree(M, k)


def _knn_faiss(M, k, use_gpu=True):
    """FAISS-based kNN (L2 distance)."""
    import faiss

    M_f32 = np.ascontiguousarray(M, dtype=np.float32)
    index = faiss.IndexFlatL2(M_f32.shape[1])

    if use_gpu and _faiss_gpu_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(M_f32)
    # Query k+1 to exclude self
    dists_sq, idxs = index.search(M_f32, k + 1)
    # Drop self (first column)
    dists = np.sqrt(np.maximum(dists_sq[:, 1:], 0.0))
    idxs = idxs[:, 1:]
    return dists, idxs


def _knn_kdtree(M, k):
    """scipy KDTree fallback."""
    from scipy.spatial import KDTree

    tree = KDTree(M)
    dists, idxs = tree.query(M, k=k + 1)
    return dists[:, 1:], idxs[:, 1:]
