"""SE-CCM: Surrogate-Enhanced Convergent Cross Mapping pipeline."""

import numpy as np
from tqdm import tqdm

from ..ccm.ccm_core import ccm, convergence_score
from ..ccm.embedding import delay_embed, select_parameters
from ..ccm.network_ccm import compute_pairwise_ccm
from ..surrogate import generate_surrogate
from .hypothesis_test import compute_pvalue, compute_zscore, fdr_correction
from ..utils.backend import gpu_available, get_array_module, to_device, to_numpy


def _ccm_predict_rho(M_x, y_aligned, E):
    """Compute CCM rho using a pre-built shadow manifold M_x.

    Avoids redundant re-embedding when only the prediction target changes
    (e.g., across surrogates).
    """
    L = len(M_x)
    k = E + 1
    from scipy.spatial import KDTree

    tree = KDTree(M_x)
    dists, idxs = tree.query(M_x, k=k + 1)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    eps = 1e-12
    w = np.exp(-dists / (dists[:, 0:1] + eps))
    w = w / (w.sum(axis=1, keepdims=True) + eps)

    y_pred = np.sum(w * y_aligned[idxs], axis=1)
    rho = np.corrcoef(y_aligned, y_pred)[0, 1]
    return 0.0 if np.isnan(rho) else float(rho)


def _ccm_surrogate_batch(M_x, idxs, w, y_aligned, surrogates, offset, T_eff,
                          use_gpu=False):
    """Vectorized surrogate CCM: compute rho for all surrogates at once.

    Uses batch matrix operations instead of per-surrogate loops.
    Optionally runs on GPU via CuPy.
    """
    xp = get_array_module(use_gpu)

    surr_sliced = xp.asarray(surrogates[:, offset:offset + T_eff])  # (n_surr, T_eff)
    idxs_d = xp.asarray(idxs)   # (L, k)
    w_d = xp.asarray(w)         # (L, k)

    # Gather neighbor values for all surrogates: (n_surr, L, k)
    y_neighbors = surr_sliced[:, idxs_d]
    # Weighted prediction: (n_surr, L)
    y_pred = xp.sum(w_d[None, :, :] * y_neighbors, axis=2)

    # Batch Pearson correlation (manual formula avoids per-surrogate np.corrcoef)
    y_s_c = surr_sliced - surr_sliced.mean(axis=1, keepdims=True)
    y_p_c = y_pred - y_pred.mean(axis=1, keepdims=True)
    num = xp.sum(y_s_c * y_p_c, axis=1)
    den = xp.sqrt(xp.sum(y_s_c ** 2, axis=1) * xp.sum(y_p_c ** 2, axis=1))
    rho_surr = num / (den + 1e-12)

    rho_surr = to_numpy(rho_surr)
    rho_surr = np.nan_to_num(rho_surr, nan=0.0)
    return rho_surr


class SECCM:
    """Surrogate-Enhanced CCM for causal network inference.

    Parameters
    ----------
    surrogate_method : str
        Surrogate generation method.
    n_surrogates : int
        Number of surrogates per test.
    alpha : float
        Significance level for hypothesis testing.
    fdr : bool
        Whether to apply FDR correction.
    seed : int, optional
        Random seed.
    iaaft_max_iter : int
        Max iterations for iAAFT.
    use_gpu : bool or "auto"
        GPU acceleration via CuPy. "auto" detects availability at fit time.
    """

    def __init__(
        self,
        surrogate_method="iaaft",
        n_surrogates=100,
        alpha=0.05,
        fdr=True,
        seed=None,
        iaaft_max_iter=200,
        verbose=True,
        min_rho=0.3,
        adaptive_rho=True,
        adaptive_rho_quantile=0.95,
        theiler_w="auto",
        E_method="simplex",
        convergence_filter=True,
        convergence_threshold=0.0,
        use_gpu="auto",
    ):
        self.surrogate_method = surrogate_method
        self.n_surrogates = n_surrogates
        self.alpha = alpha
        self.fdr = fdr
        self.seed = seed
        self.iaaft_max_iter = iaaft_max_iter
        self.verbose = verbose
        self.min_rho = min_rho
        self.adaptive_rho = adaptive_rho
        self.adaptive_rho_quantile = adaptive_rho_quantile
        self.theiler_w = theiler_w
        self.E_method = E_method
        self.convergence_filter = convergence_filter
        self.convergence_threshold = convergence_threshold
        self.use_gpu = use_gpu

        # Warn if n_surrogates is too low for the chosen alpha
        import math
        min_n = math.ceil(1 / alpha) - 1
        if n_surrogates < min_n:
            import warnings
            warnings.warn(
                f"n_surrogates={n_surrogates} is too low for alpha={alpha}: "
                f"minimum p-value achievable is {1/(n_surrogates+1):.4f}. "
                f"Need at least n_surrogates>={min_n} for rejection to be possible.",
                stacklevel=2,
            )

        # Results
        self.ccm_matrix_ = None
        self.pvalue_matrix_ = None
        self.zscore_matrix_ = None
        self.detected_ = None
        self.params_ = None
        self.surrogate_distributions_ = None
        self.min_rho_matrix_ = None

    def fit(self, data):
        """Run the full SE-CCM pipeline.

        Parameters
        ----------
        data : ndarray, shape (T, N)
            Time series data, one column per node.

        Returns
        -------
        self
        """
        N = data.shape[1]
        n_pairs = N * (N - 1)

        # Resolve use_gpu
        _use_gpu = self.use_gpu
        if _use_gpu == "auto":
            _use_gpu = gpu_available()
        self.use_gpu_resolved_ = _use_gpu

        # Step 1: Select embedding parameters per node
        param_iter = range(N)
        if self.verbose:
            param_iter = tqdm(param_iter, desc="Selecting embedding params")
        params = [select_parameters(data[:, i], E_method=self.E_method)
                  for i in param_iter]
        self.params_ = params

        # Resolve theiler_w="auto" → median tau across nodes
        theiler_w = self.theiler_w
        if theiler_w == "auto":
            taus = [p[1] for p in params]
            theiler_w = int(np.median(taus))
        self.theiler_w_used_ = theiler_w

        # Step 2: Compute observed pairwise CCM
        ccm_matrix, _ = compute_pairwise_ccm(data, params_per_node=params,
                                             theiler_w=theiler_w)
        self.ccm_matrix_ = ccm_matrix

        # Step 2b: Convergence filtering (optional)
        convergence_matrix = np.full((N, N), np.nan)
        if self.convergence_filter:
            conv_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
            conv_iter = conv_pairs
            if self.verbose:
                conv_iter = tqdm(conv_pairs, desc="Convergence check")
            for i, j in conv_iter:
                E_i, tau_i = params[i]
                cs, _ = convergence_score(
                    data[:, i], data[:, j], E_i, tau_i,
                    n_points=10, theiler_w=theiler_w,
                    cross_validate=True, n_reps=3, seed=self.seed,
                )
                convergence_matrix[i, j] = cs
        self.convergence_matrix_ = convergence_matrix

        # Step 3: Surrogate testing
        pvalue_matrix = np.full((N, N), np.nan)
        zscore_matrix = np.full((N, N), np.nan)
        surrogate_dists = {}

        surr_kwargs = {}
        if self.surrogate_method == "iaaft":
            surr_kwargs["max_iter"] = self.iaaft_max_iter

        pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        pair_iter = pairs
        if self.verbose:
            pair_iter = tqdm(pairs, desc="Surrogate testing")

        # Pre-build KDTree per effect node to avoid redundant work
        from ..ccm.ccm_core import _find_neighbors_theiler
        node_trees = {}
        for i in range(N):
            E_i, tau_i = params[i]
            M_x = delay_embed(data[:, i], E_i, tau_i)
            T_eff = len(M_x)
            offset = (E_i - 1) * tau_i
            k = E_i + 1
            dists, idxs = _find_neighbors_theiler(M_x, k, theiler_w)
            eps_val = 1e-12
            w = np.exp(-dists / (dists[:, 0:1] + eps_val))
            w = w / (w.sum(axis=1, keepdims=True) + eps_val)
            node_trees[i] = (M_x, idxs, w, offset, T_eff)

        # Pre-generate surrogates per cause variable j (shared across
        # all effect nodes i), avoiding 9× redundant generation.
        surr_cache = {}
        surr_methods_used = {}
        cause_iter = range(N)
        if self.verbose:
            cause_iter = tqdm(cause_iter, desc="Generating surrogates")
        for j in cause_iter:
            seed_j = None
            if self.seed is not None:
                seed_j = self.seed + j

            # Adaptive method selection per cause variable
            method_j = self.surrogate_method
            if method_j == "auto":
                from ..surrogate.adaptive import select_surrogate_method
                method_j, _ = select_surrogate_method(data[:, j])
            surr_methods_used[j] = method_j

            kw = dict(surr_kwargs)
            if method_j == "iaaft" and "max_iter" not in kw:
                kw["max_iter"] = self.iaaft_max_iter

            surr_cache[j] = generate_surrogate(
                data[:, j],
                method=method_j,
                n_surrogates=self.n_surrogates,
                seed=seed_j,
                use_gpu=_use_gpu,
                **kw,
            )
        self.surrogate_methods_used_ = surr_methods_used

        for i, j in pair_iter:
            E_i, tau_i = params[i]
            rho_obs = ccm_matrix[i, j]
            M_x, idxs, w, offset, T_eff = node_trees[i]

            surrogates_j = surr_cache[j]

            # Compute CCM for each surrogate (reuse pre-built KDTree)
            rho_surr = _ccm_surrogate_batch(
                M_x, idxs, w,
                data[offset:offset + T_eff, j],
                surrogates_j, offset, T_eff,
                use_gpu=_use_gpu,
            )

            pvalue_matrix[i, j] = compute_pvalue(rho_obs, rho_surr)
            zscore_matrix[i, j] = compute_zscore(rho_obs, rho_surr)
            surrogate_dists[(i, j)] = rho_surr

        self.pvalue_matrix_ = pvalue_matrix
        self.zscore_matrix_ = zscore_matrix
        self.surrogate_distributions_ = surrogate_dists

        # Step 4: Determine significance
        # Extract off-diagonal p-values
        mask = ~np.eye(N, dtype=bool)
        pvals_off = pvalue_matrix[mask]

        if self.fdr:
            rejected, _ = fdr_correction(pvals_off, alpha=self.alpha)
        else:
            rejected = pvals_off <= self.alpha

        # Apply effect-size threshold
        rho_off = ccm_matrix[mask]
        if self.adaptive_rho:
            # Per-pair adaptive threshold: max(fixed_min_rho, q-th percentile
            # of surrogate ρ distribution). This prevents false positives when
            # surrogates already achieve high ρ (e.g., Rössler, FHN).
            min_rho_matrix = np.full((N, N), self.min_rho)
            for (i, j), rho_surr in surrogate_dists.items():
                surr_q = float(np.percentile(rho_surr, 100 * self.adaptive_rho_quantile))
                min_rho_matrix[i, j] = max(self.min_rho, surr_q)
            self.min_rho_matrix_ = min_rho_matrix
            rho_thresholds = min_rho_matrix[mask]
            rejected = rejected & (rho_off >= rho_thresholds)
        else:
            self.min_rho_matrix_ = np.full((N, N), self.min_rho)
            rejected = rejected & (rho_off >= self.min_rho)

        # Apply convergence filter: suppress pairs with non-converging CCM
        if self.convergence_filter:
            conv_off = convergence_matrix[mask]
            rejected = rejected & (conv_off > self.convergence_threshold)

        detected = np.zeros((N, N), dtype=int)
        detected[mask] = rejected.astype(int)
        self.detected_ = detected

        return self

    def score(self, adj_true):
        """Evaluate detection against ground truth.

        Returns metrics including AUC-ROC comparison between raw CCM rho
        and surrogate-enhanced 1-p scores.

        Parameters
        ----------
        adj_true : ndarray, shape (N, N)
            Ground truth adjacency matrix.

        Returns
        -------
        metrics : dict
            Detection performance metrics including:
            - TPR, FPR, precision, F1 (binary detection)
            - AUC_ROC_rho: AUC-ROC using raw CCM ρ as score
            - AUC_ROC_surrogate: AUC-ROC using 1-p as score
            - AUC_ROC_delta: improvement (surrogate - rho)
        """
        from ..evaluation.metrics import evaluate_detection

        if self.detected_ is None:
            raise RuntimeError("Call fit() first.")

        # Binary detection metrics
        metrics = evaluate_detection(self.detected_, adj_true)

        # AUC-ROC comparison: raw CCM ρ vs surrogate-enhanced 1-p
        N = adj_true.shape[0]
        mask = ~np.eye(N, dtype=bool)
        y_true = adj_true[mask].ravel().astype(int)

        if len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_auc_score, average_precision_score

            # Raw CCM: use ρ as score
            rho_scores = self.ccm_matrix_[mask].ravel()
            metrics["AUC_ROC_rho"] = roc_auc_score(y_true, rho_scores)
            metrics["AUC_PR_rho"] = average_precision_score(y_true, rho_scores)

            # Surrogate-enhanced: use 1-p as score (rank-based)
            p_scores = 1.0 - self.pvalue_matrix_[mask].ravel()
            metrics["AUC_ROC_surrogate"] = roc_auc_score(y_true, p_scores)

            # Surrogate-enhanced: use z-score (continuous, finer resolution)
            z_scores = self.zscore_matrix_[mask].ravel()
            z_scores = np.nan_to_num(z_scores, nan=0.0)
            metrics["AUC_ROC_zscore"] = roc_auc_score(y_true, z_scores)
            metrics["AUC_PR_zscore"] = average_precision_score(y_true, z_scores)

            metrics["AUC_ROC_delta"] = (
                metrics["AUC_ROC_surrogate"] - metrics["AUC_ROC_rho"]
            )
            metrics["AUC_ROC_delta_zscore"] = (
                metrics["AUC_ROC_zscore"] - metrics["AUC_ROC_rho"]
            )
        else:
            metrics["AUC_ROC_rho"] = np.nan
            metrics["AUC_ROC_surrogate"] = np.nan
            metrics["AUC_ROC_zscore"] = np.nan
            metrics["AUC_ROC_delta"] = np.nan
            metrics["AUC_ROC_delta_zscore"] = np.nan
            metrics["AUC_PR_rho"] = np.nan
            metrics["AUC_PR_zscore"] = np.nan

        return metrics
