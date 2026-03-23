"""Detection performance metrics."""

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_detection(detected, ground_truth, scores=None):
    """Evaluate causal detection performance.

    Parameters
    ----------
    detected : ndarray, shape (N, N)
        Binary detected adjacency matrix.
    ground_truth : ndarray, shape (N, N)
        Binary ground truth adjacency matrix.
    scores : ndarray, shape (N, N), optional
        Continuous scores (e.g., CCM rho or 1-pvalue) for AUC computation.

    Returns
    -------
    metrics : dict
        TPR, FPR, precision, recall, F1, AUC-ROC, AUC-PR.
    """
    N = detected.shape[0]
    mask = ~np.eye(N, dtype=bool)

    y_true = ground_truth[mask].ravel().astype(int)
    y_pred = detected[mask].ravel().astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    result = {
        "TPR": tpr,
        "FPR": fpr,
        "precision": precision,
        "recall": tpr,
        "F1": f1,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }

    # AUC metrics require continuous scores
    if scores is not None:
        y_scores = scores[mask].ravel()
        if len(np.unique(y_true)) > 1:
            result["AUC_ROC"] = roc_auc_score(y_true, y_scores)
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
            result["AUC_PR"] = auc(rec_curve, prec_curve)
        else:
            result["AUC_ROC"] = np.nan
            result["AUC_PR"] = np.nan
    else:
        result["AUC_ROC"] = np.nan
        result["AUC_PR"] = np.nan

    return result


def compute_cohens_d(rho_obs, rho_surr):
    """Compute Cohen's d effect size.

    Parameters
    ----------
    rho_obs : float
        Observed CCM correlation.
    rho_surr : ndarray
        Surrogate CCM correlations.

    Returns
    -------
    d : float
        Cohen's d effect size.
    """
    mu = np.mean(rho_surr)
    sigma = np.std(rho_surr)
    if sigma < 1e-12:
        return 0.0
    return (rho_obs - mu) / sigma


def compute_delta_rho(rho_obs, rho_surr):
    """Compute delta rho (observed - mean surrogate).

    Parameters
    ----------
    rho_obs : float
        Observed CCM correlation.
    rho_surr : ndarray
        Surrogate CCM correlations.

    Returns
    -------
    delta : float
    """
    return rho_obs - np.mean(rho_surr)


def compute_sso(x, surrogates):
    """Compute Surrogate Spectral Overlap (SSO) via Jensen-Shannon divergence.

    SSO quantifies how much a surrogate method preserves the spectral
    structure of the original signal. Low SSO (→ 0) means the surrogate
    spectrum closely matches the original, implying the surrogate cannot
    disrupt the attractor geometry — problematic for oscillatory systems.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Original time series.
    surrogates : ndarray, shape (n_surrogates, T)
        Array of surrogate time series.

    Returns
    -------
    sso : float
        Jensen-Shannon divergence between normalized PSDs of x and
        the surrogate ensemble mean PSD. In [0, ln(2)] ≈ [0, 0.693].
    """
    x = np.asarray(x, dtype=float).ravel()

    # Compute PSD of original
    psd_x = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    psd_x = psd_x[1:]  # exclude DC

    # Compute mean PSD across surrogates
    n_surr = surrogates.shape[0]
    psd_surr_sum = np.zeros_like(psd_x)
    for i in range(n_surr):
        s = surrogates[i]
        psd_s = np.abs(np.fft.rfft(s - np.mean(s))) ** 2
        psd_surr_sum += psd_s[1:]
    psd_surr_mean = psd_surr_sum / n_surr

    # Normalize to probability distributions
    eps = 1e-30  # avoid log(0)
    p = psd_x / (psd_x.sum() + eps) + eps
    q = psd_surr_mean / (psd_surr_mean.sum() + eps) + eps

    # Renormalize after adding eps
    p = p / p.sum()
    q = q / q.sum()

    # Jensen-Shannon divergence
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

    return float(jsd)
