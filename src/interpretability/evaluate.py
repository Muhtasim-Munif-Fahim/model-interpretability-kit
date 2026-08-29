"""Faithfulness checks for local explanations.

A local explanation is only trustworthy if the surrogate it is built from
actually approximates the model near the instance (weighted-linear fidelity)
and if the features it highlights are consistent with the model's global
behavior (overlap with global importance).
"""

import numpy as np

from ._utils import as_2d
from .local import _weighted_r2, sample_neighborhood, weighted_least_squares

__all__ = ["weighted_linear_fidelity", "top_feature_overlap"]


def weighted_linear_fidelity(
    predict,
    x_row,
    background,
    n_samples=500,
    kernel_width=0.75,
    sigma_scale=0.1,
    seed=None,
):
    """Weighted R2 of a linear surrogate on the instance's neighborhood.

    Draws the same Gaussian neighborhood used by LIME, fits a weighted linear
    surrogate, and returns the weighted coefficient of determination. A value
    near 1 means the model is locally linear around ``x_row``; low values warn
    that the LIME-style explanation is a poor fit there.

    Returns
    -------
    float
        Weighted R2 in ``[0, 1]``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background)
    Z, weights = sample_neighborhood(
        x_row,
        background,
        n_samples=n_samples,
        kernel_width=kernel_width,
        sigma_scale=sigma_scale,
        seed=seed,
    )
    y_perturbed = np.asarray(predict(Z), dtype=float).ravel()
    design = np.column_stack([np.ones(Z.shape[0]), Z])
    coefs = weighted_least_squares(design, y_perturbed, weights)
    fitted = design @ coefs
    return float(_weighted_r2(y_perturbed, fitted, weights))


def top_feature_overlap(local_weights, global_importance, k=3):
    """Agreement between the top-k local and top-k global features.

    The local top-k uses the absolute value of the surrogate coefficients
    (the LIME convention); the global top-k uses the magnitude of the
    importance scores.

    Parameters
    ----------
    local_weights : ndarray
        Surrogate coefficients, one per feature.
    global_importance : ndarray
        Global importance scores, one per feature.
    k : int
        Number of top features to compare.

    Returns
    -------
    dict
        ``{"local_top": list, "global_top": list, "overlap": int,
        "jaccard": float}``.
    """
    local_weights = np.asarray(local_weights, dtype=float)
    global_importance = np.asarray(global_importance, dtype=float)
    if local_weights.shape != global_importance.shape:
        raise ValueError("local_weights and global_importance must have the same shape")
    if k < 1:
        raise ValueError("k must be at least 1")
    n_features = local_weights.shape[0]
    k = min(k, n_features)
    local_top = list(np.argsort(-np.abs(local_weights))[:k])
    global_top = list(np.argsort(-global_importance)[:k])
    local_set = set(local_top)
    global_set = set(global_top)
    union = len(local_set | global_set)
    overlap = len(local_set & global_set)
    jaccard = overlap / union if union else 1.0
    return {
        "local_top": local_top,
        "global_top": global_top,
        "overlap": overlap,
        "jaccard": jaccard,
    }
