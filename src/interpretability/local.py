"""Local explanations: LIME-style surrogates and tree-based attributions.

``lime_explain`` fits a weighted linear surrogate over a random neighborhood
of the instance to be explained, so the explanation is interpretable at the
cost of locality.

For the small regression trees produced by ``interpretability.demo_model``,
``tree_shap_values`` computes exact interventional SHAP values: Shapley
values of the coalition value function ``v(S) = E[f(X) | X_S = x_S]`` where
unfixed features are marginalized by flowing a background dataset through the
tree. ``local_occlusion_attribution`` is a cheaper fallback for arbitrary
predictors.
"""

from itertools import combinations
from math import factorial

import numpy as np

from ._utils import as_2d

__all__ = [
    "weighted_least_squares",
    "sample_neighborhood",
    "lime_explain",
    "tree_conditional_expectation",
    "tree_shap_values",
    "local_occlusion_attribution",
]


def weighted_least_squares(design, target, weights, l2=1e-6):
    """Solve ``argmin_b sum(w * (target - design @ b)^2)``.

    A small ridge term ``l2`` keeps the normal equations well conditioned for
    collinear or constant columns. ``design`` should include an intercept
    column when one is wanted.

    Returns
    -------
    ndarray
        Coefficient vector of shape ``(design.shape[1],)``.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if design.ndim != 2:
        raise ValueError("design must be a 2-D matrix")
    if not (design.shape[0] == target.shape[0] == weights.shape[0]):
        raise ValueError("design, target and weights must have the same number of rows")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    weighted_design = weights[:, None] * design
    gram = design.T @ weighted_design
    rhs = weighted_design.T @ target
    if l2 > 0:
        gram = gram + l2 * np.eye(design.shape[1])
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def sample_neighborhood(
    x_row, background, n_samples=500, kernel_width=0.75, sigma_scale=0.1, seed=None
):
    """Draw weighted perturbations around ``x_row`` for a LIME surrogate.

    Perturbations follow a Gaussian centered at ``x_row`` with per-feature
    scale proportional to the feature's standard deviation. Distances are
    Euclidean and are mapped to weights with an exponential kernel, so rows
    far from the instance contribute little to the surrogate fit. The
    instance itself is appended with weight 1.

    Returns
    -------
    Z : ndarray of shape (n_samples + 1, n_features)
    weights : ndarray of shape (n_samples + 1,)
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background)
    if x_row.shape[0] != background.shape[1]:
        raise ValueError("x_row and background must have the same number of features")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    rng = np.random.default_rng(seed)
    stds = background.std(axis=0) * sigma_scale
    stds = np.where(stds <= 0, 0.1 * sigma_scale, stds)
    Z = rng.normal(loc=x_row, scale=stds, size=(n_samples, background.shape[1]))
    Z = np.vstack([Z, x_row[None, :]])
    distances = np.linalg.norm(Z - x_row, axis=1) / np.sqrt(background.shape[1])
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
    return Z, weights


def _weighted_r2(y, y_hat, weights):
    w_mean = np.sum(weights * y) / np.sum(weights)
    ss_res = np.sum(weights * (y - y_hat) ** 2)
    ss_tot = np.sum(weights * (y - w_mean) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def lime_explain(
    predict,
    x_row,
    background,
    n_samples=500,
    kernel_width=0.75,
    sigma_scale=0.1,
    seed=None,
    feature_names=None,
):
    """Explain one prediction with a locally weighted linear surrogate.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    x_row : sequence of float
        Instance to explain.
    background : ndarray
        Reference dataset used to set neighborhood scales.
    n_samples : int
        Number of perturbed samples around the instance.
    kernel_width : float
        Bandwidth of the exponential distance kernel.
    sigma_scale : float
        Perturbation scale as a fraction of each feature's standard deviation.
    seed : int or None
        Random seed for the perturbations.
    feature_names : list of str or None

    Returns
    -------
    dict
        ``{"coefficients": ndarray, "intercept": float, "weighted_r2": float,
        "prediction": float, "n_samples": int, "feature_names": list}``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background)
    Z, weights = sample_neighborhood(
        x_row, background, n_samples=n_samples, kernel_width=kernel_width,
        sigma_scale=sigma_scale, seed=seed,
    )
    y_perturbed = np.asarray(predict(Z), dtype=float).ravel()
    design = np.column_stack([np.ones(Z.shape[0]), Z])
    coefs = weighted_least_squares(design, y_perturbed, weights)
    intercept = float(coefs[0])
    coefficients = coefs[1:]
    fitted = design @ coefs
    prediction = float(predict(x_row[None, :])[0])
    if feature_names is None:
        feature_names = ["X%d" % j for j in range(background.shape[1])]
    return {
        "coefficients": coefficients,
        "intercept": intercept,
        "weighted_r2": _weighted_r2(y_perturbed, fitted, weights),
        "prediction": prediction,
        "n_samples": Z.shape[0],
        "feature_names": list(feature_names),
    }


def tree_conditional_expectation(node, x_row, fixed_features, background):
    """Conditional expectation ``E[f(X) | X_fixed = x_fixed]`` for a tree.

    Background rows are flowed through the tree; at a split on a fixed
    feature the instance's branch is taken, otherwise the child expectations
    are averaged by the fraction of the background that would flow each way.

    Parameters
    ----------
    node : dict
        Tree node dict as stored in ``DecisionTreeRegressor.root_``.
    x_row : sequence of float
        Instance whose values fix the ``fixed_features`` columns.
    fixed_features : iterable of int
        Features held at ``x_row`` values.
    background : ndarray
        Reference dataset for marginalization.
    """
    if not isinstance(node, dict) or "leaf" not in node:
        raise TypeError("node must be a tree node dict with a 'leaf' key")
    if node["leaf"]:
        return float(node["value"])
    feature = node["feature"]
    if feature in fixed_features:
        child = node["left"] if x_row[feature] <= node["threshold"] else node["right"]
        return tree_conditional_expectation(child, x_row, fixed_features, background)
    n_left = int(np.sum(background[:, feature] <= node["threshold"]))
    n_total = background.shape[0]
    if n_left == 0:
        return tree_conditional_expectation(node["right"], x_row, fixed_features, background)
    if n_left == n_total:
        return tree_conditional_expectation(node["left"], x_row, fixed_features, background)
    p_left = n_left / n_total
    expected_left = tree_conditional_expectation(node["left"], x_row, fixed_features, background)
    expected_right = tree_conditional_expectation(node["right"], x_row, fixed_features, background)
    return p_left * expected_left + (1.0 - p_left) * expected_right


def tree_shap_values(node, x_row, background, feature_count):
    """Exact interventional SHAP values for a single regression tree.

    Evaluates the Shapley values of ``v(S) = E[f(X) | X_S = x_S]`` over all
    feature coalitions, which for one tree is the interventional TreeSHAP
    decomposition. Exponential in the feature count, so it is intended for
    small trees (a handful of features).

    Parameters
    ----------
    node : dict
        Root node of the fitted tree.
    x_row : sequence of float
        Instance to explain.
    background : ndarray
        Reference dataset for marginalization.
    feature_count : int
        Total number of features in the model.

    Returns
    -------
    dict
        ``{"values": ndarray of shape (feature_count,), "baseline": float}``.
        The ``values`` sum to ``predict(x_row) - baseline``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background)
    feature_count = int(feature_count)
    if feature_count < 1:
        raise ValueError("feature_count must be at least 1")
    if x_row.shape[0] != feature_count:
        raise ValueError("x_row must have feature_count entries")
    if background.shape[1] != feature_count:
        raise ValueError("background must have feature_count columns")
    values = np.zeros(feature_count)
    for j in range(feature_count):
        for size in range(feature_count):
            weight = (
                factorial(size)
                * factorial(feature_count - size - 1)
                / factorial(feature_count)
            )
            for subset in combinations(range(feature_count), size):
                if j in subset:
                    continue
                with_j = tuple(sorted(subset + (j,)))
                values[j] += weight * (
                    tree_conditional_expectation(node, x_row, with_j, background)
                    - tree_conditional_expectation(node, x_row, subset, background)
                )
    baseline = tree_conditional_expectation(node, x_row, (), background)
    return {"values": values, "baseline": baseline}


def local_occlusion_attribution(predict, x_row, background, n_samples=200, seed=None):
    """Occlusion-style local attribution for arbitrary predictors.

    For each feature the background rows are redrawn with that column pinned
    to the instance's value; the drop in average prediction relative to the
    instance is the feature's attribution. This is a Monte Carlo fallback for
    models without a tree structure and is not an exact Shapley decomposition.

    Returns
    -------
    dict
        ``{"values": ndarray, "baseline": float, "prediction": float}``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background)
    if x_row.shape[0] != background.shape[1]:
        raise ValueError("x_row and background must have the same number of features")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    rng = np.random.default_rng(seed)
    draws = background[rng.integers(0, background.shape[0], size=n_samples)].copy()
    prediction = float(predict(x_row[None, :])[0])
    values = np.zeros(background.shape[1])
    for j in range(background.shape[1]):
        pinned = draws.copy()
        pinned[:, j] = x_row[j]
        values[j] = prediction - float(np.mean(predict(pinned)))
    baseline = float(np.mean(predict(draws)))
    return {"values": values, "baseline": baseline, "prediction": prediction}
