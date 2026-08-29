"""Global feature importance: permutation and drop-column.

Both methods are model-agnostic: they only need a ``predict`` callable (or a
``fit_predict`` callable for the refit-based variant) and evaluate how a
model's score degrades when a feature is perturbed or removed.
"""

import numpy as np

from ._utils import as_1d, as_2d

__all__ = ["r2_score", "permutation_importance", "drop_column_importance"]


def r2_score(y_true, y_pred):
    """Coefficient of determination between two vectors."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _metric_or_default(metric):
    if metric is None:
        return r2_score
    if not callable(metric):
        raise TypeError("metric must be a callable metric(y_true, y_pred) -> float")
    return metric


def permutation_importance(predict, X, y, metric=None, n_repeats=5, seed=None):
    """Estimate feature importance by permuting each column and scoring the drop.

    For every feature the column is shuffled ``n_repeats`` times (with an
    optional ``seed``) and the model is re-scored. Importance is the
    difference ``baseline - score_after_permutation`` averaged over repeats,
    so a positive value means the feature carries predictive signal.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` where ``X`` is a 2-D array.
    X : ndarray
        Feature matrix. A 1-D input of shape ``(n,)`` is treated as ``n``
        rows of a single feature.
    y : ndarray
        Target vector.
    metric : callable or None
        ``metric(y_true, y_pred) -> float``, higher is better.
        Defaults to ``r2_score``.
    n_repeats : int
        Number of permutations per feature.
    seed : int or None
        Random seed controlling the permutations.

    Returns
    -------
    dict
        ``{"mean": ndarray (n_features,), "std": ndarray (n_features,),
        "baseline": float, "n_repeats": int}``.
    """
    X = as_2d(X)
    y = as_1d(y, X.shape[0])
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    metric = _metric_or_default(metric)
    baseline = float(metric(y, predict(X)))
    n_features = X.shape[1]
    rng = np.random.default_rng(seed)
    scores = np.empty((n_features, n_repeats))
    for j in range(n_features):
        col = X[:, j].copy()
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(col)
            scores[j, r] = float(metric(y, predict(X_perm)))
    return {
        "mean": baseline - scores.mean(axis=1),
        "std": scores.std(axis=1),
        "baseline": baseline,
        "n_repeats": n_repeats,
    }


def drop_column_importance(fit_predict, X, y, metric=None):
    """Estimate importance by refitting the model without each feature.

    ``fit_predict`` must be a callable with signature
    ``fit_predict(X_fit, y_fit, X_eval) -> y_pred``. For every feature the
    model is retrained on the reduced matrix, so this reflects genuine loss
    of information rather than shuffle noise, at the cost of one fit per
    feature.

    Parameters
    ----------
    fit_predict : callable
        ``fit_predict(X_fit, y_fit, X_eval) -> y_pred``.
    X : ndarray
        Feature matrix (1-D treated as a single feature).
    y : ndarray
        Target vector.
    metric : callable or None
        Higher-is-better scoring function; defaults to ``r2_score``.

    Returns
    -------
    dict
        ``{"importance": ndarray (n_features,), "baseline": float,
        "scores": ndarray (n_features,)}``.
    """
    X = as_2d(X)
    y = as_1d(y, X.shape[0])
    if not callable(fit_predict):
        raise TypeError("fit_predict must be a callable fit_predict(X_fit, y_fit, X_eval)")
    metric = _metric_or_default(metric)
    baseline = float(metric(y, fit_predict(X, y, X)))
    n_features = X.shape[1]
    scores = np.empty(n_features)
    for j in range(n_features):
        cols = [c for c in range(n_features) if c != j]
        X_reduced = X[:, cols]
        scores[j] = float(metric(y, fit_predict(X_reduced, y, X_reduced)))
    return {
        "importance": baseline - scores,
        "baseline": baseline,
        "scores": scores,
    }
