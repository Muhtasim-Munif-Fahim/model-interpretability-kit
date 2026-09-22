"""Global feature importance: permutation, drop-column, and LOCO.

These methods are model-agnostic. Permutation importance only needs a
``predict`` callable. Drop-column and leave-one-covariate-out (LOCO)
importance refit through a ``fit_predict`` callable and measure how the
model degrades when a feature is removed. Permutation and drop-column score
a higher-is-better metric; LOCO scores a lower-is-better loss on held-out
rows.
"""

import numpy as np

from ._utils import as_1d, as_2d

__all__ = [
    "r2_score",
    "mean_absolute_error",
    "mean_squared_error",
    "zero_one_loss",
    "permutation_importance",
    "drop_column_importance",
    "loco_importance",
]


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


def _aligned_vectors(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    return y_true, y_pred


def mean_absolute_error(y_true, y_pred):
    """Mean absolute error. Lower is better.

    This is the default LOCO loss, matching the absolute residual used by
    leave-one-covariate-out inference.
    """
    y_true, y_pred = _aligned_vectors(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true, y_pred):
    """Mean squared error. Lower is better."""
    y_true, y_pred = _aligned_vectors(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def zero_one_loss(y_true, y_pred):
    """Misclassification rate. Lower is better.

    ``y_true`` and ``y_pred`` are compared for equality. Use numeric class
    codes such as 0 and 1: LOCO stores targets as floats, and a non-numeric
    prediction is rejected. The LOCO importance under this loss is the drop
    in accuracy on the held-out rows.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    return float(np.mean(y_true != y_pred))


def _loss_or_default(loss):
    if loss is None:
        return mean_absolute_error
    if not callable(loss):
        raise TypeError("loss must be a callable loss(y_true, y_pred) -> float")
    return loss


def _scalar_loss(loss, y_true, y_pred):
    value = loss(y_true, y_pred)
    arr = np.asarray(value, dtype=float)
    if arr.shape != () and arr.size != 1:
        raise ValueError(
            "loss must return a single float (reduce per-row losses with a mean)"
        )
    value = float(arr)
    if not np.isfinite(value):
        raise ValueError("loss must return a finite float")
    return value


def _as_predictions(y_pred, n_rows):
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_pred.shape[0] != n_rows:
        raise ValueError("fit_predict must return one prediction per evaluation row")
    return y_pred


def _split_indices(n_rows, test_size, rng):
    if n_rows < 2:
        raise ValueError("LOCO needs at least two rows to form a train and test split")
    n_test = int(round(n_rows * float(test_size)))
    n_test = min(max(n_test, 1), n_rows - 1)
    order = rng.permutation(n_rows)
    return order[n_test:], order[:n_test]


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


def _loco_one_split(fit_predict, loss, X_train, y_train, X_test, y_test):
    """Increase in held-out loss when each covariate is left out of the fit."""
    pred_full = _as_predictions(fit_predict(X_train, y_train, X_test), y_test.shape[0])
    baseline = _scalar_loss(loss, y_test, pred_full)
    n_features = X_train.shape[1]
    importance = np.empty(n_features)
    for j in range(n_features):
        train_reduced = np.delete(X_train, j, axis=1)
        test_reduced = np.delete(X_test, j, axis=1)
        pred_reduced = _as_predictions(
            fit_predict(train_reduced, y_train, test_reduced), y_test.shape[0]
        )
        importance[j] = _scalar_loss(loss, y_test, pred_reduced) - baseline
    return importance, baseline


def loco_importance(
    fit_predict,
    X,
    y,
    X_test=None,
    y_test=None,
    loss=None,
    n_repeats=1,
    test_size=0.25,
    seed=None,
):
    """Leave-one-covariate-out (LOCO) importance on held-out rows.

    For each feature the model is refit without that column. Importance is
    the increase in loss on rows the refit did not train on, so a positive
    value means the feature improves out-of-sample predictions. This follows
    the LOCO construction in Lei, G'Sell, Rinaldo, Tibshirani, and Wasserman,
    "Distribution-Free Predictive Inference for Regression" (JASA, 2018):
    the excess prediction error of the model that never saw covariate ``j``.

    Permutation importance is already provided by :func:`permutation_importance`.
    LOCO is the refit-based alternative: it changes the fitted model instead
    of shuffling a column of a frozen predictor, and it scores a
    lower-is-better loss rather than a higher-is-better metric.

    Parameters
    ----------
    fit_predict : callable
        ``fit_predict(X_fit, y_fit, X_eval) -> y_pred``. Called once for the
        full feature set and once per left-out feature, on every repeat.
    X : ndarray
        Training features when ``X_test`` is given. Otherwise the full sample
        that each repeat splits into train and test. A 1-D array of shape
        ``(n,)`` is treated as ``n`` rows of a single feature.
    y : ndarray
        Targets aligned with ``X``.
    X_test, y_test : ndarray or None
        Held-out evaluation data. Pass both, or neither. When both are
        omitted, each repeat draws a fresh split that holds out
        ``test_size`` of the rows. When both are given and ``n_repeats`` is
        1, the model is fit once on ``X, y``. When both are given and
        ``n_repeats`` is greater than 1, each repeat bootstraps the training
        rows (same size, with replacement) and scores the fixed test set.
    loss : callable or None
        ``loss(y_true, y_pred) -> float``, lower is better. Defaults to
        :func:`mean_absolute_error`. Pass :func:`zero_one_loss` to measure
        the drop in classification accuracy, or any other scalar scorer
        such as :func:`mean_squared_error`.
    n_repeats : int
        Number of splits (or training bootstraps). The reported standard
        deviation is the spread of the importance across these repeats.
        With one repeat the standard deviation is 0.
    test_size : float
        Fraction of rows held out on each random split. Used only when
        ``X_test`` is omitted. Must lie in ``(0, 1)``.
    seed : int or None
        Seed for the splits or training bootstraps.

    Returns
    -------
    dict
        ``{"mean": ndarray (n_features,), "std": ndarray (n_features,),
        "baseline": float, "n_repeats": int}``. ``mean`` is the average
        increase in held-out loss. ``std`` is the NumPy population standard
        deviation (``ddof=0``) of that increase across repeats. ``baseline``
        is the average held-out loss of the full model (lower is better).
    """
    if not callable(fit_predict):
        raise TypeError("fit_predict must be a callable fit_predict(X_fit, y_fit, X_eval)")
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, (int, np.integer)) or n_repeats < 1:
        raise ValueError("n_repeats must be an integer >= 1")
    n_repeats = int(n_repeats)
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test and y_test must be provided together")
    loss = _loss_or_default(loss)
    X = as_2d(X)
    y = as_1d(y, X.shape[0])
    if X.shape[1] < 1:
        raise ValueError("X must contain at least one feature")
    explicit_test = X_test is not None
    if explicit_test:
        X_test = as_2d(X_test, name="X_test")
        y_test = as_1d(y_test, X_test.shape[0], name="y_test")
        if X_test.shape[1] != X.shape[1]:
            raise ValueError("X_test must have the same number of features as X")
        if X.shape[0] < 1 or X_test.shape[0] < 1:
            raise ValueError("training and test splits must each contain at least one row")
    else:
        if not np.isfinite(test_size) or not 0.0 < float(test_size) < 1.0:
            raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(seed)
    importances = np.empty((n_repeats, X.shape[1]))
    baselines = np.empty(n_repeats)
    for repeat in range(n_repeats):
        if explicit_test:
            if n_repeats == 1:
                X_train, y_train = X, y
            else:
                boot = rng.integers(0, X.shape[0], size=X.shape[0])
                X_train, y_train = X[boot], y[boot]
            X_eval, y_eval = X_test, y_test
        else:
            train_idx, test_idx = _split_indices(X.shape[0], test_size, rng)
            X_train, y_train = X[train_idx], y[train_idx]
            X_eval, y_eval = X[test_idx], y[test_idx]
        importances[repeat], baselines[repeat] = _loco_one_split(
            fit_predict, loss, X_train, y_train, X_eval, y_eval
        )
    return {
        "mean": importances.mean(axis=0),
        "std": importances.std(axis=0),
        "baseline": float(baselines.mean()),
        "n_repeats": n_repeats,
    }
