"""Partial dependence (1-D and 2-D) and individual conditional expectation.

Partial dependence of feature ``f`` at value ``v`` averages the model's
prediction over all rows with ``X[:, f]`` set to ``v``. ICE curves instead
keep every row fixed and plot its prediction as ``X[:, f]`` varies, exposing
heterogeneous model behavior that the averaged curve hides. Centered ICE
(c-ICE) subtracts each curve's value at the first grid point so level
differences disappear and slope heterogeneity is easier to see. Optional
1-D PDP confidence bands are a normal approximation for that row-averaged
mean, not uncertainty in the fitted model itself.
"""

import math

import numpy as np

from ._utils import as_2d, check_feature_index, check_rows

__all__ = [
    "make_grid",
    "partial_dependence",
    "partial_dependence_2d",
    "ice_curves",
    "ice",
    "pdp",
]


def make_grid(X_column, grid_points=20, grid=None):
    """Build the evaluation grid for one feature's PDP/ICE curve.

    The default grid spans the 5th to 95th percentiles of the feature so
    outliers do not stretch the curve. A constant feature is given a tiny
    spread around its value. A caller-provided ``grid`` is used verbatim.
    """
    col = np.asarray(X_column, dtype=float)
    if grid is not None:
        grid = np.asarray(grid, dtype=float).ravel()
        if grid.size == 0:
            raise ValueError("grid must not be empty")
        return grid
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    lo, hi = np.quantile(col, [0.05, 0.95])
    if lo == hi:
        lo, hi = float(col.min()), float(col.max())
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
    return np.linspace(lo, hi, grid_points)


def _normal_ppf(p):
    """Inverse CDF of the standard normal via Newton iteration on ``math.erf``.

    Solves ``0.5 * (1 + erf(z / sqrt(2))) = p``. A logistic starting guess
    keeps the iteration in the quadratic basin for typical two-sided
    confidence levels (0.8--0.999).
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    target = 2.0 * p - 1.0
    z = 0.6 * math.log(p / (1.0 - p))
    sqrt2 = math.sqrt(2.0)
    two_over_sqrt_pi = math.sqrt(2.0 / math.pi)
    for _ in range(20):
        err = math.erf(z / sqrt2) - target
        deriv = two_over_sqrt_pi * math.exp(-0.5 * z * z)
        if deriv == 0.0:
            break
        step = err / deriv
        z -= step
        if abs(step) < 1e-12:
            break
    return z


def _predict_on_grid(predict, X, feature_index, grid):
    """Evaluate ``predict`` at each grid value of one feature, other columns fixed."""
    X_work = np.asarray(X, dtype=float).copy()
    preds = np.empty((X_work.shape[0], grid.size), dtype=float)
    for i, value in enumerate(grid):
        X_work[:, feature_index] = value
        preds[:, i] = np.asarray(predict(X_work), dtype=float).ravel()
    return preds


def partial_dependence(predict, X, feature_index, grid=None, grid_points=20, conf_level=None):
    """Average model prediction as one feature varies over a grid.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    X : ndarray
        Feature matrix the curve is averaged over.
    feature_index : int
        Column to vary.
    grid : ndarray or None
        Explicit evaluation points; defaults to percentile-based points.
    grid_points : int
        Number of default grid points.
    conf_level : float or None
        If set to a value in ``(0, 1)``, also return a normal-approximation
        confidence band for the mean curve: ``lower`` / ``upper`` are
        ``values ± z * std / sqrt(n)`` where ``std`` is the sample standard
        deviation of predictions across rows at each grid point and ``z`` is
        the two-sided normal quantile for ``conf_level``. The band captures
        sampling variability of the PDP average, not uncertainty in the
        fitted model.

    Returns
    -------
    dict
        ``{"feature": int, "grid": ndarray, "values": ndarray}``. When
        ``conf_level`` is set, also ``"std"``, ``"lower"``, ``"upper"``,
        and ``"conf_level"``.
    """
    X = as_2d(X)
    f = check_feature_index(feature_index, X.shape[1])
    grid = make_grid(X[:, f], grid_points, grid)
    preds = _predict_on_grid(predict, X, f, grid)
    values = preds.mean(axis=0)
    result = {"feature": f, "grid": grid, "values": values}
    if conf_level is None:
        return result
    conf_level = float(conf_level)
    if not (0.0 < conf_level < 1.0):
        raise ValueError("conf_level must be in (0, 1)")
    n = X.shape[0]
    if n < 2:
        std = np.zeros(grid.size, dtype=float)
    else:
        std = preds.std(axis=0, ddof=1)
    se = std / math.sqrt(max(n, 1))
    half = _normal_ppf(0.5 + 0.5 * conf_level) * se
    result["std"] = std
    result["lower"] = values - half
    result["upper"] = values + half
    result["conf_level"] = conf_level
    return result


def partial_dependence_2d(predict, X, feature_indices, grids=None, grid_points=10):
    """Average model prediction as two features vary over a joint grid.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    X : ndarray
        Feature matrix.
    feature_indices : sequence of int
        Exactly two columns ``(f0, f1)``.
    grids : tuple of ndarray or None
        Explicit grids for each feature; defaults to percentile grids.
    grid_points : int
        Number of default grid points per feature.

    Returns
    -------
    dict
        ``{"feature0": int, "feature1": int, "grid0": ndarray,
        "grid1": ndarray, "values": ndarray of shape (len(grid0),
        len(grid1))}`` where ``values[i, j]`` is the mean prediction with
        feature0 = ``grid0[i]`` and feature1 = ``grid1[j]``.
    """
    X = as_2d(X)
    f0, f1 = feature_indices
    f0 = check_feature_index(f0, X.shape[1])
    f1 = check_feature_index(f1, X.shape[1])
    if f0 == f1:
        raise ValueError("the two feature indices must differ")
    if grids is None:
        grid0 = make_grid(X[:, f0], grid_points)
        grid1 = make_grid(X[:, f1], grid_points)
    else:
        grid0 = make_grid(X[:, f0], grid_points, grids[0])
        grid1 = make_grid(X[:, f1], grid_points, grids[1])
    X_work = X.copy()
    values = np.empty((grid0.size, grid1.size))
    for i, v0 in enumerate(grid0):
        X_work[:, f0] = v0
        for j, v1 in enumerate(grid1):
            X_work[:, f1] = v1
            values[i, j] = float(np.mean(predict(X_work)))
    return {
        "feature0": f0,
        "feature1": f1,
        "grid0": grid0,
        "grid1": grid1,
        "values": values,
    }


def ice_curves(
    predict,
    X,
    feature_index,
    grid=None,
    grid_points=20,
    rows=None,
    max_rows=25,
    centered=False,
):
    """Per-row predictions as one feature varies over a grid.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    X : ndarray
        Feature matrix.
    feature_index : int
        Column to vary.
    grid : ndarray or None
        Explicit evaluation points.
    grid_points : int
        Number of default grid points.
    rows : sequence of int or None
        Row indices to trace; defaults to the first ``max_rows`` rows.
    max_rows : int
        Cap on the default row selection.
    centered : bool
        If True, subtract each curve's value at the first grid point
        (c-ICE). All curves then start at 0, which removes level
        differences so slope heterogeneity is easier to see.

    Returns
    -------
    dict
        ``{"feature": int, "grid": ndarray, "rows": ndarray,
        "curves": ndarray of shape (len(rows), len(grid)),
        "centered": bool}``.
    """
    X = as_2d(X)
    f = check_feature_index(feature_index, X.shape[1])
    grid = make_grid(X[:, f], grid_points, grid)
    idx = check_rows(rows, X.shape[0], max_rows)
    curves = _predict_on_grid(predict, X[idx], f, grid)
    if centered:
        curves = curves - curves[:, :1]
    return {
        "feature": f,
        "grid": grid,
        "rows": idx,
        "curves": curves,
        "centered": bool(centered),
    }


ice = ice_curves
pdp = partial_dependence
