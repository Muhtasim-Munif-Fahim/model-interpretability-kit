"""Partial dependence (1-D and 2-D) and individual conditional expectation.

Partial dependence of feature ``f`` at value ``v`` averages the model's
prediction over all rows with ``X[:, f]`` set to ``v``. ICE curves instead
keep every row fixed and plot its prediction as ``X[:, f]`` varies, exposing
heterogeneous model behavior that the averaged curve hides.
"""

import numpy as np

from ._utils import as_2d, check_feature_index, check_rows

__all__ = ["make_grid", "partial_dependence", "ice_curves"]


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


def partial_dependence(predict, X, feature_index, grid=None, grid_points=20):
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

    Returns
    -------
    dict
        ``{"feature": int, "grid": ndarray, "values": ndarray}``.
    """
    X = as_2d(X)
    f = check_feature_index(feature_index, X.shape[1])
    grid = make_grid(X[:, f], grid_points, grid)
    X_work = X.copy()
    values = np.empty(grid.size)
    for i, v in enumerate(grid):
        X_work[:, f] = v
        values[i] = float(np.mean(predict(X_work)))
    return {"feature": f, "grid": grid, "values": values}


def ice_curves(predict, X, feature_index, grid=None, grid_points=20, rows=None, max_rows=25):
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

    Returns
    -------
    dict
        ``{"feature": int, "grid": ndarray, "rows": ndarray,
        "curves": ndarray of shape (len(rows), len(grid))}``.
    """
    X = as_2d(X)
    f = check_feature_index(feature_index, X.shape[1])
    grid = make_grid(X[:, f], grid_points, grid)
    idx = check_rows(rows, X.shape[0], max_rows)
    X_work = X[idx].copy()
    curves = np.empty((idx.size, grid.size))
    for i, v in enumerate(grid):
        X_work[:, f] = v
        curves[:, i] = predict(X_work)
    return {"feature": f, "grid": grid, "rows": idx, "curves": curves}
