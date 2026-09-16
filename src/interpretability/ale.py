"""1-D accumulated local effects (ALE) curves.

ALE of feature ``f`` estimates the effect of moving ``f`` across its
observed range from *local* finite differences: the feature axis is split
into quantile bins, and in each bin the model is evaluated at the bin
edges while the other features stay at their actual values. Those local
effects are then accumulated and centered to have mean zero over the
data. Unlike partial dependence, ALE does not average over unrealistic
combinations of correlated features.
"""

import numpy as np

from ._utils import as_2d, check_feature_index

__all__ = ["make_ale_grid", "accumulated_local_effects", "ale"]


def make_ale_grid(X_column, grid_points=20, grid=None):
    """Build interval edges for a 1-D ALE curve.

    The default edges are the empirical quantiles of the feature so each
    bin holds a similar number of observations and the curve covers the
    full observed range. Duplicate quantiles (from a discrete or constant
    feature) are collapsed. A constant feature is given a tiny spread so
    the curve still has two endpoints. A caller-provided ``grid`` is
    sorted and uniqued; it must contain at least two distinct edges.
    """
    col = np.asarray(X_column, dtype=float)
    if col.size == 0:
        raise ValueError("X must contain at least one sample")
    if grid is not None:
        grid = np.asarray(grid, dtype=float).ravel()
        if grid.size == 0:
            raise ValueError("grid must not be empty")
        grid = np.unique(grid)
        if grid.size < 2:
            raise ValueError("grid must contain at least 2 unique edges")
        return grid
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    edges = np.quantile(col, np.linspace(0.0, 1.0, grid_points))
    edges = np.unique(edges)
    if edges.size < 2:
        value = float(col[0])
        return np.array([value - 1e-6, value + 1e-6])
    return edges


def accumulated_local_effects(predict, X, feature_index, grid=None, grid_points=20):
    """1-D accumulated local effects of one feature over quantile bins.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    X : ndarray
        Feature matrix whose rows define both the bins and the local
        neighborhoods.
    feature_index : int
        Column whose effect is accumulated.
    grid : ndarray or None
        Explicit interval edges (sorted, uniqued). Defaults to empirical
        quantiles of the feature.
    grid_points : int
        Number of default quantile edges (``grid_points - 1`` intervals).

    Returns
    -------
    dict
        ``{"feature": int, "grid": ndarray, "values": ndarray,
        "counts": ndarray}``. ``grid`` and ``values`` have one entry per
        edge; ``counts[k]`` is the number of rows in interval
        ``[grid[k], grid[k+1]]`` (the last interval is closed on the
        right). ``values`` are centered so the mean of the piecewise-linear
        curve over the rows of ``X`` is zero.
    """
    X = as_2d(X)
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    f = check_feature_index(feature_index, X.shape[1])
    col = X[:, f]
    edges = make_ale_grid(col, grid_points, grid)
    n_intervals = edges.size - 1
    counts = np.zeros(n_intervals, dtype=int)
    deltas = np.zeros(n_intervals, dtype=float)

    for k in range(n_intervals):
        left = edges[k]
        right = edges[k + 1]
        if k == n_intervals - 1:
            mask = (col >= left) & (col <= right)
        else:
            mask = (col >= left) & (col < right)
        n_k = int(np.count_nonzero(mask))
        counts[k] = n_k
        if n_k == 0:
            continue
        X_low = X[mask].copy()
        X_high = X_low.copy()
        X_low[:, f] = left
        X_high[:, f] = right
        pred_high = np.asarray(predict(X_high), dtype=float).ravel()
        pred_low = np.asarray(predict(X_low), dtype=float).ravel()
        deltas[k] = float(np.mean(pred_high - pred_low))

    uncentered = np.empty(edges.size, dtype=float)
    uncentered[0] = 0.0
    uncentered[1:] = np.cumsum(deltas)
    values = uncentered - float(np.interp(col, edges, uncentered).mean())
    return {"feature": f, "grid": edges, "values": values, "counts": counts}


ale = accumulated_local_effects
