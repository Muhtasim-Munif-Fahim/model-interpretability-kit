"""1-D and 2-D accumulated local effects (ALE).

ALE of feature ``f`` estimates the effect of moving ``f`` across its
observed range from *local* finite differences: the feature axis is split
into quantile bins, and in each bin the model is evaluated at the bin
edges while the other features stay at their actual values. Those local
effects are then accumulated and centered to have mean zero over the
data. Unlike partial dependence, ALE does not average over unrealistic
combinations of correlated features.

2-D ALE uses the same local-difference idea on a rectangular grid: each
cell's second-order (mixed) finite difference is averaged over the rows
that fall in that cell, the cell effects are accumulated, and the result
is doubly centered so the surface is a pure interaction (main effects
removed, overall mean zero).
"""

import numpy as np

from ._utils import as_2d, check_feature_index

__all__ = [
    "make_ale_grid",
    "accumulated_local_effects",
    "accumulated_local_effects_2d",
    "ale",
    "ale_2d",
]


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


def _interval_index(col, edges):
    """Map each value to its 1-D ALE interval, or -1 if it falls outside."""
    col = np.asarray(col, dtype=float)
    n_intervals = edges.size - 1
    index = np.full(col.shape, -1, dtype=int)
    for k in range(n_intervals):
        left = edges[k]
        right = edges[k + 1]
        if k == n_intervals - 1:
            mask = (col >= left) & (col <= right)
        else:
            mask = (col >= left) & (col < right)
        index[mask] = k
    return index


def _center_ale_2d(uncentered, counts):
    """Remove main effects and the overall mean from a 2-D ALE accumulation.

    Main effects of the uncentered surface are estimated with the same
    count-weighted interval averaging used by Apley & Zhu's ALEPlot, then
    subtracted. The result is shifted so the count-weighted mean of the
    four cell corners is zero.
    """
    n0, n1 = counts.shape
    row_n = counts.sum(axis=1)
    col_n = counts.sum(axis=0)

    dh0 = uncentered[1:, :] - uncentered[:-1, :]
    weighted0 = counts * (dh0[:, :-1] + dh0[:, 1:]) / 2.0
    delta0 = np.zeros(n0, dtype=float)
    nz0 = row_n > 0
    delta0[nz0] = weighted0[nz0].sum(axis=1) / row_n[nz0]
    main0 = np.empty(n0 + 1, dtype=float)
    main0[0] = 0.0
    main0[1:] = np.cumsum(delta0)

    dh1 = uncentered[:, 1:] - uncentered[:, :-1]
    weighted1 = counts * (dh1[:-1, :] + dh1[1:, :]) / 2.0
    delta1 = np.zeros(n1, dtype=float)
    nz1 = col_n > 0
    delta1[nz1] = weighted1[:, nz1].sum(axis=0) / col_n[nz1]
    main1 = np.empty(n1 + 1, dtype=float)
    main1[0] = 0.0
    main1[1:] = np.cumsum(delta1)

    centered = uncentered - main0[:, None] - main1[None, :]
    total = int(counts.sum())
    if total == 0:
        return centered
    corner_mean = (
        centered[:-1, :-1]
        + centered[:-1, 1:]
        + centered[1:, :-1]
        + centered[1:, 1:]
    ) / 4.0
    shift = float((counts * corner_mean).sum() / total)
    return centered - shift


def accumulated_local_effects_2d(predict, X, feature_indices, grids=None, grid_points=10):
    """2-D accumulated local effects (pure interaction) of two features.

    The joint range is split into a rectangular quantile grid. In each
    cell the model is evaluated at the four corners while the other
    features stay at their actual values; the mixed finite difference is
    averaged over the rows in that cell, accumulated, and doubly centered
    so main effects are removed and the surface has mean zero.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    X : ndarray
        Feature matrix whose rows define both the cells and the local
        neighborhoods.
    feature_indices : sequence of int
        Exactly two columns ``(f0, f1)``.
    grids : tuple of ndarray or None
        Explicit interval edges for each feature (sorted, uniqued).
        Defaults to empirical quantiles of each feature.
    grid_points : int
        Number of default quantile edges per feature
        (``grid_points - 1`` intervals per axis).

    Returns
    -------
    dict
        ``{"feature0": int, "feature1": int, "grid0": ndarray,
        "grid1": ndarray, "values": ndarray of shape (len(grid0),
        len(grid1)), "counts": ndarray of shape (len(grid0) - 1,
        len(grid1) - 1)}``. ``values[i, j]`` is the centered interaction
        at ``(grid0[i], grid1[j])``. ``counts[k, m]`` is the number of
        rows in cell ``[grid0[k], grid0[k+1]] × [grid1[m], grid1[m+1]]``
        (last interval on each axis closed on the right). Empty cells
        contribute no local effect.
    """
    X = as_2d(X)
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    f0, f1 = feature_indices
    f0 = check_feature_index(f0, X.shape[1])
    f1 = check_feature_index(f1, X.shape[1])
    if f0 == f1:
        raise ValueError("the two feature indices must differ")
    if grids is None:
        grid0 = make_ale_grid(X[:, f0], grid_points)
        grid1 = make_ale_grid(X[:, f1], grid_points)
    else:
        grid0 = make_ale_grid(X[:, f0], grid_points, grids[0])
        grid1 = make_ale_grid(X[:, f1], grid_points, grids[1])

    n0 = grid0.size - 1
    n1 = grid1.size - 1
    counts = np.zeros((n0, n1), dtype=int)
    deltas = np.zeros((n0, n1), dtype=float)

    i0 = _interval_index(X[:, f0], grid0)
    i1 = _interval_index(X[:, f1], grid1)
    valid = (i0 >= 0) & (i1 >= 0)
    if np.any(valid):
        i0v = i0[valid]
        i1v = i1[valid]
        X_in = X[valid]
        X11 = X_in.copy()
        X12 = X_in.copy()
        X21 = X_in.copy()
        X22 = X_in.copy()
        X11[:, f0] = grid0[i0v]
        X11[:, f1] = grid1[i1v]
        X12[:, f0] = grid0[i0v]
        X12[:, f1] = grid1[i1v + 1]
        X21[:, f0] = grid0[i0v + 1]
        X21[:, f1] = grid1[i1v]
        X22[:, f0] = grid0[i0v + 1]
        X22[:, f1] = grid1[i1v + 1]
        pred11 = np.asarray(predict(X11), dtype=float).ravel()
        pred12 = np.asarray(predict(X12), dtype=float).ravel()
        pred21 = np.asarray(predict(X21), dtype=float).ravel()
        pred22 = np.asarray(predict(X22), dtype=float).ravel()
        mixed = (pred22 - pred21) - (pred12 - pred11)
        np.add.at(counts, (i0v, i1v), 1)
        np.add.at(deltas, (i0v, i1v), mixed)
        occupied = counts > 0
        deltas[occupied] /= counts[occupied]

    uncentered = np.zeros((grid0.size, grid1.size), dtype=float)
    uncentered[1:, 1:] = np.cumsum(np.cumsum(deltas, axis=0), axis=1)
    values = _center_ale_2d(uncentered, counts)
    return {
        "feature0": f0,
        "feature1": f1,
        "grid0": grid0,
        "grid1": grid1,
        "values": values,
        "counts": counts,
    }


ale = accumulated_local_effects
ale_2d = accumulated_local_effects_2d
