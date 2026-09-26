"""Friedman H-statistic for pairwise feature interaction strength.

The H-statistic (Friedman & Popescu, 2008) measures how much of the variation
in the two-variable partial dependence is *not* explained by the sum of the
one-variable partial dependence functions. For features ``j`` and ``k``:

```text
H^2_jk = sum_i [ PD_jk(x_i^j, x_i^k) - PD_j(x_i^j) - PD_k(x_i^k) ]^2
       / sum_i [ PD_jk(x_i^j, x_i^k) ]^2
```

centered so each PD has mean zero over the evaluation rows. ``H`` is the
square root, clipped to ``[0, 1]``. A value near 0 means the joint effect is
additive; a value near 1 means almost all of the joint PD is interaction.
"""

from __future__ import annotations

import numpy as np

from ._utils import as_2d, check_feature_index
from .partial_dependence import partial_dependence, partial_dependence_2d

__all__ = [
    "friedman_h_statistic",
    "h_statistic",
]


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - float(values.mean())


def _pd1_at_rows(predict, X, feature_index, grid_points):
    """Evaluate centered 1-D PD at each row's observed feature value.

    Uses linear interpolation on the PDP grid so every training row gets a
    value without evaluating the model once per distinct level.
    """
    pdp = partial_dependence(
        predict, X, feature_index, grid_points=grid_points
    )
    grid = np.asarray(pdp["grid"], dtype=float)
    values = _center(np.asarray(pdp["values"], dtype=float))
    col = np.asarray(X[:, feature_index], dtype=float)
    # np.interp requires increasing grid; make_grid already returns linspace.
    return np.interp(col, grid, values)


def _pd2_at_rows(predict, X, feature_indices, grid_points):
    """Evaluate centered 2-D PD at each row's observed feature pair via bilinear interpolation."""
    i, j = feature_indices
    pdp2 = partial_dependence_2d(
        predict, X, feature_indices, grid_points=grid_points
    )
    g0 = np.asarray(pdp2["grid0"], dtype=float)
    g1 = np.asarray(pdp2["grid1"], dtype=float)
    surface = _center(np.asarray(pdp2["values"], dtype=float))
    x0 = np.asarray(X[:, i], dtype=float)
    x1 = np.asarray(X[:, j], dtype=float)
    # Bilinear interpolation on the rectilinear grid.
    # Locate brackets along each axis.
    i0 = np.searchsorted(g0, x0, side="right") - 1
    i1 = np.searchsorted(g1, x1, side="right") - 1
    i0 = np.clip(i0, 0, g0.size - 2)
    i1 = np.clip(i1, 0, g1.size - 2)
    u = np.where(g0[i0 + 1] > g0[i0], (x0 - g0[i0]) / (g0[i0 + 1] - g0[i0]), 0.0)
    v = np.where(g1[i1 + 1] > g1[i1], (x1 - g1[i1]) / (g1[i1 + 1] - g1[i1]), 0.0)
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    q00 = surface[i0, i1]
    q10 = surface[i0 + 1, i1]
    q01 = surface[i0, i1 + 1]
    q11 = surface[i0 + 1, i1 + 1]
    return (
        q00 * (1.0 - u) * (1.0 - v)
        + q10 * u * (1.0 - v)
        + q01 * (1.0 - u) * v
        + q11 * u * v
    )


def friedman_h_statistic(
    predict,
    X,
    feature_indices,
    grid_points: int = 20,
):
    """Friedman H-statistic for a pair of features.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    X : ndarray
        Feature matrix the statistic is averaged over.
    feature_indices : pair of int
        The two feature columns ``(j, k)``.
    grid_points : int
        Resolution of the underlying 1-D and 2-D partial dependence grids.
        Must be at least 2.

    Returns
    -------
    dict
        ``feature_indices``, ``h`` (in ``[0, 1]``), ``h_squared``,
        ``numerator``, ``denominator``. When the joint PD is constant the
        denominator is zero and ``h`` / ``h_squared`` are returned as ``0.0``.
    """
    X = as_2d(X, name="X")
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    indices = tuple(int(v) for v in feature_indices)
    if len(indices) != 2:
        raise ValueError("feature_indices must contain exactly two indices")
    j, k = indices
    if j == k:
        raise ValueError("feature_indices must name two distinct features")
    check_feature_index(j, X.shape[1])
    check_feature_index(k, X.shape[1])

    pd_j = _pd1_at_rows(predict, X, j, grid_points)
    pd_k = _pd1_at_rows(predict, X, k, grid_points)
    pd_jk = _pd2_at_rows(predict, X, (j, k), grid_points)
    resid = pd_jk - pd_j - pd_k
    numerator = float(np.dot(resid, resid))
    denominator = float(np.dot(pd_jk, pd_jk))
    if denominator <= 0.0 or not np.isfinite(denominator):
        h2 = 0.0
    else:
        h2 = numerator / denominator
    if not np.isfinite(h2) or h2 < 0.0:
        h2 = 0.0
    h2 = float(min(h2, 1.0))
    return {
        "feature_indices": (j, k),
        "h": float(np.sqrt(h2)),
        "h_squared": h2,
        "numerator": numerator,
        "denominator": denominator,
    }


h_statistic = friedman_h_statistic
