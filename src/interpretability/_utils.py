"""Shared low-level array and index helpers used across the package."""

import numpy as np

__all__ = ["as_2d", "as_1d", "check_feature_index", "check_rows"]


def as_2d(X, name="X"):
    """Coerce ``X`` to a 2-D float array; a 1-D input becomes ``(n, 1)``."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X[:, None]
    if X.ndim != 2:
        raise ValueError("%s must be a 1-D or 2-D array" % name)
    return X


def as_1d(y, n, name="y"):
    """Coerce ``y`` to a 1-D float array of length ``n``."""
    y = np.asarray(y, dtype=float).ravel()
    if y.shape[0] != n:
        raise ValueError("%s must have as many entries as X has rows" % name)
    return y


def check_feature_index(index, n_features):
    """Validate a feature index against the number of columns."""
    index = int(index)
    if index < 0 or index >= n_features:
        raise ValueError(
            "feature index %d out of range for %d feature(s)" % (index, n_features)
        )
    return index


def check_rows(rows, n_rows, max_rows=25):
    """Resolve an ICE row selection to a validated integer index array."""
    if rows is None:
        return np.arange(min(n_rows, max_rows))
    idx = np.asarray(rows, dtype=int)
    if idx.ndim == 0:
        idx = idx[None]
    if idx.size == 0:
        raise ValueError("rows must not be empty")
    if idx.min() < 0 or idx.max() >= n_rows:
        raise ValueError("row indices out of range")
    return idx
