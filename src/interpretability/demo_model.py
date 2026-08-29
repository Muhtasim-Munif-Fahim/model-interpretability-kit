"""Transparent demo model and synthetic data for exercising the toolkit.

The data generator produces a small regression problem with a known additive
structure so explanations computed by the toolkit can be checked against
ground truth:

    y = 4.0 * X0 + 3.0 * X1 + 2.0 * (X2 > 0.5) + 1.5 * X3 + noise

``X4`` never appears in the formula and should receive near-zero importance.

``DecisionTreeRegressor`` is a hand-written regression tree using greedy
variance-reduction splitting. It is small enough to be inspected directly and
supports exact interventional tree-based attribution in
``interpretability.local``.
"""

import numpy as np

__all__ = ["DecisionTreeRegressor", "fit_decision_tree", "make_synthetic_data"]


def make_synthetic_data(n_samples=300, n_features=5, noise=0.05, seed=None):
    """Generate a regression problem with a known additive structure.

    Parameters
    ----------
    n_samples : int
        Number of rows.
    n_features : int
        Number of features. The response depends on the first four columns;
        columns beyond index 3 are pure noise features.
    noise : float
        Standard deviation of the Gaussian noise added to the response.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    feature_names : list of str
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
    y = (
        4.0 * X[:, 0]
        + 3.0 * X[:, 1]
        + 2.0 * (X[:, 2] > 0.5)
        + 1.5 * X[:, 3]
    )
    if noise > 0:
        y = y + rng.normal(0.0, noise, size=n_samples)
    feature_names = ["X%d" % j for j in range(n_features)]
    return X, y, feature_names


class DecisionTreeRegressor:
    """A minimal greedy regression tree with variance-reduction splitting.

    The fitted structure is stored in ``root_`` as nested dicts::

        {"leaf": True,  "value": float, "n": int}
        {"leaf": False, "feature": int, "threshold": float,
         "left": node, "right": node, "value": float, "n": int}
    """

    def __init__(self, max_depth=5, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root_ = None
        self.n_features_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be a 1-D or 2-D array")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.n_features_ = X.shape[1]
        self.root_ = self._build(X, y, depth=0)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("predict() called before fit()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("X has the wrong number of features")
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            out[i] = self._predict_row(self.root_, X[i])
        return out

    def _predict_row(self, node, row):
        while not node["leaf"]:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]

    def _build(self, X, y, depth):
        n = X.shape[0]
        value = float(y.mean())
        node = {"leaf": True, "value": value, "n": n}
        if n < 2 * self.min_samples_leaf or depth >= self.max_depth:
            return node
        variance = float(y.var())
        if variance < 1e-12:
            return node
        best = self._best_split(X, y, variance)
        if best is None:
            return node
        _, feature, threshold = best
        mask = X[:, feature] <= threshold
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return {
            "leaf": False,
            "feature": int(feature),
            "threshold": float(threshold),
            "left": left,
            "right": right,
            "value": value,
            "n": n,
        }

    def _best_split(self, X, y, variance):
        n = X.shape[0]
        best = None
        for f in range(X.shape[1]):
            col = X[:, f]
            order = np.argsort(col, kind="mergesort")
            xs = col[order]
            ys = y[order]
            total = float(ys.sum())
            sq = float((ys * ys).sum())
            total_sse = sq - total * total / n
            left_sum = 0.0
            left_sq = 0.0
            for i in range(1, n):
                left_sum += ys[i - 1]
                left_sq += ys[i - 1] ** 2
                if xs[i] == xs[i - 1]:
                    continue
                n_left = i
                n_right = n - i
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                left_var = left_sq - left_sum * left_sum / n_left
                right_var = sq - left_sq - (total - left_sum) ** 2 / n_right
                reduction = total_sse - (left_var + right_var)
                if best is None or reduction > best[0]:
                    best = (reduction, f, (xs[i - 1] + xs[i]) / 2.0)
        if best is not None and best[0] <= 1e-12:
            return None
        return best


def fit_decision_tree(X, y, **kwargs):
    """Fit and return a :class:`DecisionTreeRegressor`."""
    return DecisionTreeRegressor(**kwargs).fit(X, y)
