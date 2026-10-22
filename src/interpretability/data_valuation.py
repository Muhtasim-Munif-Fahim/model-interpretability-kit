"""Data valuation: which *training points* help or hurt a model.

Feature attributions explain predictions; data valuation explains the
training set. Each training point ``i`` is a player in a cooperative game
whose value ``U(S)`` is the validation performance of a model trained on the
subset ``S``. The Shapley value of that game (Ghorbani & Zou, ICML 2019;
Jia et al., AISTATS 2019) is the only valuation that is efficient, symmetric,
null-player and additive. Low or negative values flag mislabelled,
duplicated or out-of-distribution rows.

* :func:`make_utility` turns any ``fit_predict(X_fit, y_fit, X_eval)`` and a
  validation set into ``U(indices)`` (accuracy, R², negative MSE or a
  callable metric), with a well-defined value for tiny/degenerate subsets.
* :func:`exact_data_shapley` enumerates all ``2^n`` subsets (small ``n``) —
  ground truth for the estimators.
* :func:`tmc_data_shapley` is Truncated Monte Carlo Shapley: random
  permutations, skipping the tail of a permutation once the running score is
  within ``truncation_tol`` of the full-data score; returns standard errors.
* :func:`leave_one_out_values` gives the classic ``U(N) - U(N \\ {i})``.
* :func:`knn_shapley` computes the *exact* Shapley values of the
  unweighted K-nearest-neighbour classifier utility in ``O(N log N)`` per
  validation point with the Jia et al. (2019) recursion

  ``s_N = 1[y_N = y] / N``,
  ``s_i = s_{i+1} + (1[y_i = y] - 1[y_{i+1} = y]) / K * min(K, i) / i``

  over training points sorted by distance to the validation point, averaged
  over the validation set.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from ._utils import as_1d, as_2d

__all__ = [
    "make_utility",
    "exact_data_shapley",
    "tmc_data_shapley",
    "leave_one_out_values",
    "knn_shapley",
    "knn_utility",
]

_METRICS = ("accuracy", "r2", "neg_mse")


def _score(metric, y_true, y_pred):
    if callable(metric):
        return float(metric(y_true, y_pred))
    if metric == "accuracy":
        return float(np.mean(np.asarray(y_pred) == np.asarray(y_true)))
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    if metric == "neg_mse":
        return -mse
    if metric == "r2":
        var = float(np.var(y_true))
        return 1.0 - mse / var if var > 0 else 0.0
    raise ValueError("metric must be one of %s or a callable" % (_METRICS,))


def make_utility(fit_predict, X_train, y_train, X_val, y_val, metric="neg_mse", min_size=1, empty_score=None):
    """Build ``U(indices) -> float`` = validation score of a model fit on ``indices``.

    ``fit_predict(X_fit, y_fit, X_eval)`` must return predictions for
    ``X_eval``. Subsets smaller than ``min_size`` — and, for ``accuracy``,
    subsets containing a single class — are scored by a constant baseline:
    ``empty_score`` if given, else the score of predicting the training
    mean (regression metrics) or the training majority class (accuracy).
    Exceptions raised by ``fit_predict`` on degenerate subsets also fall back
    to the baseline.
    """
    X_train = as_2d(X_train, name="X_train")
    y_train = np.asarray(y_train).ravel()
    if y_train.shape[0] != X_train.shape[0]:
        raise ValueError("y_train must have one entry per row of X_train")
    X_val = as_2d(X_val, name="X_val")
    y_val = np.asarray(y_val).ravel()
    if y_val.shape[0] != X_val.shape[0]:
        raise ValueError("y_val must have one entry per row of X_val")
    if X_val.shape[1] != X_train.shape[1]:
        raise ValueError("X_train and X_val must have the same number of features")
    if not callable(metric) and metric not in _METRICS:
        raise ValueError("metric must be one of %s or a callable" % (_METRICS,))
    if empty_score is None:
        if metric == "accuracy":
            values, counts = np.unique(y_train, return_counts=True)
            const = np.full(y_val.shape[0], values[np.argmax(counts)])
        else:
            const = np.full(y_val.shape[0], float(np.mean(y_train.astype(float))))
        empty_score = _score(metric, y_val, const)
    empty_score = float(empty_score)

    def utility(indices):
        idx = np.asarray(indices, dtype=int).ravel()
        if idx.size < max(int(min_size), 1):
            return empty_score
        y_fit = y_train[idx]
        if metric == "accuracy" and np.unique(y_fit).size == 1:
            return _score(metric, y_val, np.full(y_val.shape[0], y_fit[0]))
        try:
            pred = fit_predict(X_train[idx], y_fit, X_val)
        except (ValueError, np.linalg.LinAlgError):
            return empty_score
        return _score(metric, y_val, pred)

    utility.n_players = X_train.shape[0]
    utility.empty_score = empty_score
    return utility


def _n_players(utility, n):
    if n is None:
        n = getattr(utility, "n_players", None)
    if n is None:
        raise ValueError("pass n (the number of training points)")
    return int(n)


def exact_data_shapley(utility, n=None, max_points=14):
    """Exact data Shapley values by enumerating every subset (``2^n`` fits)."""
    n = _n_players(utility, n)
    if n > max_points:
        raise ValueError("exact enumeration needs 2^n utility calls; n=%d exceeds max_points=%d" % (n, max_points))
    cache = {}
    for size in range(n + 1):
        for S in combinations(range(n), size):
            cache[S] = float(utility(list(S)))
    values = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            w = 1.0 / (n * comb(n - 1, size))
            for S in combinations(others, size):
                with_i = tuple(sorted(S + (i,)))
                values[i] += w * (cache[with_i] - cache[S])
    full = cache[tuple(range(n))]
    return {"values": values, "full_score": full, "empty_score": cache[()], "n_evaluations": len(cache)}


def tmc_data_shapley(utility, n=None, n_permutations=100, truncation_tol=0.01, seed=None,
                     convergence_tol=None, min_permutations=20):
    """Truncated Monte Carlo data Shapley (Ghorbani & Zou, 2019).

    For each random permutation the training points are added one at a time
    and each receives its marginal gain in ``utility``. Once the running
    score is within ``truncation_tol * |U(N)|`` of the full-data score the
    remaining points of that permutation get a zero marginal (they cannot
    matter much). With ``convergence_tol`` the loop stops early when the
    mean absolute relative change of the running estimate over the last 10
    permutations falls below it (after ``min_permutations``).

    Returns ``values``, per-point standard errors ``std_error``, the number
    of permutations run and of utility evaluations.
    """
    n = _n_players(utility, n)
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    if truncation_tol < 0:
        raise ValueError("truncation_tol must be non-negative")
    rng = np.random.default_rng(seed)
    full = float(utility(np.arange(n)))
    empty = float(utility(np.array([], dtype=int)))
    n_eval = 2
    marginals = []
    history = []
    for t in range(n_permutations):
        perm = rng.permutation(n)
        contrib = np.zeros(n)
        prev = empty
        for j in range(n):
            if abs(full - prev) <= truncation_tol * abs(full):
                break
            cur = float(utility(perm[: j + 1]))
            n_eval += 1
            contrib[perm[j]] = cur - prev
            prev = cur
        marginals.append(contrib)
        running = np.mean(marginals, axis=0)
        history.append(running)
        if convergence_tol is not None and t + 1 >= max(min_permutations, 11):
            old = history[-11]
            denom = np.abs(running) + 1e-12
            if float(np.mean(np.abs(running - old) / denom)) < convergence_tol:
                break
    M = np.asarray(marginals)
    se = M.std(axis=0, ddof=1) / np.sqrt(M.shape[0]) if M.shape[0] > 1 else np.full(n, np.nan)
    return {"values": M.mean(axis=0), "std_error": se, "n_permutations": M.shape[0],
            "n_evaluations": n_eval, "full_score": full, "empty_score": empty}


def leave_one_out_values(utility, n=None):
    """Leave-one-out values ``U(N) - U(N \\ {i})`` (``n + 1`` utility calls)."""
    n = _n_players(utility, n)
    full = float(utility(np.arange(n)))
    values = np.array([full - float(utility(np.delete(np.arange(n), i))) for i in range(n)])
    return {"values": values, "full_score": full}


def knn_utility(X_train, y_train, X_val, y_val, k=5):
    """The K-NN utility of Jia et al. (2019) as ``U(indices)`` (for checks).

    ``U(S) = mean_val (1/K) sum_{the min(K, |S|) nearest points of S} 1[y = y_val]``.
    """
    X_train = as_2d(X_train, name="X_train")
    X_val = as_2d(X_val, name="X_val")
    y_train = np.asarray(y_train).ravel()
    y_val = np.asarray(y_val).ravel()
    D = np.sqrt(((X_val[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))

    def utility(indices):
        idx = np.asarray(indices, dtype=int).ravel()
        if idx.size == 0:
            return 0.0
        total = 0.0
        for v in range(X_val.shape[0]):
            near = idx[np.argsort(D[v, idx], kind="mergesort")[:k]]
            total += np.sum(y_train[near] == y_val[v]) / k
        return total / X_val.shape[0]

    utility.n_players = X_train.shape[0]
    return utility


def knn_shapley(X_train, y_train, X_val, y_val, k=5):
    """Exact Shapley values of the unweighted K-NN classifier utility.

    Uses the closed-form recursion of Jia et al. (2019) per validation point
    (ties in distance broken by training index) and averages over the
    validation set. Returns ``values`` (one per training row, summing to
    ``U(N)``) and ``per_validation`` (``n_val x n_train``).
    """
    X_train = as_2d(X_train, name="X_train")
    X_val = as_2d(X_val, name="X_val")
    y_train = np.asarray(y_train).ravel()
    y_val = np.asarray(y_val).ravel()
    n = X_train.shape[0]
    if y_train.shape[0] != n or y_val.shape[0] != X_val.shape[0]:
        raise ValueError("labels must match the number of rows")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("X_train and X_val must have the same number of features")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    per = np.zeros((X_val.shape[0], n))
    ranks = np.arange(1, n + 1, dtype=float)
    factor = np.minimum(k, ranks[:-1]) / ranks[:-1]  # for i = 1..N-1 (1-based)
    for v in range(X_val.shape[0]):
        d = np.sqrt(((X_train - X_val[v]) ** 2).sum(axis=1))
        order = np.argsort(d, kind="mergesort")
        match = (y_train[order] == y_val[v]).astype(float)
        s = np.zeros(n)
        s[n - 1] = match[n - 1] / n
        for i in range(n - 2, -1, -1):
            s[i] = s[i + 1] + (match[i] - match[i + 1]) / k * factor[i]
        per[v, order] = s
    return {"values": per.mean(axis=0), "per_validation": per, "k": k}
