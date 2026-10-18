"""Prototypes and criticisms with MMD-critic (Kim, Khanna & Koyejo, 2016).

Example-based explanations summarize a dataset (or the data a model was
trained on) with a handful of real rows:

* **Prototypes** are rows that together represent the data distribution
  well. MMD-critic picks them greedily so the empirical distribution of the
  prototypes is close to that of the data in maximum mean discrepancy (MMD)
  under an RBF kernel. Each greedy step adds the row that most increases
  ``J(S) = 2/(n|S|) sum_{i in [n], j in S} k(x_i, x_j) -
  1/|S|^2 sum_{i, j in S} k(x_i, x_j)``, which equals
  ``mean(K) - MMD^2(X, S)``; for the RBF kernel ``J`` is submodular and,
  under the kernel conditions in Kim et al. (2016), monotone, so greedy
  selection is near-optimal.
* **Criticisms** are rows the prototypes explain *badly*: the points where
  the witness function ``w(x) = mean_i k(x, x_i) - mean_{j in S} k(x, z_j)``
  is largest in absolute value. Positive witness means the region is
  under-represented by the prototypes (dense data, few prototypes); negative
  means it is over-represented. Criticisms are chosen greedily to maximize
  ``sum |w(c)| + log det K_CC``; the log-determinant regularizer pushes them
  apart so they do not all come from the same outlying cluster.

Prototypes and criticisms are exact row indices into ``X``, so they can be
shown to a user as concrete examples ("typical" vs. "unusual" cases) and fed
back to ``predict`` to see how a model behaves on each.

Columns are standardized before computing kernels by default so no single
feature dominates the distance; the bandwidth defaults to the median
heuristic.
"""

import numpy as np

from ._utils import as_2d

__all__ = [
    "rbf_kernel",
    "median_heuristic_gamma",
    "mmd_squared",
    "witness_function",
    "select_prototypes",
    "select_criticisms",
    "mmd_critic",
]


def rbf_kernel(A, B, gamma):
    """Gaussian RBF kernel matrix ``exp(-gamma * ||a - b||^2)``."""
    A = as_2d(A, "A")
    B = as_2d(B, "B")
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of columns")
    if not gamma > 0:
        raise ValueError("gamma must be positive")
    sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    np.maximum(sq, 0.0, out=sq)
    return np.exp(-gamma * sq)


def median_heuristic_gamma(X, max_rows=1000, seed=0):
    """Bandwidth ``gamma = 1 / median(||x_i - x_j||^2)`` over distinct pairs.

    At most ``max_rows`` rows (sampled with ``seed``) are used to bound the
    cost. Falls back to ``1.0`` when every pair coincides.
    """
    X = as_2d(X)
    if X.shape[0] > max_rows:
        X = X[np.random.default_rng(seed).choice(X.shape[0], max_rows, replace=False)]
    sq = (X * X).sum(1)[:, None] + (X * X).sum(1)[None, :] - 2.0 * X @ X.T
    iu = np.triu_indices(X.shape[0], k=1)
    d = np.maximum(sq[iu], 0.0)
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    return 1.0 / float(np.median(d))


def mmd_squared(X, Z, gamma):
    """Biased (V-statistic) squared MMD between samples ``X`` and ``Z``."""
    Kxx = rbf_kernel(X, X, gamma)
    Kzz = rbf_kernel(Z, Z, gamma)
    Kxz = rbf_kernel(X, Z, gamma)
    return float(Kxx.mean() + Kzz.mean() - 2.0 * Kxz.mean())


def witness_function(points, X, Z, gamma):
    """MMD witness ``mean_i k(p, x_i) - mean_j k(p, z_j)`` at each point."""
    return rbf_kernel(points, X, gamma).mean(1) - rbf_kernel(points, Z, gamma).mean(1)


def _check_count(k, n, name):
    k = int(k)
    if k < 0 or k > n:
        raise ValueError("%s must be between 0 and %d, got %d" % (name, n, k))
    return k


def select_prototypes(K, n_prototypes, candidates=None):
    """Greedy MMD prototype selection from a precomputed kernel matrix.

    Parameters
    ----------
    K:
        Square ``(n, n)`` kernel matrix over the data.
    n_prototypes:
        Number of prototypes to pick.
    candidates:
        Optional row indices allowed as prototypes (default: all rows).

    Returns
    -------
    (indices, objective_trace)
        Selected indices in pick order and ``J(S) = mean(K) - MMD^2`` after
        each pick.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if K.ndim != 2 or K.shape[1] != n:
        raise ValueError("K must be a square matrix")
    m = _check_count(n_prototypes, n, "n_prototypes")
    cand = np.arange(n) if candidates is None else np.unique(np.asarray(candidates, dtype=int))
    if m > cand.size:
        raise ValueError("n_prototypes exceeds the number of candidates")
    colmean = K.mean(axis=0)
    diag = np.diag(K)
    k_to_S = np.zeros(n)  # sum_{j in S} K[:, j]
    sum_colmean = 0.0
    sum_KSS = 0.0
    available = np.ones(n, dtype=bool)
    available[np.setdiff1d(np.arange(n), cand)] = False
    chosen, trace = [], []
    for t in range(1, m + 1):
        gain = 2.0 / t * (sum_colmean + colmean) - (sum_KSS + 2.0 * k_to_S + diag) / t**2
        gain[~available] = -np.inf
        c = int(np.argmax(gain))
        chosen.append(c)
        trace.append(float(gain[c]))
        available[c] = False
        sum_KSS += 2.0 * k_to_S[c] + diag[c]
        sum_colmean += colmean[c]
        k_to_S += K[:, c]
    return np.asarray(chosen, dtype=int), np.asarray(trace)


def select_criticisms(K, prototypes, n_criticisms, regularizer="logdet"):
    """Greedy criticism selection maximizing ``sum |witness| (+ log det K_CC)``.

    Parameters
    ----------
    K:
        Square ``(n, n)`` kernel matrix over the data.
    prototypes:
        Indices of the already selected prototypes (excluded as criticisms).
    n_criticisms:
        Number of criticisms to pick.
    regularizer:
        ``"logdet"`` (default, Kim et al.'s diversity term) or ``None`` for
        plain top-``|witness|`` selection.

    Returns
    -------
    (indices, witness)
        Criticism indices in pick order and the witness value at every row.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if K.ndim != 2 or K.shape[1] != n:
        raise ValueError("K must be a square matrix")
    prototypes = np.asarray(prototypes, dtype=int)
    if prototypes.size == 0:
        raise ValueError("need at least one prototype to compute criticisms")
    if regularizer not in ("logdet", None):
        raise ValueError("regularizer must be 'logdet' or None")
    c_max = n - np.unique(prototypes).size
    k = _check_count(n_criticisms, c_max, "n_criticisms")
    witness = K.mean(axis=1) - K[:, prototypes].mean(axis=1)
    score = np.abs(witness)
    available = np.ones(n, dtype=bool)
    available[prototypes] = False
    chosen = []
    diag = np.diag(K)
    for _ in range(k):
        gain = score.copy()
        if regularizer == "logdet" and chosen:
            C = np.asarray(chosen)
            Kcc = K[np.ix_(C, C)] + 1e-10 * np.eye(C.size)
            L = np.linalg.cholesky(Kcc)
            V = np.linalg.solve(L, K[C, :])  # (|C|, n)
            schur = diag - (V * V).sum(axis=0)
            with np.errstate(divide="ignore"):
                gain = gain + np.log(np.maximum(schur, 0.0))
        gain[~available] = -np.inf
        c = int(np.argmax(gain))
        if not np.isfinite(gain[c]):
            break  # every remaining row duplicates a chosen one
        chosen.append(c)
        available[c] = False
    return np.asarray(chosen, dtype=int), witness


def mmd_critic(
    X,
    n_prototypes=10,
    n_criticisms=5,
    *,
    gamma=None,
    standardize=True,
    labels=None,
    regularizer="logdet",
):
    """Select prototypes and criticisms that summarize ``X``.

    Parameters
    ----------
    X:
        Data matrix ``(n, d)``.
    n_prototypes, n_criticisms:
        How many of each to return.
    gamma:
        RBF bandwidth; default is the median heuristic on the (standardized)
        data.
    standardize:
        Z-score columns before computing kernels (constant columns are left
        at zero).
    labels:
        Optional class labels. When given, prototypes are allocated to each
        class in proportion to its size (at least one per non-empty class
        while the budget allows) and selected *within* the class, so every
        class is represented; criticisms are still chosen globally.
    regularizer:
        Passed to :func:`select_criticisms`.

    Returns
    -------
    dict
        ``prototypes`` and ``criticisms`` (row indices), ``witness`` at every
        row, ``criticism_witness``, ``mmd2`` between data and prototypes,
        ``objective_trace``, ``gamma``, ``assignment`` (index into
        ``prototypes`` of each row's most similar prototype) and
        ``coverage`` (rows per prototype).
    """
    X = as_2d(X)
    n = X.shape[0]
    if n < 2:
        raise ValueError("need at least two rows")
    n_prototypes = _check_count(n_prototypes, n, "n_prototypes")
    if n_prototypes == 0:
        raise ValueError("n_prototypes must be at least 1")
    Xs = X
    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
    if gamma is None:
        gamma = median_heuristic_gamma(Xs)
    K = rbf_kernel(Xs, Xs, gamma)

    if labels is None:
        protos, trace = select_prototypes(K, n_prototypes)
    else:
        labels = np.asarray(labels).ravel()
        if labels.shape[0] != n:
            raise ValueError("labels must have one entry per row")
        classes, counts = np.unique(labels, return_counts=True)
        alloc = _allocate(counts, n_prototypes)
        protos = []
        for cls, a in zip(classes, alloc):
            if a == 0:
                continue
            idx = np.flatnonzero(labels == cls)
            sub, _ = select_prototypes(K[np.ix_(idx, idx)], a)
            protos.extend(idx[sub].tolist())
        protos = np.asarray(protos, dtype=int)
        trace = None

    crit, witness = select_criticisms(K, protos, n_criticisms, regularizer=regularizer)
    Kp = K[:, protos]
    mmd2 = float(K.mean() + K[np.ix_(protos, protos)].mean() - 2.0 * Kp.mean())
    assignment = np.argmax(Kp, axis=1)
    return {
        "prototypes": protos,
        "criticisms": crit,
        "witness": witness,
        "criticism_witness": witness[crit],
        "mmd2": max(mmd2, 0.0),
        "objective_trace": trace,
        "gamma": float(gamma),
        "assignment": assignment,
        "coverage": np.bincount(assignment, minlength=protos.size),
    }


def _allocate(counts, budget):
    """Largest-remainder proportional allocation, >= 1 per class if possible."""
    counts = np.asarray(counts, dtype=float)
    k = counts.size
    alloc = np.zeros(k, dtype=int)
    if budget >= k:
        alloc[:] = 1
        rest = budget - k
    else:
        alloc[np.argsort(-counts, kind="stable")[:budget]] = 1
        return alloc
    capacity = counts.astype(int) - alloc
    while rest > 0:
        share = counts / counts.sum() * budget - alloc
        share[capacity <= 0] = -np.inf
        j = int(np.argmax(share))
        if not np.isfinite(share[j]):
            break
        alloc[j] += 1
        capacity[j] -= 1
        rest -= 1
    return alloc
