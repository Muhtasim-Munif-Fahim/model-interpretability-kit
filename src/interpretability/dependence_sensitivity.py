"""Moment-independent global sensitivity: HSIC indices and Borgonovo's delta.

Variance-based Sobol indices only see how a feature moves the *mean* of the
output. Two given-data measures in this module capture any change in the
output *distribution* and need only one sample ``(X, f(X))`` (no Saltelli
design), so they also work on observational data with dependent inputs:

* :func:`hsic` – the biased V-statistic of the Hilbert–Schmidt independence
  criterion (Gretton et al. 2005) ``HSIC = trace(K H L H) / n^2`` with
  Gaussian kernels and median-heuristic bandwidths.
* :func:`hsic_sensitivity` – per-feature HSIC between ``X_j`` and the model
  output, the normalized ``R2-HSIC`` index (Da Veiga 2015)
  ``HSIC(X_j, Y) / sqrt(HSIC(X_j, X_j) HSIC(Y, Y))`` in ``[0, 1]``, and a
  permutation test of independence for screening (p-values with the
  ``(b + 1) / (B + 1)`` correction).
* :func:`borgonovo_delta` – the delta moment-independent importance
  (Borgonovo 2007) ``delta_j = 1/2 E_{X_j} || f_Y - f_{Y | X_j} ||_1``,
  estimated from given data by partitioning ``X_j`` into quantile classes and
  comparing Gaussian KDEs of ``Y`` on a common grid (Plischke, Borgonovo &
  Smith 2013). An optional bootstrap gives a bias estimate and percentile CI.
* :func:`pawn_indices` – the given-data PAWN index (Pianosi & Wagener 2015,
  2018): a summary (median by default) of Kolmogorov–Smirnov distances
  between the unconditional output CDF and CDFs conditional on
  equal-probability slices of ``X_j``, with a dummy-input noise floor.
"""

import numpy as np

from ._utils import as_2d

__all__ = ["hsic", "hsic_sensitivity", "borgonovo_delta", "pawn_indices"]


def _as_column(v, name):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    if v.ndim != 2:
        raise ValueError("%s must be 1-D or 2-D" % name)
    if not np.all(np.isfinite(v)):
        raise ValueError("%s contains non-finite values" % name)
    return v


def _median_bandwidth(v):
    diffs = np.sqrt(np.maximum(_sq_dists(v), 0.0))
    upper = diffs[np.triu_indices(len(v), k=1)]
    upper = upper[upper > 0]
    return float(np.median(upper)) if upper.size else 1.0


def _sq_dists(v):
    sq = np.sum(v * v, axis=1)
    return sq[:, None] + sq[None, :] - 2.0 * v @ v.T


def _rbf_gram(v, bandwidth=None):
    h = _median_bandwidth(v) if bandwidth is None else float(bandwidth)
    if h <= 0:
        raise ValueError("bandwidth must be positive")
    return np.exp(-np.maximum(_sq_dists(v), 0.0) / (2.0 * h * h))


def _center(K):
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def _hsic_centered(Kc, L):
    n = Kc.shape[0]
    return float(np.sum(Kc * L) / (n * n))


def hsic(x, y, bandwidth_x=None, bandwidth_y=None):
    """Biased HSIC V-statistic ``trace(K H L H) / n^2`` with Gaussian kernels.

    Zero (in the population) iff ``x`` and ``y`` are independent, for
    characteristic kernels. Bandwidths default to the median heuristic.
    """
    x = _as_column(x, "x")
    y = _as_column(y, "y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if x.shape[0] < 2:
        raise ValueError("need at least two rows")
    K = _rbf_gram(x, bandwidth_x)
    L = _rbf_gram(y, bandwidth_y)
    return _hsic_centered(_center(K), L)


def hsic_sensitivity(predict, X, y=None, n_permutations=200, max_rows=500, seed=None):
    """HSIC and R2-HSIC indices of each feature against the model output.

    Parameters
    ----------
    predict : callable or None
        ``predict(X) -> y``. Ignored when ``y`` is given (then the indices
        measure dependence of the target itself on each feature).
    X : array-like (n_rows, n_features)
    y : array-like, optional
        Output sample to use instead of ``predict(X)``.
    n_permutations : int
        Permutations for the independence test (0 disables the test).
    max_rows : int
        Rows are subsampled to at most this many (Gram matrices are O(n^2)).

    Returns
    -------
    dict
        ``{"hsic", "r2_hsic", "p_values", "n_rows"}``; ``p_values`` is ``None``
        when ``n_permutations == 0``.
    """
    X = as_2d(X)
    rng = np.random.default_rng(seed)
    if isinstance(n_permutations, bool) or int(n_permutations) != n_permutations or n_permutations < 0:
        raise ValueError("n_permutations must be a non-negative integer")
    if max_rows < 3:
        raise ValueError("max_rows must be at least 3")
    if y is None:
        if predict is None:
            raise ValueError("pass predict or y")
        y = np.asarray(predict(X), dtype=float).ravel()
    else:
        y = np.asarray(y, dtype=float).ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError("output must have one value per row")
    n = X.shape[0]
    if n > max_rows:
        idx = rng.choice(n, size=max_rows, replace=False)
        X, y = X[idx], y[idx]
        n = max_rows
    L = _rbf_gram(_as_column(y, "y"))
    Lc = _center(L)
    hsic_yy = _hsic_centered(Lc, L)

    p = X.shape[1]
    values = np.zeros(p)
    r2 = np.zeros(p)
    pvals = np.ones(p) if n_permutations else None
    for j in range(p):
        xj = X[:, j : j + 1]
        if np.ptp(xj) == 0:
            continue
        K = _rbf_gram(xj)
        Kc = _center(K)
        values[j] = _hsic_centered(Kc, L)
        hsic_xx = _hsic_centered(Kc, K)
        denom = np.sqrt(max(hsic_xx * hsic_yy, 0.0))
        r2[j] = float(np.clip(values[j] / denom, 0.0, 1.0)) if denom > 0 else 0.0
        if n_permutations:
            exceed = 0
            for _ in range(int(n_permutations)):
                perm = rng.permutation(n)
                if _hsic_centered(Kc, L[np.ix_(perm, perm)]) >= values[j] - 1e-15:
                    exceed += 1
            pvals[j] = (exceed + 1.0) / (n_permutations + 1.0)
    return {"hsic": values, "r2_hsic": r2, "p_values": pvals, "n_rows": n}


def _kde(sample, grid, bandwidth):
    z = (grid[:, None] - sample[None, :]) / bandwidth
    return np.exp(-0.5 * z * z).sum(axis=1) / (sample.size * bandwidth * np.sqrt(2.0 * np.pi))


def _silverman(sample):
    sd = np.std(sample, ddof=1) if sample.size > 1 else 0.0
    iqr = np.subtract(*np.percentile(sample, [75, 25])) / 1.34 if sample.size > 1 else 0.0
    s = min(sd, iqr) if iqr > 0 else sd
    return 0.9 * s * sample.size ** (-0.2) if s > 0 else 1e-3


def _delta_one(x, y, n_classes, grid):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_classes + 1)))
    if edges.size < 2:
        return 0.0
    labels = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    dy = grid[1] - grid[0]
    h_all = _silverman(y)
    f_y = _kde(y, grid, h_all)
    total = 0.0
    n = y.size
    for c in np.unique(labels):
        yc = y[labels == c]
        if yc.size < 2:
            continue
        f_c = _kde(yc, grid, max(_silverman(yc), h_all))
        total += (yc.size / n) * 0.5 * np.sum(np.abs(f_y - f_c)) * dy
    return float(np.clip(total, 0.0, 1.0))


def borgonovo_delta(predict, X, y=None, n_classes=None, n_grid=256, n_bootstrap=0, seed=None):
    """Given-data Borgonovo delta for each feature.

    ``delta_j`` lies in ``[0, 1]``; it is 0 when the output distribution does
    not depend on ``X_j``. ``n_classes`` defaults to ``ceil(n^(1/3))`` quantile
    classes per feature. With ``n_bootstrap > 0`` rows are resampled to give a
    bias-corrected estimate (``2 delta - mean(bootstrap)``, clipped) and a 95%
    percentile interval.

    Returns
    -------
    dict
        ``{"delta", "n_classes"}`` plus ``"delta_bc"``, ``"ci_low"``,
        ``"ci_high"`` when bootstrapping.
    """
    X = as_2d(X)
    if y is None:
        if predict is None:
            raise ValueError("pass predict or y")
        y = np.asarray(predict(X), dtype=float).ravel()
    else:
        y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    if y.shape[0] != n:
        raise ValueError("output must have one value per row")
    if n < 10:
        raise ValueError("need at least 10 rows")
    if n_classes is None:
        n_classes = int(np.ceil(n ** (1.0 / 3.0)))
    if int(n_classes) < 2:
        raise ValueError("n_classes must be at least 2")
    if n_grid < 16:
        raise ValueError("n_grid must be at least 16")
    if np.ptp(y) == 0:
        out = {"delta": np.zeros(X.shape[1]), "n_classes": int(n_classes)}
        if n_bootstrap:
            z = np.zeros(X.shape[1])
            out.update(delta_bc=z, ci_low=z.copy(), ci_high=z.copy())
        return out
    pad = 3.0 * _silverman(y)
    grid = np.linspace(y.min() - pad, y.max() + pad, int(n_grid))
    delta = np.array([_delta_one(X[:, j], y, int(n_classes), grid) for j in range(X.shape[1])])
    out = {"delta": delta, "n_classes": int(n_classes)}
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        boots = np.empty((int(n_bootstrap), X.shape[1]))
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, n, n)
            yb = y[idx]
            boots[b] = [_delta_one(X[idx, j], yb, int(n_classes), grid) for j in range(X.shape[1])]
        out["delta_bc"] = np.clip(2.0 * delta - boots.mean(axis=0), 0.0, 1.0)
        out["ci_low"] = np.percentile(boots, 2.5, axis=0)
        out["ci_high"] = np.percentile(boots, 97.5, axis=0)
    return out


_PAWN_STATS = ("median", "mean", "max", "min")


def _ks_to_reference(y_sorted, y_sub):
    """Two-sample KS distance between the full output sample and a subsample.

    Both empirical CDFs are evaluated on the pooled support (the full sample
    contains the subsample), so the supremum is attained at a data point.
    """
    sub = np.sort(y_sub)
    n, m = y_sorted.size, sub.size
    f_all = np.searchsorted(y_sorted, y_sorted, side="right") / n
    f_sub = np.searchsorted(sub, y_sorted, side="right") / m
    return float(np.max(np.abs(f_all - f_sub)))


def _pawn_slices(xj, n_classes):
    """Equal-probability slices of one feature (ties kept in one slice)."""
    edges = np.quantile(xj, np.linspace(0.0, 1.0, n_classes + 1)[1:-1])
    labels = np.searchsorted(edges, xj, side="right")
    return [np.flatnonzero(labels == k) for k in range(n_classes)]


def _pawn_one(xj, y, y_sorted, n_classes, min_slice):
    ks = []
    for idx in _pawn_slices(xj, n_classes):
        if idx.size >= min_slice:
            ks.append(_ks_to_reference(y_sorted, y[idx]))
    return np.asarray(ks, dtype=float)


def _summarize(ks, stat):
    if ks.size == 0:
        return float("nan")
    return float(getattr(np, stat)(ks))


def pawn_indices(
    predict,
    X,
    y=None,
    n_classes=10,
    stat="median",
    min_slice=5,
    n_bootstrap=0,
    seed=None,
):
    """Given-data PAWN density-based sensitivity indices.

    PAWN (Pianosi & Wagener 2015, given-data form 2018) compares the
    unconditional output CDF ``F_Y`` with conditional CDFs ``F_{Y | X_j in
    I_k}`` obtained by slicing feature ``j`` into ``n_classes``
    equal-probability intervals ``I_k``. For each slice the Kolmogorov–Smirnov
    distance ``KS_jk = sup_y |F_Y(y) - F_{Y|I_k}(y)|`` is computed, and the
    index is a summary of ``KS_j1 .. KS_jK`` (``stat``: median by default, or
    mean / max / min). Like Borgonovo's delta it reacts to *any* change in the
    output distribution, but it uses CDFs instead of densities, so it needs
    no bandwidth and is robust for skewed or heavy-tailed outputs.

    Because a finite slice never matches the full sample exactly, PAWN
    indices are positive even for irrelevant features. The result therefore
    includes a ``dummy`` index computed the same way for an independent
    random input (a noise floor for screening): features whose index does not
    clearly exceed ``dummy`` can be treated as non-influential.

    Parameters
    ----------
    predict : callable or None
        Model ``predict(X)``; ignored when ``y`` (model outputs) is given.
    X : array-like of shape (n, d)
    y : array-like of shape (n,), optional
        Precomputed outputs.
    n_classes : int
        Number of equal-probability slices per feature (>= 2).
    stat : {"median", "mean", "max", "min"}
        Summary of the per-slice KS statistics reported as ``index``.
    min_slice : int
        Slices with fewer rows (e.g. after heavy ties) are dropped.
    n_bootstrap : int
        If positive, rows are resampled to give 95% percentile intervals.
    seed : int, optional

    Returns
    -------
    dict
        ``index`` (d,), ``median``, ``mean``, ``max``, ``min`` (d,) summaries,
        ``ks`` (list of per-slice KS arrays), ``dummy`` (float), ``stat``,
        ``n_classes``; plus ``ci_low`` / ``ci_high`` when bootstrapping.
    """
    X = as_2d(X)
    if stat not in _PAWN_STATS:
        raise ValueError("stat must be one of %s" % (_PAWN_STATS,))
    if y is None:
        if predict is None:
            raise ValueError("pass predict or y")
        y = np.asarray(predict(X), dtype=float).ravel()
    else:
        y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError("output must have one value per row")
    if not np.all(np.isfinite(y)):
        raise ValueError("outputs contain non-finite values")
    n_classes = int(n_classes)
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    min_slice = int(min_slice)
    if min_slice < 1:
        raise ValueError("min_slice must be positive")
    if n < n_classes * min_slice:
        raise ValueError("need at least n_classes * min_slice rows")

    rng = np.random.default_rng(seed)
    y_sorted = np.sort(y)
    ks = [_pawn_one(X[:, j], y, y_sorted, n_classes, min_slice) for j in range(d)]
    summaries = {s: np.array([_summarize(k, s) for k in ks]) for s in _PAWN_STATS}
    dummy_x = rng.random(n)
    dummy = _summarize(_pawn_one(dummy_x, y, y_sorted, n_classes, min_slice), stat)
    out = {
        "index": summaries[stat].copy(),
        "ks": ks,
        "dummy": dummy,
        "stat": stat,
        "n_classes": n_classes,
    }
    out.update(summaries)
    if n_bootstrap:
        boots = np.empty((int(n_bootstrap), d))
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, n, n)
            yb = y[idx]
            yb_sorted = np.sort(yb)
            boots[b] = [
                _summarize(_pawn_one(X[idx, j], yb, yb_sorted, n_classes, min_slice), stat)
                for j in range(d)
            ]
        out["ci_low"] = np.nanpercentile(boots, 2.5, axis=0)
        out["ci_high"] = np.nanpercentile(boots, 97.5, axis=0)
    return out
