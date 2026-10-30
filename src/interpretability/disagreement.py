"""Disagreement metrics between feature attributions.

Kernel SHAP, sampling Shapley, Integrated Gradients, SmoothGrad and LIME often
give different explanations for the same prediction. Krishna et al. (2022)
call this *the disagreement problem*. This module measures it for any two
attribution vectors ``a`` and ``b`` of one instance. Each top-``k`` metric
ranks features by ``|phi|`` and is a fraction in ``[0, 1]``:

* :func:`feature_agreement` -- the share of top-``k`` features the two
  explanations have in common.
* :func:`rank_agreement` -- the share of top-``k`` positions that hold the
  same feature.
* :func:`sign_agreement` -- the share of top-``k`` features that both have
  and that point in the same direction.
* :func:`signed_rank_agreement` -- the same feature at the same rank with
  the same sign.
* :func:`rank_correlation` -- Spearman correlation of the ``|phi|`` ranks,
  optionally restricted to a feature subset.
* :func:`pairwise_rank_agreement` -- the share of feature pairs that the two
  explanations order the same way.
* :func:`kendall_tau` -- Kendall's tau-b with tie correction, using ``O(d^2)``
  pair counting.

:func:`explanation_disagreement` takes a dict ``{method: attributions}``,
where each value is an ``(n, d)`` array (one row per explained instance). It
returns one ``methods x methods`` matrix per metric, averaged over rows,
plus per-row values.

Only ``numpy`` is required.
"""

import itertools

import numpy as np

__all__ = [
    "top_k_features",
    "feature_agreement",
    "rank_agreement",
    "sign_agreement",
    "signed_rank_agreement",
    "rank_correlation",
    "pairwise_rank_agreement",
    "kendall_tau",
    "explanation_disagreement",
    "DISAGREEMENT_METRICS",
]


def _pair(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("attribution vectors must have the same length")
    if a.size == 0:
        raise ValueError("attribution vectors must not be empty")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("attributions must be finite")
    return a, b


def _check_k(k, d):
    k = int(k)
    if not 1 <= k <= d:
        raise ValueError("k must be between 1 and the number of features")
    return k


def top_k_features(phi, k):
    """Indices of the ``k`` largest ``|phi|``, ordered by decreasing
    magnitude. Ties keep the lower feature index first, so results are
    deterministic."""

    phi = np.asarray(phi, dtype=float).ravel()
    k = _check_k(k, phi.size)
    return np.argsort(-np.abs(phi), kind="stable")[:k]


def feature_agreement(a, b, k):
    a, b = _pair(a, b)
    ta, tb = top_k_features(a, k), top_k_features(b, k)
    return len(set(ta.tolist()) & set(tb.tolist())) / len(ta)


def rank_agreement(a, b, k):
    a, b = _pair(a, b)
    ta, tb = top_k_features(a, k), top_k_features(b, k)
    return float(np.mean(ta == tb))


def sign_agreement(a, b, k):
    a, b = _pair(a, b)
    ta, tb = top_k_features(a, k), top_k_features(b, k)
    common = set(ta.tolist()) & set(tb.tolist())
    return sum(np.sign(a[f]) == np.sign(b[f]) for f in common) / len(ta)


def signed_rank_agreement(a, b, k):
    a, b = _pair(a, b)
    ta, tb = top_k_features(a, k), top_k_features(b, k)
    same = (ta == tb) & (np.sign(a[ta]) == np.sign(b[ta]))
    return float(np.mean(same))


def _ranks(x):
    """Average ranks (1 = smallest), ties get the mean rank."""

    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def rank_correlation(a, b, features=None):
    """Spearman correlation between the ``|phi|`` rankings of ``a`` and ``b``.

    ``features`` restricts the comparison to a subset of feature indices.
    Returns ``nan`` when either ranking is constant.
    """

    a, b = _pair(a, b)
    if features is not None:
        idx = np.asarray(features, dtype=int)
        a, b = a[idx], b[idx]
    if a.size < 2:
        raise ValueError("need at least 2 features")
    ra, rb = _ranks(np.abs(a)), _ranks(np.abs(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pairwise_rank_agreement(a, b, features=None):
    """Share of feature pairs ``(i, j)`` that are ordered the same way by
    ``|a|`` and ``|b|`` (ties count as agreeing only when both explanations
    tie). This is the Krishna et al. (2022) metric over the feature set of
    interest."""

    a, b = _pair(a, b)
    idx = np.arange(a.size) if features is None else np.asarray(features, dtype=int)
    if idx.size < 2:
        raise ValueError("need at least 2 features")
    A, B = np.abs(a[idx]), np.abs(b[idx])
    i, j = np.triu_indices(idx.size, k=1)
    return float(np.mean(np.sign(A[i] - A[j]) == np.sign(B[i] - B[j])))


def kendall_tau(a, b, *, absolute=True):
    """Kendall's tau-b between two attribution vectors.

    By default it compares ``|phi|``, which measures agreement on importance
    ordering. Use ``absolute=False`` to compare signed values.
    """

    a, b = _pair(a, b)
    if a.size < 2:
        raise ValueError("need at least 2 features")
    if absolute:
        a, b = np.abs(a), np.abs(b)
    i, j = np.triu_indices(a.size, k=1)
    sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    concordant_minus_discordant = float(np.sum(sa * sb))
    n0 = i.size
    n1 = float(np.sum(sa == 0))
    n2 = float(np.sum(sb == 0))
    denom = np.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return float("nan")
    return concordant_minus_discordant / denom


DISAGREEMENT_METRICS = {
    "feature_agreement": lambda a, b, k: feature_agreement(a, b, k),
    "rank_agreement": lambda a, b, k: rank_agreement(a, b, k),
    "sign_agreement": lambda a, b, k: sign_agreement(a, b, k),
    "signed_rank_agreement": lambda a, b, k: signed_rank_agreement(a, b, k),
    "rank_correlation": lambda a, b, k: rank_correlation(a, b),
    "pairwise_rank_agreement": lambda a, b, k: pairwise_rank_agreement(a, b),
    "kendall_tau": lambda a, b, k: kendall_tau(a, b),
}


def explanation_disagreement(attributions, k=3, metrics=None):
    """All pairwise disagreement metrics between several explanation methods.

    Parameters
    ----------
    attributions:
        A mapping ``{method_name: phi}``. Each ``phi`` is a ``(d,)`` vector or
        an ``(n, d)`` array of attributions for the same ``n`` instances.
    k:
        The top-``k`` size used by the agreement metrics.
    metrics:
        A subset of :data:`DISAGREEMENT_METRICS` names. Defaults to all.

    Returns
    -------
    dict
        * ``methods`` -- the method names, in order.
        * ``matrices`` -- metric name to an ``(m, m)`` array of averages over
          instances, with 1 on the diagonal (``nan`` entries are ignored).
        * ``per_instance`` -- metric name to a dict ``{(m1, m2): (n,) array}``.
        * ``k`` -- the top-``k`` size used.
    """

    if not isinstance(attributions, dict) or len(attributions) < 2:
        raise ValueError("attributions must be a dict with at least two methods")
    names = list(attributions)
    arrays = []
    for name in names:
        arr = np.asarray(attributions[name], dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError("each attribution must be a (d,) or (n, d) array")
        arrays.append(arr)
    shape = arrays[0].shape
    if any(arr.shape != shape for arr in arrays):
        raise ValueError("all methods must explain the same (n, d) instances")
    k = _check_k(k, shape[1])
    metrics = list(DISAGREEMENT_METRICS) if metrics is None else list(metrics)
    unknown = [m for m in metrics if m not in DISAGREEMENT_METRICS]
    if unknown:
        raise ValueError("unknown metrics: %s" % ", ".join(unknown))
    m = len(names)
    matrices = {met: np.eye(m) for met in metrics}
    per_instance = {met: {} for met in metrics}
    for (p, q) in itertools.combinations(range(m), 2):
        for met in metrics:
            fn = DISAGREEMENT_METRICS[met]
            vals = np.array([fn(arrays[p][r], arrays[q][r], k) for r in range(shape[0])])
            per_instance[met][(names[p], names[q])] = vals
            avg = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
            matrices[met][p, q] = matrices[met][q, p] = avg
    return {"methods": names, "matrices": matrices, "per_instance": per_instance, "k": k}
