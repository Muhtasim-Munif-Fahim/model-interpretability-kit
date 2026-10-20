"""Exact model-agnostic Shapley values, Banzhaf values and Shapley interaction values.

For a small number of features (up to ~12) the interventional value function

``v(S) = mean_z f(x_S, z_~S)``

over a background set can be evaluated on *every* coalition ``S``, which gives
exact cooperative-game attributions for any black-box ``predict``. This is the
ground truth that :func:`interpretability.local.kernel_shap` and
:func:`interpretability.local.sampling_shapley` approximate, and generalises
the exact single-tree :func:`interpretability.local.tree_shap_values` to any
model.

* Shapley value (Shapley 1953):
  ``phi_i = sum_{S ⊆ N∖{i}} |S|! (M - |S| - 1)! / M! [v(S ∪ i) - v(S)]``;
  efficient: ``sum phi = f(x) - v(∅)``.
* Banzhaf value (Banzhaf 1965): the same marginal contributions with uniform
  weight ``1 / 2^(M-1)``; it coincides with Shapley when the model has at
  most pairwise interactions and is not efficient beyond that, but its contributions are not
  reweighted by coalition size, which some prefer for robustness (Wang & Jia,
  "Data Banzhaf", AISTATS 2023).
* Shapley interaction values (Grabisch & Roubens 1999; Lundberg et al.,
  Nature MI 2020): for ``i != j``

  ``Phi_ij = sum_{S ⊆ N∖{i,j}} |S|! (M - |S| - 2)! / (2 (M - 1)!) ∇_ij(S)``

  with ``∇_ij(S) = v(S ∪ ij) - v(S ∪ i) - v(S ∪ j) + v(S)``, and
  ``Phi_ii = phi_i - sum_{j != i} Phi_ij``, so every row sums to ``phi_i`` and
  the whole matrix sums to ``f(x) - v(∅)``. Off-diagonal entries split each
  pairwise interaction equally between the two features.

All ``2^M * n_background`` model inputs are built at once and predicted in
chunks of ``batch_size`` rows.
"""

from __future__ import annotations

from math import factorial

import numpy as np

from ._utils import as_2d

__all__ = [
    "coalition_values",
    "exact_shapley",
    "banzhaf_values",
    "shapley_interaction_values",
]

MAX_FEATURES = 16


def _prepare(x_row, background, feature_names, max_features):
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background, name="background")
    m = background.shape[1]
    if x_row.shape[0] != m:
        raise ValueError("x_row and background must have the same number of features")
    if background.shape[0] < 1:
        raise ValueError("background must contain at least one row")
    if not np.all(np.isfinite(x_row)):
        raise ValueError("x_row must be finite")
    limit = min(int(max_features), MAX_FEATURES)
    if m > limit:
        raise ValueError(
            "exact enumeration needs 2^M coalitions; %d features exceeds max_features=%d "
            "(use sampling_shapley or kernel_shap instead)" % (m, limit)
        )
    if feature_names is None:
        feature_names = ["X%d" % j for j in range(m)]
    elif len(feature_names) != m:
        raise ValueError("feature_names must have one entry per feature")
    return x_row, background, list(feature_names)


def coalition_values(predict, x_row, background, batch_size=65536, max_features=12):
    """``v(S)`` for every coalition, indexed by bitmask (bit ``j`` = feature ``j`` present).

    Returns ``(values, n_evaluations)`` where ``values`` has length ``2^M``;
    ``values[0]`` is the mean background prediction and ``values[-1]`` is
    ``f(x_row)``.
    """

    x_row, background, _ = _prepare(x_row, background, None, max_features)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    m = x_row.shape[0]
    n_bg = background.shape[0]
    n_coal = 1 << m
    masks = ((np.arange(n_coal)[:, None] >> np.arange(m)) & 1).astype(bool)
    out = np.empty(n_coal, dtype=float)
    per_chunk = max(1, int(batch_size) // n_bg)
    for start in range(0, n_coal, per_chunk):
        block = masks[start : start + per_chunk]
        rows = np.where(block[:, None, :], x_row[None, None, :], background[None, :, :])
        preds = np.asarray(predict(rows.reshape(-1, m)), dtype=float).ravel()
        if preds.shape[0] != rows.shape[0] * n_bg:
            raise ValueError("predict must return one value per input row")
        out[start : start + block.shape[0]] = preds.reshape(block.shape[0], n_bg).mean(axis=1)
    if not np.all(np.isfinite(out)):
        raise ValueError("predict returned non-finite values")
    return out, int(n_coal * n_bg)


def _popcounts(m):
    idx = np.arange(1 << m)
    return ((idx[:, None] >> np.arange(m)) & 1).sum(axis=1)


def _marginal_values(v, m, weights_by_size):
    sizes = _popcounts(m)
    idx = np.arange(1 << m)
    values = np.zeros(m)
    for i in range(m):
        bit = 1 << i
        without = idx[(idx & bit) == 0]
        values[i] = np.sum(weights_by_size[sizes[without]] * (v[without | bit] - v[without]))
    return values


def _result(values, v, n_eval, names, extra=None):
    out = {
        "values": values,
        "baseline": float(v[0]),
        "prediction": float(v[-1]),
        "n_evaluations": n_eval,
        "feature_names": names,
    }
    if extra:
        out.update(extra)
    return out


def exact_shapley(predict, x_row, background, feature_names=None, batch_size=65536, max_features=12):
    """Exact interventional Shapley values of ``predict`` at ``x_row``.

    Returns ``{"values", "baseline", "prediction", "n_evaluations",
    "feature_names"}``; ``values.sum() == prediction - baseline`` to rounding.
    """

    x_row, background, names = _prepare(x_row, background, feature_names, max_features)
    m = x_row.shape[0]
    v, n_eval = coalition_values(predict, x_row, background, batch_size, max_features)
    w = np.array([factorial(s) * factorial(m - s - 1) / factorial(m) for s in range(m)] + [0.0])
    return _result(_marginal_values(v, m, w), v, n_eval, names)


def banzhaf_values(predict, x_row, background, feature_names=None, batch_size=65536, max_features=12):
    """Exact Banzhaf values (uniform average of marginal contributions).

    Also reports ``"sum"`` and ``"efficiency_gap" = prediction - baseline - sum``,
    which is generally non-zero when features interact.
    """

    x_row, background, names = _prepare(x_row, background, feature_names, max_features)
    m = x_row.shape[0]
    v, n_eval = coalition_values(predict, x_row, background, batch_size, max_features)
    w = np.full(m + 1, 1.0 / (1 << (m - 1)))
    values = _marginal_values(v, m, w)
    total = float(values.sum())
    return _result(values, v, n_eval, names,
                   {"sum": total, "efficiency_gap": float(v[-1] - v[0] - total)})


def shapley_interaction_values(
    predict, x_row, background, feature_names=None, batch_size=65536, max_features=12
):
    """Exact Shapley interaction value matrix (``M x M``).

    ``interactions[i, j]`` (``i != j``) is half of the pairwise Shapley
    interaction index of ``i`` and ``j``; the diagonal holds the main effects
    ``phi_i - sum_{j != i} Phi_ij``. Rows sum to the Shapley values, which are
    also returned, and the matrix sums to ``prediction - baseline``.
    ``"top_interactions"`` lists feature pairs by ``|2 * Phi_ij|``.
    """

    x_row, background, names = _prepare(x_row, background, feature_names, max_features)
    m = x_row.shape[0]
    v, n_eval = coalition_values(predict, x_row, background, batch_size, max_features)
    w_phi = np.array([factorial(s) * factorial(m - s - 1) / factorial(m) for s in range(m)] + [0.0])
    phi = _marginal_values(v, m, w_phi)
    inter = np.zeros((m, m))
    if m >= 2:
        sizes = _popcounts(m)
        idx = np.arange(1 << m)
        w = np.array(
            [factorial(s) * factorial(m - s - 2) / (2.0 * factorial(m - 1)) for s in range(m - 1)]
            + [0.0, 0.0]
        )
        for i in range(m):
            bi = 1 << i
            for j in range(i + 1, m):
                bj = 1 << j
                rest = idx[((idx & bi) == 0) & ((idx & bj) == 0)]
                delta = v[rest | bi | bj] - v[rest | bi] - v[rest | bj] + v[rest]
                inter[i, j] = inter[j, i] = np.sum(w[sizes[rest]] * delta)
    for i in range(m):
        inter[i, i] = phi[i] - (inter[i].sum() - inter[i, i])
    pairs = [
        (names[i], names[j], float(2.0 * inter[i, j]))
        for i in range(m) for j in range(i + 1, m)
    ]
    pairs.sort(key=lambda t: -abs(t[2]))
    return _result(phi, v, n_eval, names,
                   {"interactions": inter, "main_effects": np.diag(inter).copy(),
                    "top_interactions": pairs})
