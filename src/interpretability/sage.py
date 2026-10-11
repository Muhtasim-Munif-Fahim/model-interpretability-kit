"""SAGE: Shapley Additive Global importancE (Covert, Lundberg and Lee, 2020).

SAGE assigns each feature a share of the model's *predictive power*: the
reduction in expected loss that the features provide over predicting with
no information. For a feature subset ``S`` the cooperative game is

    v(S) = E_{x,y}[ loss(y, f_S(x_S)) ],   f_S(x_S) = E_z[ f(x_S, z_~S) ]

with "absent" features marginalized over a background set (marginal or
interventional imputation). The SAGE value of feature ``j`` is its
Shapley value in the game ``-v``:

    phi_j = sum_S |S|!(d-|S|-1)!/d! * (v(S) - v(S + {j}))

so ``phi_j > 0`` means feature ``j`` lowers the loss. The values satisfy
*efficiency*: ``sum_j phi_j = v(empty) - v(all)``, the loss of the
constant mean prediction minus the model's loss.

:func:`sage_values` uses the permutation estimator from the paper. Each
draw takes one evaluation row ``(x, y)`` and a random permutation. It
adds the features one at a time and credits each one with the drop in
loss it causes. The imputation expectation averages over the **whole**
background set (keep it small, e.g. 32-128 rows), so the estimate is
unbiased for that background distribution and each draw telescopes
exactly to ``loss(empty) - loss(all)`` for its row.

Unlike permutation importance, SAGE accounts for interactions and
correlated features by averaging over orderings, and it is additive.
Unlike :func:`~interpretability.local.sampling_shapley`, it explains the
*loss* over a dataset, not one prediction.
"""

import numpy as np

from ._utils import as_1d, as_2d

__all__ = ["sage_values", "sage_value_function", "SAGE_LOSSES"]


def _mse(y_true, y_pred):
    return (y_true - y_pred) ** 2


def _mae(y_true, y_pred):
    return np.abs(y_true - y_pred)


def _log_loss(y_true, y_pred):
    p = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


SAGE_LOSSES = {"mse": _mse, "mae": _mae, "log_loss": _log_loss}


def _resolve_loss(loss):
    if isinstance(loss, str):
        if loss not in SAGE_LOSSES:
            raise ValueError("loss must be one of %s or a callable" % sorted(SAGE_LOSSES))
        return SAGE_LOSSES[loss]
    if not callable(loss):
        raise TypeError("loss must be a string or a per-row callable loss(y_true, y_pred)")

    def _wrapped(y_true, y_pred):
        values = np.asarray(loss(y_true, y_pred), dtype=float).ravel()
        if values.shape[0] != np.asarray(y_true).shape[0]:
            raise ValueError("a callable SAGE loss must return one loss per row")
        return values

    return _wrapped


def _check_positive_int(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError("%s must be a positive integer" % name)
    return int(value)


def _prepare(predict, X, y, background, n_background, rng):
    X = as_2d(X)
    y = as_1d(y, X.shape[0])
    if background is None:
        k = min(X.shape[0], _check_positive_int(n_background, "n_background"))
        background = X[rng.choice(X.shape[0], size=k, replace=False)]
    else:
        background = as_2d(background, name="background")
        if background.shape[1] != X.shape[1]:
            raise ValueError("background must have the same number of features as X")
    if not callable(predict):
        raise TypeError("predict must be a callable predict(X) -> y_pred")
    return X, y, background


def _imputed_means(predict, x_rows, masks, background):
    """Mean prediction over the background with features in ``masks`` set from ``x_rows``.

    ``x_rows`` is ``(q, d)`` and ``masks`` is a boolean ``(q, d)`` array.
    Returns a length-``q`` vector.
    """
    q, d = x_rows.shape
    k = background.shape[0]
    stacked = np.broadcast_to(background, (q, k, d)).copy()
    stacked = np.where(masks[:, None, :], x_rows[:, None, :], stacked)
    preds = np.asarray(predict(stacked.reshape(q * k, d)), dtype=float).ravel()
    if preds.shape[0] != q * k:
        raise ValueError("predict must return one value per input row")
    return preds.reshape(q, k).mean(axis=1)


def sage_value_function(predict, X, y, subset, background=None, loss="mse", n_background=64, seed=None):
    """Expected loss ``v(S)`` with only the features in ``subset`` known.

    The exact (non-sampled) value of the SAGE game on the rows ``X, y``,
    with the other features marginalized over ``background``. Useful for
    small-``d`` exact checks and for reading off ``v(empty)`` / ``v(all)``.
    """
    rng = np.random.default_rng(seed)
    X, y, background = _prepare(predict, X, y, background, n_background, rng)
    loss_fn = _resolve_loss(loss)
    mask = np.zeros(X.shape[1], dtype=bool)
    for j in subset:
        if not 0 <= int(j) < X.shape[1]:
            raise ValueError("subset contains an out-of-range feature index")
        mask[int(j)] = True
    means = _imputed_means(predict, X, np.broadcast_to(mask, X.shape), background)
    return float(np.mean(loss_fn(y, means)))


def sage_values(
    predict,
    X,
    y,
    background=None,
    loss="mse",
    n_permutations=256,
    n_background=64,
    batch_size=32,
    seed=None,
    feature_names=None,
):
    """Permutation-sampling SAGE values.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred``. For ``loss="log_loss"`` it must return the
        probability of the positive class.
    X, y : array-like
        Evaluation rows and targets; each draw samples one row.
    background : ndarray or None
        Rows that supply the values of absent features. Defaults to
        ``n_background`` rows sampled from ``X`` without replacement.
    loss : {"mse", "mae", "log_loss"} or callable
        A callable must return **per-row** losses ``loss(y_true, y_pred)``.
    n_permutations : int
        Number of (row, permutation) draws.
    batch_size : int
        Draws evaluated per ``predict`` call. Each draw needs
        ``(d + 1) * len(background)`` predictions.
    seed : int or None
    feature_names : list of str or None

    Returns
    -------
    dict
        ``values`` (SAGE value per feature, in loss units), ``std_error``
        (Monte Carlo standard error per feature), ``ratio_to_total`` (each
        value divided by ``total``), ``total`` (the sum of the values, equal
        to ``loss_empty - loss_full`` over the sampled rows), ``loss_empty``
        and ``loss_full``, ``n_permutations``, ``n_evaluations`` and
        ``feature_names``.
    """
    rng = np.random.default_rng(seed)
    X, y, background = _prepare(predict, X, y, background, n_background, rng)
    loss_fn = _resolve_loss(loss)
    n_permutations = _check_positive_int(n_permutations, "n_permutations")
    batch_size = _check_positive_int(batch_size, "batch_size")
    n, d = X.shape
    if feature_names is None:
        feature_names = ["X%d" % j for j in range(d)]
    elif len(feature_names) != d:
        raise ValueError("feature_names must have one entry per feature")

    contrib = np.zeros((n_permutations, d), dtype=float)
    loss_empty = np.zeros(n_permutations, dtype=float)
    loss_full = np.zeros(n_permutations, dtype=float)
    rows = rng.integers(0, n, size=n_permutations)
    perms = np.array([rng.permutation(d) for _ in range(n_permutations)], dtype=int)
    n_evaluations = 0

    steps = np.arange(d + 1)
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        b = stop - start
        # rank[w, j] = position of feature j in permutation w; a feature is known
        # at step k when its rank is < k.
        rank = np.empty((b, d), dtype=int)
        rank[np.arange(b)[:, None], perms[start:stop]] = np.arange(d)[None, :]
        masks = rank[:, None, :] < steps[None, :, None]  # (b, d + 1, d)
        x_rows = np.repeat(X[rows[start:stop]][:, None, :], d + 1, axis=1)
        means = _imputed_means(
            predict, x_rows.reshape(b * (d + 1), d), masks.reshape(b * (d + 1), d), background
        ).reshape(b, d + 1)
        n_evaluations += b * (d + 1) * background.shape[0]
        y_rep = np.repeat(y[rows[start:stop]][:, None], d + 1, axis=1)
        losses = loss_fn(y_rep.ravel(), means.ravel()).reshape(b, d + 1)
        drops = losses[:, :-1] - losses[:, 1:]  # drops[w, k] belongs to perms[w, k]
        contrib[np.arange(start, stop)[:, None], perms[start:stop]] = drops
        loss_empty[start:stop] = losses[:, 0]
        loss_full[start:stop] = losses[:, -1]

    values = contrib.mean(axis=0)
    if n_permutations > 1:
        std_error = contrib.std(axis=0, ddof=1) / np.sqrt(n_permutations)
    else:
        std_error = np.full(d, np.nan)
    total = float(values.sum())
    ratio = values / total if total != 0 else np.full(d, np.nan)
    return {
        "values": values,
        "std_error": std_error,
        "ratio_to_total": ratio,
        "total": total,
        "loss_empty": float(loss_empty.mean()),
        "loss_full": float(loss_full.mean()),
        "n_permutations": n_permutations,
        "n_evaluations": int(n_evaluations),
        "feature_names": list(feature_names),
    }
