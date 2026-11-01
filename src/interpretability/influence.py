"""Influence functions for L2-regularised linear and logistic regression.

Data valuation by retraining (``data_valuation``) costs one fit per training
point or subset. *Influence functions* (Cook & Weisberg, 1982; Koh & Liang,
"Understanding Black-box Predictions via Influence Functions", ICML 2017)
answer "what happens if training point ``i`` is removed?" from the fitted
model alone, using one Hessian solve.

The model minimises the summed objective

``L(theta) = sum_i l(z_i, theta) + (l2 / 2) * ||w||^2``

where ``theta = (w, b)`` and the intercept ``b`` is not penalised. The loss
``l`` is the squared error ``(y - x.theta)^2 / 2`` (ridge) or the logistic
loss ``log(1 + exp(f)) - y f`` with ``y`` in ``{0, 1}``. With ``H`` the
Hessian of ``L`` at the optimum and ``g_i`` the gradient of ``l(z_i)``:

* the first-order *influence* estimate of the parameter change when ``z_i``
  is removed is ``theta_{-i} - theta ~ H^{-1} g_i``;
* the *Newton-step* estimate (Pregibon, 1981) uses the Hessian without the
  point, ``(H - H_i)^{-1} g_i``. For ridge this is the exact
  leave-one-out solution, ``-H^{-1} x_i r_i / (1 - h_i)``, and for logistic
  regression it is a one-step approximation to it that is usually very close;
* the change in the loss on a test point is ``grad l(z_test) . delta_i``.
  Koh & Liang's ``I_up,loss`` is the same quantity for *up*-weighting, with
  the sign flipped;
* *self-influence* ``g_i^T H^{-1} g_i`` flags points that the model fits only
  by bending around them (outliers, mislabelled rows).

:func:`loo_refit` refits the model ``n`` times. It is the ground truth that
the estimates are checked against.
"""

from __future__ import annotations

import numpy as np

from ._utils import as_1d, as_2d

__all__ = ["fit_regularized_glm", "influence_functions", "loo_refit"]

_LOSSES = ("squared", "logistic")


def _design(X, fit_intercept):
    return np.hstack([X, np.ones((X.shape[0], 1))]) if fit_intercept else X


def _penalty(p, l2, fit_intercept):
    d = np.full(p, float(l2))
    if fit_intercept:
        d[-1] = 0.0
    return np.diag(d)


def _sigmoid(f):
    out = np.empty_like(f)
    pos = f >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-f[pos]))
    e = np.exp(f[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _check(X, y, loss, l2):
    if loss not in _LOSSES:
        raise ValueError("loss must be one of %s" % (_LOSSES,))
    X = as_2d(X)
    y = as_1d(y, X.shape[0])
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    if loss == "logistic":
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("logistic loss needs labels in {0, 1}")
        if l2 <= 0:
            raise ValueError("logistic loss needs l2 > 0 (the unpenalised optimum may not exist)")
    return X, y


def _solve_theta(Z, y, loss, P, max_iter=100, tol=1e-10):
    if loss == "squared":
        return np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    theta = np.zeros(Z.shape[1])
    for _ in range(max_iter):
        p = _sigmoid(Z @ theta)
        grad = Z.T @ (p - y) + P @ theta
        H = (Z * (p * (1 - p))[:, None]).T @ Z + P
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.max(np.abs(step)) < tol:
            break
    return theta


def fit_regularized_glm(X, y, loss="squared", l2=1.0, fit_intercept=True):
    """Fit ridge (``loss="squared"``) or L2 logistic regression.

    Returns the parameter vector ``theta``, with the intercept last when
    ``fit_intercept`` is set.
    """
    X, y = _check(X, y, loss, l2)
    Z = _design(X, fit_intercept)
    return _solve_theta(Z, y, loss, _penalty(Z.shape[1], l2, fit_intercept))


def _point_terms(Z, y, theta, loss):
    """Per-row loss, gradient (n, p) and Hessian weight (n,)."""
    f = Z @ theta
    if loss == "squared":
        r = y - f
        return 0.5 * r ** 2, -r[:, None] * Z, np.ones_like(f)
    p = _sigmoid(f)
    losses = np.logaddexp(0.0, f) - y * f
    return losses, (p - y)[:, None] * Z, p * (1 - p)


def influence_functions(X, y, loss="squared", l2=1.0, X_test=None, y_test=None,
                        fit_intercept=True, method="influence"):
    """Estimate the effect of removing each training row without refitting.

    Parameters
    ----------
    X, y : training data (``y`` in ``{0, 1}`` for ``loss="logistic"``).
    loss : ``"squared"`` (ridge) or ``"logistic"``.
    l2 : ridge penalty on the weights; the intercept is not penalised.
    X_test, y_test : optional evaluation set. When given, the result has the
        estimated change in each test loss when each training row is removed.
    method : ``"influence"`` (first order, ``H^{-1} g_i``) or ``"newton"``
        (leave-one-out Hessian; exact for ridge).

    Returns
    -------
    dict with keys

    ``theta``
        fitted parameters (intercept last).
    ``param_change``
        ``(n, p)`` estimated ``theta_{-i} - theta``.
    ``self_influence``
        ``g_i^T H^{-1} g_i`` per row.
    ``leverage``
        ``w_i x_i^T H^{-1} x_i``, with ``w_i = 1`` (ridge) or ``p_i (1 - p_i)``.
    ``train_loss``
        per-row training loss at ``theta``.
    ``cooks_distance``
        (ridge only) ``r_i^2 h_i / (p s^2 (1 - h_i)^2)``.
    ``test_loss_change``
        ``(n, m)`` estimated change in each test loss when row ``i`` is removed.
        Only present when a test set is given.
    ``mean_test_loss_change``
        mean over the test set. Positive means the row *helps* (removing it
        raises the test loss); negative means it hurts.
    """
    if method not in ("influence", "newton"):
        raise ValueError("method must be 'influence' or 'newton'")
    X, y = _check(X, y, loss, l2)
    Z = _design(X, fit_intercept)
    n, p = Z.shape
    P = _penalty(p, l2, fit_intercept)
    theta = _solve_theta(Z, y, loss, P)
    losses, G, w = _point_terms(Z, y, theta, loss)
    H = (Z * w[:, None]).T @ Z + P
    try:
        Hinv_G = np.linalg.solve(H, G.T).T  # (n, p)
        Hinv_Z = np.linalg.solve(H, Z.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("Hessian is singular; increase l2") from exc
    self_infl = np.einsum("ij,ij->i", G, Hinv_G)
    lev = w * np.einsum("ij,ij->i", Z, Hinv_Z)
    if method == "influence":
        delta = Hinv_G
    else:
        # Sherman-Morrison on H - w_i z_i z_i^T:
        # (H - w z z^T)^{-1} g = H^{-1} g + w H^{-1} z (z^T H^{-1} g) / (1 - w z^T H^{-1} z)
        if np.any(lev >= 1 - 1e-12):
            raise ValueError("a row has leverage 1; its leave-one-out fit is not identified")
        zHg = np.einsum("ij,ij->i", Z, Hinv_G)
        delta = Hinv_G + (w * zHg / (1 - lev))[:, None] * Hinv_Z
    out = {
        "theta": theta,
        "param_change": delta,
        "self_influence": self_infl,
        "leverage": lev,
        "train_loss": losses,
        "method": method,
        "loss": loss,
    }
    if loss == "squared":
        dof = n - p
        resid2 = 2 * losses
        s2 = resid2.sum() / dof if dof > 0 else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            out["cooks_distance"] = resid2 * lev / (p * s2 * (1 - lev) ** 2)
    if X_test is not None:
        if y_test is None:
            raise ValueError("y_test is required with X_test")
        Xt = as_2d(X_test, name="X_test")
        if Xt.shape[1] != X.shape[1]:
            raise ValueError("X_test must have the same number of columns as X")
        yt = as_1d(y_test, Xt.shape[0], name="y_test")
        _, Gt, _ = _point_terms(_design(Xt, fit_intercept), yt, theta, loss)
        change = delta @ Gt.T
        out["test_loss_change"] = change
        out["mean_test_loss_change"] = change.mean(axis=1)
    return out


def loo_refit(X, y, loss="squared", l2=1.0, X_test=None, y_test=None, fit_intercept=True):
    """Exact leave-one-out by refitting ``n`` times (ground truth for checks).

    Returns ``theta``, ``param_change`` ``(n, p)`` and, with a test set,
    ``test_loss_change`` ``(n, m)`` (exact change in test loss, not linearised)
    and ``mean_test_loss_change``.
    """
    X, y = _check(X, y, loss, l2)
    Z = _design(X, fit_intercept)
    n, p = Z.shape
    P = _penalty(p, l2, fit_intercept)
    theta = _solve_theta(Z, y, loss, P)
    mask = np.ones(n, dtype=bool)
    thetas = np.empty((n, p))
    for i in range(n):
        mask[i] = False
        thetas[i] = _solve_theta(Z[mask], y[mask], loss, P)
        mask[i] = True
    out = {"theta": theta, "param_change": thetas - theta}
    if X_test is not None:
        if y_test is None:
            raise ValueError("y_test is required with X_test")
        Xt = as_2d(X_test, name="X_test")
        yt = as_1d(y_test, Xt.shape[0], name="y_test")
        Zt = _design(Xt, fit_intercept)
        base, _, _ = _point_terms(Zt, yt, theta, loss)
        change = np.array([_point_terms(Zt, yt, t, loss)[0] - base for t in thetas])
        out["test_loss_change"] = change
        out["mean_test_loss_change"] = change.mean(axis=1)
    return out
