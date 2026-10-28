"""Quantitative quality metrics for local feature attributions.

Different attribution methods (Kernel SHAP, sampling Shapley, Integrated
Gradients, SmoothGrad, ...) often disagree on the same instance. These metrics
score an attribution vector ``phi`` for one instance ``x`` without ground
truth:

* :func:`deletion_insertion` -- MoRF deletion / insertion curves, their
  normalised areas and AOPC (Samek et al., 2017; Petsiuk et al., 2018).
* :func:`attribution_infidelity` -- the explanation infidelity of
  Yeh et al. (NeurIPS 2019): ``E_I[(I^T phi - (f(x) - f(x - I)))^2]`` under a
  Gaussian (gradient-style) or subset-removal (additive-style) perturbation,
  plus the optimally rescaled variant.
* :func:`attribution_sensitivity` -- max-/average-sensitivity (Yeh et al.,
  2019) and a local Lipschitz estimate (Alvarez-Melis & Jaakkola, 2018) of
  the explainer itself under small input perturbations.

Every function treats the model as a plain ``predict(X) -> y`` callable and
the explainer (for sensitivity) as ``explain(x_row) -> phi``.
"""

import numpy as np

__all__ = [
    "deletion_insertion",
    "attribution_infidelity",
    "attribution_sensitivity",
]


def _check_row(x_row, phi):
    x_row = np.asarray(x_row, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    if x_row.size == 0:
        raise ValueError("x_row must not be empty")
    if phi.shape != x_row.shape:
        raise ValueError("attributions must have one value per feature")
    if not np.all(np.isfinite(phi)):
        raise ValueError("attributions must be finite")
    return x_row, phi


def _resolve_baseline(baseline, x_row):
    if baseline is None:
        return np.zeros_like(x_row)
    baseline = np.asarray(baseline, dtype=float)
    if baseline.ndim == 2:
        baseline = baseline.mean(axis=0)
    baseline = baseline.ravel()
    if baseline.shape != x_row.shape:
        raise ValueError("baseline must have one value per feature (or be a 2-D background)")
    return baseline


def _area(y):
    """Trapezoidal area of ``y`` over an evenly spaced grid on ``[0, 1]``."""
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return float(y[0]) if y.size else 0.0
    return float(np.sum((y[1:] + y[:-1]) / 2.0) / (y.size - 1))


def deletion_insertion(predict, x_row, attributions, baseline=None, order="signed"):
    """Deletion and insertion curves for a local attribution.

    Features are ranked most-relevant-first (MoRF). The *deletion* curve
    starts at ``x`` and replaces the top-ranked features with ``baseline``
    one at a time; a faithful attribution makes the prediction fall fast
    (small deletion area). The *insertion* curve starts at the baseline and
    inserts the top-ranked features from ``x``; a faithful attribution makes
    the prediction rise fast (large insertion area).

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y`` for a 2-D array.
    x_row : array-like, shape (d,)
        Instance being explained.
    attributions : array-like, shape (d,)
        Attribution per feature.
    baseline : array-like, optional
        Reference values: a vector of length ``d`` or a 2-D background whose
        column means are used. Defaults to zeros.
    order : {"signed", "abs"}
        ``"signed"`` ranks by attribution value (positive evidence first, the
        AOPC convention); ``"abs"`` ranks by magnitude.

    Returns
    -------
    dict
        ``order`` (feature ranking), ``deletion`` and ``insertion`` curves
        (length ``d + 1``), their normalised areas ``deletion_auc`` /
        ``insertion_auc`` on the fraction-of-features axis, ``aopc`` (mean
        drop ``f(x) - f(x_k)`` over ``k = 1..d``), ``prediction`` and
        ``baseline_prediction``.
    """
    x_row, phi = _check_row(x_row, attributions)
    base = _resolve_baseline(baseline, x_row)
    if order == "signed":
        ranking = np.argsort(-phi, kind="stable")
    elif order == "abs":
        ranking = np.argsort(-np.abs(phi), kind="stable")
    else:
        raise ValueError("order must be 'signed' or 'abs'")
    d = x_row.size
    deleted = np.tile(x_row, (d + 1, 1))
    inserted = np.tile(base, (d + 1, 1))
    for k in range(1, d + 1):
        top = ranking[:k]
        deleted[k, top] = base[top]
        inserted[k, top] = x_row[top]
    del_curve = np.asarray(predict(deleted), dtype=float).ravel()
    ins_curve = np.asarray(predict(inserted), dtype=float).ravel()
    if del_curve.shape[0] != d + 1 or ins_curve.shape[0] != d + 1:
        raise ValueError("predict must return one value per row")
    prediction = float(del_curve[0])
    return {
        "order": [int(j) for j in ranking],
        "deletion": del_curve,
        "insertion": ins_curve,
        "deletion_auc": _area(del_curve),
        "insertion_auc": _area(ins_curve),
        "aopc": float(np.mean(prediction - del_curve[1:])),
        "prediction": prediction,
        "baseline_prediction": float(ins_curve[0]),
    }


def attribution_infidelity(
    predict,
    x_row,
    attributions,
    baseline=None,
    perturbation="subset",
    n_samples=200,
    noise_sigma=0.1,
    subset_prob=0.5,
    seed=None,
):
    """Explanation infidelity of Yeh et al. (NeurIPS 2019).

    ``INFD(phi, f, x) = E_I[(I^T phi - (f(x) - f(x - I)))^2]``.

    Two perturbation families are supported:

    * ``"gaussian"``: ``I ~ N(0, noise_sigma^2 I_d)``. Suited to gradient-type
      attributions (SmoothGrad, raw gradients); for a linear model the
      gradient has zero infidelity.
    * ``"subset"``: a random subset ``S`` of features is switched to the
      baseline, ``x - I = x_S<-baseline``, and ``I^T phi`` is read as
      ``sum_{j in S} phi_j``. Suited to additive attributions that satisfy
      completeness relative to a baseline (Shapley values, Integrated
      Gradients); for a linear model ``phi_j = w_j (x_j - b_j)`` has zero
      infidelity. Each feature joins ``S`` independently with probability
      ``subset_prob`` (empty draws are resampled).

    Returns
    -------
    dict
        ``infidelity`` (mean squared error), ``scaled_infidelity`` (after the
        optimal rescaling ``beta * phi`` of Yeh et al., which removes the
        penalty for attributions that are right up to a constant factor),
        ``beta``, ``perturbation`` and ``n_samples``.
    """
    x_row, phi = _check_row(x_row, attributions)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    rng = np.random.default_rng(seed)
    d = x_row.size
    if perturbation == "gaussian":
        if noise_sigma <= 0:
            raise ValueError("noise_sigma must be positive")
        noise = rng.normal(scale=noise_sigma, size=(n_samples, d))
        perturbed = x_row[None, :] - noise
        explained = noise @ phi
    elif perturbation == "subset":
        if not 0.0 < subset_prob <= 1.0:
            raise ValueError("subset_prob must be in (0, 1]")
        base = _resolve_baseline(baseline, x_row)
        masks = rng.random((n_samples, d)) < subset_prob
        empty = ~masks.any(axis=1)
        while empty.any():
            masks[empty] = rng.random((int(empty.sum()), d)) < subset_prob
            empty = ~masks.any(axis=1)
        perturbed = np.where(masks, base[None, :], x_row[None, :])
        explained = masks.astype(float) @ phi
    else:
        raise ValueError("perturbation must be 'gaussian' or 'subset'")
    f_x = float(np.asarray(predict(x_row[None, :]), dtype=float).ravel()[0])
    f_pert = np.asarray(predict(perturbed), dtype=float).ravel()
    if f_pert.shape[0] != n_samples:
        raise ValueError("predict must return one value per row")
    delta = f_x - f_pert
    infd = float(np.mean((explained - delta) ** 2))
    denom = float(np.mean(explained ** 2))
    beta = float(np.mean(explained * delta) / denom) if denom > 0 else 0.0
    scaled = float(np.mean((beta * explained - delta) ** 2))
    return {
        "infidelity": infd,
        "scaled_infidelity": scaled,
        "beta": beta,
        "perturbation": perturbation,
        "n_samples": int(n_samples),
    }


def attribution_sensitivity(
    explain,
    x_row,
    radius=0.1,
    n_samples=20,
    scale=None,
    normalize=False,
    seed=None,
):
    """Max-/average-sensitivity and local Lipschitz estimate of an explainer.

    Draws ``n_samples`` points ``x'`` uniformly from the L-infinity ball of
    radius ``radius`` around ``x`` (per-feature radii ``radius * scale`` when
    ``scale`` is given, e.g. the feature standard deviations) and recomputes
    the explanation. Robust explainers change little:

    * ``max_sensitivity = max ||phi(x') - phi(x)||_2`` (Yeh et al., 2019)
    * ``avg_sensitivity = mean ||phi(x') - phi(x)||_2``
    * ``lipschitz = max ||phi(x') - phi(x)||_2 / ||x' - x||_2``
      (Alvarez-Melis & Jaakkola, 2018)

    Parameters
    ----------
    explain : callable
        ``explain(x_row) -> phi`` returning a length-``d`` attribution (a
        dict with a ``"values"`` key, as returned by the explainers in
        :mod:`interpretability.local`, is also accepted).
    normalize : bool
        If true, sensitivities are divided by ``||phi(x)||_2``.

    Returns
    -------
    dict
        ``max_sensitivity``, ``avg_sensitivity``, ``lipschitz``,
        ``attribution`` (``phi(x)``), ``distances`` (per-sample
        ``||phi(x') - phi(x)||``) and ``n_samples``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    d = x_row.size
    if scale is None:
        widths = np.full(d, float(radius))
    else:
        scale = np.asarray(scale, dtype=float).ravel()
        if scale.shape != (d,) or np.any(scale < 0):
            raise ValueError("scale must be a non-negative vector with one value per feature")
        widths = radius * scale

    def _phi(row):
        out = explain(row)
        if isinstance(out, dict):
            out = out["values"]
        out = np.asarray(out, dtype=float).ravel()
        if out.shape != (d,):
            raise ValueError("explain must return one attribution per feature")
        return out

    rng = np.random.default_rng(seed)
    phi0 = _phi(x_row)
    norm0 = float(np.linalg.norm(phi0))
    dists = np.empty(n_samples)
    ratios = np.zeros(n_samples)
    for s in range(n_samples):
        step = rng.uniform(-1.0, 1.0, size=d) * widths
        diff = float(np.linalg.norm(_phi(x_row + step) - phi0))
        dists[s] = diff
        step_norm = float(np.linalg.norm(step))
        if step_norm > 0:
            ratios[s] = diff / step_norm
    if normalize:
        if norm0 == 0:
            raise ValueError("cannot normalize: attribution at x_row is all zeros")
        dists = dists / norm0
    return {
        "max_sensitivity": float(dists.max()),
        "avg_sensitivity": float(dists.mean()),
        "lipschitz": float(ratios.max()),
        "attribution": phi0,
        "distances": dists,
        "n_samples": int(n_samples),
    }
