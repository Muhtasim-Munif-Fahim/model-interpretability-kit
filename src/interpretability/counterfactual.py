"""Counterfactual explanations with Growing Spheres (Laugel et al., 2018).

A counterfactual explanation answers "what is the smallest change to this
input that would change the model's decision?" -- e.g. *the loan would be
approved if income were 4,000 higher*. Growing Spheres (Laugel, Lesot,
Marsala, Renard and Detyniecki, "Inverse Classification for Comparison-based
Interpretability in Machine Learning", 2018) finds one for any black-box
``predict`` without gradients:

1. **Generation.** Work in a standardized space ``z = (x - x0) / scale``.
   Sample points uniformly inside a ball of radius ``eta`` around the
   instance; while that ball already contains "enemies" (points whose
   prediction meets the target), halve ``eta``. Then grow outward through
   spherical layers ``[r, r + eta]``, sampling each uniformly, until a layer
   contains an enemy. The closest enemy (L2 in the standardized space) is the
   raw counterfactual.
2. **Feature selection.** Sparsify it: try resetting each changed feature to
   its original value, smallest change first, and keep the reset whenever the
   prediction still meets the target. Fewer changed features means a more
   actionable explanation.

Immutable features (age, protected attributes) are excluded from the search
space, and candidates are clipped to per-feature bounds (by default the
range observed in the reference data). The target is a prediction interval
(regression or score thresholds), a class label, or any callable predicate.
"""

import numpy as np

from ._utils import as_2d

__all__ = ["growing_spheres_counterfactual", "sample_spherical_layer"]


def sample_spherical_layer(n, dim, r_low, r_high, rng):
    """Draw ``n`` points uniformly from the shell ``r_low <= ||z|| <= r_high``.

    Directions are normalized Gaussians; radii use the inverse CDF of the
    volume ``r^dim`` so the density is uniform in the shell, not
    concentrated near its inner surface.
    """
    if r_low < 0 or r_high < r_low:
        raise ValueError("need 0 <= r_low <= r_high")
    direction = rng.normal(size=(n, dim))
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    direction /= norms
    u = rng.uniform(size=(n, 1))
    radius = (u * (r_high**dim - r_low**dim) + r_low**dim) ** (1.0 / dim)
    return direction * radius


def _target_predicate(target_range, target_class, target):
    given = sum(v is not None for v in (target_range, target_class, target))
    if given != 1:
        raise ValueError("provide exactly one of target_range, target_class or target")
    if target is not None:
        if not callable(target):
            raise TypeError("target must be a callable predicate on predictions")
        return lambda pred: np.asarray(target(pred), dtype=bool).ravel()
    if target_class is not None:
        return lambda pred: np.asarray(pred).ravel() == target_class
    low, high = target_range
    low = -np.inf if low is None else float(low)
    high = np.inf if high is None else float(high)
    if low > high:
        raise ValueError("target_range lower bound exceeds upper bound")

    def in_range(pred):
        p = np.asarray(pred, dtype=float).ravel()
        return (p >= low) & (p <= high)

    return in_range


def growing_spheres_counterfactual(
    predict,
    x,
    X_reference,
    *,
    target_range=None,
    target_class=None,
    target=None,
    immutable=None,
    bounds="reference",
    n_samples=500,
    eta=None,
    max_layers=200,
    sparsify=True,
    seed=None,
    feature_names=None,
):
    """Find a sparse, nearby counterfactual for one instance.

    Parameters
    ----------
    predict:
        Callable ``predict(X) -> predictions`` on 2-D arrays.
    x:
        The instance to explain (1-D).
    X_reference:
        Reference data used for per-feature scales (standard deviations) and,
        with ``bounds="reference"``, the feasible ranges.
    target_range, target_class, target:
        Exactly one: an inclusive ``(low, high)`` prediction interval (either
        end may be ``None``), a class label, or a callable that maps a
        prediction array to booleans.
    immutable:
        Feature indices that must not change.
    bounds:
        ``"reference"`` (min/max of ``X_reference``), ``None`` (unbounded), or
        an array of shape ``(n_features, 2)``.
    n_samples:
        Points sampled per ball / layer.
    eta:
        Initial radius in standardized units (default ``1.0``).
    max_layers:
        Maximum number of radius halvings plus growth layers.
    sparsify:
        Run the feature-selection step.

    Returns
    -------
    dict with ``found``, ``counterfactual``, ``original``,
    ``prediction_original``, ``prediction_counterfactual``, ``delta``,
    ``changed_features`` (indices), ``changed_feature_names``,
    ``distance_l2`` (standardized), ``n_changed``, ``radius`` and
    ``n_evaluations``.
    """
    X_reference = as_2d(X_reference, "X_reference")
    x = np.asarray(x, dtype=float).ravel()
    d = X_reference.shape[1]
    if x.shape[0] != d:
        raise ValueError("x must have as many features as X_reference")
    if isinstance(n_samples, bool) or int(n_samples) < 1:
        raise ValueError("n_samples must be a positive integer")
    if isinstance(max_layers, bool) or int(max_layers) < 1:
        raise ValueError("max_layers must be a positive integer")
    eta = 1.0 if eta is None else float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be a positive number")
    is_target = _target_predicate(target_range, target_class, target)
    rng = np.random.default_rng(seed)

    scale = X_reference.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    frozen = np.zeros(d, dtype=bool)
    if immutable is not None:
        for j in np.atleast_1d(immutable):
            j = int(j)
            if j < 0 or j >= d:
                raise ValueError("immutable feature index %d out of range" % j)
            frozen[j] = True
    mutable = np.flatnonzero(~frozen)
    if mutable.size == 0:
        raise ValueError("at least one feature must be mutable")

    if isinstance(bounds, str):
        if bounds != "reference":
            raise ValueError("bounds must be 'reference', None or an (n_features, 2) array")
        lo, hi = X_reference.min(axis=0), X_reference.max(axis=0)
        lo, hi = np.minimum(lo, x), np.maximum(hi, x)
    elif bounds is None:
        lo, hi = np.full(d, -np.inf), np.full(d, np.inf)
    else:
        b = np.asarray(bounds, dtype=float)
        if b.shape != (d, 2) or np.any(b[:, 0] > b[:, 1]):
            raise ValueError("bounds must have shape (n_features, 2) with low <= high")
        lo, hi = b[:, 0], b[:, 1]

    n_eval = 0

    def evaluate(points):
        nonlocal n_eval
        n_eval += points.shape[0]
        return np.asarray(predict(points)).ravel()

    def candidates(r_low, r_high):
        z = sample_spherical_layer(int(n_samples), mutable.size, r_low, r_high, rng)
        pts = np.repeat(x[None, :], z.shape[0], axis=0)
        pts[:, mutable] = x[mutable] + z * scale[mutable]
        return np.clip(pts, lo, hi)

    def std_dist(points):
        return np.linalg.norm((points - x) / scale, axis=1)

    names = list(feature_names) if feature_names is not None else ["x%d" % j for j in range(d)]
    pred_x = evaluate(x[None, :])
    base = {
        "original": x.copy(),
        "prediction_original": pred_x[0].item() if hasattr(pred_x[0], "item") else pred_x[0],
    }
    if is_target(pred_x)[0]:
        return {
            **base,
            "found": True,
            "counterfactual": x.copy(),
            "prediction_counterfactual": base["prediction_original"],
            "delta": np.zeros(d),
            "changed_features": [],
            "changed_feature_names": [],
            "distance_l2": 0.0,
            "n_changed": 0,
            "radius": 0.0,
            "n_evaluations": n_eval,
        }

    # Step 1a: shrink the initial ball until it holds no enemies.
    layers = 0
    radius = eta
    while layers < int(max_layers):
        pts = candidates(0.0, radius)
        hits = is_target(evaluate(pts))
        layers += 1
        if not hits.any():
            break
        radius /= 2.0
    # Step 1b: grow outward layer by layer until an enemy appears.
    enemy = None
    r_low = radius
    while layers < int(max_layers):
        pts = candidates(r_low, r_low + radius)
        hits = is_target(evaluate(pts))
        layers += 1
        if hits.any():
            enemies = pts[hits]
            enemy = enemies[np.argmin(std_dist(enemies))]
            break
        r_low += radius
    if enemy is None:
        return {
            **base,
            "found": False,
            "counterfactual": None,
            "prediction_counterfactual": None,
            "delta": None,
            "changed_features": [],
            "changed_feature_names": [],
            "distance_l2": None,
            "n_changed": None,
            "radius": r_low,
            "n_evaluations": n_eval,
        }

    # Step 2: feature selection, smallest standardized change first.
    cf = enemy.copy()
    if sparsify:
        order = np.argsort(np.abs((cf - x) / scale))
        for j in order:
            if cf[j] == x[j]:
                continue
            trial = cf.copy()
            trial[j] = x[j]
            if is_target(evaluate(trial[None, :]))[0]:
                cf = trial
    pred_cf = evaluate(cf[None, :])[0]
    delta = cf - x
    changed = [int(j) for j in np.flatnonzero(delta != 0)]
    return {
        **base,
        "found": True,
        "counterfactual": cf,
        "prediction_counterfactual": pred_cf.item() if hasattr(pred_cf, "item") else pred_cf,
        "delta": delta,
        "changed_features": changed,
        "changed_feature_names": [names[j] for j in changed],
        "distance_l2": float(std_dist(cf[None, :])[0]),
        "n_changed": len(changed),
        "radius": float(r_low + radius),
        "n_evaluations": n_eval,
    }
