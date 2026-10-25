"""Global surrogate trees and rule extraction.

A *global surrogate* explains a black box by fitting an interpretable
model to the black box's own predictions (Craven & Shavlik 1996; Molnar,
*Interpretable Machine Learning*, ch. "Global Surrogate"). Its usefulness
is measured by **fidelity** — how well the surrogate reproduces the black
box, not the ground truth — reported as

* ``R^2`` of surrogate vs. black-box predictions (``1`` = perfect mimic),
* RMSE, and, for classifiers with a decision threshold, the *agreement*
  rate of the thresholded labels.

:func:`global_surrogate` fits a shallow :class:`~interpretability.demo_model.DecisionTreeRegressor`
on ``predict(X)``, optionally scoring fidelity on held-out rows, and
returns:

* **rules** — one conjunctive rule per leaf ("x1 <= 0.3 and x4 > 1.2 ->
  0.81"), with redundant bounds on the same feature merged to the tightest
  interval, plus the leaf's support and coverage;
* **importances** — impurity (SSE) reduction attributable to each feature
  in the surrogate, normalised to sum to one.

:func:`surrogate_depth_curve` traces the interpretability/fidelity
trade-off over a range of depths.
"""

import numpy as np

from ._utils import as_2d
from .demo_model import DecisionTreeRegressor

__all__ = [
    "extract_rules",
    "format_rule",
    "surrogate_fidelity",
    "surrogate_importances",
    "global_surrogate",
    "surrogate_depth_curve",
]


def _names(feature_names, n_features):
    if feature_names is None:
        return ["x%d" % j for j in range(n_features)]
    names = list(feature_names)
    if len(names) != n_features:
        raise ValueError("feature_names must have one entry per feature")
    return names


def extract_rules(tree, feature_names=None):
    """Return one rule per leaf of a fitted :class:`DecisionTreeRegressor`.

    Each rule is a dict ``{"conditions": {feature: (low, high)}, "value",
    "n", "coverage"}`` where the feature satisfies ``low < x <= high``
    (``-inf`` / ``inf`` for open sides). Multiple splits on one feature are
    merged into the tightest interval. Rules are sorted by prediction.
    """

    if not getattr(tree, "is_fitted_", False):
        raise ValueError("tree must be fitted")
    names = _names(feature_names, tree.n_features_)
    total = tree.root_["n"]
    rules = []

    def walk(node, bounds):
        if node["leaf"]:
            conds = {names[f]: (lo, hi) for f, (lo, hi) in sorted(bounds.items())}
            rules.append({
                "conditions": conds,
                "value": float(node["value"]),
                "n": int(node["n"]),
                "coverage": node["n"] / total,
            })
            return
        f, t = node["feature"], node["threshold"]
        lo, hi = bounds.get(f, (-np.inf, np.inf))
        left = dict(bounds)
        left[f] = (lo, min(hi, t))
        right = dict(bounds)
        right[f] = (max(lo, t), hi)
        walk(node["left"], left)
        walk(node["right"], right)

    walk(tree.root_, {})
    rules.sort(key=lambda r: r["value"])
    return rules


def format_rule(rule, digits=4):
    """Human-readable ``if ... then`` text for a rule from :func:`extract_rules`."""

    parts = []
    for name, (lo, hi) in rule["conditions"].items():
        if np.isfinite(lo) and np.isfinite(hi):
            parts.append("%.*g < %s <= %.*g" % (digits, lo, name, digits, hi))
        elif np.isfinite(hi):
            parts.append("%s <= %.*g" % (name, digits, hi))
        else:
            parts.append("%s > %.*g" % (name, digits, lo))
    cond = " and ".join(parts) if parts else "always"
    return "if %s then %.*g  (n=%d, %.1f%%)" % (
        cond, digits, rule["value"], rule["n"], 100 * rule["coverage"])


def surrogate_fidelity(black_box_pred, surrogate_pred, threshold=None):
    """Fidelity of surrogate predictions to black-box predictions.

    Returns ``{"r2", "rmse"}`` plus ``"agreement"`` (share of rows where both
    predictions fall on the same side of ``threshold``) when a threshold is
    given. ``r2`` is ``1`` when the black box is constant and matched exactly.
    """

    a = np.asarray(black_box_pred, dtype=float).ravel()
    b = np.asarray(surrogate_pred, dtype=float).ravel()
    if a.shape != b.shape or a.size == 0:
        raise ValueError("prediction vectors must be non-empty and of equal length")
    sse = float(np.sum((a - b) ** 2))
    sst = float(np.sum((a - a.mean()) ** 2))
    if sst == 0:
        r2 = 1.0 if sse == 0 else 0.0
    else:
        r2 = 1.0 - sse / sst
    out = {"r2": r2, "rmse": float(np.sqrt(sse / a.size))}
    if threshold is not None:
        out["agreement"] = float(np.mean((a > threshold) == (b > threshold)))
    return out


def surrogate_importances(tree, X, target):
    """SSE reduction per feature of a fitted tree on ``(X, target)``, summing to one."""

    X = as_2d(X)
    target = np.asarray(target, dtype=float).ravel()
    gains = np.zeros(tree.n_features_)

    def sse(v):
        return float(np.sum((v - v.mean()) ** 2)) if v.size else 0.0

    def walk(node, idx):
        if node["leaf"] or idx.size == 0:
            return
        mask = X[idx, node["feature"]] <= node["threshold"]
        li, ri = idx[mask], idx[~mask]
        gains[node["feature"]] += sse(target[idx]) - sse(target[li]) - sse(target[ri])
        walk(node["left"], li)
        walk(node["right"], ri)

    walk(tree.root_, np.arange(X.shape[0]))
    total = gains.sum()
    return gains / total if total > 0 else gains


def global_surrogate(
    predict,
    X,
    max_depth=3,
    min_samples_leaf=5,
    X_test=None,
    threshold=None,
    feature_names=None,
):
    """Fit a shallow regression-tree surrogate to ``predict(X)``.

    Parameters
    ----------
    predict : callable
        Black-box prediction function ``predict(X) -> (n,)``.
    X : array-like of shape (n, d)
        Rows on which the black box is queried and the surrogate fitted.
    max_depth, min_samples_leaf : int
        Surrogate tree complexity (depth 2-4 keeps rules readable).
    X_test : array-like, optional
        Held-out rows for an out-of-sample fidelity estimate.
    threshold : float, optional
        Decision threshold for label agreement (e.g. 0.5 for probabilities).
    feature_names : list of str, optional

    Returns
    -------
    dict with ``tree``, ``rules``, ``fidelity``, ``test_fidelity`` (or
    ``None``), ``importances`` (dict name -> share), ``n_leaves`` and
    ``max_depth``.
    """

    X = as_2d(X)
    if int(max_depth) != max_depth or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    names = _names(feature_names, X.shape[1])
    target = np.asarray(predict(X), dtype=float).ravel()
    if target.shape[0] != X.shape[0]:
        raise ValueError("predict must return one value per row")
    tree = DecisionTreeRegressor(max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf))
    tree.fit(X, target)
    fid = surrogate_fidelity(target, tree.predict(X), threshold)
    test_fid = None
    if X_test is not None:
        Xt = as_2d(X_test)
        test_fid = surrogate_fidelity(predict(Xt), tree.predict(Xt), threshold)
    rules = extract_rules(tree, names)
    imp = surrogate_importances(tree, X, target)
    return {
        "tree": tree,
        "rules": rules,
        "fidelity": fid,
        "test_fidelity": test_fid,
        "importances": dict(zip(names, (float(v) for v in imp))),
        "n_leaves": len(rules),
        "max_depth": int(max_depth),
    }


def surrogate_depth_curve(predict, X, depths=(1, 2, 3, 4, 5, 6), min_samples_leaf=5, X_test=None):
    """Fidelity ``R^2`` (train and optionally test) and leaf count per depth."""

    X = as_2d(X)
    out = []
    for d in depths:
        res = global_surrogate(predict, X, max_depth=d, min_samples_leaf=min_samples_leaf, X_test=X_test)
        out.append({
            "max_depth": int(d),
            "n_leaves": res["n_leaves"],
            "r2": res["fidelity"]["r2"],
            "test_r2": None if res["test_fidelity"] is None else res["test_fidelity"]["r2"],
        })
    return out
