"""RuleFit: sparse linear models on tree-derived rules (Friedman & Popescu 2008).

RuleFit turns a tree ensemble into an interpretable additive model:

1. **Rule generation.** A boosted ensemble of shallow regression trees is
   grown on random subsamples (ISLE-style gradient boosting with shrinkage,
   Friedman & Popescu 2003). Every non-root node of every tree defines a
   *rule*: the conjunction of split conditions on its path, e.g.
   ``x1 > 0.3 and x4 <= 1.2``, which is ``1`` for rows that reach the node.
2. **Linear terms.** Each feature is winsorised at the ``trim`` /
   ``1 - trim`` quantiles (robust to outliers) and scaled by ``0.4 / std``
   so it is on the same footing as a typical rule (Friedman & Popescu §5).
3. **Sparse fit.** A lasso on ``[rules, linear terms]`` selects a small set
   of terms. ``alpha=None`` chooses the penalty by K-fold cross-validation
   over a log-spaced path from ``alpha_max`` down to ``alpha_max / 100``.

Importances follow §6 of the paper: a rule ``r_k`` with support ``s_k``
gets ``I_k = |a_k| sqrt(s_k (1 - s_k))``; a linear term gets
``|b_j| std(l_j)``. Feature importance is the linear importance plus each
rule's importance split equally among the features it uses.

Pass the black box's ``predict(X)`` as ``y`` to use RuleFit as a global
surrogate (fidelity is then reported against those predictions).
Only numpy is required; the trees are
:class:`~interpretability.demo_model.DecisionTreeRegressor`.
"""

import numpy as np

from ._utils import as_1d, as_2d
from .demo_model import DecisionTreeRegressor

__all__ = [
    "extract_tree_rules",
    "rule_matrix",
    "lasso_coordinate_descent",
    "rulefit",
    "format_rulefit_rule",
]


def _names(feature_names, n_features):
    if feature_names is None:
        return ["x%d" % j for j in range(n_features)]
    names = list(feature_names)
    if len(names) != n_features:
        raise ValueError("feature_names must have one entry per feature")
    return names


def extract_tree_rules(tree):
    """All non-root node rules of a fitted tree.

    Returns a list of tuples of ``(feature, op, threshold)`` with ``op`` in
    ``{"<=", ">"}``; redundant conditions on the same feature and side are
    merged to the tightest one.
    """
    if not getattr(tree, "is_fitted_", False):
        raise ValueError("tree must be fitted")
    rules = []

    def walk(node, conds):
        if node["leaf"]:
            return
        f, t = node["feature"], node["threshold"]
        for child, op in ((node["left"], "<="), (node["right"], ">")):
            new = _merge(conds, (f, op, t))
            rules.append(new)
            walk(child, new)

    walk(tree.root_, ())
    return rules


def _merge(conds, cond):
    f, op, t = cond
    out = []
    replaced = False
    for g, o, s in conds:
        if g == f and o == op:
            s = min(s, t) if op == "<=" else max(s, t)
            replaced = True
        out.append((g, o, s))
    if not replaced:
        out.append(cond)
    return tuple(sorted(out))


def rule_matrix(X, rules):
    """Binary ``(n, n_rules)`` matrix of rule activations."""
    X = as_2d(X)
    R = np.ones((X.shape[0], len(rules)))
    for k, rule in enumerate(rules):
        for f, op, t in rule:
            R[:, k] *= (X[:, f] <= t) if op == "<=" else (X[:, f] > t)
    return R


def lasso_coordinate_descent(Z, y, alpha, max_iter=1000, tol=1e-7, warm_start=None):
    """Lasso ``(1/2n)||y - b0 - Z b||^2 + alpha ||b||_1`` by cyclic coordinate descent.

    Returns ``(intercept, coef)``. Columns with zero variance get a zero
    coefficient.
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    n, p = Z.shape
    zm = Z.mean(axis=0)
    ym = float(y.mean())
    Zc = Z - zm
    yc = y - ym
    # Covariance updates: each coordinate step costs O(p) instead of O(n).
    G = Zc.T @ Zc / n
    c = Zc.T @ yc / n
    norms = np.diag(G).copy()
    b = np.zeros(p) if warm_start is None else np.array(warm_start, dtype=float)
    Gb = G @ b
    active_only = False
    for _ in range(max_iter):
        max_change = 0.0
        coords = np.flatnonzero(b) if active_only else range(p)
        for j in coords:
            if norms[j] <= 1e-12:
                b[j] = 0.0
                continue
            old = b[j]
            rho = c[j] - Gb[j] + norms[j] * old
            new = np.sign(rho) * max(abs(rho) - alpha, 0.0) / norms[j]
            if new != old:
                Gb += G[:, j] * (new - old)
                b[j] = new
                max_change = max(max_change, abs(new - old) * np.sqrt(norms[j]))
        if max_change < tol:
            if not active_only:
                break
            active_only = False  # confirm with a full sweep
        else:
            active_only = True
    return ym - float(zm @ b), b


def _alpha_max(Z, y):
    Zc = Z - Z.mean(axis=0)
    return float(np.max(np.abs(Zc.T @ (y - y.mean()))) / Z.shape[0])


def _cv_alpha(Z, y, n_alphas, n_folds, rng):
    n = Z.shape[0]
    a_max = max(_alpha_max(Z, y), 1e-12)
    alphas = a_max * np.logspace(0, -2, n_alphas)
    folds = np.array_split(rng.permutation(n), n_folds)
    errors = np.zeros(n_alphas)
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        warm = None
        for i, a in enumerate(alphas):
            b0, b = lasso_coordinate_descent(Z[train], y[train], a, max_iter=300, tol=1e-5, warm_start=warm)
            warm = b
            errors[i] += float(np.sum((y[test] - b0 - Z[test] @ b) ** 2))
    errors /= n
    return float(alphas[int(np.argmin(errors))]), alphas, errors


def rulefit(
    X,
    y,
    n_trees=30,
    max_depth=3,
    sample_fraction=0.5,
    learning_rate=0.1,
    min_samples_leaf=5,
    include_linear=True,
    trim=0.025,
    alpha=None,
    n_alphas=20,
    n_folds=5,
    max_rules=None,
    feature_names=None,
    seed=None,
):
    """Fit RuleFit and return a dict describing the sparse rule model.

    Parameters
    ----------
    X, y :
        Data and target (or black-box predictions for a surrogate).
    n_trees, max_depth, sample_fraction, learning_rate, min_samples_leaf :
        Rule-generating boosted ensemble. Each tree has a random depth in
        ``1..max_depth`` (a cheap stand-in for the paper's exponential tree
        size distribution) and is fit on a ``sample_fraction`` subsample.
    include_linear, trim :
        Add winsorised, ``0.4/std``-scaled linear terms.
    alpha :
        Lasso penalty; ``None`` = K-fold CV over ``n_alphas`` values.
    max_rules :
        Optional cap on candidate rules (most frequent kept first).

    Returns
    -------
    dict with ``intercept``, ``rules`` (non-zero rules sorted by importance:
    ``conditions`` (named), ``coef``, ``support``, ``importance``, ``text``),
    ``linear`` (non-zero linear terms), ``feature_importance`` (normalised to
    sum to one), ``alpha``, ``n_candidate_rules``, ``r2`` (in-sample, against
    ``y``) and ``predict`` (callable on new rows).
    """
    X = as_2d(X)
    n, d = X.shape
    y = as_1d(y, n)
    names = _names(feature_names, d)
    if n_trees < 1 or max_depth < 1:
        raise ValueError("n_trees and max_depth must be >= 1")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1]")
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    if alpha is not None and alpha < 0:
        raise ValueError("alpha must be non-negative")
    rng = np.random.default_rng(seed)

    # 1. Rule generation with an ISLE / shrunken boosting ensemble.
    F = np.full(n, float(y.mean()))
    m = max(2 * min_samples_leaf, int(round(sample_fraction * n)))
    m = min(m, n)
    seen = {}
    for _ in range(n_trees):
        idx = rng.choice(n, size=m, replace=False)
        depth = int(rng.integers(1, max_depth + 1))
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples_leaf)
        tree.fit(X[idx], (y - F)[idx])
        F += learning_rate * tree.predict(X)
        for rule in extract_tree_rules(tree):
            seen[rule] = seen.get(rule, 0) + 1
    rules = list(seen)
    R = rule_matrix(X, rules) if rules else np.zeros((n, 0))
    support = R.mean(axis=0) if rules else np.zeros(0)
    # Drop rules that are constant on the training data and duplicate columns.
    keep, signatures = [], set()
    order = sorted(range(len(rules)), key=lambda k: (-seen[rules[k]], k))
    for k in order:
        if support[k] <= 0.0 or support[k] >= 1.0:
            continue
        sig = R[:, k].tobytes()
        if sig in signatures:
            continue
        signatures.add(sig)
        keep.append(k)
    if max_rules is not None:
        keep = keep[: int(max_rules)]
    rules = [rules[k] for k in keep]
    R = R[:, keep]
    support = support[keep]

    # 2. Linear terms.
    if include_linear:
        lo = np.quantile(X, trim, axis=0)
        hi = np.quantile(X, 1.0 - trim, axis=0)
        W = np.clip(X, lo, hi)
        sd = W.std(axis=0)
        scale = np.where(sd > 0, 0.4 / np.where(sd > 0, sd, 1.0), 0.0)
        L = W * scale
    else:
        lo = hi = scale = None
        L = np.zeros((n, 0))
    Z = np.hstack([R, L])
    if Z.shape[1] == 0:
        raise ValueError("no rules or linear terms to fit")

    # 3. Lasso (columns standardised internally so the penalty is fair).
    zsd = Z.std(axis=0)
    zsd_safe = np.where(zsd > 0, zsd, 1.0)
    Zs = Z / zsd_safe
    if alpha is None:
        alpha_used, _, _ = _cv_alpha(Zs, y, n_alphas, n_folds, rng)
    else:
        alpha_used = float(alpha)
    b0, bs = lasso_coordinate_descent(Zs, y, alpha_used)
    coef = np.where(zsd > 0, bs / zsd_safe, 0.0)

    n_rules = len(rules)
    a_rule, b_lin = coef[:n_rules], coef[n_rules:]
    feat_imp = np.zeros(d)
    rule_out = []
    for k in np.flatnonzero(a_rule):
        s = float(support[k])
        imp = abs(a_rule[k]) * np.sqrt(s * (1.0 - s))
        feats = sorted({f for f, _, _ in rules[k]})
        for f in feats:
            feat_imp[f] += imp / len(feats)
        conds = [(names[f], op, float(t)) for f, op, t in rules[k]]
        rule_out.append({
            "conditions": conds,
            "coef": float(a_rule[k]),
            "support": s,
            "importance": float(imp),
            "text": format_rulefit_rule(conds),
        })
    rule_out.sort(key=lambda r: -r["importance"])
    linear_out = []
    for j in np.flatnonzero(b_lin):
        imp = abs(b_lin[j]) * float(L[:, j].std())
        feat_imp[j] += imp
        linear_out.append({
            "feature": names[j],
            "coef": float(b_lin[j] * scale[j]),  # per unit of the (winsorised) raw feature
            "importance": float(imp),
        })
    linear_out.sort(key=lambda r: -r["importance"])
    total = feat_imp.sum()

    def predict(Xnew):
        Xnew = as_2d(Xnew)
        if Xnew.shape[1] != d:
            raise ValueError("X has the wrong number of features")
        parts = [rule_matrix(Xnew, rules) if rules else np.zeros((Xnew.shape[0], 0))]
        if include_linear:
            parts.append(np.clip(Xnew, lo, hi) * scale)
        return b0 + np.hstack(parts) @ coef

    fitted = predict(X)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return {
        "intercept": float(b0),
        "rules": rule_out,
        "linear": linear_out,
        "feature_importance": {nm: float(v / total) if total > 0 else 0.0 for nm, v in zip(names, feat_imp)},
        "alpha": alpha_used,
        "n_candidate_rules": n_rules,
        "n_terms": int(np.count_nonzero(coef)),
        "r2": r2,
        "predict": predict,
    }


def format_rulefit_rule(conditions, digits=4):
    """``"x1 > 0.3 and x4 <= 1.2"`` from ``[(name, op, threshold), ...]``."""
    return " and ".join("%s %s %.*g" % (nm, op, digits, t) for nm, op, t in conditions)
