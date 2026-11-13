import numpy as np
import pytest

from interpretability import (
    extract_tree_rules,
    lasso_coordinate_descent,
    rule_matrix,
    rulefit,
)
from interpretability.demo_model import DecisionTreeRegressor


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 4))
    y = 3.0 * (X[:, 0] > 0.5) + 2.0 * X[:, 1] + 0.05 * rng.normal(size=n)
    return X, y


def test_tree_rules_cover_every_node():
    X, y = _data()
    tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5).fit(X, y)
    rules = extract_tree_rules(tree)
    assert len(rules) == 6  # 3 internal nodes -> 6 children
    R = rule_matrix(X, rules)
    root_children = [k for k, r in enumerate(rules) if len(r) == 1 and r[0][0] == tree.root_["feature"]]
    np.testing.assert_allclose(R[:, root_children].sum(axis=1), 1.0)
    assert set(np.unique(R)) <= {0.0, 1.0}


def test_rule_merge_keeps_tightest_bound():
    X = np.array([[0.1], [0.4], [0.6], [0.9]])
    R = rule_matrix(X, [((0, "<=", 0.5),), ((0, ">", 0.3), (0, "<=", 0.7))])
    np.testing.assert_array_equal(R, [[1, 0], [1, 1], [0, 1], [0, 0]])


def test_lasso_matches_ols_and_soft_threshold():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(200, 3))
    y = Z @ [1.5, 0.0, -2.0] + 0.3 + 0.01 * rng.normal(size=200)
    b0, b = lasso_coordinate_descent(Z, y, alpha=0.0, tol=1e-12, max_iter=5000)
    A = np.column_stack([np.ones(200), Z])
    ols = np.linalg.lstsq(A, y, rcond=None)[0]
    np.testing.assert_allclose([b0, *b], ols, atol=1e-6)
    # Orthogonal, unit-variance design: lasso = soft-thresholded OLS.
    Q = np.linalg.qr(rng.normal(size=(100, 3)))[0]
    Q = (Q - Q.mean(axis=0))
    Q = np.linalg.qr(Q)[0] * np.sqrt(100)
    yy = Q @ [2.0, 0.5, -1.0]
    _, bb = lasso_coordinate_descent(Q, yy, alpha=0.7, tol=1e-12)
    ols_b = np.linalg.lstsq(np.column_stack([np.ones(100), Q]), yy, rcond=None)[0][1:]
    np.testing.assert_allclose(bb, np.sign(ols_b) * np.maximum(np.abs(ols_b) - 0.7, 0), atol=1e-6)
    with pytest.raises(ValueError):
        lasso_coordinate_descent(Z, y, alpha=-1)


def test_rulefit_recovers_structure():
    X, y = _data()
    res = rulefit(X, y, n_trees=20, max_depth=2, seed=0, feature_names=["a", "b", "c", "d"])
    assert res["r2"] > 0.95
    fi = res["feature_importance"]
    assert pytest.approx(sum(fi.values())) == 1.0
    assert fi["a"] + fi["b"] > 0.85
    top = res["rules"][0]
    assert any(nm == "a" and abs(t - 0.5) < 0.1 for nm, _, t in top["conditions"])
    lin = {t["feature"]: t["coef"] for t in res["linear"]}
    assert lin.get("b", 0.0) == pytest.approx(2.0, abs=0.6)
    assert 0 < res["n_terms"] <= res["n_candidate_rules"] + 4
    pred = res["predict"](X[:10])
    assert pred.shape == (10,)
    assert isinstance(top["text"], str) and "a" in top["text"]


def test_rulefit_as_surrogate_and_determinism():
    X, _ = _data(n=200, seed=3)
    black_box = lambda Z: np.where(Z[:, 2] > 0.3, 1.0, -1.0) * (1 + Z[:, 3])
    target = black_box(X)
    a = rulefit(X, target, n_trees=15, max_depth=3, seed=5)
    b = rulefit(X, target, n_trees=15, max_depth=3, seed=5)
    assert a["alpha"] == b["alpha"]
    np.testing.assert_allclose(a["predict"](X), b["predict"](X))
    assert a["r2"] > 0.8
    imp = a["feature_importance"]
    assert max(imp, key=imp.get) == "x2"


def test_rulefit_options():
    X, y = _data(n=150)
    no_lin = rulefit(X, y, n_trees=10, include_linear=False, alpha=0.01, seed=0)
    assert no_lin["linear"] == []
    capped = rulefit(X, y, n_trees=10, max_rules=5, alpha=0.01, seed=0)
    assert capped["n_candidate_rules"] == 5
    big_alpha = rulefit(X, y, n_trees=5, alpha=1e6, seed=0)
    assert big_alpha["n_terms"] == 0
    np.testing.assert_allclose(big_alpha["predict"](X), y.mean())


def test_rulefit_validation():
    X, y = _data(n=60)
    with pytest.raises(ValueError):
        rulefit(X, y[:5])
    with pytest.raises(ValueError):
        rulefit(X, y, sample_fraction=0)
    with pytest.raises(ValueError):
        rulefit(X, y, learning_rate=2)
    with pytest.raises(ValueError):
        rulefit(X, y, trim=0.6)
    with pytest.raises(ValueError):
        rulefit(X, y, alpha=-1)
    with pytest.raises(ValueError):
        rulefit(X, y, feature_names=["a"])
    res = rulefit(X, y, n_trees=3, alpha=0.01, seed=0)
    with pytest.raises(ValueError):
        res["predict"](X[:, :2])
