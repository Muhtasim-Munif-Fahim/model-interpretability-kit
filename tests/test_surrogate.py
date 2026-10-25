import numpy as np
import pytest

from interpretability import (
    extract_rules,
    format_rule,
    global_surrogate,
    surrogate_depth_curve,
    surrogate_fidelity,
    surrogate_importances,
)
from interpretability.cli import main
from interpretability.demo_model import DecisionTreeRegressor


def _step(X):
    return np.where(X[:, 0] > 0.2, 2.0, -1.0) + np.where(X[:, 1] > 0.0, 1.0, 0.0)


@pytest.fixture
def X():
    return np.random.default_rng(0).uniform(-1, 1, size=(400, 3))


def test_surrogate_recovers_step_function(X):
    Xt = np.random.default_rng(1).uniform(-1, 1, size=(300, 3))
    res = global_surrogate(_step, X, max_depth=2, X_test=Xt, feature_names=["a", "b", "c"])
    assert res["fidelity"]["r2"] == pytest.approx(1.0)
    assert res["test_fidelity"]["r2"] > 0.95
    assert res["n_leaves"] == 4
    assert sorted(round(r["value"]) for r in res["rules"]) == [-1, 0, 2, 3]
    assert res["importances"]["c"] == 0.0
    assert res["importances"]["a"] > res["importances"]["b"] > 0
    assert sum(res["importances"].values()) == pytest.approx(1.0)
    assert sum(r["coverage"] for r in res["rules"]) == pytest.approx(1.0)


def test_rules_reproduce_tree_predictions(X):
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5).fit(X, y)
    rules = extract_rules(tree)
    pred = np.full(X.shape[0], np.nan)
    for rule in rules:
        mask = np.ones(X.shape[0], dtype=bool)
        for name, (lo, hi) in rule["conditions"].items():
            j = int(name[1:])
            mask &= (X[:, j] > lo) & (X[:, j] <= hi)
        assert mask.sum() == rule["n"]
        pred[mask] = rule["value"]
    assert np.allclose(pred, tree.predict(X))
    # merged bounds: each feature appears at most once per rule
    for rule in rules:
        assert len(rule["conditions"]) <= 4


def test_format_rule_text():
    rule = {"conditions": {"x0": (-np.inf, 0.5), "x1": (0.1, 0.9), "x2": (2.0, np.inf)},
            "value": 1.25, "n": 10, "coverage": 0.25}
    text = format_rule(rule)
    assert text == "if x0 <= 0.5 and 0.1 < x1 <= 0.9 and x2 > 2 then 1.25  (n=10, 25.0%)"
    assert format_rule({"conditions": {}, "value": 3.0, "n": 4, "coverage": 1.0}).startswith("if always")


def test_fidelity_metrics():
    a = np.array([0.1, 0.4, 0.6, 0.9])
    assert surrogate_fidelity(a, a, threshold=0.5) == {"r2": 1.0, "rmse": 0.0, "agreement": 1.0}
    f = surrogate_fidelity(a, [0.6, 0.4, 0.6, 0.9], threshold=0.5)
    assert f["agreement"] == 0.75
    assert f["r2"] == pytest.approx(1 - 0.25 / np.sum((a - a.mean()) ** 2))
    assert surrogate_fidelity([1, 1], [1, 1])["r2"] == 1.0
    assert surrogate_fidelity([1, 1], [1, 2])["r2"] == 0.0
    with pytest.raises(ValueError):
        surrogate_fidelity([1, 2], [1])


def test_depth_curve_is_monotone_in_fidelity(X):
    f = lambda Z: np.sin(3 * Z[:, 0]) + Z[:, 1] ** 2  # noqa: E731
    curve = surrogate_depth_curve(f, X, depths=(1, 2, 3, 4, 5))
    r2 = [row["r2"] for row in curve]
    assert all(b >= a - 1e-12 for a, b in zip(r2, r2[1:]))
    assert [row["n_leaves"] for row in curve][0] == 2
    assert curve[0]["test_r2"] is None


def test_importances_zero_for_stump_free_tree():
    X = np.random.default_rng(2).normal(size=(30, 2))
    tree = DecisionTreeRegressor(max_depth=2).fit(X, np.ones(30))
    assert np.all(surrogate_importances(tree, X, np.ones(30)) == 0)


def test_validation(X):
    with pytest.raises(ValueError):
        global_surrogate(_step, X, max_depth=0)
    with pytest.raises(ValueError):
        global_surrogate(lambda Z: np.zeros(3), X)
    with pytest.raises(ValueError):
        global_surrogate(_step, X, feature_names=["a"])
    with pytest.raises(ValueError):
        extract_rules(DecisionTreeRegressor())


def test_cli_surrogate(capsys):
    assert main(["--seed", "1", "--n-samples", "200", "surrogate", "--max-depth", "2", "--curve"]) == 0
    out = capsys.readouterr().out
    assert "Global surrogate (depth 2" in out and "Rules:" in out and "Depth curve" in out
    assert out.count("  if ") >= 2
