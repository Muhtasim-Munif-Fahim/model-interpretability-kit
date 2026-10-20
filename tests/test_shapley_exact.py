"""Tests for exact Shapley, Banzhaf and Shapley interaction values."""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pytest

from interpretability import (
    banzhaf_values,
    coalition_values,
    exact_shapley,
    sampling_shapley,
    shapley_interaction_values,
)
from interpretability.cli import main
from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.local import tree_shap_values


def _value(predict, x_row, background, subset):
    Z = background.copy()
    Z[:, list(subset)] = x_row[list(subset)]
    return float(np.mean(predict(Z)))


def _brute_shapley(predict, x_row, background, weight_fn):
    m = x_row.shape[0]
    phi = np.zeros(m)
    for j in range(m):
        others = [k for k in range(m) if k != j]
        for size in range(m):
            for subset in combinations(others, size):
                phi[j] += weight_fn(size, m) * (
                    _value(predict, x_row, background, subset + (j,))
                    - _value(predict, x_row, background, subset)
                )
    return phi


def _shapley_weight(s, m):
    return factorial(s) * factorial(m - s - 1) / factorial(m)


def _nonlinear(X):
    X = np.asarray(X)
    return np.sin(X[:, 0]) * X[:, 1] + X[:, 2] ** 2 - 0.5 * X[:, 0] * X[:, 3] + X[:, 1]


def test_coalition_values_bitmask_layout_and_batching():
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(7, 3))
    x = rng.normal(size=3)
    v, n_eval = coalition_values(_nonlinear_3, x, bg)
    assert n_eval == 8 * 7 and v.shape == (8,)
    for mask in range(8):
        subset = tuple(j for j in range(3) if mask >> j & 1)
        assert v[mask] == pytest.approx(_value(_nonlinear_3, x, bg, subset))
    small, _ = coalition_values(_nonlinear_3, x, bg, batch_size=5)
    assert np.allclose(small, v)


def _nonlinear_3(X):
    X = np.asarray(X)
    return X[:, 0] * X[:, 1] + np.exp(0.3 * X[:, 2])


def test_exact_shapley_matches_brute_force_and_is_efficient():
    rng = np.random.default_rng(1)
    bg = rng.normal(size=(12, 4))
    x = rng.normal(size=4)
    res = exact_shapley(_nonlinear, x, bg)
    assert np.allclose(res["values"], _brute_shapley(_nonlinear, x, bg, _shapley_weight))
    assert res["values"].sum() == pytest.approx(res["prediction"] - res["baseline"])
    assert res["prediction"] == pytest.approx(float(_nonlinear(x[None])[0]))
    assert res["baseline"] == pytest.approx(float(np.mean(_nonlinear(bg))))
    assert res["feature_names"] == ["X0", "X1", "X2", "X3"]


def test_linear_model_closed_form_and_dummy_feature():
    w = np.array([2.0, -1.0, 0.0, 0.5])
    rng = np.random.default_rng(2)
    bg = rng.normal(size=(20, 4))
    x = rng.normal(size=4)

    def predict(X):
        return np.asarray(X) @ w + 3.0

    expected = w * (x - bg.mean(axis=0))
    assert np.allclose(exact_shapley(predict, x, bg)["values"], expected)
    assert np.allclose(banzhaf_values(predict, x, bg)["values"], expected)
    inter = shapley_interaction_values(predict, x, bg)["interactions"]
    assert np.allclose(inter - np.diag(np.diag(inter)), 0.0)
    assert np.allclose(np.diag(inter), expected)
    assert exact_shapley(predict, x, bg)["values"][2] == 0.0


def test_product_interaction_closed_form():
    b = np.array([[0.5, -1.0, 2.0]])
    x = np.array([2.0, 3.0, 7.0])

    def predict(X):
        X = np.asarray(X)
        return X[:, 0] * X[:, 1]

    res = shapley_interaction_values(predict, x, b)
    inter = res["interactions"]
    assert 2 * inter[0, 1] == pytest.approx((x[0] - b[0, 0]) * (x[1] - b[0, 1]))
    assert inter[0, 1] == pytest.approx(inter[1, 0])
    assert np.allclose(inter[2], 0.0)
    assert np.allclose(inter.sum(axis=1), res["values"])
    assert inter.sum() == pytest.approx(res["prediction"] - res["baseline"])
    # main effects: x0 * b1 - b0 * b1 and b0 * x1 - b0 * b1
    assert inter[0, 0] == pytest.approx((x[0] - b[0, 0]) * b[0, 1])
    assert inter[1, 1] == pytest.approx(b[0, 0] * (x[1] - b[0, 1]))
    assert res["top_interactions"][0][:2] == ("X0", "X1")


def test_interaction_matrix_properties_on_nonlinear_model():
    rng = np.random.default_rng(3)
    bg = rng.normal(size=(10, 4))
    x = rng.normal(size=4)
    res = shapley_interaction_values(_nonlinear, x, bg, feature_names=list("abcd"))
    inter = res["interactions"]
    assert np.allclose(inter, inter.T)
    assert np.allclose(inter.sum(axis=1), exact_shapley(_nonlinear, x, bg)["values"])
    assert inter.sum() == pytest.approx(res["prediction"] - res["baseline"])
    # x2 enters additively: no interaction with anything
    assert np.allclose(np.delete(inter[2], 2), 0.0, atol=1e-12)
    # brute-force check of one pairwise index
    m = 4
    i, j = 0, 1
    others = [k for k in range(m) if k not in (i, j)]
    direct = 0.0
    for size in range(m - 1):
        w = factorial(size) * factorial(m - size - 2) / (2 * factorial(m - 1))
        for S in combinations(others, size):
            direct += w * (
                _value(_nonlinear, x, bg, S + (i, j)) - _value(_nonlinear, x, bg, S + (i,))
                - _value(_nonlinear, x, bg, S + (j,)) + _value(_nonlinear, x, bg, S)
            )
    assert inter[0, 1] == pytest.approx(direct)
    names = {(a, b) for a, b, _ in res["top_interactions"]}
    assert len(names) == 6


def test_banzhaf_matches_brute_force_and_reports_efficiency_gap():
    rng = np.random.default_rng(4)
    bg = rng.normal(size=(9, 4))
    x = rng.normal(size=4)
    res = banzhaf_values(_nonlinear, x, bg)
    expected = _brute_shapley(_nonlinear, x, bg, lambda s, m: 1.0 / 2 ** (m - 1))
    assert np.allclose(res["values"], expected)
    assert res["efficiency_gap"] == pytest.approx(res["prediction"] - res["baseline"] - res["sum"])
    # Banzhaf is efficient for games with at most pairwise interactions ...
    assert abs(res["efficiency_gap"]) < 1e-9

    def three_way(X):
        X = np.asarray(X)
        return X[:, 0] * X[:, 1] * X[:, 2] + X[:, 3]

    # ... but not once a three-way interaction is present
    gap = banzhaf_values(three_way, x, bg)
    assert abs(gap["efficiency_gap"]) > 1e-3
    assert exact_shapley(three_way, x, bg)["values"].sum() == pytest.approx(
        gap["prediction"] - gap["baseline"])
    # with two features Banzhaf and Shapley coincide
    two = bg[:, :2]

    def f2(X):
        X = np.asarray(X)
        return X[:, 0] * X[:, 1] + X[:, 0]

    assert np.allclose(banzhaf_values(f2, x[:2], two)["values"], exact_shapley(f2, x[:2], two)["values"])


def test_matches_tree_shap_with_single_reference_and_sampling_shapley():
    X, y, _ = make_synthetic_data(n_samples=200, seed=0)
    model = fit_decision_tree(X, y, max_depth=4, min_samples_leaf=5)
    for r in range(3):
        ref = X[60 + r : 61 + r]
        exact = exact_shapley(model.predict, X[r], ref)
        tree = tree_shap_values(model.root_, X[r], ref, feature_count=5)
        assert np.allclose(exact["values"], tree["values"], atol=1e-10)
        assert exact["baseline"] == pytest.approx(tree["baseline"])
    bg = X[:40]
    exact = exact_shapley(model.predict, X[5], bg)
    approx = sampling_shapley(model.predict, X[5], bg, n_permutations=3000, seed=0)
    assert np.allclose(approx["values"], exact["values"], atol=4 * approx["std_error"].max() + 1e-3)


def test_validation():
    bg = np.zeros((3, 2))
    with pytest.raises(ValueError, match="same number"):
        exact_shapley(_nonlinear_3, np.zeros(3), bg)
    with pytest.raises(ValueError, match="max_features"):
        exact_shapley(lambda X: X.sum(axis=1), np.zeros(13), np.zeros((2, 13)))
    with pytest.raises(ValueError, match="feature_names"):
        exact_shapley(lambda X: X.sum(axis=1), np.zeros(2), bg, feature_names=["a"])
    with pytest.raises(ValueError, match="one value per input row"):
        exact_shapley(lambda X: np.zeros(1), np.zeros(2), bg)
    with pytest.raises(ValueError, match="non-finite"):
        exact_shapley(lambda X: np.full(len(X), np.nan), np.zeros(2), bg)
    with pytest.raises(ValueError, match="batch_size"):
        coalition_values(lambda X: X.sum(axis=1), np.zeros(2), bg, batch_size=0)
    with pytest.raises(ValueError, match="finite"):
        exact_shapley(lambda X: X.sum(axis=1), np.array([np.nan, 0.0]), bg)
    single = shapley_interaction_values(lambda X: 2 * np.asarray(X)[:, 0], np.ones(1), np.zeros((1, 1)))
    assert single["interactions"].shape == (1, 1)
    assert single["interactions"][0, 0] == pytest.approx(2.0)


def test_cli_exact_shapley(capsys):
    code = main(["--seed", "1", "--n-samples", "120", "exact-shapley", "--rows", "0",
                 "--n-background", "15", "--interactions", "--banzhaf", "--top", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Exact Shapley for row 0" in out
    assert "banzhaf" in out and "top pairwise interactions" in out
    line = [ln for ln in out.splitlines() if "sum(phi)" in ln][0]
    left, right = line.split("f(x)-baseline=")
    assert float(left.split("=")[1]) == pytest.approx(float(right), abs=1e-4)
