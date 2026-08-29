"""Tests for LIME-style surrogates and tree-based local attributions."""

import numpy as np
import pytest

from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.local import (
    lime_explain,
    local_occlusion_attribution,
    sample_neighborhood,
    tree_conditional_expectation,
    tree_shap_values,
    weighted_least_squares,
)


def _linear_predict(Z, coefs=(2.0, 1.0)):
    return Z @ np.asarray(coefs, dtype=float)


def _make_background(n=200, seed=0, n_features=2):
    return np.random.default_rng(seed).uniform(size=(n, n_features))


def test_weighted_least_squares_matches_ordinary_least_squares():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(60, 2))
    design = np.column_stack([np.ones(60), X])
    y = design @ np.array([0.5, 2.0, -1.0])
    weights = np.ones(60)
    coefs = weighted_least_squares(design, y, weights, l2=0.0)
    assert coefs == pytest.approx(np.array([0.5, 2.0, -1.0]), abs=1e-8)


def test_weighted_least_squares_biased_towards_high_weight_rows():
    design = np.column_stack([np.ones(3), np.array([0.0, 1.0, 2.0])])
    y = np.array([0.0, 0.0, 10.0])
    weights = np.array([1.0, 1.0, 100.0])
    coefs = weighted_least_squares(design, y, weights, l2=0.0)
    slope = coefs[1]
    assert slope > 4.0


def test_weighted_least_squares_handles_collinear_columns():
    rng = np.random.default_rng(2)
    base = rng.uniform(size=(40, 2))
    design = np.column_stack([base, base[:, 0]])
    y = base[:, 0]
    coefs = weighted_least_squares(design, y, np.ones(40))
    assert np.all(np.isfinite(coefs))


def test_weighted_least_squares_rejects_bad_inputs():
    with pytest.raises(ValueError):
        weighted_least_squares(np.ones((4, 2)), np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        weighted_least_squares(np.ones((4, 2)), np.ones(4), np.array([-1.0, 1, 1, 1]))


def test_sample_neighborhood_center_has_unit_weight():
    background = _make_background()
    x_row = background[3]
    Z, weights = sample_neighborhood(x_row, background, n_samples=100, seed=5)
    assert Z.shape == (101, 2)
    assert weights.shape == (101,)
    assert weights[-1] == pytest.approx(1.0)
    assert np.all(weights > 0.0)
    assert np.all(weights <= 1.0 + 1e-12)


def test_sample_neighborhood_reproducible_with_seed():
    background = _make_background()
    Z1, _ = sample_neighborhood(background[0], background, n_samples=50, seed=11)
    Z2, _ = sample_neighborhood(background[0], background, n_samples=50, seed=11)
    assert np.array_equal(Z1, Z2)


def test_sample_neighborhood_rejects_mismatched_width():
    background = _make_background()
    with pytest.raises(ValueError):
        sample_neighborhood(np.zeros(3), background, n_samples=10)


def test_lime_recovers_linear_model():
    background = _make_background(n=300, seed=3)
    x_row = background[7]
    result = lime_explain(_linear_predict, x_row, background, n_samples=500, seed=9)
    assert result["coefficients"] == pytest.approx(np.array([2.0, 1.0]), abs=1e-4)
    assert result["intercept"] == pytest.approx(0.0, abs=1e-4)
    assert result["weighted_r2"] == pytest.approx(1.0, abs=1e-8)
    assert result["prediction"] == pytest.approx(_linear_predict(x_row[None, :])[0])


def test_lime_reproducible_with_seed():
    background = _make_background(n=200, seed=4)
    x_row = background[5]
    r1 = lime_explain(_linear_predict, x_row, background, n_samples=100, seed=42)
    r2 = lime_explain(_linear_predict, x_row, background, n_samples=100, seed=42)
    assert np.allclose(r1["coefficients"], r2["coefficients"])
    assert r1["weighted_r2"] == pytest.approx(r2["weighted_r2"])


def test_lime_feature_names_passthrough():
    background = _make_background(n=100, seed=6)
    names = ["age", "income"]
    result = lime_explain(_linear_predict, background[0], background, n_samples=50, seed=1, feature_names=names)
    assert result["feature_names"] == names
    assert result["coefficients"].shape == (2,)


def test_lime_demo_tree_smoke():
    X, y, names = make_synthetic_data(n_samples=300, seed=8)
    model = fit_decision_tree(X, y, max_depth=5, min_samples_leaf=5)
    result = lime_explain(model.predict, X[0], X, n_samples=150, seed=2, feature_names=names)
    assert result["coefficients"].shape == (5,)
    assert np.all(np.isfinite(result["coefficients"]))
    assert 0.0 <= result["weighted_r2"] <= 1.0
    assert result["prediction"] == pytest.approx(model.predict(X[0:1])[0])


def _leaf(value):
    return {"leaf": True, "value": value, "n": 1}


def _stump():
    return {
        "leaf": False,
        "feature": 0,
        "threshold": 0.5,
        "left": _leaf(2.0),
        "right": _leaf(4.0),
        "value": 3.0,
        "n": 4,
    }


def test_tree_conditional_expectation_empty_fixed_is_background_mean():
    node = _stump()
    background = np.array([[0.1, 0.9], [0.2, 0.1], [0.8, 0.3], [0.9, 0.7]])
    expected = tree_conditional_expectation(node, background[0], (), background)
    assert expected == pytest.approx(3.0)
    assert expected == pytest.approx(
        np.mean(np.where(background[:, 0] <= 0.5, 2.0, 4.0))
    )


def test_tree_conditional_expectation_all_fixed_is_point_prediction():
    node = _stump()
    background = np.array([[0.1, 0.9], [0.2, 0.1], [0.8, 0.3], [0.9, 0.7]])
    x_row = np.array([0.1, 0.5])
    expected = tree_conditional_expectation(node, x_row, (0, 1), background)
    assert expected == pytest.approx(2.0)


def test_tree_conditional_expectation_single_fixed_feature():
    node = _stump()
    background = np.array([[0.1, 0.9], [0.2, 0.1], [0.8, 0.3], [0.9, 0.7]])
    x_row = np.array([0.8, 0.5])
    expected = tree_conditional_expectation(node, x_row, (0,), background)
    assert expected == pytest.approx(4.0)


def test_tree_shap_single_feature_tree_additivity():
    node = _stump()
    background = np.array([[0.1, 0.9], [0.2, 0.1], [0.8, 0.3], [0.9, 0.7]])
    x_row = np.array([0.9, 0.5])
    result = tree_shap_values(node, x_row, background, feature_count=2)
    prediction = 4.0
    assert result["baseline"] == pytest.approx(3.0)
    assert np.sum(result["values"]) == pytest.approx(prediction - result["baseline"])
    assert result["values"][1] == pytest.approx(0.0)


def _interventional_mean(node, background):
    """Reference implementation: leaf values weighted by marginal split flow."""

    def walk(node, prob):
        if node["leaf"]:
            return prob * node["value"]
        feature = node["feature"]
        frac = float(np.mean(background[:, feature] <= node["threshold"]))
        return walk(node["left"], prob * frac) + walk(node["right"], prob * (1.0 - frac))

    return walk(node, 1.0)


def test_tree_shap_demo_tree_additivity_and_baseline():
    X, y, _ = make_synthetic_data(n_samples=300, seed=10)
    model = fit_decision_tree(X, y, max_depth=5, min_samples_leaf=5)
    background = X[::2]
    x_row = X[1]
    result = tree_shap_values(model.root_, x_row, background, feature_count=5)
    prediction = model.predict(x_row[None, :])[0]
    assert result["baseline"] == pytest.approx(
        _interventional_mean(model.root_, background), abs=1e-8
    )
    assert np.sum(result["values"]) == pytest.approx(
        prediction - result["baseline"], abs=1e-8
    )


def test_tree_shap_unused_feature_gets_zero_attribution():
    X, y, _ = make_synthetic_data(n_samples=250, seed=12)
    model = fit_decision_tree(X[:, :4], y, max_depth=5, min_samples_leaf=5)
    x_row = np.append(X[3, :4], 0.25)
    background = np.column_stack([X[::2, :4], np.zeros(X[::2].shape[0])])
    result = tree_shap_values(model.root_, x_row, background, feature_count=5)
    assert result["values"][4] == pytest.approx(0.0, abs=1e-8)
    assert np.sum(np.abs(result["values"])) > 0.5
    assert np.argmax(np.abs(result["values"])) != 4


def test_tree_shap_rejects_bad_inputs():
    node = _stump()
    background = np.array([[0.1, 0.9], [0.2, 0.1], [0.8, 0.3], [0.9, 0.7]])
    with pytest.raises(ValueError):
        tree_shap_values(node, np.zeros(2), background, feature_count=0)
    with pytest.raises(ValueError):
        tree_shap_values(node, np.zeros(3), background, feature_count=2)
    with pytest.raises(TypeError):
        tree_conditional_expectation("not a node", np.zeros(2), (), background)


def test_occlusion_attribution_two_feature_linear_model():
    background = _make_background(n=500, seed=14)
    x_row = background[10]
    result = local_occlusion_attribution(_linear_predict, x_row, background, n_samples=800, seed=3)
    w = np.array([2.0, 1.0])
    mean_bg = background.mean(axis=0)
    expected0 = w[1] * (x_row[1] - mean_bg[1])
    expected1 = w[0] * (x_row[0] - mean_bg[0])
    assert result["values"][0] == pytest.approx(expected0, abs=0.05)
    assert result["values"][1] == pytest.approx(expected1, abs=0.05)


def test_occlusion_attribution_reproducible_with_seed():
    background = _make_background(n=150, seed=15)
    x_row = background[2]
    r1 = local_occlusion_attribution(_linear_predict, x_row, background, n_samples=100, seed=7)
    r2 = local_occlusion_attribution(_linear_predict, x_row, background, n_samples=100, seed=7)
    assert np.allclose(r1["values"], r2["values"])
