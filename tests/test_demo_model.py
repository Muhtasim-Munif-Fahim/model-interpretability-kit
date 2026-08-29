"""Tests for the transparent demo model and synthetic data generator."""

import numpy as np
import pytest

from interpretability.demo_model import (
    DecisionTreeRegressor,
    fit_decision_tree,
    make_synthetic_data,
)


def _tree_depth(node):
    if node["leaf"]:
        return 0
    return 1 + max(_tree_depth(node["left"]), _tree_depth(node["right"]))


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def test_synthetic_data_shape_and_names():
    X, y, names = make_synthetic_data(n_samples=120, seed=0)
    assert X.shape == (120, 5)
    assert y.shape == (120,)
    assert names == ["X0", "X1", "X2", "X3", "X4"]


def test_synthetic_data_reproducible_with_seed():
    X1, y1, _ = make_synthetic_data(n_samples=50, seed=7)
    X2, y2, _ = make_synthetic_data(n_samples=50, seed=7)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_synthetic_data_varies_without_seed():
    X1, _, _ = make_synthetic_data(n_samples=20)
    X2, _, _ = make_synthetic_data(n_samples=20)
    assert not np.array_equal(X1, X2)


def test_synthetic_ground_truth_feature_ranking():
    X, y, _ = make_synthetic_data(n_samples=2000, noise=0.0, seed=3)
    corrs = [np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])]
    assert corrs[0] > corrs[1]
    assert corrs[1] > corrs[3]
    assert corrs[2] > corrs[4]
    assert corrs[4] < 0.1


def test_synthetic_noise_degrades_fit():
    clean = make_synthetic_data(n_samples=300, noise=0.0, seed=1)[1]
    noisy = make_synthetic_data(n_samples=300, noise=0.05, seed=1)[1]
    assert np.any(clean != noisy)


def test_tree_predicts_stored_leaf_means():
    X, y, _ = make_synthetic_data(n_samples=200, seed=0)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10).fit(X, y)
    preds = tree.predict(X)
    for i in range(X.shape[0]):
        node = tree.root_
        while not node["leaf"]:
            if X[i, node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        assert preds[i] == pytest.approx(node["value"])


def test_tree_fits_constant_feature_column():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.full(60, 0.3), rng.uniform(size=60)])
    y = 2.0 * X[:, 1]
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5).fit(X, y)
    preds = tree.predict(X)
    assert preds.shape == (60,)
    assert tree.n_features_ == 2


def test_tree_constant_target_produces_single_leaf():
    X = np.random.default_rng(1).uniform(size=(30, 2))
    y = np.full(30, 5.0)
    tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2).fit(X, y)
    assert tree.root_["leaf"] is True
    assert tree.root_["value"] == pytest.approx(5.0)


def test_tree_max_depth_enforced():
    X, y, _ = make_synthetic_data(n_samples=300, seed=2)
    tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=2).fit(X, y)
    assert _tree_depth(tree.root_) <= 2


def test_tree_small_training_set_stays_leaf():
    X, y, _ = make_synthetic_data(n_samples=6, seed=8)
    tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5).fit(X, y)
    assert tree.root_["leaf"] is True


def test_tree_fits_known_axis_aligned_structure():
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(500, 2))
    y = 3.0 * X[:, 0] + 2.0 * (X[:, 1] > 0.5)
    tree = fit_decision_tree(X, y, max_depth=6, min_samples_leaf=5)
    assert _r2(y, tree.predict(X)) > 0.9


def test_tree_single_row_prediction():
    X, y, _ = make_synthetic_data(n_samples=50, seed=5)
    tree = fit_decision_tree(X, y)
    out = tree.predict(X[0])
    assert out.shape == (1,)
    assert np.isscalar(out[0]) or np.ndim(out[0]) == 0


def test_tree_rejects_wrong_feature_count():
    X, y, _ = make_synthetic_data(n_samples=50, seed=6)
    tree = fit_decision_tree(X, y)
    with pytest.raises(ValueError):
        tree.predict(np.zeros((3, 2)))


def test_tree_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        DecisionTreeRegressor().predict(np.zeros((2, 2)))


def test_fit_decision_tree_is_callable_predictor():
    X, y, _ = make_synthetic_data(n_samples=60, seed=9)
    predictor = fit_decision_tree(X, y, max_depth=3)
    assert callable(predictor.predict)
    assert predictor.predict(X[:5]).shape == (5,)
