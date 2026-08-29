"""Tests for permutation and drop-column feature importance."""

import numpy as np
import pytest

from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.importance import (
    drop_column_importance,
    permutation_importance,
    r2_score,
)


def _split(X, y, n_train=150):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _fit_predict_demo(X_fit, y_fit, X_eval):
    return fit_decision_tree(X_fit, y_fit, max_depth=5, min_samples_leaf=5).predict(X_eval)


def test_r2_score_perfect_and_constant_targets():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    assert r2_score(y, y.mean() * np.ones_like(y)) == pytest.approx(0.0)
    assert r2_score(np.ones(4), np.ones(4)) == pytest.approx(1.0)


def test_permutation_importance_output_structure():
    X, y, _ = make_synthetic_data(n_samples=200, seed=0)
    Xtr, ytr, Xev, yev = _split(X, y)
    model = fit_decision_tree(Xtr, ytr, max_depth=5, min_samples_leaf=5)
    result = permutation_importance(model.predict, Xev, yev, n_repeats=3, seed=1)
    assert set(result) == {"mean", "std", "baseline", "n_repeats"}
    assert result["mean"].shape == (5,)
    assert result["std"].shape == (5,)
    assert result["n_repeats"] == 3
    assert np.all(result["std"] >= 0.0)


def test_permutation_importance_deterministic_with_seed():
    X, y, _ = make_synthetic_data(n_samples=200, seed=0)
    Xtr, ytr, Xev, yev = _split(X, y)
    model = fit_decision_tree(Xtr, ytr, max_depth=5, min_samples_leaf=5)
    r1 = permutation_importance(model.predict, Xev, yev, n_repeats=4, seed=42)
    r2 = permutation_importance(model.predict, Xev, yev, n_repeats=4, seed=42)
    assert np.allclose(r1["mean"], r2["mean"])
    assert np.allclose(r1["std"], r2["std"])


def test_permutation_importance_ranks_known_truth():
    X, y, _ = make_synthetic_data(n_samples=300, seed=11)
    Xtr, ytr, Xev, yev = _split(X, y)
    model = fit_decision_tree(Xtr, ytr, max_depth=5, min_samples_leaf=5)
    imp = permutation_importance(model.predict, Xev, yev, n_repeats=5, seed=3)["mean"]
    assert imp[0] > imp[1]
    assert imp[1] > imp[3]
    assert imp[2] > 0.0
    assert imp[4] < imp[2]
    assert imp[4] < 0.02


def test_permutation_importance_constant_feature_is_zero():
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.uniform(size=150), np.full(150, 0.7)])
    y = 3.0 * X[:, 0] + rng.normal(0.0, 0.05, size=150)
    Xtr, ytr, Xev, yev = _split(X, y, n_train=100)
    model = fit_decision_tree(Xtr, ytr, max_depth=5, min_samples_leaf=3)
    imp = permutation_importance(model.predict, Xev, yev, n_repeats=3, seed=2)["mean"]
    assert imp[1] == pytest.approx(0.0, abs=1e-12)
    assert imp[0] > 0.5


def test_permutation_importance_single_feature_1d_input():
    X, y, _ = make_synthetic_data(n_samples=120, seed=6)
    Xtr, ytr, Xev, yev = _split(X, y, n_train=80)
    model = fit_decision_tree(Xtr[:, :1], ytr, max_depth=4, min_samples_leaf=3)
    result = permutation_importance(model.predict, Xev[:, 0], yev, n_repeats=2, seed=0)
    assert result["mean"].shape == (1,)
    assert result["mean"][0] > 0.0


def test_permutation_importance_with_custom_metric():
    def neg_mse(y_true, y_pred):
        return -float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    X, y, _ = make_synthetic_data(n_samples=300, seed=9)
    Xtr, ytr, Xev, yev = _split(X, y)
    model = fit_decision_tree(Xtr, ytr, max_depth=5, min_samples_leaf=5)
    imp = permutation_importance(model.predict, Xev, yev, metric=neg_mse, n_repeats=3, seed=4)
    assert imp["baseline"] > imp["baseline"] - imp["mean"][0]
    assert np.all(imp["mean"] >= -1e-9)
    assert imp["mean"][0] > imp["mean"][4]


def test_permutation_importance_tiny_dataset():
    X, y, _ = make_synthetic_data(n_samples=10, seed=12)
    model = fit_decision_tree(X, y, max_depth=2, min_samples_leaf=2)
    result = permutation_importance(model.predict, X, y, n_repeats=2, seed=0)
    assert result["mean"].shape == (5,)
    assert np.all(np.isfinite(result["mean"]))


def test_permutation_importance_rejects_bad_inputs():
    X, y, _ = make_synthetic_data(n_samples=30, seed=13)
    with pytest.raises(ValueError):
        permutation_importance(lambda z: z[:, 0], X, y, n_repeats=0)
    with pytest.raises(TypeError):
        permutation_importance(42, X, y, n_repeats=2)
    with pytest.raises(TypeError):
        permutation_importance(lambda z: z[:, 0], X, y, metric="r2", n_repeats=2)


def test_drop_column_importance_ranks_known_truth():
    X, y, _ = make_synthetic_data(n_samples=240, seed=14)
    Xtr, ytr, Xev, yev = _split(X, y, n_train=160)
    result = drop_column_importance(_fit_predict_demo, Xtr, ytr)
    imp = result["importance"]
    assert np.all(imp >= -1e-9)
    assert imp[0] > imp[1]
    assert imp[1] > imp[3]
    assert imp[4] < imp[2]
    assert imp[4] < 0.05


def test_drop_column_importance_baseline_equals_full_model_score():
    X, y, _ = make_synthetic_data(n_samples=150, seed=15)
    result = drop_column_importance(_fit_predict_demo, X, y)
    full_r2 = r2_score(y, _fit_predict_demo(X, y, X))
    assert result["baseline"] == pytest.approx(full_r2)


def test_drop_column_importance_rejects_bad_inputs():
    X, y, _ = make_synthetic_data(n_samples=40, seed=16)
    with pytest.raises(TypeError):
        drop_column_importance("not callable", X, y)
    with pytest.raises(ValueError):
        drop_column_importance(_fit_predict_demo, np.ones((5, 2, 2)), y[:5])
