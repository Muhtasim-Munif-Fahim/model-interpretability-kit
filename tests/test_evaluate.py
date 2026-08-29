"""Tests for faithfulness metrics."""

import numpy as np
import pytest

from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.evaluate import top_feature_overlap, weighted_linear_fidelity


def _linear_predict(Z, coefs=(2.0, 1.0)):
    return Z @ np.asarray(coefs, dtype=float)


def _background(n=200, seed=0):
    return np.random.default_rng(seed).uniform(size=(n, 2))


def test_fidelity_perfect_for_linear_model():
    background = _background()
    x_row = background[4]
    r2 = weighted_linear_fidelity(_linear_predict, x_row, background, n_samples=400, seed=1)
    assert r2 == pytest.approx(1.0, abs=1e-8)


def test_fidelity_reproducible_with_seed():
    background = _background(n=150, seed=2)
    x_row = background[6]
    r1 = weighted_linear_fidelity(_linear_predict, x_row, background, n_samples=200, seed=7)
    r2 = weighted_linear_fidelity(_linear_predict, x_row, background, n_samples=200, seed=7)
    assert r1 == pytest.approx(r2)


def test_fidelity_bounded_for_demo_tree():
    X, y, _ = make_synthetic_data(n_samples=300, seed=3)
    model = fit_decision_tree(X, y, max_depth=5, min_samples_leaf=5)
    r2 = weighted_linear_fidelity(model.predict, X[0], X, n_samples=150, seed=4)
    assert 0.0 <= r2 <= 1.0


def test_fidelity_positive_for_piecewise_constant_demo_tree():
    X, y, _ = make_synthetic_data(n_samples=300, seed=5)
    model = fit_decision_tree(X, y, max_depth=5, min_samples_leaf=5)
    r2 = weighted_linear_fidelity(model.predict, X[1], X, n_samples=200, seed=6)
    assert 0.0 < r2 <= 1.0
    assert r2 > 0.3


def test_fidelity_lower_for_nonlinear_neighborhood():
    background = np.random.default_rng(8).uniform(size=(200, 2))
    x_row = background[np.argmin(np.abs(background[:, 0] - 0.5))]
    r2 = weighted_linear_fidelity(
        lambda Z: np.abs(Z[:, 0] - 0.5), x_row, background, n_samples=300, seed=9
    )
    assert 0.0 <= r2 < 0.9


def test_top_feature_overlap_identical_rankings():
    result = top_feature_overlap(np.array([1.0, 0.5, 0.2]), np.array([0.9, 0.4, 0.1]), k=2)
    assert result["overlap"] == 2
    assert result["jaccard"] == pytest.approx(1.0)
    assert result["local_top"] == [0, 1]
    assert result["global_top"] == [0, 1]


def test_top_feature_overlap_disjoint_rankings():
    local = np.array([0.2, 0.5, 1.0])
    global_ = np.array([1.0, 0.5, 0.2])
    result = top_feature_overlap(local, global_, k=1)
    assert result["overlap"] == 0
    assert result["jaccard"] == pytest.approx(0.0)
    assert result["local_top"] == [2]
    assert result["global_top"] == [0]


def test_top_feature_overlap_uses_absolute_local_weights():
    local = np.array([-1.0, 0.01, 0.0])
    global_ = np.array([0.9, 0.2, 0.1])
    result = top_feature_overlap(local, global_, k=1)
    assert result["local_top"] == [0]
    assert result["overlap"] == 1


def test_top_feature_overlap_k_capped_at_feature_count():
    local = np.array([1.0, 0.0])
    global_ = np.array([0.0, 1.0])
    result = top_feature_overlap(local, global_, k=10)
    assert result["overlap"] == 2
    assert result["jaccard"] == pytest.approx(1.0)


def test_top_feature_overlap_rejects_bad_inputs():
    with pytest.raises(ValueError):
        top_feature_overlap(np.ones(3), np.ones(2), k=2)
    with pytest.raises(ValueError):
        top_feature_overlap(np.ones(3), np.ones(3), k=0)
