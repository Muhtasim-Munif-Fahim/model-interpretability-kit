"""Tests for the Friedman H-statistic."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability import friedman_h_statistic as h_from_package
from interpretability import h_statistic
from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.interaction import friedman_h_statistic


def test_exported_from_public_api():
    assert h_from_package is friedman_h_statistic
    assert h_statistic is friedman_h_statistic


def test_additive_model_has_near_zero_h():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return 2.0 * Z[:, 0] - 1.5 * Z[:, 1]

    result = friedman_h_statistic(predict, X, (0, 1), grid_points=15)
    assert result["feature_indices"] == (0, 1)
    assert 0.0 <= result["h"] <= 1.0
    assert result["h"] == pytest.approx(result["h_squared"] ** 0.5)
    assert result["h"] < 0.05


def test_multiplicative_interaction_has_large_h():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, size=(250, 2))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0] * Z[:, 1]

    result = friedman_h_statistic(predict, X, (0, 1), grid_points=20)
    assert result["h"] > 0.5


def test_rejects_bad_inputs():
    X = np.zeros((10, 3))
    predict = lambda Z: Z[:, 0]
    with pytest.raises(ValueError, match="exactly two"):
        friedman_h_statistic(predict, X, (0,))
    with pytest.raises(ValueError, match="distinct"):
        friedman_h_statistic(predict, X, (1, 1))
    with pytest.raises(ValueError, match="grid_points"):
        friedman_h_statistic(predict, X, (0, 1), grid_points=1)


def test_tree_model_runs_and_is_bounded():
    X, y, _names = make_synthetic_data(n_samples=180, seed=4)
    model = fit_decision_tree(X, y, max_depth=4, min_samples_leaf=5)
    result = friedman_h_statistic(model.predict, X, (0, 1), grid_points=10)
    assert 0.0 <= result["h"] <= 1.0
    assert result["numerator"] >= 0.0
    assert result["denominator"] >= 0.0
