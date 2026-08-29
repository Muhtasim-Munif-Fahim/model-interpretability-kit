"""Tests for 1-D partial dependence, 2-D surfaces, and ICE curves."""

import numpy as np
import pytest

from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.partial_dependence import (
    ice_curves,
    make_grid,
    partial_dependence,
    partial_dependence_2d,
)


def _linear_predict(Z, coefs=(2.0, 1.0)):
    return Z @ np.asarray(coefs, dtype=float)


def test_make_grid_default_is_percentile_based():
    col = np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.9, 0.95, 0.99])
    grid = make_grid(col, grid_points=5)
    assert grid.shape == (5,)
    assert grid[0] >= 0.0
    assert grid[-1] <= 0.99
    assert np.all(np.diff(grid) > 0)


def test_make_grid_constant_feature_has_small_spread():
    grid = make_grid(np.full(40, 3.0), grid_points=4)
    assert grid.shape == (4,)
    assert np.allclose(grid, 3.0, atol=1e-5)


def test_make_grid_explicit_grid_used_verbatim():
    grid = make_grid(np.arange(10.0), grid_points=5, grid=[1.0, 2.0, 3.0])
    assert np.array_equal(grid, np.array([1.0, 2.0, 3.0]))


def test_make_grid_rejects_empty_and_tiny():
    with pytest.raises(ValueError):
        make_grid(np.arange(5.0), grid=[])
    with pytest.raises(ValueError):
        make_grid(np.arange(5.0), grid_points=1)


def test_pdp_1d_monotonic_for_linear_model():
    X = np.random.default_rng(0).uniform(size=(80, 2))
    result = partial_dependence(_linear_predict, X, feature_index=0, grid_points=10)
    assert np.all(np.diff(result["values"]) > 0)
    assert result["feature"] == 0
    assert result["grid"].shape == result["values"].shape == (10,)
    slope = np.gradient(result["values"], result["grid"])
    assert np.allclose(slope, 2.0, atol=1e-8)


def test_pdp_flat_for_unused_feature():
    X = np.random.default_rng(1).uniform(size=(60, 3))
    coefs = (2.0, 1.0, 0.0)
    result = partial_dependence(lambda Z: _linear_predict(Z, coefs), X, feature_index=2)
    assert np.allclose(result["values"], result["values"][0], atol=1e-10)


def test_pdp_custom_grid():
    X = np.random.default_rng(2).uniform(size=(40, 2))
    grid = np.linspace(0.1, 0.9, 7)
    result = partial_dependence(_linear_predict, X, feature_index=1, grid=grid)
    assert np.array_equal(result["grid"], grid)
    assert np.allclose(result["values"], grid + 2.0 * X[:, 0].mean(), atol=1e-8)


def test_pdp_single_feature_1d_input():
    X = np.random.default_rng(3).uniform(size=(50, 1))
    result = partial_dependence(lambda Z: Z[:, 0] ** 2, X, feature_index=0)
    assert np.all(np.diff(result["values"]) > 0)


def test_pdp_constant_feature_flat_curve():
    X = np.column_stack([np.full(30, 0.4), np.random.default_rng(4).uniform(size=30)])
    result = partial_dependence(_linear_predict, X, feature_index=0, grid_points=5)
    assert np.allclose(result["values"], result["values"][0], atol=1e-6)


def test_pdp_invalid_feature_index():
    X = np.random.default_rng(5).uniform(size=(20, 2))
    with pytest.raises(ValueError):
        partial_dependence(_linear_predict, X, feature_index=5)


def test_pdp_averages_over_all_rows():
    X = np.random.default_rng(6).uniform(size=(100, 2))
    result = partial_dependence(_linear_predict, X, feature_index=0, grid_points=3)
    expected_mean = (2.0 * result["grid"] + X[:, 1].mean())
    assert np.allclose(result["values"], expected_mean, atol=1e-8)


def test_pdp_2d_surface_shape():
    X, y, _ = make_synthetic_data(n_samples=120, seed=7)
    model = fit_decision_tree(X, y, max_depth=4, min_samples_leaf=3)
    result = partial_dependence_2d(model.predict, X, (0, 1), grid_points=8)
    assert result["values"].shape == (8, 8)
    assert result["feature0"] == 0
    assert result["feature1"] == 1
    assert result["grid0"].shape == (8,)
    assert result["grid1"].shape == (8,)


def test_pdp_2d_surface_values_match_manual_computation():
    X = np.random.default_rng(8).uniform(size=(50, 2))
    result = partial_dependence_2d(_linear_predict, X, (0, 1), grid_points=5)
    for i in range(5):
        for j in range(5):
            expected = 2.0 * result["grid0"][i] + 1.0 * result["grid1"][j]
            assert result["values"][i, j] == pytest.approx(expected)


def test_pdp_2d_separable_for_additive_model():
    X = np.random.default_rng(9).uniform(size=(40, 2))
    result = partial_dependence_2d(_linear_predict, X, (0, 1), grid_points=6)
    row_diff = np.diff(result["values"], axis=1)
    assert np.allclose(row_diff, row_diff[0], atol=1e-8)


def test_pdp_2d_rejects_same_or_invalid_features():
    X = np.random.default_rng(10).uniform(size=(30, 2))
    with pytest.raises(ValueError):
        partial_dependence_2d(_linear_predict, X, (0, 0))
    with pytest.raises(ValueError):
        partial_dependence_2d(_linear_predict, X, (0, 2))


def test_ice_curves_shape_and_rows():
    X = np.random.default_rng(11).uniform(size=(60, 2))
    result = ice_curves(_linear_predict, X, feature_index=0, grid_points=7, rows=[0, 1, 2])
    assert result["curves"].shape == (3, 7)
    assert np.array_equal(result["rows"], np.array([0, 1, 2]))


def test_ice_curves_default_max_rows():
    X = np.random.default_rng(12).uniform(size=(60, 2))
    result = ice_curves(_linear_predict, X, feature_index=0, grid_points=4, max_rows=5)
    assert result["curves"].shape == (5, 4)
    assert np.array_equal(result["rows"], np.arange(5))


def test_ice_curves_parallel_lines_for_linear_model():
    X = np.random.default_rng(13).uniform(size=(30, 2))
    result = ice_curves(_linear_predict, X, feature_index=0, grid_points=6, rows=[0, 5, 12])
    for k in range(3):
        assert np.allclose(
            np.gradient(result["curves"][k], result["grid"]), 2.0, atol=1e-8
        )


def test_ice_curves_capture_heterogeneous_levels():
    X = np.random.default_rng(14).uniform(size=(20, 2))
    result = ice_curves(_linear_predict, X, feature_index=0, grid_points=3, rows=[0, 9])
    assert result["curves"][0, 0] != pytest.approx(result["curves"][1, 0])


def test_ice_curves_invalid_rows():
    X = np.random.default_rng(15).uniform(size=(20, 2))
    with pytest.raises(ValueError):
        ice_curves(_linear_predict, X, feature_index=0, rows=[0, 100])


def test_ice_curves_single_row():
    X = np.random.default_rng(16).uniform(size=(20, 2))
    result = ice_curves(_linear_predict, X, feature_index=1, grid_points=5, rows=3)
    assert result["curves"].shape == (1, 5)
    assert result["rows"][0] == 3


def test_ice_curves_constant_feature_flat():
    X = np.column_stack([np.full(25, 0.7), np.random.default_rng(17).uniform(size=25)])
    result = ice_curves(_linear_predict, X, feature_index=0, grid_points=3, rows=[0])
    assert np.allclose(result["curves"][0], result["curves"][0, 0], atol=1e-6)
