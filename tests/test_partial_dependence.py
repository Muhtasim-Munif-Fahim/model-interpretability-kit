"""Tests for 1-D partial dependence, 2-D surfaces, and ICE curves."""

import numpy as np
import pytest

from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.partial_dependence import (
    _normal_ppf,
    ice,
    ice_curves,
    make_grid,
    partial_dependence,
    partial_dependence_2d,
    pdp,
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


def test_pdp_and_ice_exported_from_public_api():
    from interpretability import ice as ice_from_package
    from interpretability import ice_curves as ice_curves_from_package
    from interpretability import partial_dependence as pdp_from_package
    from interpretability import pdp as pdp_alias_from_package

    assert ice_from_package is ice_curves
    assert ice_curves_from_package is ice_curves
    assert ice is ice_curves
    assert pdp_from_package is partial_dependence
    assert pdp_alias_from_package is partial_dependence
    assert pdp is partial_dependence


def test_normal_ppf_known_quantiles():
    assert _normal_ppf(0.5) == pytest.approx(0.0, abs=1e-10)
    assert _normal_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-8)
    assert _normal_ppf(0.025) == pytest.approx(-1.959963984540054, abs=1e-8)
    with pytest.raises(ValueError):
        _normal_ppf(0.0)
    with pytest.raises(ValueError):
        _normal_ppf(1.0)


def test_pdp_without_conf_level_omits_band_keys():
    X = np.random.default_rng(18).uniform(size=(40, 2))
    result = partial_dependence(_linear_predict, X, feature_index=0, grid_points=5)
    assert "lower" not in result
    assert "upper" not in result
    assert "std" not in result
    assert "conf_level" not in result


def test_pdp_confidence_bands_match_row_standard_error():
    rng = np.random.default_rng(19)
    X = rng.uniform(size=(80, 2))
    result = partial_dependence(
        _linear_predict, X, feature_index=0, grid_points=10, conf_level=0.95
    )
    assert result["conf_level"] == pytest.approx(0.95)
    assert result["std"].shape == result["lower"].shape == result["upper"].shape == (10,)
    expected_mean = 2.0 * result["grid"] + X[:, 1].mean()
    expected_std = np.full(10, X[:, 1].std(ddof=1))
    z = 1.959963984540054
    se = expected_std / np.sqrt(X.shape[0])
    assert np.allclose(result["values"], expected_mean, atol=1e-8)
    assert np.allclose(result["std"], expected_std, atol=1e-8)
    assert np.allclose(result["lower"], expected_mean - z * se, atol=1e-6)
    assert np.allclose(result["upper"], expected_mean + z * se, atol=1e-6)
    assert np.all(result["lower"] < result["values"])
    assert np.all(result["values"] < result["upper"])


def test_pdp_confidence_band_width_constant_for_additive_model():
    X = np.random.default_rng(20).uniform(size=(50, 2))
    result = partial_dependence(
        _linear_predict, X, feature_index=0, grid_points=6, conf_level=0.9
    )
    width = result["upper"] - result["lower"]
    assert np.allclose(width, width[0], atol=1e-10)


def test_pdp_confidence_bands_flat_std_for_unused_feature():
    X = np.random.default_rng(21).uniform(size=(60, 3))
    coefs = (2.0, 1.0, 0.0)
    result = partial_dependence(
        lambda Z: _linear_predict(Z, coefs), X, feature_index=2, conf_level=0.95
    )
    assert np.allclose(result["values"], result["values"][0], atol=1e-10)
    assert np.allclose(result["std"], result["std"][0], atol=1e-10)
    assert np.all(result["upper"] > result["lower"])


def test_pdp_confidence_bands_single_row_have_zero_width():
    X = np.array([[0.2, 0.4]])
    result = partial_dependence(
        _linear_predict, X, feature_index=0, grid_points=4, conf_level=0.95
    )
    assert np.allclose(result["std"], 0.0)
    assert np.allclose(result["lower"], result["values"])
    assert np.allclose(result["upper"], result["values"])


def test_pdp_rejects_invalid_conf_level():
    X = np.random.default_rng(22).uniform(size=(20, 2))
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            partial_dependence(_linear_predict, X, feature_index=0, conf_level=bad)


def test_pdp_is_mean_of_ice_curves():
    X = np.random.default_rng(23).uniform(size=(40, 2))
    grid = np.linspace(0.1, 0.9, 8)
    pdp_result = partial_dependence(_linear_predict, X, feature_index=1, grid=grid)
    ice_result = ice_curves(
        _linear_predict, X, feature_index=1, grid=grid, rows=np.arange(X.shape[0])
    )
    assert np.allclose(pdp_result["values"], ice_result["curves"].mean(axis=0), atol=1e-10)


def test_ice_centered_curves_start_at_zero():
    X = np.random.default_rng(24).uniform(size=(30, 2))
    result = ice_curves(
        _linear_predict, X, feature_index=0, grid_points=6, rows=[0, 4, 9], centered=True
    )
    assert result["centered"] is True
    assert np.allclose(result["curves"][:, 0], 0.0, atol=1e-12)


def test_ice_centered_preserves_within_curve_deltas():
    X = np.random.default_rng(25).uniform(size=(20, 2))
    raw = ice_curves(_linear_predict, X, feature_index=1, grid_points=5, rows=[0, 3])
    centered = ice_curves(
        _linear_predict, X, feature_index=1, grid_points=5, rows=[0, 3], centered=True
    )
    assert np.allclose(
        np.diff(raw["curves"], axis=1), np.diff(centered["curves"], axis=1), atol=1e-12
    )
    assert centered["centered"] is True
    assert raw["centered"] is False


def test_ice_centered_collapses_parallel_linear_curves():
    X = np.random.default_rng(26).uniform(size=(25, 2))
    result = ice_curves(
        _linear_predict, X, feature_index=0, grid_points=7, rows=[0, 1, 2, 3], centered=True
    )
    assert np.allclose(result["curves"], result["curves"][0], atol=1e-10)
    slope = np.gradient(result["curves"][0], result["grid"])
    assert np.allclose(slope, 2.0, atol=1e-8)
