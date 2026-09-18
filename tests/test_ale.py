"""Tests for 1-D accumulated local effects curves and 2-D interaction surfaces."""

import numpy as np
import pytest

from interpretability import accumulated_local_effects as ale_from_package
from interpretability import accumulated_local_effects_2d as ale_2d_from_package
from interpretability.ale import (
    ale,
    ale_2d,
    accumulated_local_effects,
    accumulated_local_effects_2d,
    make_ale_grid,
)
from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.partial_dependence import partial_dependence


def _linear_predict(Z, coefs=(2.0, 1.0)):
    return Z @ np.asarray(coefs, dtype=float)


def test_ale_exported_from_public_api():
    assert ale_from_package is accumulated_local_effects
    assert ale is accumulated_local_effects
    assert ale_2d_from_package is accumulated_local_effects_2d
    assert ale_2d is accumulated_local_effects_2d


def test_make_ale_grid_default_is_quantile_edges():
    col = np.linspace(0.0, 1.0, 101)
    grid = make_ale_grid(col, grid_points=5)
    assert grid.shape == (5,)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(1.0)
    assert np.all(np.diff(grid) > 0)


def test_make_ale_grid_constant_feature_has_small_spread():
    grid = make_ale_grid(np.full(40, 3.0), grid_points=4)
    assert grid.shape == (2,)
    assert np.allclose(grid, 3.0, atol=1e-5)


def test_make_ale_grid_explicit_grid_sorted_unique():
    grid = make_ale_grid(np.arange(10.0), grid_points=5, grid=[3.0, 1.0, 2.0, 1.0])
    assert np.array_equal(grid, np.array([1.0, 2.0, 3.0]))


def test_make_ale_grid_rejects_empty_tiny_and_no_samples():
    with pytest.raises(ValueError):
        make_ale_grid(np.arange(5.0), grid=[])
    with pytest.raises(ValueError):
        make_ale_grid(np.arange(5.0), grid=[1.0, 1.0])
    with pytest.raises(ValueError):
        make_ale_grid(np.arange(5.0), grid_points=1)
    with pytest.raises(ValueError):
        make_ale_grid(np.array([]), grid_points=4)


def test_ale_recovers_centered_linear_effect():
    X = np.random.default_rng(0).uniform(size=(80, 2))
    result = accumulated_local_effects(_linear_predict, X, feature_index=0, grid_points=10)
    assert result["feature"] == 0
    assert result["grid"].shape == result["values"].shape == (10,)
    assert result["counts"].shape == (9,)
    assert int(result["counts"].sum()) == X.shape[0]
    expected = 2.0 * (result["grid"] - X[:, 0].mean())
    assert np.allclose(result["values"], expected, atol=1e-8)
    slope = np.gradient(result["values"], result["grid"])
    assert np.allclose(slope, 2.0, atol=1e-8)


def test_ale_centered_over_observations():
    X = np.random.default_rng(1).uniform(size=(60, 2))
    result = accumulated_local_effects(_linear_predict, X, feature_index=1, grid_points=8)
    ale_at_obs = np.interp(X[:, 1], result["grid"], result["values"])
    assert float(ale_at_obs.mean()) == pytest.approx(0.0, abs=1e-10)


def test_ale_flat_zero_for_unused_feature():
    X = np.random.default_rng(2).uniform(size=(60, 3))
    coefs = (2.0, 1.0, 0.0)
    result = accumulated_local_effects(
        lambda Z: _linear_predict(Z, coefs), X, feature_index=2
    )
    assert np.allclose(result["values"], 0.0, atol=1e-10)


def test_ale_custom_grid():
    X = np.random.default_rng(3).uniform(size=(40, 2))
    grid = np.linspace(0.1, 0.9, 7)
    result = accumulated_local_effects(_linear_predict, X, feature_index=1, grid=grid)
    assert np.array_equal(result["grid"], grid)
    clipped = np.clip(X[:, 1], grid[0], grid[-1])
    expected = 1.0 * (grid - clipped.mean())
    assert np.allclose(result["values"], expected, atol=1e-8)


def test_ale_single_feature_1d_input():
    X = np.random.default_rng(4).uniform(size=(50, 1))
    result = accumulated_local_effects(lambda Z: Z[:, 0] ** 2, X, feature_index=0)
    assert np.all(np.diff(result["values"]) > 0)


def test_ale_constant_feature_flat_curve():
    X = np.column_stack([np.full(30, 0.4), np.random.default_rng(5).uniform(size=30)])
    result = accumulated_local_effects(_linear_predict, X, feature_index=0, grid_points=5)
    assert np.allclose(result["values"], 0.0, atol=1e-5)


def test_ale_invalid_feature_index():
    X = np.random.default_rng(6).uniform(size=(20, 2))
    with pytest.raises(ValueError):
        accumulated_local_effects(_linear_predict, X, feature_index=5)


def test_ale_empty_X_rejected():
    with pytest.raises(ValueError):
        accumulated_local_effects(_linear_predict, np.empty((0, 2)), feature_index=0)


def test_ale_matches_centered_pdp_for_additive_model():
    X = np.random.default_rng(7).uniform(size=(70, 2))
    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 12)
    ale_result = accumulated_local_effects(_linear_predict, X, 0, grid=grid)
    pdp = partial_dependence(_linear_predict, X, 0, grid=grid)
    centered_pdp = pdp["values"] - float(np.interp(X[:, 0], grid, pdp["values"]).mean())
    assert np.allclose(ale_result["values"], centered_pdp, atol=1e-8)


def test_ale_step_function_jump():
    rng = np.random.default_rng(8)
    X = rng.uniform(size=(200, 2))

    def step_predict(Z):
        return 2.0 * (Z[:, 0] > 0.5).astype(float)

    result = accumulated_local_effects(step_predict, X, feature_index=0, grid_points=21)
    assert result["values"][-1] - result["values"][0] == pytest.approx(2.0, abs=1e-8)
    jump_idx = int(np.argmax(np.diff(result["values"])))
    assert result["grid"][jump_idx] <= 0.5 <= result["grid"][jump_idx + 1]


def test_ale_empty_interval_has_zero_local_effect():
    X = np.random.default_rng(9).uniform(0.0, 0.2, size=(40, 2))
    grid = np.array([0.0, 0.2, 1.0])
    result = accumulated_local_effects(_linear_predict, X, feature_index=0, grid=grid)
    assert result["counts"][1] == 0
    assert result["values"][2] == pytest.approx(result["values"][1])


def test_ale_on_demo_tree_tracks_synthetic_signal():
    X, y, _ = make_synthetic_data(n_samples=250, seed=11)
    model = fit_decision_tree(X, y, max_depth=6, min_samples_leaf=5)
    ale0 = accumulated_local_effects(model.predict, X, feature_index=0, grid_points=15)
    ale1 = accumulated_local_effects(model.predict, X, feature_index=1, grid_points=15)
    ale4 = accumulated_local_effects(model.predict, X, feature_index=4, grid_points=15)
    assert ale0["grid"].shape == ale0["values"].shape
    assert ale0["values"][-1] > ale0["values"][0]
    assert ale1["values"][-1] > ale1["values"][0]
    signal_span = max(
        float(np.ptp(ale0["values"])),
        float(np.ptp(ale1["values"])),
    )
    assert float(np.ptp(ale4["values"])) < 0.25 * signal_span
    assert float(np.max(np.abs(ale4["values"]))) < 0.5


def test_ale_on_synthetic_linear_predict_ranks_features():
    X, _, _ = make_synthetic_data(n_samples=180, seed=12, noise=0.0)

    def additive_predict(Z):
        return 4.0 * Z[:, 0] + 3.0 * Z[:, 1] + 2.0 * (Z[:, 2] > 0.5) + 1.5 * Z[:, 3]

    spans = []
    for f in range(5):
        result = accumulated_local_effects(additive_predict, X, feature_index=f, grid_points=16)
        spans.append(float(np.ptp(result["values"])))
    assert spans[0] == pytest.approx(4.0 * (X[:, 0].max() - X[:, 0].min()), abs=1e-6)
    assert spans[4] == pytest.approx(0.0, abs=1e-10)
    assert spans[0] > spans[1] > spans[3] > spans[4]


def _corner_weighted_mean(values, counts):
    corners = (
        values[:-1, :-1] + values[:-1, 1:] + values[1:, :-1] + values[1:, 1:]
    ) / 4.0
    total = float(counts.sum())
    return float((counts * corners).sum() / total)


def test_ale_2d_surface_shape_and_counts():
    X = np.random.default_rng(20).uniform(size=(80, 3))
    result = accumulated_local_effects_2d(
        lambda Z: _linear_predict(Z, (2.0, 1.0, 0.0)), X, (0, 1), grid_points=8
    )
    assert result["feature0"] == 0
    assert result["feature1"] == 1
    assert result["grid0"].shape == (8,)
    assert result["grid1"].shape == (8,)
    assert result["values"].shape == (8, 8)
    assert result["counts"].shape == (7, 7)
    assert int(result["counts"].sum()) == X.shape[0]


def test_ale_2d_zero_for_additive_model():
    X = np.random.default_rng(21).uniform(size=(100, 2))
    result = accumulated_local_effects_2d(_linear_predict, X, (0, 1), grid_points=10)
    assert np.allclose(result["values"], 0.0, atol=1e-10)


def test_ale_2d_zero_when_one_feature_unused():
    X = np.random.default_rng(22).uniform(size=(80, 3))
    result = accumulated_local_effects_2d(
        lambda Z: _linear_predict(Z, (2.0, 1.0, 0.0)), X, (0, 2), grid_points=8
    )
    assert np.allclose(result["values"], 0.0, atol=1e-10)


def test_ale_2d_recovers_centered_multiplicative_interaction():
    grid0 = np.linspace(0.0, 1.0, 9)
    grid1 = np.linspace(0.0, 1.0, 8)
    mid0 = 0.5 * (grid0[:-1] + grid0[1:])
    mid1 = 0.5 * (grid1[:-1] + grid1[1:])
    xx, yy = np.meshgrid(mid0, mid1, indexing="ij")
    X = np.column_stack([xx.ravel(), yy.ravel()])

    def product_predict(Z):
        return Z[:, 0] * Z[:, 1]

    result = accumulated_local_effects_2d(
        product_predict, X, (0, 1), grids=(grid0, grid1)
    )
    assert np.array_equal(result["grid0"], grid0)
    assert np.array_equal(result["grid1"], grid1)
    assert np.all(result["counts"] == 1)
    expected = np.outer(grid0 - X[:, 0].mean(), grid1 - X[:, 1].mean())
    expected = expected - _corner_weighted_mean(expected, result["counts"])
    assert np.allclose(result["values"], expected, atol=1e-10)


def test_ale_2d_centered_over_cells():
    X = np.random.default_rng(24).uniform(size=(90, 2))
    result = accumulated_local_effects_2d(
        lambda Z: Z[:, 0] * Z[:, 1], X, (0, 1), grid_points=7
    )
    assert _corner_weighted_mean(result["values"], result["counts"]) == pytest.approx(
        0.0, abs=1e-10
    )


def test_ale_2d_custom_grids():
    X = np.random.default_rng(25).uniform(size=(50, 2))
    grids = (np.linspace(0.1, 0.9, 5), np.linspace(0.2, 0.8, 4))
    result = accumulated_local_effects_2d(_linear_predict, X, (0, 1), grids=grids)
    assert np.array_equal(result["grid0"], grids[0])
    assert np.array_equal(result["grid1"], grids[1])
    assert result["values"].shape == (5, 4)
    assert result["counts"].shape == (4, 3)
    assert np.allclose(result["values"], 0.0, atol=1e-10)


def test_ale_2d_empty_cell_has_zero_local_effect():
    X = np.random.default_rng(26).uniform(0.0, 0.2, size=(40, 2))
    grids = (np.array([0.0, 0.2, 1.0]), np.array([0.0, 0.2, 1.0]))
    result = accumulated_local_effects_2d(
        lambda Z: Z[:, 0] * Z[:, 1], X, (0, 1), grids=grids
    )
    assert result["counts"][1, 1] == 0
    assert result["counts"][0, 1] == 0
    assert result["counts"][1, 0] == 0
    assert int(result["counts"][0, 0]) == X.shape[0]
    second = (
        result["values"][1:, 1:]
        - result["values"][:-1, 1:]
        - result["values"][1:, :-1]
        + result["values"][:-1, :-1]
    )
    assert second[1, 1] == pytest.approx(0.0, abs=1e-12)
    assert second[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert second[1, 0] == pytest.approx(0.0, abs=1e-12)


def test_ale_2d_rejects_same_or_invalid_features():
    X = np.random.default_rng(27).uniform(size=(30, 2))
    with pytest.raises(ValueError):
        accumulated_local_effects_2d(_linear_predict, X, (0, 0))
    with pytest.raises(ValueError):
        accumulated_local_effects_2d(_linear_predict, X, (0, 2))


def test_ale_2d_empty_X_rejected():
    with pytest.raises(ValueError):
        accumulated_local_effects_2d(_linear_predict, np.empty((0, 2)), (0, 1))


def test_ale_2d_constant_feature_flat_surface():
    rng = np.random.default_rng(28)
    X = np.column_stack([np.full(40, 0.4), rng.uniform(size=40)])
    result = accumulated_local_effects_2d(_linear_predict, X, (0, 1), grid_points=5)
    assert np.allclose(result["values"], 0.0, atol=1e-8)


def test_ale_2d_interaction_larger_than_additive():
    X = np.random.default_rng(29).uniform(size=(150, 3))
    additive = accumulated_local_effects_2d(
        lambda Z: 4.0 * Z[:, 0] + 3.0 * Z[:, 1], X, (0, 1), grid_points=8
    )
    interaction = accumulated_local_effects_2d(
        lambda Z: 4.0 * Z[:, 0] * Z[:, 1], X, (0, 1), grid_points=8
    )
    assert float(np.max(np.abs(additive["values"]))) < 1e-10
    assert float(np.ptp(interaction["values"])) > 0.2


def test_ale_2d_on_demo_tree_additive_pair_is_small():
    X, y, _ = make_synthetic_data(n_samples=250, seed=30)
    model = fit_decision_tree(X, y, max_depth=6, min_samples_leaf=5)
    surface = accumulated_local_effects_2d(model.predict, X, (0, 1), grid_points=8)
    noise = accumulated_local_effects_2d(model.predict, X, (0, 4), grid_points=8)
    assert surface["values"].shape == (8, 8)
    assert np.isfinite(surface["values"]).all()
    assert int(surface["counts"].sum()) == X.shape[0]
    assert float(np.max(np.abs(noise["values"]))) <= float(np.max(np.abs(surface["values"]))) + 0.5


def test_ale_2d_on_synthetic_product_predict_ranks_pairs():
    X, _, _ = make_synthetic_data(n_samples=180, seed=31, noise=0.0)

    def with_interaction(Z):
        return 4.0 * Z[:, 0] + 3.0 * Z[:, 1] + 2.5 * Z[:, 0] * Z[:, 1]

    interacting = accumulated_local_effects_2d(with_interaction, X, (0, 1), grid_points=10)
    unused = accumulated_local_effects_2d(with_interaction, X, (0, 4), grid_points=10)
    assert float(np.ptp(interacting["values"])) > 5.0 * float(np.ptp(unused["values"]))
    assert np.allclose(unused["values"], 0.0, atol=1e-10)
