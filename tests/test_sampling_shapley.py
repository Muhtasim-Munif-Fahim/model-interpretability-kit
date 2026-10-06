"""Tests for permutation-sampling Shapley values."""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pytest

from interpretability import sampling_shapley as exported_sampling_shapley
from interpretability.cli import main
from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.local import sampling_shapley, tree_shap_values


def _exact_interventional_shapley(predict, x_row, background):
    """Brute-force Shapley values of v(S) = mean_z f(x_S, z_~S)."""
    m = x_row.shape[0]

    def value(subset):
        Z = background.copy()
        Z[:, list(subset)] = x_row[list(subset)]
        return float(np.mean(predict(Z)))

    phi = np.zeros(m)
    for j in range(m):
        others = [k for k in range(m) if k != j]
        for size in range(m):
            weight = factorial(size) * factorial(m - size - 1) / factorial(m)
            for subset in combinations(others, size):
                phi[j] += weight * (value(subset + (j,)) - value(subset))
    return phi


def _nonlinear(Z):
    return np.sin(Z[:, 0]) * Z[:, 1] + Z[:, 2] ** 2 - 0.5 * Z[:, 0] * Z[:, 2]


def test_export():
    assert exported_sampling_shapley is sampling_shapley


@pytest.mark.parametrize("antithetic", [True, False])
def test_efficiency_holds_exactly(antithetic):
    rng = np.random.default_rng(0)
    background = rng.normal(size=(50, 4))
    x_row = np.array([0.5, -1.0, 2.0, 0.3])
    res = sampling_shapley(
        lambda Z: _nonlinear(Z) + Z[:, 3], x_row, background,
        n_permutations=7, antithetic=antithetic, seed=1,
    )
    assert res["prediction"] == pytest.approx(float(_nonlinear(x_row[None])[0] + x_row[3]))
    assert float(np.sum(res["values"])) == pytest.approx(res["prediction"] - res["baseline"], abs=1e-12)


def test_linear_model_single_background_row_is_exact():
    coefs = np.array([2.0, -1.0, 0.5])
    z = np.array([[1.0, 1.0, 1.0]])
    x_row = np.array([0.3, -0.7, 1.2])
    res = sampling_shapley(lambda Z: Z @ coefs, x_row, z, n_permutations=3, seed=0)
    np.testing.assert_allclose(res["values"], coefs * (x_row - z[0]), atol=1e-12)
    np.testing.assert_allclose(res["std_error"], 0.0, atol=1e-12)


def test_antithetic_splits_pure_interaction_exactly():
    # f = x0 * x1 with z = 0: each ordering credits the whole product to the
    # feature switched second; a permutation plus its reverse splits it evenly.
    res = sampling_shapley(
        lambda Z: Z[:, 0] * Z[:, 1], np.array([2.0, 3.0]), np.zeros((1, 2)),
        n_permutations=5, antithetic=True, seed=0,
    )
    np.testing.assert_allclose(res["values"], [3.0, 3.0], atol=1e-12)


def test_dummy_feature_gets_zero():
    rng = np.random.default_rng(3)
    background = rng.normal(size=(30, 4))
    res = sampling_shapley(
        lambda Z: _nonlinear(Z[:, :3]), np.array([1.0, 2.0, -1.0, 9.0]),
        background, n_permutations=20, seed=0,
    )
    assert res["values"][3] == pytest.approx(0.0, abs=1e-12)
    assert res["std_error"][3] == pytest.approx(0.0, abs=1e-12)


def test_converges_to_brute_force_interventional_shapley():
    rng = np.random.default_rng(4)
    background = rng.normal(size=(12, 3))
    x_row = np.array([1.2, -0.4, 0.8])
    exact = _exact_interventional_shapley(_nonlinear, x_row, background)
    res = sampling_shapley(_nonlinear, x_row, background, n_permutations=4000, seed=5)
    err = np.abs(res["values"] - exact)
    assert np.all(err <= 5 * res["std_error"] + 1e-9)
    assert np.max(err) < 0.1


def test_demo_tree_matches_brute_force_and_zeroes_unused_features():
    X, y, _ = make_synthetic_data(n_samples=200, seed=7)
    model = fit_decision_tree(X, y, max_depth=4, min_samples_leaf=10)
    background = X[:20]
    x_row = X[150]
    exact = _exact_interventional_shapley(model.predict, x_row, background)
    res = sampling_shapley(model.predict, x_row, background, n_permutations=3000, seed=0)
    np.testing.assert_allclose(res["values"], exact, atol=5 * res["std_error"].max() + 1e-9)
    # Features the tree never splits on get exactly zero credit, as with TreeSHAP.
    tree = tree_shap_values(model.root_, x_row, background, feature_count=X.shape[1])
    unused = np.asarray(tree["values"]) == 0.0
    assert unused.any()
    np.testing.assert_array_equal(res["values"][unused], 0.0)


def test_antithetic_reduces_error_on_additive_model():
    rng = np.random.default_rng(8)
    background = rng.normal(size=(1, 6))
    coefs = rng.normal(size=6)
    x_row = rng.normal(size=6)

    def f(Z):
        return Z @ coefs + 0.1 * Z[:, 0] * Z[:, 1]

    exact = _exact_interventional_shapley(f, x_row, background)
    err_plain = []
    err_anti = []
    for seed in range(10):
        plain = sampling_shapley(f, x_row, background, n_permutations=10, antithetic=False, seed=seed)
        anti = sampling_shapley(f, x_row, background, n_permutations=10, antithetic=True, seed=seed)
        err_plain.append(np.abs(plain["values"] - exact).max())
        err_anti.append(np.abs(anti["values"] - exact).max())
    assert np.mean(err_anti) < np.mean(err_plain)


def test_single_batched_predict_call_and_determinism():
    calls = []

    def f(Z):
        calls.append(Z.shape[0])
        return _nonlinear(Z)

    rng = np.random.default_rng(9)
    background = rng.normal(size=(15, 3))
    a = sampling_shapley(f, np.ones(3), background, n_permutations=6, seed=11)
    assert calls == [2 * 6 * 4]
    assert a["n_evaluations"] == 48
    assert a["n_permutations"] == 6
    b = sampling_shapley(f, np.ones(3), background, n_permutations=6, seed=11)
    np.testing.assert_array_equal(a["values"], b["values"])
    assert a["feature_names"] == ["X0", "X1", "X2"]


def test_single_draw_without_antithetic_has_nan_std_error():
    res = sampling_shapley(_nonlinear, np.ones(3), np.zeros((2, 3)), n_permutations=1, antithetic=False, seed=0)
    assert np.isnan(res["std_error"]).all()


def test_validation():
    bg = np.zeros((4, 3))
    with pytest.raises(ValueError, match="same number of features"):
        sampling_shapley(_nonlinear, np.ones(2), bg)
    with pytest.raises(ValueError, match="n_permutations"):
        sampling_shapley(_nonlinear, np.ones(3), bg, n_permutations=0)
    with pytest.raises(ValueError, match="n_permutations"):
        sampling_shapley(_nonlinear, np.ones(3), bg, n_permutations=2.5)
    with pytest.raises(ValueError, match="n_permutations"):
        sampling_shapley(_nonlinear, np.ones(3), bg, n_permutations=True)
    with pytest.raises(ValueError, match="feature_names"):
        sampling_shapley(_nonlinear, np.ones(3), bg, feature_names=["a"])
    with pytest.raises(ValueError, match="one value per input row"):
        sampling_shapley(lambda Z: np.zeros(1), np.ones(3), bg)


def test_cli_subcommand(capsys):
    code = main(["--seed", "3", "--n-samples", "120", "sampling-shapley", "--rows", "0", "--n-permutations", "20"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Sampling Shapley for row 0" in out
    assert "sum(phi)" in out
    line = [ln for ln in out.splitlines() if "sum(phi)" in ln][0]
    parts = line.replace("=", " ").split()
    assert float(parts[1]) == pytest.approx(float(parts[3]), abs=1e-3)
