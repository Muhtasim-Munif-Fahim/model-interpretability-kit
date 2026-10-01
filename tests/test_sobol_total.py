"""Tests for Sobol total-order sensitivity indices."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability import sobol_total_order as exported_st
from interpretability.importance import sobol_first_order, sobol_total_order


def test_export():
    assert exported_st is sobol_total_order


def test_additive_linear_model_ranks_features():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(200, 3))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return 3.0 * Z[:, 0] + 0.1 * Z[:, 1] + 0.0 * Z[:, 2]

    result = sobol_total_order(predict, X, n_samples=800, seed=1)
    assert result["ST"].shape == (3,)
    assert result["S1"].shape == (3,)
    assert result["ST"][0] > result["ST"][1]
    assert result["ST"][0] > result["ST"][2]
    assert result["ST"][0] > 0.7
    assert result["ST"][1] < 0.2
    assert result["ST"][2] < 0.15
    assert result["variance"] > 0.0
    # Additive model: ST ≈ S1
    assert result["ST"][0] == pytest.approx(result["S1"][0], abs=0.15)


def test_irrelevant_feature_near_zero():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(150, 2))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0] ** 2

    result = sobol_total_order(predict, X, n_samples=600, seed=3)
    assert result["ST"][0] > 0.8
    assert result["ST"][1] < 0.15


def test_interaction_inflates_st_over_s1():
    """For f = x0 * x1, first-order indices are small but total-order are large."""
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(200, 2))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0] * Z[:, 1]

    result = sobol_total_order(predict, X, n_samples=1000, seed=8)
    # Pure interaction: S1 near 0, ST near 1 for both features.
    assert result["S1"][0] < 0.25
    assert result["S1"][1] < 0.25
    assert result["ST"][0] > 0.5
    assert result["ST"][1] > 0.5
    assert result["ST"][0] > result["S1"][0]
    assert result["ST"][1] > result["S1"][1]


def test_reproducible_with_seed():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 3))

    def predict(Z):
        return np.asarray(Z, dtype=float).sum(axis=1)

    r1 = sobol_total_order(predict, X, n_samples=100, seed=9)
    r2 = sobol_total_order(predict, X, n_samples=100, seed=9)
    assert np.allclose(r1["ST"], r2["ST"])
    assert np.allclose(r1["S1"], r2["S1"])


def test_rejects_bad_n_samples():
    X = np.random.randn(20, 2)
    with pytest.raises(ValueError):
        sobol_total_order(lambda Z: Z[:, 0], X, n_samples=1)
    with pytest.raises(ValueError):
        sobol_total_order(lambda Z: Z[:, 0], X, n_samples=True)  # type: ignore[arg-type]


def test_rejects_1d_X():
    with pytest.raises(ValueError):
        sobol_total_order(lambda Z: Z[:, 0], np.ones(10), n_samples=20)


def test_indices_in_unit_interval():
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(100, 4))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0] * Z[:, 1] + Z[:, 2]

    result = sobol_total_order(predict, X, n_samples=400, seed=6)
    assert np.all(result["ST"] >= 0.0)
    assert np.all(result["ST"] <= 1.0)
    assert np.all(result["S1"] >= 0.0)
    assert np.all(result["S1"] <= 1.0)


def test_s1_matches_sobol_first_order_same_seed():
    rng = np.random.default_rng(11)
    X = rng.uniform(size=(120, 3))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return 2.0 * Z[:, 0] + Z[:, 1]

    st = sobol_total_order(predict, X, n_samples=500, seed=12)
    s1 = sobol_first_order(predict, X, n_samples=500, seed=12)
    assert np.allclose(st["S1"], s1["S1"], atol=1e-10)


def test_cli_sobol_total(capsys):
    from interpretability.cli import main

    assert main(["--seed", "0", "sobol-total", "--n-samples", "64"]) == 0
    out = capsys.readouterr().out
    assert "Sobol total-order" in out
    assert "ST" in out
