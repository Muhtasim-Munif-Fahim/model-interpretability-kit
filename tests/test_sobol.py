"""Tests for Sobol first-order sensitivity indices."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability import sobol_first_order as exported_sobol
from interpretability.importance import sobol_first_order


def test_export():
    assert exported_sobol is sobol_first_order


def test_additive_linear_model_ranks_features():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(200, 3))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return 3.0 * Z[:, 0] + 0.1 * Z[:, 1] + 0.0 * Z[:, 2]

    result = sobol_first_order(predict, X, n_samples=800, seed=1)
    assert result["S1"].shape == (3,)
    assert result["S1"][0] > result["S1"][1]
    assert result["S1"][0] > result["S1"][2]
    assert result["S1"][0] > 0.7
    assert result["S1"][1] < 0.15
    assert result["S1"][2] < 0.15
    assert result["variance"] > 0.0


def test_irrelevant_feature_near_zero():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(150, 2))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0] ** 2

    result = sobol_first_order(predict, X, n_samples=600, seed=3)
    assert result["S1"][0] > 0.8
    assert result["S1"][1] < 0.15


def test_reproducible_with_seed():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 3))

    def predict(Z):
        return np.asarray(Z, dtype=float).sum(axis=1)

    r1 = sobol_first_order(predict, X, n_samples=100, seed=9)
    r2 = sobol_first_order(predict, X, n_samples=100, seed=9)
    assert np.allclose(r1["S1"], r2["S1"])


def test_rejects_bad_n_samples():
    X = np.random.randn(20, 2)
    with pytest.raises(ValueError):
        sobol_first_order(lambda Z: Z[:, 0], X, n_samples=1)
    with pytest.raises(ValueError):
        sobol_first_order(lambda Z: Z[:, 0], X, n_samples=True)  # type: ignore[arg-type]


def test_rejects_1d_X():
    with pytest.raises(ValueError):
        sobol_first_order(lambda Z: Z[:, 0], np.ones(10), n_samples=20)


def test_indices_in_unit_interval():
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(100, 4))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0] * Z[:, 1] + Z[:, 2]

    result = sobol_first_order(predict, X, n_samples=400, seed=6)
    assert np.all(result["S1"] >= 0.0)
    assert np.all(result["S1"] <= 1.0)


def test_cli_sobol(capsys):
    from interpretability.cli import main

    assert main(["--seed", "0", "sobol", "--n-samples", "64"]) == 0
    out = capsys.readouterr().out
    assert "Sobol first-order" in out
