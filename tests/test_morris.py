"""Tests for Morris elementary-effects screening."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability import morris_screening as exported_morris
from interpretability.importance import morris_elementary_effects, morris_screening


def test_export():
    assert exported_morris is morris_screening
    assert morris_elementary_effects is not None


def test_additive_linear_model_ranks_features():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(200, 3))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return 3.0 * Z[:, 0] + 0.1 * Z[:, 1] + 0.0 * Z[:, 2]

    result = morris_screening(predict, X, n_trajectories=40, n_levels=4, seed=1)
    assert result["mu_star"].shape == (3,)
    assert result["mu"].shape == (3,)
    assert result["sigma"].shape == (3,)
    assert result["mu_star"][0] > result["mu_star"][1]
    assert result["mu_star"][0] > result["mu_star"][2]
    assert result["mu_star"][2] < 0.05


def test_irrelevant_feature_near_zero():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(150, 2))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0] ** 2

    result = morris_screening(predict, X, n_trajectories=30, n_levels=4, seed=3)
    assert result["mu_star"][0] > result["mu_star"][1]
    assert result["mu_star"][1] < 0.05


def test_reproducible_with_seed():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 3))

    def predict(Z):
        return np.asarray(Z, dtype=float).sum(axis=1)

    r1 = morris_screening(predict, X, n_trajectories=15, seed=9)
    r2 = morris_screening(predict, X, n_trajectories=15, seed=9)
    assert np.allclose(r1["mu_star"], r2["mu_star"])
    assert np.allclose(r1["mu"], r2["mu"])
    assert np.allclose(r1["sigma"], r2["sigma"])


def test_alias_matches():
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(50, 2))

    def predict(Z):
        return np.asarray(Z, dtype=float)[:, 0]

    a = morris_screening(predict, X, n_trajectories=10, seed=1)
    b = morris_elementary_effects(predict, X, n_trajectories=10, seed=1)
    assert np.allclose(a["mu_star"], b["mu_star"])


def test_rejects_bad_params():
    X = np.random.randn(20, 2)
    with pytest.raises(ValueError):
        morris_screening(lambda Z: Z[:, 0], X, n_trajectories=0)
    with pytest.raises(ValueError):
        morris_screening(lambda Z: Z[:, 0], X, n_levels=3)
    with pytest.raises(ValueError):
        morris_screening(lambda Z: Z[:, 0], X, n_levels=True)  # type: ignore[arg-type]


def test_rejects_1d_X():
    with pytest.raises(ValueError):
        morris_screening(lambda Z: Z[:, 0], np.ones(10), n_trajectories=5)


def test_cli_morris(capsys):
    from interpretability.cli import main

    assert main(["--seed", "0", "morris", "--n-trajectories", "8"]) == 0
    out = capsys.readouterr().out
    assert "Morris screening" in out
    assert "mu*" in out
