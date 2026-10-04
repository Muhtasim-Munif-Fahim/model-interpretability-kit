"""Tests for SmoothGrad."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability.local import smoothgrad, integrated_gradients
from interpretability import smoothgrad as exported_sg


def _linear_predict(Z, coefs=(2.0, -1.0, 0.5)):
    return Z @ np.asarray(coefs, dtype=float)


def test_export():
    assert exported_sg is smoothgrad


def test_shape_and_metadata():
    x_row = np.array([0.3, -0.7, 1.2])
    result = smoothgrad(
        lambda Z: _linear_predict(Z),
        x_row,
        n_samples=20,
        noise_sigma=0.05,
        seed=0,
        feature_names=["a", "b", "c"],
    )
    assert result["values"].shape == (3,)
    assert result["n_samples"] == 20
    assert result["noise_sigma"] == pytest.approx(0.05)
    assert result["feature_names"] == ["a", "b", "c"]
    assert result["prediction"] == pytest.approx(float(_linear_predict(x_row[None, :])[0]))
    assert np.all(np.isfinite(result["values"]))


def test_seed_reproducibility():
    x_row = np.array([1.0, -0.5, 0.25, 2.0])

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0] ** 2 - 0.5 * Z[:, 1] + Z[:, 2] * Z[:, 3]

    a = smoothgrad(predict, x_row, n_samples=30, noise_sigma=0.2, seed=7)
    b = smoothgrad(predict, x_row, n_samples=30, noise_sigma=0.2, seed=7)
    c = smoothgrad(predict, x_row, n_samples=30, noise_sigma=0.2, seed=8)
    assert np.allclose(a["values"], b["values"])
    assert not np.allclose(a["values"], c["values"])


def test_sigma_zero_matches_raw_gradient():
    coefs = np.array([2.0, -1.0, 0.5])
    x_row = np.array([0.3, -0.7, 1.2])
    result = smoothgrad(
        lambda Z: _linear_predict(Z, coefs),
        x_row,
        n_samples=5,
        noise_sigma=0.0,
        eps=1e-5,
        seed=0,
    )
    # For a linear model the FD gradient equals the coefficients.
    assert result["values"] == pytest.approx(coefs, abs=1e-4)


def test_finite_outputs_nonlinear():
    rng = np.random.default_rng(3)
    x_row = rng.normal(size=4)

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return np.sin(Z[:, 0]) + Z[:, 1] * Z[:, 2] - Z[:, 3] ** 2

    result = smoothgrad(predict, x_row, n_samples=40, noise_sigma=0.15, seed=1)
    assert np.all(np.isfinite(result["values"]))
    assert np.isfinite(result["prediction"])


def test_rejects_bad_n_samples():
    with pytest.raises(ValueError):
        smoothgrad(lambda Z: Z.sum(axis=1), np.zeros(2), n_samples=0)


def test_rejects_negative_sigma():
    with pytest.raises(ValueError):
        smoothgrad(lambda Z: Z.sum(axis=1), np.zeros(2), noise_sigma=-0.1)


def test_cli_smoothgrad(capsys):
    from interpretability.cli import main

    rc = main(["--seed", "3", "smoothgrad", "--rows", "0", "--n-samples", "10", "--noise-sigma", "0.05"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SmoothGrad for row 0" in out
    assert "sigma" in out
