"""Tests for Kernel SHAP-lite."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability.local import kernel_shap, shapley_kernel_weight
from interpretability import kernel_shap as exported_kernel_shap


def _linear_predict(Z, coefs=(2.0, -1.0, 0.5)):
    return Z @ np.asarray(coefs, dtype=float)


def test_export():
    assert exported_kernel_shap is kernel_shap


def test_shapley_kernel_weight_symmetric():
    m = 5
    for s in range(1, m):
        assert shapley_kernel_weight(s, m) == pytest.approx(shapley_kernel_weight(m - s, m))
    assert shapley_kernel_weight(0, m) == float("inf")
    assert shapley_kernel_weight(m, m) == float("inf")


def test_shapley_kernel_weight_rejects_bad_size():
    with pytest.raises(ValueError):
        shapley_kernel_weight(-1, 3)
    with pytest.raises(ValueError):
        shapley_kernel_weight(1, 0)


def test_recovers_linear_model_exactly():
    rng = np.random.default_rng(0)
    background = rng.normal(size=(400, 3))
    coefs = np.array([2.0, -1.0, 0.5])
    x_row = np.array([0.3, -0.7, 1.2])
    result = kernel_shap(
        lambda Z: _linear_predict(Z, coefs),
        x_row,
        background,
        n_samples=256,
        seed=1,
        l2=0.0,
    )
    # For a linear model without intercept under mean-imputation Kernel SHAP,
    # φ_j = coef_j * (x_j - E[X_j]).
    expected = coefs * (x_row - background.mean(axis=0))
    assert result["values"] == pytest.approx(expected, abs=1e-5)
    assert result["prediction"] == pytest.approx(float(coefs @ x_row))
    assert result["values"].sum() == pytest.approx(
        result["prediction"] - result["baseline"], abs=1e-5
    )


def test_efficiency_approximately_holds():
    rng = np.random.default_rng(2)
    background = rng.uniform(size=(150, 4))
    x_row = background[10]

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0] * Z[:, 1] + 0.3 * Z[:, 2] - Z[:, 3]

    result = kernel_shap(predict, x_row, background, n_samples=300, seed=3)
    gap = result["prediction"] - result["baseline"]
    assert result["values"].sum() == pytest.approx(gap, abs=0.15)


def test_reproducible_with_seed():
    rng = np.random.default_rng(4)
    background = rng.normal(size=(80, 3))
    x_row = background[0]
    r1 = kernel_shap(_linear_predict, x_row, background, n_samples=50, seed=9)
    r2 = kernel_shap(_linear_predict, x_row, background, n_samples=50, seed=9)
    assert np.allclose(r1["values"], r2["values"])


def test_rejects_mismatched_width():
    background = np.zeros((10, 2))
    with pytest.raises(ValueError):
        kernel_shap(_linear_predict, np.zeros(3), background, n_samples=10)


def test_rejects_bad_n_samples():
    background = np.zeros((10, 2))
    with pytest.raises(ValueError):
        kernel_shap(_linear_predict, np.zeros(2), background, n_samples=0)


def test_cli_kernel_shap(capsys):
    from interpretability.cli import main

    rc = main(["--seed", "3", "--n-samples", "80", "kernel-shap", "--rows", "0", "--n-samples", "64"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Kernel SHAP for row 0" in out
    assert "sum(phi)" in out
