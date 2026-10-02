"""Tests for Integrated Gradients."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability.local import integrated_gradients
from interpretability import integrated_gradients as exported_ig


def _linear_predict(Z, coefs=(2.0, -1.0, 0.5)):
    return Z @ np.asarray(coefs, dtype=float)


def test_export():
    assert exported_ig is integrated_gradients


def test_recovers_linear_model_exactly():
    coefs = np.array([2.0, -1.0, 0.5])
    x_row = np.array([0.3, -0.7, 1.2])
    baseline = np.zeros(3)
    result = integrated_gradients(
        lambda Z: _linear_predict(Z, coefs),
        x_row,
        baseline=baseline,
        n_steps=16,
        eps=1e-5,
    )
    expected = coefs * (x_row - baseline)
    assert result["values"] == pytest.approx(expected, abs=1e-4)
    assert result["prediction"] == pytest.approx(float(coefs @ x_row))
    assert result["values"].sum() == pytest.approx(
        result["prediction"] - result["baseline"], abs=1e-4
    )


def test_completeness_approximately_holds():
    rng = np.random.default_rng(2)
    x_row = rng.normal(size=4)
    baseline = rng.normal(size=4)

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0] * Z[:, 1] + 0.3 * Z[:, 2] - Z[:, 3]

    result = integrated_gradients(predict, x_row, baseline=baseline, n_steps=64, eps=1e-4)
    gap = result["prediction"] - result["baseline"]
    assert result["values"].sum() == pytest.approx(gap, abs=0.05)


def test_default_baseline_is_zeros():
    x_row = np.array([1.0, 2.0])
    result = integrated_gradients(lambda Z: Z.sum(axis=1), x_row, n_steps=8)
    assert np.allclose(result["baseline_input"], 0.0)
    assert result["n_steps"] == 8


def test_rejects_mismatched_baseline():
    with pytest.raises(ValueError):
        integrated_gradients(lambda Z: Z.sum(axis=1), np.zeros(3), baseline=np.zeros(2))


def test_rejects_bad_n_steps():
    with pytest.raises(ValueError):
        integrated_gradients(lambda Z: Z.sum(axis=1), np.zeros(2), n_steps=0)


def test_cli_integrated_gradients(capsys):
    from interpretability.cli import main

    rc = main(["--seed", "3", "integrated-gradients", "--rows", "0", "--n-steps", "16"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Integrated Gradients for row 0" in out
    assert "sum(phi)" in out
