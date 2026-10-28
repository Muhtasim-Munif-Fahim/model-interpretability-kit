"""Tests for deletion/insertion, infidelity and sensitivity metrics."""

import numpy as np
import pytest

from interpretability import (
    attribution_infidelity,
    attribution_sensitivity,
    deletion_insertion,
    exact_shapley,
)
from interpretability.cli import main

W = np.array([3.0, -1.0, 0.5, 0.0])


def _linear(Z):
    return np.asarray(Z, dtype=float) @ W


def test_deletion_insertion_linear_exact_values():
    x = np.array([1.0, 2.0, 2.0, 5.0])
    base = np.zeros(4)
    phi = W * (x - base)  # [3, -2, 1, 0]
    res = deletion_insertion(_linear, x, phi, baseline=base)
    assert res["order"] == [0, 2, 3, 1]
    np.testing.assert_allclose(res["deletion"], [2.0, -1.0, -2.0, -2.0, 0.0])
    np.testing.assert_allclose(res["insertion"], [0.0, 3.0, 4.0, 4.0, 2.0])
    assert res["prediction"] == pytest.approx(2.0)
    assert res["baseline_prediction"] == pytest.approx(0.0)
    assert res["aopc"] == pytest.approx(np.mean([3.0, 4.0, 4.0, 2.0]))
    # trapezoid on [0, 1] with 5 points
    d = res["deletion"]
    assert res["deletion_auc"] == pytest.approx(np.sum((d[1:] + d[:-1]) / 2) / 4)


def test_faithful_ranking_beats_reversed_ranking():
    x = np.array([1.0, 2.0, 2.0, 5.0])
    phi = W * x
    good = deletion_insertion(_linear, x, phi)
    bad = deletion_insertion(_linear, x, -phi)
    assert good["deletion_auc"] < bad["deletion_auc"]
    assert good["insertion_auc"] > bad["insertion_auc"]
    assert good["aopc"] > bad["aopc"]


def test_abs_order_and_background_baseline():
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(50, 4))
    x = np.array([1.0, 2.0, 2.0, 5.0])
    phi = W * (x - bg.mean(axis=0))
    res = deletion_insertion(_linear, x, phi, baseline=bg, order="abs")
    assert res["order"][0] == 0
    assert res["deletion"][-1] == pytest.approx(_linear(bg.mean(axis=0)[None])[0])


def test_deletion_insertion_validation():
    with pytest.raises(ValueError):
        deletion_insertion(_linear, np.ones(4), np.ones(3))
    with pytest.raises(ValueError):
        deletion_insertion(_linear, np.ones(4), np.ones(4), order="bogus")
    with pytest.raises(ValueError):
        deletion_insertion(_linear, np.ones(4), np.ones(4), baseline=np.ones(2))


def test_subset_infidelity_zero_for_exact_shapley_of_linear_model():
    x = np.array([1.0, 2.0, 2.0, 5.0])
    base = np.array([0.5, 0.0, 1.0, 1.0])
    phi = exact_shapley(_linear, x, base[None, :])["values"]
    res = attribution_infidelity(_linear, x, phi, baseline=base, n_samples=100, seed=1)
    assert res["infidelity"] == pytest.approx(0.0, abs=1e-18)
    assert res["beta"] == pytest.approx(1.0)


def test_gaussian_infidelity_zero_for_gradient_of_linear_model():
    x = np.array([1.0, 2.0, 2.0, 5.0])
    res = attribution_infidelity(_linear, x, W, perturbation="gaussian", n_samples=100, seed=2)
    assert res["infidelity"] == pytest.approx(0.0, abs=1e-18)


def test_scaled_infidelity_forgives_constant_factor():
    x = np.array([1.0, 2.0, 2.0, 5.0])
    phi = 3.0 * W * x
    res = attribution_infidelity(_linear, x, phi, n_samples=200, seed=3)
    assert res["infidelity"] > 1.0
    assert res["scaled_infidelity"] == pytest.approx(0.0, abs=1e-12)
    assert res["beta"] == pytest.approx(1.0 / 3.0)


def test_infidelity_ranks_wrong_attribution_worse_and_is_seeded():
    x = np.array([1.0, 2.0, 2.0, 5.0])
    good = attribution_infidelity(_linear, x, W * x, n_samples=200, seed=4)
    wrong = attribution_infidelity(_linear, x, np.array([0.0, 0.0, 0.0, 3.0]), n_samples=200, seed=4)
    again = attribution_infidelity(_linear, x, np.array([0.0, 0.0, 0.0, 3.0]), n_samples=200, seed=4)
    assert wrong["infidelity"] > good["infidelity"]
    assert wrong["infidelity"] == again["infidelity"]


def test_infidelity_validation():
    x = np.ones(4)
    with pytest.raises(ValueError):
        attribution_infidelity(_linear, x, np.ones(4), perturbation="nope")
    with pytest.raises(ValueError):
        attribution_infidelity(_linear, x, np.ones(4), n_samples=1)
    with pytest.raises(ValueError):
        attribution_infidelity(_linear, x, np.ones(4), subset_prob=0.0)
    with pytest.raises(ValueError):
        attribution_infidelity(_linear, x, np.ones(4), perturbation="gaussian", noise_sigma=0)


def test_sensitivity_zero_for_constant_explainer():
    res = attribution_sensitivity(lambda r: W, np.ones(4), radius=0.2, n_samples=10, seed=0)
    assert res["max_sensitivity"] == 0.0
    assert res["avg_sensitivity"] == 0.0
    assert res["lipschitz"] == 0.0


def test_sensitivity_lipschitz_of_linear_explainer():
    # phi(x) = W * x has Lipschitz constant max|W| = 3 in the L2 sense.
    res = attribution_sensitivity(lambda r: W * r, np.ones(4), radius=0.1, n_samples=200, seed=5)
    assert 0.0 < res["lipschitz"] <= 3.0 + 1e-12
    assert res["lipschitz"] > 2.0
    assert res["max_sensitivity"] <= 3.0 * 0.1 * 2.0 + 1e-12
    assert res["avg_sensitivity"] <= res["max_sensitivity"]


def test_sensitivity_accepts_dict_scale_and_normalize():
    x = np.array([1.0, 2.0, 2.0, 5.0])
    res = attribution_sensitivity(
        lambda r: {"values": W * r}, x, radius=0.1, scale=np.array([1.0, 0.0, 0.0, 0.0]),
        n_samples=20, normalize=True, seed=6,
    )
    np.testing.assert_allclose(res["attribution"], W * x)
    # only feature 0 moves: |3 * delta| <= 0.3, normalised by ||W*x||
    assert res["max_sensitivity"] <= 0.3 / np.linalg.norm(W * x) + 1e-12
    with pytest.raises(ValueError):
        attribution_sensitivity(lambda r: np.zeros(4), x, normalize=True)
    with pytest.raises(ValueError):
        attribution_sensitivity(lambda r: W, x, radius=0)
    with pytest.raises(ValueError):
        attribution_sensitivity(lambda r: W[:2], x)


def test_cli_attribution_quality(capsys):
    assert main(["--seed", "3", "--n-samples", "200", "attribution-quality", "--rows", "0", "--n-perturb", "50"]) == 0
    out = capsys.readouterr().out
    assert "kernel-shap" in out and "integrated-gradients" in out and "infidelity" in out
