"""Tests for Growing Spheres counterfactual explanations."""

import numpy as np
import pytest

from interpretability import growing_spheres_counterfactual, sample_spherical_layer
from interpretability.cli import main


def _linear(seed=0, n=500):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) * np.array([1.0, 2.0, 1.0])
    w = np.array([1.0, 0.5, 0.0])
    return X, (lambda Z: np.asarray(Z) @ w), w


def test_layer_sampler_is_uniform_in_the_shell():
    rng = np.random.default_rng(0)
    z = sample_spherical_layer(20000, 3, 1.0, 2.0, rng)
    r = np.linalg.norm(z, axis=1)
    assert r.min() >= 1.0 - 1e-12 and r.max() <= 2.0 + 1e-12
    # uniform in volume: P(r <= 1.5) = (1.5^3 - 1) / (2^3 - 1)
    assert np.mean(r <= 1.5) == pytest.approx((1.5**3 - 1) / 7, abs=0.01)
    assert np.abs(z.mean(axis=0)).max() < 0.05
    with pytest.raises(ValueError):
        sample_spherical_layer(5, 2, 2.0, 1.0, rng)


def test_regression_target_reached_near_minimal_distance():
    X, predict, w = _linear()
    x = np.zeros(3)
    res = growing_spheres_counterfactual(
        predict, x, X, target_range=(2.0, None), bounds=None, sparsify=False, seed=1, n_samples=2000
    )
    assert res["found"]
    assert res["prediction_counterfactual"] >= 2.0
    # closed-form minimum standardized distance to {w.x >= 2}
    scale = X.std(axis=0)
    optimum = 2.0 / np.linalg.norm(w * scale)
    assert optimum <= res["distance_l2"] <= 1.35 * optimum
    assert res["delta"][2] != 0.0  # unsparsified: irrelevant feature also moved


def test_sparsify_drops_irrelevant_features():
    X, predict, _ = _linear()
    res = growing_spheres_counterfactual(predict, np.zeros(3), X, target_range=(2.0, None), seed=1)
    assert res["found"]
    assert 2 not in res["changed_features"]
    assert res["n_changed"] <= 2
    assert predict(res["counterfactual"][None, :])[0] >= 2.0


def test_immutable_and_bounds_respected():
    X, predict, _ = _linear()
    x = np.zeros(3)
    res = growing_spheres_counterfactual(
        predict, x, X, target_range=(1.5, None), immutable=[0], seed=2
    )
    assert res["found"]
    assert res["counterfactual"][0] == 0.0
    assert res["changed_features"] == [1]
    tight = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=float)
    bounded = growing_spheres_counterfactual(
        predict, x, X, target_range=(1.2, None), bounds=tight, seed=3
    )
    assert bounded["found"]
    assert np.all(bounded["counterfactual"] >= -1) and np.all(bounded["counterfactual"] <= 1)
    impossible = growing_spheres_counterfactual(
        predict, x, X, target_range=(10.0, None), bounds=tight, seed=3, max_layers=20
    )
    assert impossible["found"] is False and impossible["counterfactual"] is None


def test_classification_target_class_and_callable():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 2))

    def classify(Z):
        Z = np.asarray(Z)
        return (Z[:, 0] ** 2 + Z[:, 1] ** 2 > 1.0).astype(int)

    x = np.array([0.1, 0.0])
    res = growing_spheres_counterfactual(classify, x, X, target_class=1, seed=5)
    assert res["found"] and res["prediction_counterfactual"] == 1
    assert res["distance_l2"] < 1.5
    via_callable = growing_spheres_counterfactual(
        classify, x, X, target=lambda p: p == 1, seed=5
    )
    assert np.allclose(via_callable["counterfactual"], res["counterfactual"])


def test_already_in_target_and_validation():
    X, predict, _ = _linear()
    res = growing_spheres_counterfactual(predict, np.zeros(3), X, target_range=(-1.0, 1.0))
    assert res["found"] and res["n_changed"] == 0 and res["distance_l2"] == 0.0
    with pytest.raises(ValueError, match="exactly one"):
        growing_spheres_counterfactual(predict, np.zeros(3), X)
    with pytest.raises(ValueError, match="exactly one"):
        growing_spheres_counterfactual(predict, np.zeros(3), X, target_range=(0, 1), target_class=1)
    with pytest.raises(ValueError):
        growing_spheres_counterfactual(predict, np.zeros(2), X, target_range=(0, 1))
    with pytest.raises(ValueError):
        growing_spheres_counterfactual(predict, np.zeros(3), X, target_range=(0, 1), immutable=[0, 1, 2])
    with pytest.raises(ValueError):
        growing_spheres_counterfactual(predict, np.zeros(3), X, target_range=(2, 1))
    with pytest.raises(ValueError):
        growing_spheres_counterfactual(predict, np.zeros(3), X, target_range=(0, 1), eta=0)


def test_cli_counterfactual(capsys):
    code = main(["--n-samples", "200", "counterfactual", "--row", "0", "--target-max", "4.0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Counterfactual for row 0" in out
    assert "feature(s) changed" in out
    new_pred = float(out.split("new prediction ")[1].split(",")[0])
    assert new_pred <= 4.0
    with pytest.raises(SystemExit):
        main(["counterfactual", "--row", "0"])
