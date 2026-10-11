"""Tests for SAGE global importance."""

import itertools
import math

import numpy as np
import pytest

from interpretability import sage_values
from interpretability.cli import main
from interpretability.sage import sage_value_function


def _linear_problem(n=400, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) * np.array([1.0, 3.0, 0.5])
    beta = np.array([1.0, 0.5, 0.0])
    y = X @ beta + noise * rng.normal(size=n)
    return X, y, beta, (lambda Z: Z @ beta)


def _exact_sage(predict, X, y, background, loss="mse"):
    d = X.shape[1]
    v = {}
    for r in range(d + 1):
        for S in itertools.combinations(range(d), r):
            v[S] = sage_value_function(predict, X, y, S, background=background, loss=loss)
    phi = np.zeros(d)
    for j in range(d):
        others = [k for k in range(d) if k != j]
        for r in range(d):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(d - r - 1) / math.factorial(d)
                with_j = tuple(sorted(S + (j,)))
                phi[j] += w * (v[S] - v[with_j])
    return phi, v


def test_efficiency_is_exact_per_draw():
    X, y, _, f = _linear_problem()
    res = sage_values(f, X, y, n_permutations=50, n_background=32, seed=1)
    assert res["values"].sum() == pytest.approx(res["total"])
    assert res["total"] == pytest.approx(res["loss_empty"] - res["loss_full"], rel=1e-10)
    assert np.nansum(res["ratio_to_total"]) == pytest.approx(1.0)


def test_matches_exact_shapley_of_the_loss_game():
    X, y, _, f = _linear_problem(n=60, seed=2)
    background = X[:20]
    exact, v = _exact_sage(f, X, y, background)
    assert exact.sum() == pytest.approx(v[()] - v[(0, 1, 2)])
    res = sage_values(f, X, y, background=background, n_permutations=3000, seed=3)
    assert np.all(np.abs(res["values"] - exact) < 4 * res["std_error"] + 1e-9)


def test_linear_independent_features_recover_beta_squared_variance():
    X, y, beta, f = _linear_problem(n=2000, seed=4)
    res = sage_values(f, X, y, n_permutations=4000, n_background=128, seed=5)
    expected = beta**2 * X.var(axis=0)
    # Monte Carlo error dominates; allow 4 standard errors plus 5% background bias.
    assert np.all(np.abs(res["values"] - expected) <= 4 * res["std_error"] + 0.05 * expected)
    # Feature 2 has zero coefficient: SAGE value is ~0.
    assert abs(res["values"][2]) < 0.02
    assert np.argmax(res["values"]) == np.argmax(expected)


def test_pure_interaction_is_split_evenly():
    rng = np.random.default_rng(6)
    X = rng.choice([-1.0, 1.0], size=(600, 3))
    y = X[:, 0] * X[:, 1]

    def f(Z):
        return Z[:, 0] * Z[:, 1]

    res = sage_values(f, X, y, n_permutations=1200, n_background=64, seed=7)
    assert res["values"][0] == pytest.approx(res["values"][1], abs=0.08)
    assert res["values"][0] == pytest.approx(0.5, abs=0.08)
    assert abs(res["values"][2]) < 1e-12


def test_log_loss_for_a_probabilistic_classifier():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(800, 2))
    p = 1.0 / (1.0 + np.exp(-3.0 * X[:, 0]))
    y = (rng.random(800) < p).astype(float)

    def proba(Z):
        return 1.0 / (1.0 + np.exp(-3.0 * Z[:, 0]))

    res = sage_values(proba, X, y, loss="log_loss", n_permutations=400, n_background=64, seed=9)
    assert res["values"][0] > 0.2
    assert abs(res["values"][1]) < 1e-12
    assert res["loss_empty"] == pytest.approx(np.log(2.0), abs=0.08)


def test_callable_per_row_loss_and_mae():
    X, y, _, f = _linear_problem(n=200, seed=10)
    a = sage_values(f, X, y, loss="mae", n_permutations=40, seed=11)
    b = sage_values(f, X, y, loss=lambda t, p: np.abs(t - p), n_permutations=40, seed=11)
    assert np.allclose(a["values"], b["values"])
    with pytest.raises(ValueError):
        sage_values(f, X, y, loss=lambda t, p: float(np.mean(np.abs(t - p))), n_permutations=4)


def test_batching_and_seed_do_not_change_results():
    X, y, _, f = _linear_problem(n=150, seed=12)
    a = sage_values(f, X, y, n_permutations=37, batch_size=5, seed=13)
    b = sage_values(f, X, y, n_permutations=37, batch_size=100, seed=13)
    assert np.allclose(a["values"], b["values"])
    assert a["n_evaluations"] == 37 * 4 * 64
    c = sage_values(f, X, y, n_permutations=37, seed=14)
    assert not np.allclose(a["values"], c["values"])


def test_validation():
    X, y, _, f = _linear_problem(n=50, seed=15)
    with pytest.raises(ValueError):
        sage_values(f, X, y, loss="hinge")
    with pytest.raises(ValueError):
        sage_values(f, X, y, n_permutations=0)
    with pytest.raises(ValueError):
        sage_values(f, X, y, background=np.zeros((4, 2)))
    with pytest.raises(ValueError):
        sage_values(f, X, y[:-1])
    with pytest.raises(ValueError):
        sage_values(f, X, y, feature_names=["a"])
    with pytest.raises(ValueError):
        sage_value_function(f, X, y, [7])
    single = sage_values(f, X, y, n_permutations=1, seed=0)
    assert np.all(np.isnan(single["std_error"]))


def test_cli_sage(capsys):
    assert main(["--n-samples", "200", "sage", "--n-permutations", "64", "--n-background", "16"]) == 0
    out = capsys.readouterr().out
    assert "SAGE values" in out
    assert "X0" in out
