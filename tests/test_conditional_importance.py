"""Tests for conditional permutation importance."""

import numpy as np
import pytest

from interpretability import conditional_permutation_importance
from interpretability.cli import main
from interpretability.importance import mean_squared_error, permutation_importance


def _neg_mse(y_true, y_pred):
    return -mean_squared_error(y_true, y_pred)


def _correlated(n=2000, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = x0 + noise * rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x0, x1, x2])
    y = X.sum(axis=1) + 0.1 * rng.normal(size=n)
    return X, y


def _additive(X):
    return X[:, 0] + X[:, 1] + X[:, 2]


def test_residual_matches_gaussian_closed_form():
    # For f = sum(x) and the MSE loss, replacing x_j by an independent draw
    # from x_j | x_{-j} raises the expected loss by 2 Var(x_j | x_{-j}).
    noise = 0.1
    X, y = _correlated(n=4000, noise=noise)
    out = conditional_permutation_importance(
        _additive, X, y, metric=_neg_mse, n_repeats=10, seed=0
    )
    cond_var_x0 = noise**2 / (1 + noise**2)  # Var(x0 | x1)
    cond_var_x1 = noise**2  # Var(x1 | x0)
    assert out["mean"][0] == pytest.approx(2 * cond_var_x0, rel=0.15)
    assert out["mean"][1] == pytest.approx(2 * cond_var_x1, rel=0.15)
    assert out["mean"][2] == pytest.approx(2.0, rel=0.1)
    assert out["conditioning"] == [[1, 2], [0, 2], [0, 1]]


def test_correlated_duplicates_lose_importance_but_marginal_does_not():
    X, y = _correlated()
    marginal = permutation_importance(_additive, X, y, metric=_neg_mse, n_repeats=5, seed=1)
    for method in ("residual", "grid"):
        cond = conditional_permutation_importance(
            _additive, X, y, metric=_neg_mse, n_repeats=5, method=method, n_bins=10, seed=1
        )
        assert np.all(cond["mean"][:2] < 0.1 * marginal["mean"][:2])
        assert cond["mean"][2] == pytest.approx(marginal["mean"][2], rel=0.1)


def test_single_feature_reduces_to_marginal_permutation():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 1))
    y = 3.0 * X[:, 0] + rng.normal(size=200)

    def model(Z):
        return 3.0 * Z[:, 0]

    marginal = permutation_importance(model, X, y, n_repeats=4, seed=7)
    for method in ("residual", "grid"):
        cond = conditional_permutation_importance(model, X, y, n_repeats=4, method=method, seed=7)
        np.testing.assert_allclose(cond["mean"], marginal["mean"])
        np.testing.assert_allclose(cond["std"], marginal["std"])
        assert cond["baseline"] == marginal["baseline"]


def test_independent_features_match_marginal_importance():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 3))
    y = 2 * X[:, 0] + X[:, 1] + 0.1 * rng.normal(size=3000)

    def model(Z):
        return 2 * Z[:, 0] + Z[:, 1]

    marginal = permutation_importance(model, X, y, metric=_neg_mse, n_repeats=8, seed=0)
    cond = conditional_permutation_importance(model, X, y, metric=_neg_mse, n_repeats=8, seed=0)
    np.testing.assert_allclose(cond["mean"][:2], marginal["mean"][:2], rtol=0.08)
    assert abs(cond["mean"][2]) < 1e-12


def test_grid_permutes_only_within_strata():
    rng = np.random.default_rng(4)
    group = np.repeat([0.0, 1.0, 2.0, 3.0], 50)
    X = np.column_stack([group + 0.01 * rng.normal(size=200), rng.normal(size=200)])
    seen = []

    def model(Z):
        seen.append(Z.copy())
        return Z[:, 1]

    y = X[:, 1]
    out = conditional_permutation_importance(
        model, X, y, n_repeats=2, method="grid", n_bins=4, conditioning=[0], seed=0
    )
    assert out["conditioning"] == [[], [0]]
    # Calls: baseline, 2 marginal draws for x0 (no conditioning), 2 for x1.
    for Z in seen[3:]:
        for g in range(4):
            rows = slice(50 * g, 50 * (g + 1))
            np.testing.assert_allclose(np.sort(Z[rows, 1]), np.sort(X[rows, 1]))
        assert not np.allclose(Z[:, 1], X[:, 1])


def test_explicit_conditioning_mapping_and_auto_selection():
    X, y = _correlated(n=500)
    out = conditional_permutation_importance(
        _additive, X, y, method="grid", conditioning={0: [2], 1: [0], 2: [0, 1]}, seed=0
    )
    assert out["conditioning"] == [[2], [0], [0, 1]]
    auto = conditional_permutation_importance(_additive, X, y, method="grid", seed=0)
    assert auto["conditioning"][0] == [1]
    assert auto["conditioning"][1] == [0]


def test_seed_reproducibility_and_validation():
    X, y = _correlated(n=300)
    a = conditional_permutation_importance(_additive, X, y, seed=5)
    b = conditional_permutation_importance(_additive, X, y, seed=5)
    np.testing.assert_array_equal(a["mean"], b["mean"])
    with pytest.raises(ValueError, match="n_repeats"):
        conditional_permutation_importance(_additive, X, y, n_repeats=0)
    with pytest.raises(ValueError, match="method"):
        conditional_permutation_importance(_additive, X, y, method="knockoff")
    with pytest.raises(ValueError, match="n_bins"):
        conditional_permutation_importance(_additive, X, y, method="grid", n_bins=1)
    with pytest.raises(ValueError, match="n_conditioning"):
        conditional_permutation_importance(_additive, X, y, method="grid", n_conditioning=0)
    with pytest.raises(ValueError, match="out of range"):
        conditional_permutation_importance(_additive, X, y, method="grid", conditioning=[7])


@pytest.mark.parametrize("method", ["residual", "grid"])
def test_cli_conditional_importance(capsys, method):
    code = main(
        [
            "--seed",
            "1",
            "--n-samples",
            "120",
            "conditional-importance",
            "--n-repeats",
            "2",
            "--method",
            method,
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Conditional permutation importance (%s sampler" % method in out
    assert "marginal" in out
    assert "X0" in out and "X4" in out
