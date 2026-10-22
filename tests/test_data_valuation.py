"""Tests for data valuation (data Shapley, TMC, LOO, K-NN Shapley)."""

import numpy as np
import pytest

from interpretability import (
    exact_data_shapley,
    knn_shapley,
    knn_utility,
    leave_one_out_values,
    make_utility,
    tmc_data_shapley,
)
from interpretability.cli import main


def _mean_fit_predict(X_fit, y_fit, X_eval):
    return np.full(X_eval.shape[0], float(np.mean(y_fit)))


def _knn_problem(seed, n=9, n_val=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    Xv = rng.normal(size=(n_val, 2))
    yv = (Xv[:, 0] > 0).astype(int)
    return X, y, Xv, yv


@pytest.mark.parametrize("seed,k", [(0, 1), (1, 2), (2, 3), (3, 5), (4, 9)])
def test_knn_shapley_matches_exact_enumeration(seed, k):
    X, y, Xv, yv = _knn_problem(seed)
    closed = knn_shapley(X, y, Xv, yv, k=k)
    exact = exact_data_shapley(knn_utility(X, y, Xv, yv, k=k))
    np.testing.assert_allclose(closed["values"], exact["values"], atol=1e-12)
    # Efficiency: values sum to U(N) - U(empty) = U(N)
    assert closed["values"].sum() == pytest.approx(exact["full_score"])
    assert closed["per_validation"].shape == (6, 9)


def test_knn_shapley_flags_flipped_labels():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    flipped = rng.choice(200, size=20, replace=False)
    y[flipped] = 1 - y[flipped]
    Xv = rng.normal(size=(100, 2))
    yv = (Xv[:, 0] > 0).astype(int)
    vals = knn_shapley(X, y, Xv, yv, k=5)["values"]
    clean = np.setdiff1d(np.arange(200), flipped)
    assert vals[flipped].mean() < vals[clean].mean()
    lowest = np.argsort(vals)[:20]
    assert len(np.intersect1d(lowest, flipped)) >= 10


def test_exact_shapley_on_additive_game():
    # U(S) = sum of weights in S -> Shapley value = weight.
    w = np.array([0.5, -1.0, 2.0, 0.0, 0.25])

    def utility(idx):
        return float(w[np.asarray(idx, dtype=int)].sum())

    res = exact_data_shapley(utility, n=5)
    np.testing.assert_allclose(res["values"], w, atol=1e-12)
    assert res["n_evaluations"] == 32
    loo = leave_one_out_values(utility, n=5)
    np.testing.assert_allclose(loo["values"], w, atol=1e-12)
    with pytest.raises(ValueError, match="max_points"):
        exact_data_shapley(utility, n=20)
    with pytest.raises(ValueError, match="pass n"):
        exact_data_shapley(utility)


def test_tmc_converges_to_exact_without_truncation():
    X, y, Xv, yv = _knn_problem(11, n=8, n_val=10)
    U = knn_utility(X, y, Xv, yv, k=3)
    exact = knn_shapley(X, y, Xv, yv, k=3)["values"]
    tmc = tmc_data_shapley(U, n_permutations=1500, truncation_tol=0.0, seed=0)
    assert np.all(np.abs(tmc["values"] - exact) < 4 * tmc["std_error"] + 1e-3)
    # Every permutation telescopes: the sum is exact.
    assert tmc["values"].sum() == pytest.approx(tmc["full_score"] - tmc["empty_score"])
    assert tmc["n_permutations"] == 1500


def test_tmc_truncation_saves_evaluations_and_converges_early():
    X, y, Xv, yv = _knn_problem(12, n=30, n_val=20)
    U = knn_utility(X, y, Xv, yv, k=3)
    full = tmc_data_shapley(U, n_permutations=10, truncation_tol=0.0, seed=1)
    trunc = tmc_data_shapley(U, n_permutations=10, truncation_tol=0.2, seed=1)
    assert trunc["n_evaluations"] < full["n_evaluations"]
    early = tmc_data_shapley(U, n_permutations=500, truncation_tol=0.0, seed=2,
                             convergence_tol=0.05, min_permutations=20)
    assert 20 <= early["n_permutations"] < 500
    with pytest.raises(ValueError):
        tmc_data_shapley(U, n_permutations=0)


def test_make_utility_baselines_and_metrics():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 2))
    y = X[:, 0] * 2.0
    Xv = rng.normal(size=(15, 2))
    yv = Xv[:, 0] * 2.0
    U = make_utility(_mean_fit_predict, X, y, Xv, yv, metric="neg_mse", min_size=3)
    base = -np.mean((yv - y.mean()) ** 2)
    assert U([]) == pytest.approx(base)
    assert U([0, 1]) == pytest.approx(base)  # below min_size
    assert U(np.arange(20)) == pytest.approx(base)
    assert U.n_players == 20
    r2 = make_utility(_mean_fit_predict, X, y, Xv, yv, metric="r2", empty_score=0.0)
    assert r2([]) == 0.0
    custom = make_utility(_mean_fit_predict, X, y, Xv, yv, metric=lambda a, b: -np.mean(np.abs(a - b)))
    assert custom(np.arange(5)) < 0

    yc = (X[:, 0] > 0).astype(int)
    yvc = (Xv[:, 0] > 0).astype(int)

    def nn(X_fit, y_fit, X_eval):
        d = ((X_eval[:, None, :] - X_fit[None]) ** 2).sum(-1)
        return y_fit[np.argmin(d, axis=1)]

    acc = make_utility(nn, X, yc, Xv, yvc, metric="accuracy")
    one_class = np.where(yc == 1)[0][:3]
    assert acc(one_class) == pytest.approx(np.mean(yvc == 1))
    assert acc(np.arange(20)) > 0.7

    def broken(X_fit, y_fit, X_eval):
        raise ValueError("singular")

    fallback = make_utility(broken, X, y, Xv, yv, empty_score=-5.0)
    assert fallback(np.arange(10)) == -5.0
    with pytest.raises(ValueError, match="metric"):
        make_utility(_mean_fit_predict, X, y, Xv, yv, metric="auc")
    with pytest.raises(ValueError):
        make_utility(_mean_fit_predict, X, y[:5], Xv, yv)


def test_knn_shapley_validation():
    X, y, Xv, yv = _knn_problem(0)
    with pytest.raises(ValueError, match="k must"):
        knn_shapley(X, y, Xv, yv, k=0)
    with pytest.raises(ValueError):
        knn_shapley(X, y[:3], Xv, yv)
    with pytest.raises(ValueError):
        knn_shapley(X, y, Xv[:, :1], yv)


def test_cli_data_values(capsys):
    assert main(["--n-samples", "80", "data-values", "--n-train", "40", "--permutations", "5", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "Data Shapley (TMC, 5 permutations" in out
    assert "least valuable" in out
