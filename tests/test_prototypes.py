"""Tests for MMD-critic prototypes and criticisms."""

import itertools

import numpy as np
import pytest

from interpretability import mmd_critic, mmd_squared, select_criticisms, select_prototypes, witness_function
from interpretability.cli import main
from interpretability.prototypes import median_heuristic_gamma, rbf_kernel


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(0.0, 0.0), scale=0.3, size=(150, 2))
    b = rng.normal(loc=(5.0, 5.0), scale=0.3, size=(50, 2))
    return np.vstack([a, b]), np.r_[np.zeros(150, int), np.ones(50, int)]


def test_rbf_kernel_and_mmd_basics():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    K = rbf_kernel(X, X, 0.5)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert mmd_squared(X, X, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert mmd_squared(X, X + 3.0, 0.5) > 0.5
    with pytest.raises(ValueError):
        rbf_kernel(X, X, 0.0)
    with pytest.raises(ValueError):
        rbf_kernel(X, X[:, :2], 1.0)


def test_median_heuristic():
    X = np.array([[0.0], [1.0], [3.0]])
    # squared distances 1, 9, 4 -> median 4
    assert median_heuristic_gamma(X) == pytest.approx(0.25)
    assert median_heuristic_gamma(np.zeros((4, 2))) == 1.0


def test_greedy_objective_matches_negative_mmd_and_is_monotone():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 2))
    K = rbf_kernel(X, X, 0.7)
    idx, trace = select_prototypes(K, 6)
    assert len(set(idx.tolist())) == 6
    for t in range(1, 7):
        S = idx[:t]
        assert trace[t - 1] == pytest.approx(K.mean() - mmd_squared(X, X[S], 0.7), abs=1e-10)


def test_greedy_picks_best_extension_each_step():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(15, 2))
    K = rbf_kernel(X, X, 1.0)
    idx, _ = select_prototypes(K, 3)
    for t in range(3):
        prev = list(idx[:t])
        options = [c for c in range(15) if c not in prev]
        best = min(options, key=lambda c: mmd_squared(X, X[prev + [c]], 1.0))
        assert idx[t] == best
    # greedy MMD with 3 prototypes beats the average random 3-subset
    rand = np.mean([mmd_squared(X, X[list(p)], 1.0) for p in itertools.combinations(range(15), 3)])
    assert mmd_squared(X, X[idx], 1.0) < rand


def test_candidates_restrict_prototypes():
    X, _ = _blobs(7)
    K = rbf_kernel(X, X, 0.5)
    idx, _ = select_prototypes(K, 4, candidates=np.arange(150, 200))
    assert np.all(idx >= 150)
    with pytest.raises(ValueError):
        select_prototypes(K, 5, candidates=[0, 1])


def test_prototypes_cover_both_clusters():
    X, _ = _blobs()
    res = mmd_critic(X, n_prototypes=4, n_criticisms=2)
    protos = X[res["prototypes"]]
    assert np.any(protos[:, 0] < 2) and np.any(protos[:, 0] > 3)
    # big cluster gets more prototypes than the small one
    assert np.sum(protos[:, 0] < 2) > np.sum(protos[:, 0] > 3)
    assert not set(res["criticisms"].tolist()) & set(res["prototypes"].tolist())
    assert res["coverage"].sum() == X.shape[0]
    assert res["mmd2"] >= 0
    assert np.allclose(res["criticism_witness"], res["witness"][res["criticisms"]])


def test_witness_function_matches_kernel_means():
    X, _ = _blobs(4)
    Z = X[:5]
    w = witness_function(X[:10], X, Z, 0.2)
    expected = rbf_kernel(X[:10], X, 0.2).mean(1) - rbf_kernel(X[:10], Z, 0.2).mean(1)
    assert np.allclose(w, expected)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_criticisms_find_unrepresented_clusters_and_logdet_spreads_them(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=0.5, size=(150, 2))
    B1 = rng.normal(loc=(5, 5), scale=0.2, size=(30, 2))  # dense, no prototypes
    B2 = rng.normal(loc=(5, -5), scale=0.4, size=(20, 2))  # sparser, no prototypes
    X = np.vstack([A, B1, B2])
    K = rbf_kernel(X, X, 0.5)
    protos, _ = select_prototypes(K, 15, candidates=np.arange(150))
    plain, w = select_criticisms(K, protos, 3, regularizer=None)
    reg, _ = select_criticisms(K, protos, 3, regularizer="logdet")
    # plain top-|witness| piles every criticism into the densest gap
    assert np.all((plain >= 150) & (plain < 180))
    assert np.all(w[plain] > 0)  # under-represented
    # the log-det term spreads them out to the second missing cluster too
    assert np.any((reg >= 150) & (reg < 180)) and np.any(reg >= 180)


def test_per_class_allocation_represents_every_class():
    X, y = _blobs(6)
    res = mmd_critic(X, n_prototypes=5, n_criticisms=0, labels=y)
    lab = y[res["prototypes"]]
    assert sorted(np.bincount(lab).tolist()) == [1, 4]
    res2 = mmd_critic(X, n_prototypes=1, n_criticisms=0, labels=y)
    assert y[res2["prototypes"][0]] == 0  # budget smaller than classes -> largest class


def test_validation():
    X, _ = _blobs()
    with pytest.raises(ValueError):
        mmd_critic(X, n_prototypes=0)
    with pytest.raises(ValueError):
        mmd_critic(X, n_prototypes=10_000)
    with pytest.raises(ValueError):
        mmd_critic(X, labels=np.zeros(3))
    K = rbf_kernel(X, X, 1.0)
    with pytest.raises(ValueError):
        select_criticisms(K, [], 2)
    with pytest.raises(ValueError):
        select_criticisms(K, [0], 1, regularizer="bogus")
    with pytest.raises(ValueError):
        select_prototypes(K[:, :3], 1)


def test_cli_prototypes(capsys):
    assert main(["--seed", "0", "--n-samples", "120", "prototypes", "--n-prototypes", "3", "--n-criticisms", "2"]) == 0
    out = capsys.readouterr().out
    assert "MMD-critic on 120 rows" in out
    assert "Prototypes" in out and "Criticisms" in out
