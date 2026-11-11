import numpy as np
import pytest

from interpretability import pawn_indices
from interpretability.cli import main


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 3))
    y = 3.0 * X[:, 0] + 0.5 * X[:, 1] + 0.05 * rng.normal(size=n)  # X2 irrelevant
    return X, y


def test_ranks_features_and_irrelevant_near_dummy():
    X, y = _data()
    res = pawn_indices(None, X, y=y, n_classes=10, seed=0)
    assert res["index"][0] > res["index"][1] > res["index"][2]
    assert res["index"][0] > 0.5
    # an irrelevant input sits at the noise floor given by the dummy
    assert res["index"][2] < 2.0 * res["dummy"] + 0.02
    assert res["index"][1] > res["dummy"]
    assert np.all((res["index"] >= 0) & (res["index"] <= 1))
    assert res["stat"] == "median" and res["n_classes"] == 10
    assert len(res["ks"]) == 3 and all(k.size == 10 for k in res["ks"])


def test_slice_ks_matches_scipy_two_sample_statistic():
    stats = pytest.importorskip("scipy.stats")
    X, y = _data(n=600, seed=1)
    res = pawn_indices(None, X, y=y, n_classes=6)
    edges = np.quantile(X[:, 0], np.linspace(0, 1, 7)[1:-1])
    labels = np.searchsorted(edges, X[:, 0], side="right")
    expected = [stats.ks_2samp(y, y[labels == k]).statistic for k in range(6)]
    np.testing.assert_allclose(res["ks"][0], expected, atol=1e-12)
    np.testing.assert_allclose(res["median"][0], np.median(expected))
    np.testing.assert_allclose(res["max"][0], np.max(expected))


def test_detects_variance_only_effect_missed_by_mean():
    rng = np.random.default_rng(2)
    n = 3000
    X = rng.uniform(size=(n, 2))
    y = rng.normal(size=n) * (0.1 + 2.0 * X[:, 0])  # E[y | x0] = 0 for every x0
    res = pawn_indices(None, X, y=y, n_classes=8, stat="mean", seed=0)
    assert abs(np.corrcoef(X[:, 0], y)[0, 1]) < 0.05
    assert res["index"][0] > 2.5 * res["index"][1]
    assert res["index"][0] == pytest.approx(res["mean"][0])


def test_predict_callable_and_bootstrap_ci():
    X, y = _data(n=800, seed=3)
    res = pawn_indices(lambda Z: 3.0 * Z[:, 0] + 0.5 * Z[:, 1], X, n_classes=8, n_bootstrap=30, seed=4)
    assert np.all(res["ci_low"] <= res["ci_high"])
    assert res["ci_low"][0] > res["ci_high"][2]


def test_constant_output_and_validation():
    X, _ = _data(n=200)
    res = pawn_indices(None, X, y=np.ones(200), n_classes=5)
    np.testing.assert_allclose(res["index"], 0.0)
    with pytest.raises(ValueError, match="stat"):
        pawn_indices(None, X, y=np.ones(200), stat="mode")
    with pytest.raises(ValueError, match="n_classes"):
        pawn_indices(None, X, y=np.ones(200), n_classes=1)
    with pytest.raises(ValueError, match="rows"):
        pawn_indices(None, X[:20], y=np.ones(20), n_classes=10)
    with pytest.raises(ValueError, match="one value per row"):
        pawn_indices(None, X, y=np.ones(10))
    with pytest.raises(ValueError, match="predict or y"):
        pawn_indices(None, X)


def test_cli_pawn(capsys):
    code = main(["--seed", "1", "--n-samples", "200", "pawn", "--n-classes", "5", "--n-bootstrap", "5"])
    assert code == 0
    out = capsys.readouterr().out
    assert "PAWN sensitivity" in out and "dummy floor" in out and "[" in out
