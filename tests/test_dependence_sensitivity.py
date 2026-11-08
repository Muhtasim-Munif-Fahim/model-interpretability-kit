import numpy as np
import pytest

from interpretability import borgonovo_delta, hsic, hsic_sensitivity
from interpretability.cli import main
from interpretability.dependence_sensitivity import _center, _rbf_gram


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n, 3))


def test_hsic_matches_trace_formula():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    y = x ** 2 + 0.1 * rng.normal(size=40)
    K = _rbf_gram(x[:, None])
    L = _rbf_gram(y[:, None])
    H = np.eye(40) - 1.0 / 40
    expected = np.trace(K @ H @ L @ H) / 40 ** 2
    assert hsic(x, y) == pytest.approx(expected)
    np.testing.assert_allclose(_center(K), H @ K @ H, atol=1e-12)


def test_hsic_detects_nonlinear_zero_correlation_dependence():
    X = _data()
    predict = lambda Z: Z[:, 0] ** 2  # symmetric: corr(X0, Y) ~ 0
    assert abs(np.corrcoef(X[:, 0], predict(X))[0, 1]) < 0.15
    res = hsic_sensitivity(predict, X, n_permutations=99, seed=0)
    assert res["p_values"][0] <= 0.02
    assert res["p_values"][1] > 0.05 and res["p_values"][2] > 0.05
    assert res["r2_hsic"][0] > 5 * max(res["r2_hsic"][1], res["r2_hsic"][2])
    assert np.all((res["r2_hsic"] >= 0) & (res["r2_hsic"] <= 1))


def test_r2_hsic_is_one_for_identity():
    X = _data(n=80)
    res = hsic_sensitivity(None, X, y=X[:, 1], n_permutations=0)
    assert res["r2_hsic"][1] == pytest.approx(1.0)
    assert res["p_values"] is None


def test_hsic_subsampling_and_validation():
    X = _data(n=200)
    res = hsic_sensitivity(lambda Z: Z[:, 0], X, n_permutations=0, max_rows=50, seed=0)
    assert res["n_rows"] == 50
    with pytest.raises(ValueError):
        hsic_sensitivity(None, X)
    with pytest.raises(ValueError):
        hsic([1.0, 2.0], [1.0])


def test_borgonovo_delta_ranks_features():
    X = _data(n=600, seed=3)
    predict = lambda Z: 3 * Z[:, 0] + 0.5 * Z[:, 1]
    res = borgonovo_delta(predict, X)
    d = res["delta"]
    assert np.all((d >= 0) & (d <= 1))
    assert d[0] > d[1] > d[2]
    assert d[0] > 0.3
    assert d[2] < 0.12


def test_borgonovo_bootstrap_and_constant_output():
    X = _data(n=200, seed=4)
    res = borgonovo_delta(lambda Z: Z[:, 0], X, n_bootstrap=20, seed=0)
    assert np.all(res["ci_low"] <= res["ci_high"])
    assert res["delta_bc"].shape == (3,)
    const = borgonovo_delta(lambda Z: np.ones(len(Z)), X)
    np.testing.assert_allclose(const["delta"], 0.0)
    with pytest.raises(ValueError):
        borgonovo_delta(lambda Z: Z[:, 0], X[:5])


def test_cli_hsic(capsys):
    code = main(["--seed", "1", "--n-samples", "150", "hsic", "--n-permutations", "19"])
    out = capsys.readouterr().out
    assert code == 0
    assert "R2-HSIC" in out and "delta" in out
    assert "X0" in out
