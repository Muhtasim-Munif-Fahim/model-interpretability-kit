import numpy as np
import pytest

from interpretability import grouped_permutation_importance
from interpretability.importance import mean_squared_error, permutation_importance


def _data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    X = np.column_stack([x0, x0 + 0.05 * rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)])
    y = X[:, 0] + X[:, 1] + 0.5 * X[:, 2]
    return X, y


def _model(Z):
    return Z[:, 0] + Z[:, 1] + 0.5 * Z[:, 2]


def test_singleton_groups_match_permutation_importance():
    X, y = _data()
    single = permutation_importance(_model, X, y, n_repeats=4, seed=3)
    grouped = grouped_permutation_importance(_model, X, y, range(X.shape[1]), n_repeats=4, seed=3)
    np.testing.assert_allclose(grouped["mean"], single["mean"])
    np.testing.assert_allclose(grouped["std"], single["std"])
    assert grouped["baseline"] == pytest.approx(single["baseline"])
    assert grouped["groups"] == ["g0", "g1", "g2", "g3"]


def test_group_exceeds_sum_of_parts_for_correlated_block():
    X, y = _data()
    neg_mse = lambda a, b: -mean_squared_error(a, b)
    single = permutation_importance(_model, X, y, metric=neg_mse, n_repeats=5, seed=0)["mean"]
    res = grouped_permutation_importance(_model, X, y, {"pair": [0, 1], "x2": [2], "noise": [3]},
                                         metric=neg_mse, n_repeats=5, seed=0)
    imp = dict(zip(res["groups"], res["mean"]))
    # f = x0 + x1 with x0 ~= x1: joint shuffle gives E[(s - s')^2] = 2 Var(s) ~= 8,
    # single shuffles give ~2 each.
    assert imp["pair"] == pytest.approx(8.0, rel=0.15)
    assert imp["pair"] > single[0] + single[1] * 1.5
    assert imp["x2"] == pytest.approx(0.5, rel=0.2)
    assert imp["noise"] == pytest.approx(0.0, abs=1e-12)


def test_joint_permutation_keeps_rows_together():
    X, y = _data(n=200)
    seen = []

    def spy(Z):
        seen.append(Z.copy())
        return _model(Z)

    grouped_permutation_importance(spy, X, y, [[0, 2]], n_repeats=2, seed=1)
    original = {tuple(r) for r in X[:, [0, 2]].round(12)}
    for Z in seen[1:]:
        assert {tuple(r) for r in Z[:, [0, 2]].round(12)} == original
        np.testing.assert_array_equal(Z[:, [1, 3]], X[:, [1, 3]])
        assert not np.array_equal(Z[:, [0, 2]], X[:, [0, 2]])


def test_output_shapes_and_overlapping_groups():
    X, y = _data(n=300)
    res = grouped_permutation_importance(_model, X, y, {"a": [0, 1], "b": [1, 2], "c": 3}, n_repeats=3, seed=0)
    assert res["scores"].shape == (3, 3)
    assert res["columns"] == [[0, 1], [1, 2], [3]]
    assert res["n_repeats"] == 3


def test_validation():
    X, y = _data(n=50)
    with pytest.raises(ValueError, match="at least one group"):
        grouped_permutation_importance(_model, X, y, {})
    with pytest.raises(ValueError, match="empty"):
        grouped_permutation_importance(_model, X, y, {"a": []})
    with pytest.raises(ValueError, match="out of range"):
        grouped_permutation_importance(_model, X, y, [[0, 9]])
    with pytest.raises(ValueError, match="repeats"):
        grouped_permutation_importance(_model, X, y, [[1, 1]])
    with pytest.raises(ValueError, match="n_repeats"):
        grouped_permutation_importance(_model, X, y, [[0]], n_repeats=0)
