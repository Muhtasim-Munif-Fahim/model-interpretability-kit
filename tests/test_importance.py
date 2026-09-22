"""Tests for permutation, drop-column, and LOCO feature importance."""

import numpy as np
import pytest

from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.importance import (
    drop_column_importance,
    loco_importance,
    mean_absolute_error,
    mean_squared_error,
    permutation_importance,
    r2_score,
    zero_one_loss,
)


def _split(X, y, n_train=150):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _fit_predict_demo(X_fit, y_fit, X_eval):
    return fit_decision_tree(X_fit, y_fit, max_depth=5, min_samples_leaf=5).predict(X_eval)


def test_r2_score_perfect_and_constant_targets():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    assert r2_score(y, y.mean() * np.ones_like(y)) == pytest.approx(0.0)
    assert r2_score(np.ones(4), np.ones(4)) == pytest.approx(1.0)


def test_permutation_importance_output_structure():
    X, y, _ = make_synthetic_data(n_samples=200, seed=0)
    Xtr, ytr, Xev, yev = _split(X, y)
    model = fit_decision_tree(Xtr, ytr, max_depth=5, min_samples_leaf=5)
    result = permutation_importance(model.predict, Xev, yev, n_repeats=3, seed=1)
    assert set(result) == {"mean", "std", "baseline", "n_repeats"}
    assert result["mean"].shape == (5,)
    assert result["std"].shape == (5,)
    assert result["n_repeats"] == 3
    assert np.all(result["std"] >= 0.0)


def test_permutation_importance_deterministic_with_seed():
    X, y, _ = make_synthetic_data(n_samples=200, seed=0)
    Xtr, ytr, Xev, yev = _split(X, y)
    model = fit_decision_tree(Xtr, ytr, max_depth=5, min_samples_leaf=5)
    r1 = permutation_importance(model.predict, Xev, yev, n_repeats=4, seed=42)
    r2 = permutation_importance(model.predict, Xev, yev, n_repeats=4, seed=42)
    assert np.allclose(r1["mean"], r2["mean"])
    assert np.allclose(r1["std"], r2["std"])


def test_permutation_importance_ranks_known_truth():
    X, y, _ = make_synthetic_data(n_samples=300, seed=11)
    Xtr, ytr, Xev, yev = _split(X, y)
    model = fit_decision_tree(Xtr, ytr, max_depth=5, min_samples_leaf=5)
    imp = permutation_importance(model.predict, Xev, yev, n_repeats=5, seed=3)["mean"]
    assert imp[0] > imp[1]
    assert imp[1] > imp[3]
    assert imp[2] > 0.0
    assert imp[4] < imp[2]
    assert imp[4] < 0.02


def test_permutation_importance_constant_feature_is_zero():
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.uniform(size=150), np.full(150, 0.7)])
    y = 3.0 * X[:, 0] + rng.normal(0.0, 0.05, size=150)
    Xtr, ytr, Xev, yev = _split(X, y, n_train=100)
    model = fit_decision_tree(Xtr, ytr, max_depth=5, min_samples_leaf=3)
    imp = permutation_importance(model.predict, Xev, yev, n_repeats=3, seed=2)["mean"]
    assert imp[1] == pytest.approx(0.0, abs=1e-12)
    assert imp[0] > 0.5


def test_permutation_importance_single_feature_1d_input():
    X, y, _ = make_synthetic_data(n_samples=120, seed=6)
    Xtr, ytr, Xev, yev = _split(X, y, n_train=80)
    model = fit_decision_tree(Xtr[:, :1], ytr, max_depth=4, min_samples_leaf=3)
    result = permutation_importance(model.predict, Xev[:, 0], yev, n_repeats=2, seed=0)
    assert result["mean"].shape == (1,)
    assert result["mean"][0] > 0.0


def test_permutation_importance_with_custom_metric():
    def neg_mse(y_true, y_pred):
        return -float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    X, y, _ = make_synthetic_data(n_samples=300, seed=9)
    Xtr, ytr, Xev, yev = _split(X, y)
    model = fit_decision_tree(Xtr, ytr, max_depth=5, min_samples_leaf=5)
    imp = permutation_importance(model.predict, Xev, yev, metric=neg_mse, n_repeats=3, seed=4)
    assert imp["baseline"] > imp["baseline"] - imp["mean"][0]
    assert np.all(imp["mean"] >= -1e-9)
    assert imp["mean"][0] > imp["mean"][4]


def test_permutation_importance_tiny_dataset():
    X, y, _ = make_synthetic_data(n_samples=10, seed=12)
    model = fit_decision_tree(X, y, max_depth=2, min_samples_leaf=2)
    result = permutation_importance(model.predict, X, y, n_repeats=2, seed=0)
    assert result["mean"].shape == (5,)
    assert np.all(np.isfinite(result["mean"]))


def test_permutation_importance_rejects_bad_inputs():
    X, y, _ = make_synthetic_data(n_samples=30, seed=13)
    with pytest.raises(ValueError):
        permutation_importance(lambda z: z[:, 0], X, y, n_repeats=0)
    with pytest.raises(TypeError):
        permutation_importance(42, X, y, n_repeats=2)
    with pytest.raises(TypeError):
        permutation_importance(lambda z: z[:, 0], X, y, metric="r2", n_repeats=2)


def test_drop_column_importance_ranks_known_truth():
    X, y, _ = make_synthetic_data(n_samples=240, seed=14)
    Xtr, ytr, Xev, yev = _split(X, y, n_train=160)
    result = drop_column_importance(_fit_predict_demo, Xtr, ytr)
    imp = result["importance"]
    assert np.all(imp >= -1e-9)
    assert imp[0] > imp[1]
    assert imp[1] > imp[3]
    assert imp[4] < imp[2]
    assert imp[4] < 0.05


def test_drop_column_importance_baseline_equals_full_model_score():
    X, y, _ = make_synthetic_data(n_samples=150, seed=15)
    result = drop_column_importance(_fit_predict_demo, X, y)
    full_r2 = r2_score(y, _fit_predict_demo(X, y, X))
    assert result["baseline"] == pytest.approx(full_r2)


def test_drop_column_importance_rejects_bad_inputs():
    X, y, _ = make_synthetic_data(n_samples=40, seed=16)
    with pytest.raises(TypeError):
        drop_column_importance("not callable", X, y)
    with pytest.raises(ValueError):
        drop_column_importance(_fit_predict_demo, np.ones((5, 2, 2)), y[:5])


def _ols_fit_predict(X_fit, y_fit, X_eval):
    X_fit = np.asarray(X_fit, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    if X_fit.shape[1] == 0:
        return np.full(X_eval.shape[0], float(y_fit.mean()))
    design = np.column_stack([np.ones(X_fit.shape[0]), X_fit])
    coef, _, _, _ = np.linalg.lstsq(design, y_fit, rcond=None)
    return np.column_stack([np.ones(X_eval.shape[0]), X_eval]) @ coef


def _threshold_classifier(X_fit, y_fit, X_eval):
    """Classify from column 0; an empty matrix predicts the majority class."""
    X_fit = np.asarray(X_fit, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    y_fit = np.asarray(y_fit).ravel()
    if X_fit.shape[1] == 0:
        classes, counts = np.unique(y_fit, return_counts=True)
        majority = classes[int(np.argmax(counts))]
        return np.full(X_eval.shape[0], majority)
    column = X_fit[:, 0]
    order = np.argsort(column, kind="mergesort")
    ordered = column[order]
    best_threshold = float(ordered[0] - 1.0)
    best_accuracy = -1.0
    for i in range(1, ordered.shape[0]):
        if ordered[i] == ordered[i - 1]:
            continue
        threshold = 0.5 * (ordered[i - 1] + ordered[i])
        accuracy = float(np.mean((column > threshold).astype(float) == y_fit))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return (X_eval[:, 0] > best_threshold).astype(float)


def test_loco_importance_matches_manual_excess_loss():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = 2.0 * X[:, 0] - X[:, 1]
    X_train, y_train = X[:30], y[:30]
    X_test, y_test = X[30:], y[30:]
    result = loco_importance(
        _ols_fit_predict,
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        loss=mean_squared_error,
    )
    pred_full = _ols_fit_predict(X_train, y_train, X_test)
    baseline = mean_squared_error(y_test, pred_full)
    assert result["baseline"] == pytest.approx(baseline)
    assert result["n_repeats"] == 1
    assert np.all(result["std"] == 0.0)
    for j in range(2):
        pred_j = _ols_fit_predict(np.delete(X_train, j, axis=1), y_train, np.delete(X_test, j, axis=1))
        expected = mean_squared_error(y_test, pred_j) - baseline
        assert result["mean"][j] == pytest.approx(expected)
    assert result["mean"][0] > result["mean"][1] > 0.0


def test_loco_importance_output_structure_and_repeat_std():
    X, y, _ = make_synthetic_data(n_samples=160, seed=0)
    result = loco_importance(_fit_predict_demo, X, y, n_repeats=4, test_size=0.25, seed=1)
    assert set(result) == {"mean", "std", "baseline", "n_repeats"}
    assert result["mean"].shape == (5,)
    assert result["std"].shape == (5,)
    assert result["n_repeats"] == 4
    assert np.all(result["std"] >= 0.0)
    assert result["std"][0] > 0.0
    assert np.all(np.isfinite(result["mean"]))
    assert result["baseline"] > 0.0


def test_loco_importance_single_repeat_std_is_zero():
    X, y, _ = make_synthetic_data(n_samples=80, seed=2)
    result = loco_importance(_fit_predict_demo, X, y, n_repeats=1, seed=2)
    assert np.all(result["std"] == 0.0)
    assert result["n_repeats"] == 1


def test_loco_importance_deterministic_with_seed():
    X, y, _ = make_synthetic_data(n_samples=120, seed=3)
    first = loco_importance(_fit_predict_demo, X, y, n_repeats=3, test_size=0.3, seed=7)
    second = loco_importance(_fit_predict_demo, X, y, n_repeats=3, test_size=0.3, seed=7)
    assert np.allclose(first["mean"], second["mean"])
    assert np.allclose(first["std"], second["std"])
    assert first["baseline"] == pytest.approx(second["baseline"])


def test_loco_importance_explicit_split_ignores_seed():
    X, y, _ = make_synthetic_data(n_samples=100, seed=4)
    kwargs = dict(X_test=X[70:], y_test=y[70:], loss=mean_absolute_error, n_repeats=1)
    first = loco_importance(_fit_predict_demo, X[:70], y[:70], seed=1, **kwargs)
    second = loco_importance(_fit_predict_demo, X[:70], y[:70], seed=99, **kwargs)
    assert np.allclose(first["mean"], second["mean"])


def test_loco_importance_bootstrap_repeats_are_seeded():
    X, y, _ = make_synthetic_data(n_samples=90, seed=5)
    X_train, y_train, X_test, y_test = X[:60], y[:60], X[60:], y[60:]
    first = loco_importance(
        _fit_predict_demo, X_train, y_train, X_test=X_test, y_test=y_test, n_repeats=3, seed=8
    )
    second = loco_importance(
        _fit_predict_demo, X_train, y_train, X_test=X_test, y_test=y_test, n_repeats=3, seed=8
    )
    assert np.allclose(first["mean"], second["mean"])
    assert np.allclose(first["std"], second["std"])
    assert first["std"][0] > 0.0


def test_loco_importance_ranks_known_truth():
    X, y, _ = make_synthetic_data(n_samples=240, seed=11)
    result = loco_importance(
        _fit_predict_demo,
        X[:160],
        y[:160],
        X_test=X[160:],
        y_test=y[160:],
        loss=mean_absolute_error,
    )
    imp = result["mean"]
    assert imp[0] > imp[1]
    assert imp[1] > imp[3]
    assert imp[2] > 0.0
    assert imp[4] < imp[2]
    assert imp[4] < 0.05


def test_loco_importance_baseline_is_held_out_loss():
    X, y, _ = make_synthetic_data(n_samples=80, noise=0.3, seed=15)

    def overfit(X_fit, y_fit, X_eval):
        return fit_decision_tree(X_fit, y_fit, max_depth=20, min_samples_leaf=1).predict(X_eval)

    X_train, y_train, X_test, y_test = X[:50], y[:50], X[50:], y[50:]
    result = loco_importance(overfit, X_train, y_train, X_test=X_test, y_test=y_test)
    test_loss = mean_absolute_error(y_test, overfit(X_train, y_train, X_test))
    train_loss = mean_absolute_error(y_train, overfit(X_train, y_train, X_train))
    assert result["baseline"] == pytest.approx(test_loss)
    assert result["baseline"] != pytest.approx(train_loss)


def test_loco_importance_constant_feature_is_zero():
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.uniform(size=120), np.full(120, 0.7)])
    y = 3.0 * X[:, 0] + rng.normal(0.0, 0.05, size=120)
    result = loco_importance(
        _fit_predict_demo,
        X[:80],
        y[:80],
        X_test=X[80:],
        y_test=y[80:],
        n_repeats=1,
    )
    assert result["mean"][1] == pytest.approx(0.0, abs=1e-12)
    assert result["mean"][0] > 0.2


def test_loco_importance_single_feature_1d_input():
    X, y, _ = make_synthetic_data(n_samples=90, seed=6)
    result = loco_importance(
        _fit_predict_demo,
        X[:60, 0],
        y[:60],
        X_test=X[60:, 0],
        y_test=y[60:],
    )
    assert result["mean"].shape == (1,)
    assert result["mean"][0] > 0.0


def test_loco_importance_squared_error_loss():
    X, y, _ = make_synthetic_data(n_samples=180, seed=9)
    imp = loco_importance(
        _fit_predict_demo,
        X[:120],
        y[:120],
        X_test=X[120:],
        y_test=y[120:],
        loss=mean_squared_error,
    )["mean"]
    assert imp[0] > imp[4]
    assert imp[0] > 0.0


def test_loco_importance_zero_one_loss_is_accuracy_drop():
    rng = np.random.default_rng(1)
    signal = rng.uniform(size=160)
    noise = rng.uniform(size=160)
    X = np.column_stack([signal, noise])
    y = (signal > 0.5).astype(float)
    result = loco_importance(
        _threshold_classifier,
        X[:110],
        y[:110],
        X_test=X[110:],
        y_test=y[110:],
        loss=zero_one_loss,
    )
    assert result["baseline"] == pytest.approx(0.0, abs=1e-12)
    assert result["mean"][0] > 0.3
    assert result["mean"][1] == pytest.approx(0.0, abs=1e-12)
    full_accuracy = 1.0 - result["baseline"]
    reduced_accuracy = full_accuracy - result["mean"][0]
    assert reduced_accuracy < 0.7


def test_loco_importance_does_not_mutate_inputs():
    X, y, _ = make_synthetic_data(n_samples=40, seed=17)
    X_train, y_train = X[:30].copy(), y[:30].copy()
    X_test, y_test = X[30:].copy(), y[30:].copy()
    original = (X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
    loco_importance(
        _ols_fit_predict, X_train, y_train, X_test=X_test, y_test=y_test, loss=mean_squared_error
    )
    assert np.allclose(X_train, original[0])
    assert np.allclose(y_train, original[1])
    assert np.allclose(X_test, original[2])
    assert np.allclose(y_test, original[3])


def test_loco_importance_rejects_bad_inputs():
    X, y, _ = make_synthetic_data(n_samples=30, seed=13)
    with pytest.raises(ValueError):
        loco_importance(_fit_predict_demo, X, y, n_repeats=0)
    with pytest.raises(ValueError):
        loco_importance(_fit_predict_demo, X, y, n_repeats=True)
    with pytest.raises(TypeError):
        loco_importance("not callable", X, y)
    with pytest.raises(TypeError):
        loco_importance(_fit_predict_demo, X, y, loss="mae")
    with pytest.raises(ValueError):
        loco_importance(_fit_predict_demo, X, y, loss=lambda yt, yp: np.abs(yt - yp))
    with pytest.raises(ValueError):
        loco_importance(_fit_predict_demo, X, y, test_size=0.0)
    with pytest.raises(ValueError):
        loco_importance(_fit_predict_demo, X, y, test_size=1.0)
    with pytest.raises(ValueError):
        loco_importance(_fit_predict_demo, X, y, X_test=X)
    with pytest.raises(ValueError):
        loco_importance(_fit_predict_demo, X[:20], y[:20], X_test=X[:10, :2], y_test=y[:10])
    with pytest.raises(ValueError):
        loco_importance(_fit_predict_demo, X[:1], y[:1])
    with pytest.raises(ValueError):
        loco_importance(_fit_predict_demo, np.ones((5, 2, 2)), y[:5])
