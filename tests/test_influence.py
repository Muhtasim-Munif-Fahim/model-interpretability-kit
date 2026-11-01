import numpy as np
import pytest

from interpretability import fit_regularized_glm, influence_functions, loo_refit
from interpretability.cli import main


def _reg_data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.3 + rng.normal(size=n)
    Xt = rng.normal(size=(15, 3))
    yt = Xt @ np.array([1.0, -2.0, 0.5]) + 0.3
    return X, y, Xt, yt


def _clf_data(n=80, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    f = X @ np.array([1.5, -1.0, 0.5])
    y = (f + rng.logistic(size=n) > 0).astype(float)
    Xt = rng.normal(size=(20, 3))
    yt = (Xt @ np.array([1.5, -1.0, 0.5]) > 0).astype(float)
    return X, y, Xt, yt


def test_ridge_fit_matches_closed_form_and_unpenalised_intercept():
    X, y, _, _ = _reg_data()
    theta = fit_regularized_glm(X, y, "squared", l2=2.0)
    Z = np.hstack([X, np.ones((len(X), 1))])
    P = np.diag([2.0, 2.0, 2.0, 0.0])
    np.testing.assert_allclose(theta, np.linalg.solve(Z.T @ Z + P, Z.T @ y))
    # l2 = 0 reduces to ordinary least squares
    np.testing.assert_allclose(fit_regularized_glm(X, y, l2=0.0), np.linalg.lstsq(Z, y, rcond=None)[0])


def test_logistic_fit_is_stationary():
    X, y, _, _ = _clf_data()
    theta = fit_regularized_glm(X, y, "logistic", l2=0.5)
    Z = np.hstack([X, np.ones((len(X), 1))])
    p = 1 / (1 + np.exp(-Z @ theta))
    grad = Z.T @ (p - y) + 0.5 * np.r_[theta[:3], 0.0]
    assert np.abs(grad).max() < 1e-8


def test_newton_is_exact_loo_for_ridge():
    X, y, Xt, yt = _reg_data()
    est = influence_functions(X, y, "squared", 1.0, Xt, yt, method="newton")
    ex = loo_refit(X, y, "squared", 1.0, Xt, yt)
    np.testing.assert_allclose(est["param_change"], ex["param_change"], atol=1e-10)


def test_first_order_ridge_equals_exact_times_one_minus_leverage():
    X, y, _, _ = _reg_data()
    est = influence_functions(X, y, "squared", 1.0)
    ex = loo_refit(X, y, "squared", 1.0)
    np.testing.assert_allclose(est["param_change"], ex["param_change"] * (1 - est["leverage"])[:, None],
                               atol=1e-10)


def test_test_loss_change_tracks_exact_loo():
    X, y, Xt, yt = _reg_data(n=80, seed=3)
    est = influence_functions(X, y, "squared", 1.0, Xt, yt)
    ex = loo_refit(X, y, "squared", 1.0, Xt, yt)
    assert est["test_loss_change"].shape == (80, 15)
    assert np.corrcoef(est["mean_test_loss_change"], ex["mean_test_loss_change"])[0, 1] > 0.9


@pytest.mark.parametrize("method", ["influence", "newton"])
def test_logistic_estimates_close_to_refit(method):
    X, y, Xt, yt = _clf_data()
    est = influence_functions(X, y, "logistic", 1.0, Xt, yt, method=method)
    ex = loo_refit(X, y, "logistic", 1.0, Xt, yt)
    corr = np.corrcoef(est["param_change"].ravel(), ex["param_change"].ravel())[0, 1]
    assert corr > 0.98
    assert np.corrcoef(est["mean_test_loss_change"], ex["mean_test_loss_change"])[0, 1] > 0.95


def test_newton_beats_first_order_for_logistic():
    X, y, _, _ = _clf_data()
    ex = loo_refit(X, y, "logistic", 1.0)["param_change"]
    e1 = np.abs(influence_functions(X, y, "logistic", 1.0)["param_change"] - ex).max()
    e2 = np.abs(influence_functions(X, y, "logistic", 1.0, method="newton")["param_change"] - ex).max()
    assert e2 < e1


def test_outlier_has_largest_self_influence_and_hurts():
    X, y, Xt, yt = _reg_data(n=60, seed=5)
    y = y.copy()
    y[7] += 25.0  # gross label error
    est = influence_functions(X, y, "squared", 1.0, Xt, yt)
    assert np.argmax(est["self_influence"]) == 7
    assert np.argmax(est["cooks_distance"]) == 7
    assert np.argmin(est["mean_test_loss_change"]) == 7  # removing it lowers test loss most


def test_leverage_sums_to_effective_dof():
    X, y, _, _ = _reg_data()
    est = influence_functions(X, y, "squared", 0.0)
    assert est["leverage"].sum() == pytest.approx(4.0)  # OLS hat matrix trace = p
    est = influence_functions(X, y, "squared", 5.0)
    assert est["leverage"].sum() < 4.0


def test_no_intercept_option():
    X, y, _, _ = _reg_data()
    est = influence_functions(X, y, "squared", 1.0, fit_intercept=False, method="newton")
    ex = loo_refit(X, y, "squared", 1.0, fit_intercept=False)
    assert est["theta"].shape == (3,)
    np.testing.assert_allclose(est["param_change"], ex["param_change"], atol=1e-10)


def test_validation_errors():
    X, y, Xt, yt = _reg_data()
    with pytest.raises(ValueError, match="loss"):
        influence_functions(X, y, "hinge")
    with pytest.raises(ValueError, match="method"):
        influence_functions(X, y, method="exact")
    with pytest.raises(ValueError, match="non-negative"):
        influence_functions(X, y, l2=-1)
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        influence_functions(X, y, "logistic")
    with pytest.raises(ValueError, match="l2 > 0"):
        influence_functions(X, (y > 0).astype(float), "logistic", l2=0.0)
    with pytest.raises(ValueError, match="y_test"):
        influence_functions(X, y, X_test=Xt)
    with pytest.raises(ValueError, match="columns"):
        influence_functions(X, y, X_test=Xt[:, :2], y_test=yt)
    with pytest.raises(ValueError, match="finite"):
        Xn = X.copy()
        Xn[0, 0] = np.nan
        influence_functions(Xn, y)


def test_cli_influence(capsys):
    assert main(["influence", "--n-train", "80", "--top", "2", "--method", "newton"]) == 0
    out = capsys.readouterr().out
    assert "most harmful" in out and "most helpful" in out
    assert "max |param change - exact LOO| = " in out
