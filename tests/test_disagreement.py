import itertools

import numpy as np
import pytest

from interpretability.disagreement import (
    explanation_disagreement,
    feature_agreement,
    kendall_tau,
    pairwise_rank_agreement,
    rank_agreement,
    rank_correlation,
    sign_agreement,
    signed_rank_agreement,
    top_k_features,
)


A = np.array([5.0, -4.0, 3.0, 0.5, 0.1])
B = np.array([-4.0, 5.0, 0.2, 3.0, 0.1])  # top-3 {1, 0, 3}; ranks 0/1 swapped; sign flips


def test_top_k_features_by_magnitude_with_stable_ties():
    assert top_k_features(A, 3).tolist() == [0, 1, 2]
    assert top_k_features([1.0, -1.0, 1.0], 2).tolist() == [0, 1]
    with pytest.raises(ValueError):
        top_k_features(A, 0)


def test_hand_computed_top_k_metrics():
    # top3(A) = [0, 1, 2]; top3(B) = [1, 0, 3]
    assert feature_agreement(A, B, 3) == pytest.approx(2 / 3)
    assert rank_agreement(A, B, 3) == 0.0
    # common {0, 1}: sign(A0)=+, sign(B0)=- ; sign(A1)=-, sign(B1)=+
    assert sign_agreement(A, B, 3) == 0.0
    assert sign_agreement(A, np.abs(B) * np.sign(A), 3) == pytest.approx(2 / 3)
    assert signed_rank_agreement(A, A, 3) == 1.0
    assert rank_agreement(A, B, 5) == pytest.approx(1 / 5)  # only feature 4 in place


def test_identical_and_scaled_explanations_agree_fully():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=8)
    for fn in (feature_agreement, rank_agreement, sign_agreement, signed_rank_agreement):
        assert fn(phi, 3 * phi, 4) == 1.0
    assert rank_correlation(phi, 2 * phi) == pytest.approx(1.0)
    assert kendall_tau(phi, 0.5 * phi) == pytest.approx(1.0)
    assert pairwise_rank_agreement(phi, phi) == 1.0
    # signs only matter for the sign metrics
    assert feature_agreement(phi, -phi, 4) == 1.0 and sign_agreement(phi, -phi, 4) == 0.0


def test_kendall_tau_matches_bruteforce_tau_b_with_ties():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.integers(0, 4, 9).astype(float)
        b = rng.integers(0, 4, 9).astype(float)
        conc = disc = ta = tb = 0
        for i, j in itertools.combinations(range(9), 2):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            conc += s > 0
            disc += s < 0
            ta += a[i] == a[j]
            tb += b[i] == b[j]
        n0 = 36
        expected = (conc - disc) / np.sqrt((n0 - ta) * (n0 - tb)) if (n0 - ta) * (n0 - tb) else np.nan
        got = kendall_tau(a, b)
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected)
    assert kendall_tau([1, 2, 3], [3, 2, 1], absolute=False) == pytest.approx(-1.0)


def test_spearman_matches_pearson_of_average_ranks_and_subset():
    a = np.array([0.1, 0.4, 0.4, 2.0, -3.0])
    b = np.array([0.2, 0.3, 0.5, 1.0, 0.0])
    # |a| ranks: [1, 2.5, 2.5, 4, 5]; |b| ranks: [2, 3, 4, 5, 1]
    ra, rb = np.array([1, 2.5, 2.5, 4, 5]), np.array([2, 3, 4, 5, 1])
    assert rank_correlation(a, b) == pytest.approx(np.corrcoef(ra, rb)[0, 1])
    assert rank_correlation(a, b, features=[0, 1, 3]) == pytest.approx(1.0)
    assert np.isnan(rank_correlation([1, 1, 1], [1, 2, 3]))
    assert pairwise_rank_agreement(a, b, features=[0, 3, 4]) == pytest.approx(1 / 3)


def test_explanation_disagreement_matrices():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(6, 5))
    attributions = {
        "shap": base,
        "noisy": base + rng.normal(0, 0.05, base.shape),
        "random": rng.normal(size=(6, 5)),
    }
    out = explanation_disagreement(attributions, k=2)
    assert out["methods"] == ["shap", "noisy", "random"] and out["k"] == 2
    for name, M in out["matrices"].items():
        assert M.shape == (3, 3)
        np.testing.assert_allclose(M, M.T)
        np.testing.assert_allclose(np.diag(M), 1.0)
    fa = out["matrices"]["feature_agreement"]
    assert fa[0, 1] > fa[0, 2]
    kt = out["matrices"]["kendall_tau"]
    assert kt[0, 1] > 0.8 and kt[0, 1] > kt[0, 2]
    assert out["per_instance"]["rank_agreement"][("shap", "noisy")].shape == (6,)
    one_row = explanation_disagreement({"a": A, "b": B}, k=3, metrics=["feature_agreement"])
    assert one_row["matrices"]["feature_agreement"][0, 1] == pytest.approx(2 / 3)


def test_validation():
    with pytest.raises(ValueError, match="same length"):
        feature_agreement([1, 2], [1, 2, 3], 1)
    with pytest.raises(ValueError, match="finite"):
        rank_agreement([1, np.nan], [1, 2], 1)
    with pytest.raises(ValueError, match="two methods"):
        explanation_disagreement({"a": A})
    with pytest.raises(ValueError, match="same"):
        explanation_disagreement({"a": np.ones((2, 5)), "b": np.ones((3, 5))})
    with pytest.raises(ValueError, match="unknown"):
        explanation_disagreement({"a": A, "b": B}, metrics=["nope"])


def test_cli_explanation_disagreement(capsys):
    from interpretability.cli import main

    assert main(["--n-samples", "120", "explanation-disagreement", "--rows", "0,1", "--k", "2"]) == 0
    out = capsys.readouterr().out
    assert "feature_agreement" in out and "kernel-shap" in out and "integrated-gradients" in out
