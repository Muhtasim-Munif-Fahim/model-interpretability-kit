"""Tests for tabular Anchors explanations."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.local import anchor_explain


def test_returns_expected_keys():
    X, y, _ = make_synthetic_data(n_samples=300, seed=0)
    model = fit_decision_tree(X, y, max_depth=4)
    result = anchor_explain(model.predict, X[0], X, n_samples=400, seed=1)
    assert set(result) >= {
        "predicates",
        "precision",
        "coverage",
        "prediction",
        "tolerance",
        "n_samples",
    }
    assert 0.0 <= result["precision"] <= 1.0 + 1e-9
    assert 0.0 <= result["coverage"] <= 1.0 + 1e-9


def test_precision_meets_threshold_when_possible():
    # Linear model on one feature: anchoring that feature should be precise.
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(500, 3))

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return Z[:, 0]

    x = np.array([0.55, 0.1, 0.1])
    result = anchor_explain(
        predict,
        x,
        X,
        n_samples=1000,
        n_bins=10,
        precision_threshold=0.9,
        tolerance=0.08,
        seed=2,
    )
    assert result["precision"] >= 0.85
    feats = {p["feature"] for p in result["predicates"]}
    assert 0 in feats


def test_predicates_reference_instance_bin():
    X, y, _ = make_synthetic_data(n_samples=250, seed=3)
    model = fit_decision_tree(X, y, max_depth=3)
    result = anchor_explain(model.predict, X[5], X, n_samples=300, seed=4)
    for pred in result["predicates"]:
        assert pred["low"] <= X[5, pred["feature"]] <= pred["high"] + 1e-9


def test_reproducible_with_seed():
    X, y, _ = make_synthetic_data(n_samples=200, seed=5)
    model = fit_decision_tree(X, y, max_depth=3)
    a = anchor_explain(model.predict, X[0], X, n_samples=200, seed=7)
    b = anchor_explain(model.predict, X[0], X, n_samples=200, seed=7)
    assert a["predicates"] == b["predicates"]
    assert a["precision"] == b["precision"]


def test_invalid_inputs():
    X = np.random.default_rng(0).uniform(size=(50, 2))

    def predict(Z):
        return np.asarray(Z, dtype=float).sum(axis=1)

    with pytest.raises(ValueError, match="n_samples"):
        anchor_explain(predict, X[0], X, n_samples=5)
    with pytest.raises(ValueError, match="n_bins"):
        anchor_explain(predict, X[0], X, n_bins=1)
    with pytest.raises(ValueError, match="width"):
        anchor_explain(predict, np.zeros(3), X)
