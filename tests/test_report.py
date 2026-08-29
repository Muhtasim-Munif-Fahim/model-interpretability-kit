"""Tests for the markdown report renderer."""

import numpy as np
import pytest

from interpretability.report import DEFAULT_CAVEATS, render_report


def _importance():
    return {"mean": np.array([0.4, 0.2, 0.05]), "std": np.array([0.01, 0.02, 0.0]), "baseline": 0.9}


def _pdp():
    grid = np.linspace(0.0, 1.0, 5)
    return [{"feature": "X0", "grid": grid, "values": grid * 2.0}]


def _local():
    return [
        {
            "description": "Row 0",
            "prediction": 3.2,
            "weighted_r2": 0.97,
            "coefficients": np.array([2.1, -0.4]),
            "feature_names": ["X0", "X1"],
        }
    ]


def _faithfulness():
    return [{"description": "Row 0", "fidelity_r2": 0.97, "overlap": 2, "jaccard": 0.67}]


def test_report_contains_required_sections():
    report = render_report(_importance(), _pdp(), _local(), _faithfulness())
    assert "Global feature importance" in report
    assert "Partial dependence summaries" in report
    assert "Example local explanations" in report
    assert "Faithfulness of local explanations" in report
    assert "Caveats" in report


def test_report_importance_table_sorted_descending():
    report = render_report(_importance(), [], [], [])
    lines = [line for line in report.splitlines() if line.startswith("| X")]
    assert len(lines) == 3
    assert "X0" in lines[0]
    assert "X1" in lines[1]
    assert "X2" in lines[2]


def test_report_renders_baseline_score():
    report = render_report(_importance(), [], [], [])
    assert "0.9000" in report
    assert "**3** features" in report

def test_report_pdp_trend_and_grid():
    report = render_report(_importance(), _pdp(), [], [])
    assert "increasing" in report
    assert "0.0000 ... 1.0000" in report


def test_report_local_explanations_table():
    report = render_report(_importance(), [], _local(), [])
    assert "Row 0" in report
    assert "| X0 | 2.1000 |" in report
    assert "| X1 | -0.4000 |" in report


def test_report_faithfulness_table():
    report = render_report(_importance(), [], [], _faithfulness())
    assert "| Row 0 | 0.9700 | 2 | 0.6700 |" in report


def test_report_default_caveats_present():
    report = render_report(_importance(), [], [], [])
    for caveat in DEFAULT_CAVEATS:
        assert caveat in report


def test_report_custom_caveats_replace_defaults():
    report = render_report(_importance(), [], [], [], caveats=["Custom note."])
    assert "Custom note." in report
    assert DEFAULT_CAVEATS[0] not in report


def test_report_default_feature_names_when_omitted():
    report = render_report({"mean": np.array([0.1, 0.2])}, [], [], [])
    assert "| X1 | 0.2000 |" in report


def test_report_accepts_plain_array_importance():
    report = render_report(np.array([0.3, 0.1]), [], [], [], feature_names=["age", "size"])
    assert "| age | 0.3000 |" in report


def test_report_accepts_lime_explain_style_dict():
    explanation = {
        "prediction": 1.5,
        "weighted_r2": 0.9,
        "coefficients": np.array([0.5, 0.25]),
    }
    report = render_report(_importance(), [], [explanation], [])
    assert "Instance 1" in report
    assert "| X1 | 0.2500 |" in report


def test_report_feature_name_mismatch_raises():
    with pytest.raises(ValueError):
        render_report({"mean": np.array([0.1, 0.2, 0.3])}, [], [], [], feature_names=["a", "b"])


def test_report_empty_sections_are_annotated():
    report = render_report({"mean": np.array([0.1])}, [], [], [])
    assert "No partial dependence summaries provided." in report
    assert "No local explanations provided." in report
    assert "No faithfulness results provided." in report


def test_report_is_valid_markdown_shape():
    report = render_report(_importance(), _pdp(), _local(), _faithfulness())
    assert report.startswith("Model Interpretability Report")
    assert report.strip()
    assert "\n\n" in report
