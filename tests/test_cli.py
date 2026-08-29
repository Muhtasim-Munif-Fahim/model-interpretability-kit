"""Tests for the command-line interface."""

import numpy as np
import pytest

from interpretability.cli import main


def test_importance_subcommand_prints_table(capsys):
    code = main(["--seed", "1", "--n-samples", "120", "importance", "--n-repeats", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Permutation importance" in out
    assert "X0" in out
    assert "X4" in out


def test_importance_subcommand_ranks_signal_over_noise(capsys):
    main(["--seed", "2", "--n-samples", "120", "importance", "--n-repeats", "2"])
    out = capsys.readouterr().out
    x0_line = next(line for line in out.splitlines() if "X0" in line)
    x4_line = next(line for line in out.splitlines() if "X4" in line)
    x0_score = float(x0_line.split()[1].split("+/-")[0])
    x4_score = float(x4_line.split()[1].split("+/-")[0])
    assert x0_score > x4_score
    assert x4_score < 0.05


def test_pdp_subcommand_single_feature(capsys):
    code = main(["--seed", "3", "--n-samples", "100", "pdp", "--features", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Partial dependence for X1" in out
    assert "->" in out


def test_pdp_subcommand_two_features_surface(capsys):
    code = main(["--seed", "3", "--n-samples", "100", "pdp", "--features", "0,2", "--grid-points", "5"])
    out = capsys.readouterr().out
    assert code == 0
    assert "2-D partial dependence X0 x X2" in out
    assert "5x5 surface" in out


def test_explain_subcommand(capsys):
    code = main(["--seed", "4", "--n-samples", "100", "explain", "--rows", "0", "--n-samples", "120"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Local explanation for row 0" in out
    assert "fidelity" in out


def test_report_subcommand_writes_file(tmp_path, capsys):
    out_path = str(tmp_path / "cli_report.md")
    code = main(["--seed", "5", "--n-samples", "100", "report", "--out", out_path])
    out = capsys.readouterr().out
    assert code == 0
    assert "Wrote %s" % out_path in out
    content = open(out_path, encoding="utf-8").read()
    assert "Global feature importance" in content
    assert "Caveats" in content


def test_cli_missing_command_raises_system_exit():
    with pytest.raises(SystemExit):
        main(["--seed", "1"])
