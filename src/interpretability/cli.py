"""Command-line interface for the interpretability toolkit.

Subcommands build a demo decision tree on synthetic (or CSV) data and then
expose a single explanation type each: ``importance``, ``pdp``, ``explain``
and ``report``. All randomness is seeded through ``--seed`` so runs are
reproducible.
"""

import argparse
import sys

import numpy as np

from .demo_model import fit_decision_tree, make_synthetic_data
from .evaluate import top_feature_overlap
from .importance import permutation_importance
from .local import lime_explain
from .partial_dependence import partial_dependence, partial_dependence_2d
from .report import render_report

__all__ = ["main", "build_parser"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="interpretability",
        description="Model-agnostic interpretability for a demo regression tree.",
    )
    parser.add_argument("--data", default=None, help="CSV path; the last column is the target")
    parser.add_argument("--n-samples", type=int, default=300, help="samples when no CSV is given")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    sub = parser.add_subparsers(dest="command", required=True)

    p_importance = sub.add_parser("importance", help="permutation feature importance")
    p_importance.add_argument("--n-repeats", type=int, default=5)

    p_pdp = sub.add_parser("pdp", help="partial dependence curves")
    p_pdp.add_argument("--features", default="0,1", help="one or two feature indices")
    p_pdp.add_argument("--grid-points", type=int, default=15)

    p_explain = sub.add_parser("explain", help="local LIME-style explanations")
    p_explain.add_argument("--rows", default="0,1", help="comma-separated row indices")
    p_explain.add_argument("--n-samples", type=int, default=300, help="perturbations per row")

    p_report = sub.add_parser("report", help="write a markdown report")
    p_report.add_argument("--out", default="demo_report.md")
    return parser


def _load_data(args):
    if args.data:
        data = np.loadtxt(args.data, delimiter=",", ndmin=2)
        if data.shape[1] < 2:
            sys.exit("CSV data must contain at least one feature and a target column")
        X, y = data[:, :-1], data[:, -1]
        feature_names = ["X%d" % j for j in range(X.shape[1])]
    else:
        X, y, feature_names = make_synthetic_data(n_samples=args.n_samples, seed=args.seed)
    return X, y, feature_names


def _fit(X, y, seed):
    return fit_decision_tree(X, y, max_depth=6, min_samples_leaf=5)


def _parse_indices(value, label):
    try:
        return [int(item) for item in value.split(",")]
    except ValueError:
        sys.exit("--%s expects comma-separated integers, got %r" % (label, value))


def _cmd_importance(args):
    X, y, names = _load_data(args)
    model = _fit(X, y, args.seed)
    result = permutation_importance(
        model.predict, X, y, n_repeats=args.n_repeats, seed=args.seed
    )
    order = np.argsort(-result["mean"])
    print("Permutation importance (baseline R2 = %.4f):" % result["baseline"])
    for j in order:
        print("  %-6s %8.4f +/- %.4f" % (names[j], result["mean"][j], result["std"][j]))
    return 0


def _cmd_pdp(args):
    X, y, names = _load_data(args)
    model = _fit(X, y, args.seed)
    features = _parse_indices(args.features, "features")
    if len(features) == 1:
        f = features[0]
        pdp = partial_dependence(model.predict, X, f, grid_points=args.grid_points)
        print("Partial dependence for %s:" % names[f])
        for grid_value, value in zip(pdp["grid"], pdp["values"]):
            print("  %.4f -> %.4f" % (grid_value, value))
    elif len(features) == 2:
        surface = partial_dependence_2d(
            model.predict, X, tuple(features), grid_points=args.grid_points
        )
        print(
            "2-D partial dependence %s x %s: %dx%d surface, min %.4f, max %.4f"
            % (
                names[surface["feature0"]],
                names[surface["feature1"]],
                surface["values"].shape[0],
                surface["values"].shape[1],
                surface["values"].min(),
                surface["values"].max(),
            )
        )
    else:
        sys.exit("--features expects one or two indices")
    return 0


def _cmd_explain(args):
    X, y, names = _load_data(args)
    model = _fit(X, y, args.seed)
    rows = _parse_indices(args.rows, "rows")
    for i in rows:
        if i < 0 or i >= X.shape[0]:
            sys.exit("row index %d out of range" % i)
        result = lime_explain(
            model.predict,
            X[i],
            X,
            n_samples=args.n_samples,
            seed=args.seed,
            feature_names=names,
        )
        print(
            "Local explanation for row %d (prediction %.4f, fidelity %.3f):"
            % (i, result["prediction"], result["weighted_r2"])
        )
        for j, coef in enumerate(result["coefficients"]):
            print("  %-6s %8.4f" % (names[j], coef))
    return 0


def _cmd_report(args):
    X, y, names = _load_data(args)
    model = _fit(X, y, args.seed)
    importance = permutation_importance(
        model.predict, X, y, n_repeats=5, seed=args.seed
    )
    pdp_summaries = []
    for f in (0, 1, 2):
        pdp = partial_dependence(model.predict, X, f, grid_points=15)
        pdp_summaries.append(
            {"feature": names[f], "grid": pdp["grid"], "values": pdp["values"]}
        )
    local_explanations = []
    faithfulness = []
    for i in (0, 1):
        explanation = lime_explain(
            model.predict,
            X[i],
            X,
            n_samples=args.n_samples,
            seed=args.seed,
            feature_names=names,
        )
        explanation["description"] = "Row %d" % i
        overlap = top_feature_overlap(explanation["coefficients"], importance["mean"], k=3)
        local_explanations.append(explanation)
        faithfulness.append(
            {
                "description": "Row %d" % i,
                "fidelity_r2": explanation["weighted_r2"],
                "overlap": overlap["overlap"],
                "jaccard": overlap["jaccard"],
            }
        )
    report = render_report(
        importance,
        pdp_summaries,
        local_explanations,
        faithfulness,
        feature_names=names,
        title="Demo interpretability report",
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)
    print("Wrote %s" % args.out)
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "importance":
        return _cmd_importance(args)
    if args.command == "pdp":
        return _cmd_pdp(args)
    if args.command == "explain":
        return _cmd_explain(args)
    if args.command == "report":
        return _cmd_report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
