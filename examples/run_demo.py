"""End-to-end demo of the interpretability toolkit.

Builds the synthetic regression problem, fits the transparent decision tree,
then computes permutation importance, 1-D partial dependence and ALE for
three features, ICE curves for a few rows, LIME-style local explanations (plus
exact interventional tree SHAP values) for two evaluation rows, a
faithfulness summary, and writes ``output/demo_report.md``.

Run from the repository root::

    python examples/run_demo.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from interpretability.ale import accumulated_local_effects
from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.evaluate import top_feature_overlap
from interpretability.importance import permutation_importance
from interpretability.local import lime_explain, tree_shap_values
from interpretability.partial_dependence import ice_curves, partial_dependence
from interpretability.report import render_report

SEED = 42
N_SAMPLES = 300
N_TRAIN = 200


def main():
    X, y, feature_names = make_synthetic_data(n_samples=N_SAMPLES, seed=SEED)
    X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
    X_eval, y_eval = X[N_TRAIN:], y[N_TRAIN:]

    model = fit_decision_tree(X_train, y_train, max_depth=6, min_samples_leaf=5)

    importance = permutation_importance(
        model.predict, X_eval, y_eval, n_repeats=10, seed=SEED
    )
    order = np.argsort(-importance["mean"])
    print("=== Permutation importance (R2 drop on the evaluation set) ===")
    for j in order:
        print(
            "  %-6s %8.4f +/- %.4f" % (feature_names[j], importance["mean"][j], importance["std"][j])
        )
    print("  baseline R2 = %.4f" % importance["baseline"])
    top3 = [feature_names[j] for j in order[:3]]
    print("Top-3 features: %s\n" % ", ".join(top3))

    print("=== Partial dependence ===")
    pdp_summaries = []
    for f in (0, 1, 2):
        pdp = partial_dependence(model.predict, X_eval, f, grid_points=15)
        pdp_summaries.append(
            {"feature": feature_names[f], "grid": pdp["grid"], "values": pdp["values"]}
        )
        print(
            "  %s: grid [%.3f, %.3f] -> predictions [%.3f, %.3f]"
            % (
                feature_names[f],
                pdp["grid"][0],
                pdp["grid"][-1],
                pdp["values"][0],
                pdp["values"][-1],
            )
        )

    print("=== Accumulated local effects ===")
    for f in (0, 1, 2):
        ale = accumulated_local_effects(model.predict, X_eval, f, grid_points=15)
        print(
            "  %s: grid [%.3f, %.3f] -> centered effects [%.3f, %.3f]"
            % (
                feature_names[f],
                ale["grid"][0],
                ale["grid"][-1],
                ale["values"][0],
                ale["values"][-1],
            )
        )

    ice = ice_curves(model.predict, X_eval, 0, grid_points=15, rows=[0, 1, 2, 3, 4])
    print(
        "  ICE curves for %s over %d rows, %d grid points each\n"
        % (feature_names[ice["feature"]], len(ice["rows"]), ice["curves"].shape[1])
    )

    print("=== Local explanations (LIME-style + interventional tree SHAP) ===")
    local_explanations = []
    faithfulness = []
    for i in (0, 1):
        explanation = lime_explain(
            model.predict,
            X_eval[i],
            X_eval,
            n_samples=600,
            sigma_scale=0.25,
            seed=SEED,
            feature_names=feature_names,
        )
        explanation["description"] = "Row %d" % i
        shap = tree_shap_values(
            model.root_, X_eval[i], X_eval, feature_count=X_eval.shape[1]
        )
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
        weights = "  ".join(
            "%s=%+.3f" % (feature_names[j], explanation["coefficients"][j])
            for j in range(len(feature_names))
        )
        print(
            "  Row %d: prediction %.4f, surrogate R2 %.3f" % (i, explanation["prediction"], explanation["weighted_r2"])
        )
        print("    LIME weights: %s" % weights)
        print(
            "    Tree SHAP: %s (baseline %.4f, sum %.4f)"
            % (
                "  ".join("%s=%+.3f" % (feature_names[j], shap["values"][j]) for j in range(len(feature_names))),
                shap["baseline"],
                shap["values"].sum(),
            )
        )

    mean_fidelity = float(np.mean([row["fidelity_r2"] for row in faithfulness]))
    print("\n=== Faithfulness summary ===")
    for row in faithfulness:
        print(
            "  %s: fidelity R2 %.3f, top-3 overlap %d/%d, Jaccard %.2f"
            % (
                row["description"],
                row["fidelity_r2"],
                row["overlap"],
                min(3, X_eval.shape[1]),
                row["jaccard"],
            )
        )
    print("  Mean local-explanation fidelity R2: %.4f" % mean_fidelity)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_report.md")
    report = render_report(
        importance,
        pdp_summaries,
        local_explanations,
        faithfulness,
        feature_names=feature_names,
        title="Demo interpretability report",
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print("\nWrote %s" % out_path)


if __name__ == "__main__":
    main()
