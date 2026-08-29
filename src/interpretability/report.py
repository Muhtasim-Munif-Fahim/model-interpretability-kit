"""Markdown report rendering for interpretability results."""

import numpy as np

__all__ = ["render_report"]

DEFAULT_CAVEATS = [
    "Permutation importance measures the drop in score under random column "
    "shuffling; correlated features receive diluted or inflated values "
    "depending on how strongly they share signal, and single shuffles are "
    "noisy, so repeated shuffles and a seeded run are recommended.",
    "Partial dependence averages the model over all rows at each grid value, "
    "which implicitly assumes the varied feature is independent of the "
    "others; in low-density regions of the data the average extrapolates "
    "outside the training distribution.",
    "LIME-style local explanations are linear approximations around a single "
    "instance: their weights are only meaningful within the neighborhood "
    "defined by the sampling width and kernel, and different widths can "
    "produce different stories.",
    "Tree SHAP attributions here are interventional: unfixed features are "
    "marginalized over the supplied background data assuming independence, "
    "which differs from path-dependent variants for correlated features.",
]


def _fmt(value, ndigits=4):
    return "%.*f" % (ndigits, value)


def _trend(values):
    values = np.asarray(values, dtype=float)
    span = float(np.max(values) - np.min(values))
    if span < 1e-8:
        return "flat"
    if values[-1] > values[0]:
        return "increasing"
    if values[-1] < values[0]:
        return "decreasing"
    return "flat"


def _importance_arrays(global_importance):
    """Normalize importance input to (mean, std, baseline)."""
    if isinstance(global_importance, dict):
        mean = np.asarray(global_importance.get("mean"), dtype=float)
        std = np.asarray(global_importance.get("std", np.zeros(mean.shape[0])), dtype=float)
        baseline = global_importance.get("baseline", None)
    else:
        mean = np.asarray(global_importance, dtype=float)
        std = np.zeros(mean.shape[0])
        baseline = None
    return mean, std, baseline


def _importance_table(mean, std, names):
    order = np.argsort(-mean)
    rows = ["| Feature | Mean importance | Std |", "| --- | ---: | ---: |"]
    for idx in order:
        rows.append("| %s | %s | %s |" % (names[idx], _fmt(mean[idx]), _fmt(std[idx])))
    return "\n".join(rows)


def render_report(
    global_importance,
    pdp_summaries,
    local_explanations,
    faithfulness,
    feature_names=None,
    title="Model Interpretability Report",
    caveats=None,
):
    """Render a markdown report from interpretation results.

    Parameters
    ----------
    global_importance : dict or ndarray
        ``{"mean": array, "std": array, "baseline": float}`` as returned by
        ``importance.permutation_importance``, or a plain mean array.
    pdp_summaries : list of dict
        Each entry has ``{"feature": str, "grid": array, "values": array}``.
    local_explanations : list of dict
        ``lime_explain``-style dicts with ``"coefficients"``,
        ``"intercept"``, ``"prediction"``, ``"weighted_r2"`` and optional
        ``"feature_names"`` and ``"description"``.
    faithfulness : list of dict
        Entries with ``"description"``, ``"fidelity_r2"``, ``"overlap"`` and
        ``"jaccard"``.
    feature_names : list of str or None
        Names for all features; defaults to ``X0``, ``X1``, ...
    title : str
        Report heading.
    caveats : list of str or None
        Replaces the default caveats when given.

    Returns
    -------
    str
        Markdown document.
    """
    mean, std, baseline = _importance_arrays(global_importance)
    n_features = mean.shape[0]
    if feature_names is None:
        feature_names = ["X%d" % j for j in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("feature_names must match the number of features")

    lines = [title, "=" * len(title), ""]
    if baseline is not None:
        lines.append(
            "Baseline model score (higher is better): **%s** (over **%d** "
            "features).\n" % (_fmt(baseline), n_features)
        )

    lines.append("## Global feature importance")
    lines.append("")
    lines.append(_importance_table(mean, std, feature_names))
    lines.append("")

    lines.append("## Partial dependence summaries")
    lines.append("")
    if pdp_summaries:
        for summary in pdp_summaries:
            grid = np.asarray(summary["grid"], dtype=float)
            values = np.asarray(summary["values"], dtype=float)
            name = summary.get("feature", "feature")
            lines.append("### %s" % name)
            lines.append("")
            lines.append("| Property | Value |")
            lines.append("| --- | ---: |")
            lines.append("| Grid range | %s ... %s |" % (_fmt(grid[0]), _fmt(grid[-1])))
            lines.append("| Prediction at grid min | %s |" % _fmt(values[0]))
            lines.append("| Prediction at grid max | %s |" % _fmt(values[-1]))
            lines.append("| Overall range | %s ... %s |" % (_fmt(values.min()), _fmt(values.max())))
            lines.append("| Trend | %s |" % _trend(values))
            lines.append("")
    else:
        lines.append("_No partial dependence summaries provided._")
        lines.append("")

    lines.append("## Example local explanations")
    lines.append("")
    if local_explanations:
        for i, explanation in enumerate(local_explanations):
            description = explanation.get("description", "Instance %d" % (i + 1))
            names = explanation.get(
                "feature_names",
                ["X%d" % j for j in range(explanation["coefficients"].shape[0])],
            )
            coefficients = np.asarray(explanation["coefficients"], dtype=float)
            lines.append("### %s" % description)
            lines.append("")
            lines.append(
                "Prediction: **%s** | Surrogate fidelity (weighted R2): **%s**"
                % (_fmt(explanation["prediction"]), _fmt(explanation["weighted_r2"]))
            )
            lines.append("")
            lines.append("| Feature | Local weight |")
            lines.append("| --- | ---: |")
            for j in range(coefficients.shape[0]):
                lines.append("| %s | %s |" % (names[j], _fmt(coefficients[j])))
            lines.append("")
    else:
        lines.append("_No local explanations provided._")
        lines.append("")

    lines.append("## Faithfulness of local explanations")
    lines.append("")
    if faithfulness:
        lines.append("| Instance | Fidelity R2 | Top-3 overlap | Jaccard |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in faithfulness:
            lines.append(
                "| %s | %s | %d | %s |"
                % (
                    row["description"],
                    _fmt(row["fidelity_r2"]),
                    row["overlap"],
                    _fmt(row["jaccard"]),
                )
            )
        lines.append("")
    else:
        lines.append("_No faithfulness results provided._")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for caveat in caveats if caveats is not None else DEFAULT_CAVEATS:
        lines.append("- %s" % caveat)
    lines.append("")
    return "\n".join(lines)
