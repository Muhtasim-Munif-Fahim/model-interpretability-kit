# model-interpretability-kit

Model-agnostic machine learning interpretability in pure Python and NumPy.

This kit explains predictions of any model that can be wrapped as a plain
`predict(X) -> y` callable, without a scikit-learn dependency. It ships a
small hand-written regression tree as a transparent demo model whose ground
truth is known, so every explanation can be checked against reality.

## Methods

| Method | What it measures | Output | Reference |
| --- | --- | --- | --- |
| Permutation importance | Drop in model score when a feature's column is shuffled | mean ± std importance per feature | Fisher, Rudin & Dominici, "All Models are Wrong, but Many Are Useful" (JMLR, 2019); Breiman, "Random Forests" (2001) |
| Drop-column importance | Drop in score when a feature is removed and the model is refit | importance per feature | Same class of variable-importance measures |
| Partial dependence (1-D / 2-D) | Average prediction as one or two features vary over a grid; 1-D can include a normal-approximation confidence band for that mean | curve / surface arrays | Friedman, "Greedy Function Approximation" (Annals of Statistics, 2001) |
| Accumulated local effects (1-D / 2-D) | Accumulated local prediction change as one feature (or a pair) moves across quantile bins, centered to mean zero; 2-D is the pure interaction after main effects are removed | curve / surface arrays | Apley & Zhu, "Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models" (JASA, 2020) |
| ICE / centered ICE | Per-row predictions as one feature varies; optional centering at the first grid point so every curve starts at 0 | curve per row | Goldstein et al., "Peeking Inside the Black Box" (JCGS, 2015) |
| LIME-style surrogate | Locally weighted linear fit around an instance | feature weights + intercept + local R2 | Ribeiro, Singh & Guestrin, "Why Should I Trust You?" (KDD, 2016) |
| Interventional tree SHAP | Exact Shapley decomposition for one regression tree | per-feature attributions summing to `prediction - baseline` | Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS, 2017); Lundberg et al., "From Local Explanations to Global Understanding" (Nature MI, 2020) |
| Faithfulness checks | Local surrogate R2; top-feature overlap between local and global attributions | per-instance numbers | Molnar, "Interpretable Machine Learning" (2022) |

The broader conceptual framework and caveats are described in Christoph
Molnar's book, *Interpretable Machine Learning*,
https://christophm.github.io/interpretable-ml-book/.

## Install

Requires Python 3.9+ and NumPy.

```bash
pip install -r requirements.txt
pip install -e .            # optional, installs the package for import
```

The tests additionally use pytest:

```bash
pip install pytest
python -m pytest tests -q
```

## Quickstart

```python
import numpy as np
from interpretability.ale import accumulated_local_effects, accumulated_local_effects_2d
from interpretability.demo_model import fit_decision_tree, make_synthetic_data
from interpretability.importance import permutation_importance
from interpretability.local import lime_explain
from interpretability.partial_dependence import ice_curves, partial_dependence

X, y, names = make_synthetic_data(n_samples=300, seed=42)
model = fit_decision_tree(X[:200], y[:200], max_depth=6, min_samples_leaf=5)

# Global: which features matter?
imp = permutation_importance(model.predict, X[200:], y[200:], n_repeats=10, seed=42)
print(imp["mean"], imp["std"])

# Global: how does the prediction respond to feature 0?
pdp = partial_dependence(model.predict, X[200:], 0, grid_points=20, conf_level=0.95)
print(pdp["grid"], pdp["values"], pdp["lower"], pdp["upper"])

# Global: same question without the PDP independence assumption
ale = accumulated_local_effects(model.predict, X[200:], 0, grid_points=20)
print(ale["grid"], ale["values"])

# Global: extra interaction of features 0 and 1 after main effects are removed
ale2 = accumulated_local_effects_2d(model.predict, X[200:], (0, 1), grid_points=10)
print(ale2["grid0"], ale2["grid1"], ale2["values"].shape)

# Per row: does the average hide heterogeneous responses?
ice = ice_curves(model.predict, X[200:], 0, grid_points=20, rows=range(8), centered=True)
print(ice["grid"], ice["curves"].shape)

# Local: why did row 0 get its prediction?
exp = lime_explain(model.predict, X[200], X[200:], n_samples=400, seed=42, feature_names=names)
print(exp["coefficients"], exp["intercept"], exp["weighted_r2"])
```

### Command line

```bash
python -m interpretability.cli --seed 42 importance --n-repeats 10
python -m interpretability.cli --seed 42 pdp --features 0,2
python -m interpretability.cli --seed 42 pdp --features 0 --conf-level 0.95
python -m interpretability.cli --seed 42 ice --features 0 --rows 0,1,2 --centered
python -m interpretability.cli --seed 42 ale --features 0
python -m interpretability.cli --seed 42 ale --features 0,2
python -m interpretability.cli --seed 42 explain --rows 0,1
python -m interpretability.cli --seed 42 report --out examples/output/demo_report.md
```

Every subcommand accepts `--data path/to/file.csv` (comma-separated, target
as the last column) instead of the built-in synthetic data. All randomness is
seeded, so runs are reproducible.

### Demo

```bash
python examples/run_demo.py
```

fits the transparent tree, computes all explanation types, prints a summary,
and writes `examples/output/demo_report.md`.

## How the demo model is grounded

`make_synthetic_data` generates features `X0..X4` uniform in `[0, 1]` and a
target with a known additive structure:

```
y = 4.0 * X0 + 3.0 * X1 + 2.0 * (X2 > 0.5) + 1.5 * X3 + noise
```

`X4` does not appear in the formula, so a correct explanation must rank it
last and the additive coefficients give a reference for what LIME weights
should look like. The demo tree is a hand-written greedy regression tree
(variance-reduction splits) that stays small enough to inspect directly and
to support exact tree SHAP attribution.

## Caveats

- **Importance variance.** Permutation importance uses random shuffles; with
  few repeats the estimates are noisy, and for correlated features the
  reported drop is diluted or inflated depending on how the signal is shared.
  Drop-column importance refits the model, which is more faithful but much
  more expensive.
- **Partial dependence independence assumption.** PDP averages over all rows
  at each grid value, implicitly assuming the varied feature is independent
  of the others. In low-density regions of the data the average extrapolates
  outside the training distribution, and ICE curves help reveal whether the
  averaged curve hides heterogeneous behavior. Optional 1-D confidence bands
  are a normal approximation for sampling variability of that row average
  (`mean ± z * std / sqrt(n)`); they are not a statement about parameter
  uncertainty in the fitted model. Centered ICE subtracts each curve's value
  at the first grid point so differences in slope are easier to compare.
- **ALE vs partial dependence.** ALE estimates a feature's effect from finite
  differences inside quantile bins, so it does not require independence from
  the other features. The curve is centered to have mean zero over the data
  and should not be read as the raw predicted value at a grid point (unlike
  PDP). Two-feature ALE is a second-order interaction surface: main effects
  of each feature are subtracted, so an additive model is near zero even when
  both features matter individually. Coarse bins can miss sharp jumps; empty
  bins contribute no local effect.
- **LIME sampling sensitivity.** Local surrogate weights depend on the
  neighborhood width, the kernel bandwidth, and the number of samples. The
  weights are only meaningful locally, and the surrogate's own R2 (reported
  by the kit) says how much of the model is actually linear near the
  instance.
- **Tree SHAP variant.** The tree attributions are interventional: unfixed
  features are marginalized over the supplied background assuming
  independence, which differs from path-dependent variants when features are
  correlated.

## License

MIT.
