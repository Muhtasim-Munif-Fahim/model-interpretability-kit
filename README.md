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
| Drop-column importance | Drop in score when a feature is removed and the model is refit on the same rows | importance per feature | Same class of variable-importance measures |
| Leave-one-covariate-out (LOCO) | Increase in held-out loss when a covariate is left out of the fit | mean ± std importance per feature | Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, "Distribution-Free Predictive Inference for Regression" (JASA, 2018) |
| Conditional permutation importance | Drop in score when a feature is replaced by a draw from (an approximation of) its conditional distribution given the other features, either by permuting linear-regression residuals or by permuting within joint quantile strata of the most-correlated features | mean ± std importance per feature, plus the conditioning sets | Strobl, Boulesteix, Kneib, Augustin & Zeileis, "Conditional variable importance for random forests" (BMC Bioinformatics, 2008); Watson & Wright, "Testing conditional independence in supervised learning algorithms" (Machine Learning, 2021); Molnar et al., "Model-agnostic feature importance and effects with dependent features: a conditional subgroup approach" (DMKD, 2023) |
| SAGE | Shapley values of the model's *loss*: each feature's share of the drop from the constant-prediction loss to the model's loss, with absent features marginalized over a background set (permutation estimator) | per-feature values summing exactly to `loss(empty) - loss(model)`, plus Monte Carlo standard errors | Covert, Lundberg & Lee, "Understanding Global Feature Importance by Additive Importance Measures" (NeurIPS, 2020) |
| Partial dependence (1-D / 2-D) | Average prediction as one or two features vary over a grid; 1-D can include a normal-approximation confidence band for that mean | curve / surface arrays | Friedman, "Greedy Function Approximation" (Annals of Statistics, 2001) |
| Accumulated local effects (1-D / 2-D) | Accumulated local prediction change as one feature (or a pair) moves across quantile bins, centered to mean zero; 2-D is the pure interaction after main effects are removed | curve / surface arrays | Apley & Zhu, "Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models" (JASA, 2020) |
| Friedman H-statistic | Strength of pairwise interaction: share of joint PD variation not explained by the sum of main-effect PDs | scalar in [0, 1] | Friedman & Popescu, "Predictive Learning via Rule Ensembles" (Annals of Applied Statistics, 2008) |
| ICE / centered ICE | Per-row predictions as one feature varies; optional centering at the first grid point so every curve starts at 0 | curve per row | Goldstein et al., "Peeking Inside the Black Box" (JCGS, 2015) |
| Anchors | High-precision sufficient rule (feature-bin predicates) for an instance | predicates + precision + coverage | Ribeiro, Singh & Guestrin, "Anchors: High-Precision Model-Agnostic Explanations" (AAAI, 2018) |
| LIME-style surrogate | Locally weighted linear fit around an instance | feature weights + intercept + local R2 | Ribeiro, Singh & Guestrin, "Why Should I Trust You?" (KDD, 2016) |
| Interventional tree SHAP | Exact Shapley decomposition for one regression tree | per-feature attributions summing to `prediction - baseline` | Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS, 2017); Lundberg et al., "From Local Explanations to Global Understanding" (Nature MI, 2020) |
| Kernel SHAP-lite | Model-agnostic Shapley values via coalition sampling + Shapley-kernel WLS | per-feature attributions summing to `prediction - baseline` | Lundberg & Lee (NeurIPS, 2017) |
| Exact Shapley / Banzhaf / Shapley interaction values | Interventional value function `v(S) = mean_z f(x_S, z_~S)` evaluated on all `2^M` coalitions (M ≤ 12). Gives exact Shapley values, Banzhaf values, and the `M x M` Shapley interaction matrix (main effects on the diagonal, pairwise interactions split off-diagonal) | per-feature values summing exactly to `prediction - baseline`, an interaction matrix that sums to the same total, and the Banzhaf efficiency gap | Shapley (1953); Banzhaf (1965); Grabisch & Roubens, "An axiomatic approach to the concept of interaction among players in cooperative games" (IJGT, 1999); Lundberg et al., "From Local Explanations to Global Understanding with Explainable AI for Trees" (Nature MI, 2020) |
| Permutation-sampling Shapley | Monte Carlo interventional Shapley values: average marginal contributions over random feature orderings, with absent features taken from real background rows; optional antithetic (reversed) permutations | per-feature attributions summing exactly to `prediction - baseline`, plus Monte Carlo standard errors | Štrumbelj & Kononenko, "Explaining prediction models and individual predictions with feature contributions" (KAIS, 2014); Mitchell et al., "Sampling Permutations for Shapley Value Estimation" (JMLR, 2022) |
| Integrated Gradients | Path-integral of gradients from baseline→input (Riemann / finite-diff) | per-feature attributions summing to `f(x)-f(baseline)` | Sundararajan, Taly & Yan (ICML, 2017) |
| SmoothGrad | Average finite-difference gradients over Gaussian-noised copies of the input | per-feature smoothed saliency | Smilkov et al., "SmoothGrad: removing noise by adding noise" (2017) |
| Sobol first-order | Fraction of output variance explained by each feature alone (Saltelli / pick-freeze) | `S1` per feature in `[0, 1]` | Sobol (1993); Saltelli et al., "Global Sensitivity Analysis" (2008) |
| Sobol total-order | Fraction of variance from a feature and all its interactions (`1 - Var(E[Y|X_~i])/Var(Y)`) | `ST` (and optional `S1`) per feature in `[0, 1]` | Sobol (1993); Jansen / Saltelli pick-freeze |
| Growing Spheres counterfactuals | Smallest (standardized L2) change to an instance that moves its prediction into a target interval / class, found by sampling spherical layers outward, then sparsified by resetting features | counterfactual point, changed features, deltas, distance, evaluations | Laugel, Lesot, Marsala, Renard & Detyniecki, "Inverse Classification for Comparison-based Interpretability in Machine Learning" (2018); Wachter, Mittelstadt & Russell, "Counterfactual Explanations without Opening the Black Box" (Harvard JOLT, 2018) |
| MMD-critic prototypes & criticisms | Greedy rows whose empirical distribution minimizes RBF-kernel MMD to the data (prototypes), plus rows with the largest witness-function magnitude under a log-det diversity term (criticisms); optional per-class prototype allocation | row indices, witness values, MMD^2, coverage per prototype | Kim, Khanna & Koyejo, "Examples are not Enough, Learn to Criticize! Criticism for Interpretability" (NeurIPS, 2016) |
| Faithfulness checks | Local surrogate R2; top-feature overlap between local and global attributions | per-instance numbers | Molnar, "Interpretable Machine Learning" (2022) |

The broader conceptual framework and caveats are described in Christoph
Molnar's book, *Interpretable Machine Learning*,
https://christophm.github.io/interpretable-ml-book/.

### Why LOCO is here

Permutation importance is already part of this kit. `permutation_importance`
shuffles one column at a time, measures the drop in a higher-is-better score
(R² by default, or a custom scorer such as accuracy), and reports the mean
and standard deviation across optional `n_repeats`. This change does not add
a second permutation implementation. It adds leave-one-covariate-out (LOCO)
importance instead: refit without each covariate and record how much
held-out loss increases. Drop-column importance is the related in-sample
refit; LOCO is the out-of-sample version, with repeats and a standard
deviation across those repeats. The default loss is mean absolute error.
`mean_squared_error` is a drop-in replacement, and `zero_one_loss` makes
the importance the drop in classification accuracy.


### Friedman H-statistic

`friedman_h_statistic` (alias `h_statistic`) measures pairwise interaction
strength from partial dependence. For features `j` and `k`:

```text
H^2_jk = sum_i [PD_jk(x_i^j, x_i^k) - PD_j(x_i^j) - PD_k(x_i^k)]^2
       / sum_i [PD_jk(x_i^j, x_i^k)]^2
```

with each PD centered to mean zero over the evaluation rows. `H` near 0 means
the joint effect is additive; near 1 means almost all of the joint PD is
interaction. This sits alongside PDP, ALE, and LOCO as a global diagnostic.

```python
from interpretability import friedman_h_statistic
from interpretability.demo_model import fit_decision_tree, make_synthetic_data

X, y, names = make_synthetic_data(n_samples=300, seed=42)
model = fit_decision_tree(X, y, max_depth=6, min_samples_leaf=5)
h = friedman_h_statistic(model.predict, X, (0, 1), grid_points=15)
print(names[0], names[1], h["h"], h["h_squared"])
```

```bash
python -m interpretability.cli h-statistic --features 0,1 --grid-points 15
```

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
from interpretability.importance import loco_importance, permutation_importance
from interpretability.local import lime_explain
from interpretability.interaction import friedman_h_statistic
from interpretability.partial_dependence import ice_curves, partial_dependence
from interpretability.local import integrated_gradients, kernel_shap, sampling_shapley, smoothgrad

X, y, names = make_synthetic_data(n_samples=300, seed=42)
model = fit_decision_tree(X[:200], y[:200], max_depth=6, min_samples_leaf=5)

# Global: which features matter?
imp = permutation_importance(model.predict, X[200:], y[200:], n_repeats=10, seed=42)
print(imp["mean"], imp["std"])

# Global: same question by refitting without each covariate (held-out loss)
def fit_predict(X_fit, y_fit, X_eval):
    return fit_decision_tree(X_fit, y_fit, max_depth=6, min_samples_leaf=5).predict(X_eval)

loco = loco_importance(fit_predict, X[:200], y[:200], X_test=X[200:], y_test=y[200:])
print(loco["mean"], loco["std"], loco["baseline"])  # baseline is MAE, lower is better

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

ks = kernel_shap(model.predict, X[200], X[200:], n_samples=256, seed=0)
print(ks["values"], ks["baseline"], ks["prediction"])

# Local: permutation-sampling Shapley (interventional, exact efficiency, with SEs)
ss = sampling_shapley(model.predict, X[200], X[200:], n_permutations=200, seed=0)
print(ss["values"], ss["std_error"], ss["values"].sum(), ss["prediction"] - ss["baseline"])

# Local: Integrated Gradients from mean baseline
ig = integrated_gradients(model.predict, X[200], baseline=X[200:].mean(axis=0), n_steps=32)
print(ig["values"], ig["values"].sum(), ig["prediction"] - ig["baseline"])

# Local: SmoothGrad (denoised saliency)
sg = smoothgrad(model.predict, X[200], n_samples=50, noise_sigma=0.1, seed=0)
print(sg["values"], sg["n_samples"], sg["noise_sigma"])


# Pairwise interaction strength (Friedman H)
h = friedman_h_statistic(model.predict, X[200:], (0, 1), grid_points=15)
print(h["h"], h["h_squared"])

# Local: why did row 0 get its prediction?
exp = lime_explain(model.predict, X[200], X[200:], n_samples=400, seed=42, feature_names=names)
print(exp["coefficients"], exp["intercept"], exp["weighted_r2"])
```

### Command line

```bash
python -m interpretability.cli --seed 42 importance --n-repeats 10
python -m interpretability.cli --seed 42 loco --n-repeats 5 --test-size 0.25
python -m interpretability.cli --seed 42 conditional-importance --method residual --n-repeats 10
python -m interpretability.cli --seed 42 conditional-importance --method grid --n-bins 4 --n-conditioning 2
python -m interpretability.cli --seed 42 pdp --features 0,2
python -m interpretability.cli --seed 42 pdp --features 0 --conf-level 0.95
python -m interpretability.cli --seed 42 ice --features 0 --rows 0,1,2 --centered
python -m interpretability.cli --seed 42 kernel-shap --rows 0,1 --n-samples 200
python -m interpretability.cli --seed 42 sampling-shapley --rows 0,1 --n-permutations 100
python -m interpretability.cli --seed 42 integrated-gradients --rows 0,1 --n-steps 32
python -m interpretability.cli --seed 42 smoothgrad --rows 0,1 --n-samples 50 --noise-sigma 0.1
python -m interpretability.cli --seed 42 sage --n-permutations 256 --n-background 64
python -m interpretability.cli --seed 42 counterfactual --row 0 --target-max 4.0 --immutable 4
python -m interpretability.cli --seed 42 prototypes --n-prototypes 5 --n-criticisms 3
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
  Drop-column importance refits the model on the same rows, which is more
  faithful to a removed column but much more expensive, and it can look
  optimistic because the score is not held out. LOCO also refits once per
  feature (and once per repeat), then measures the increase in loss on rows
  that were not used for that fit. A single repeat has a standard deviation
  of zero; raise `n_repeats` to see split-to-split variability. Correlated
  covariates can look unimportant because the features that remain still
  carry the signal, and a noisy feature can receive a small negative value
  when leaving it out slightly improves the held-out loss. The LOCO
  `baseline` is that held-out loss (lower is better), not the higher-is-better
  score used by permutation importance.
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


## Anchors

`anchor_explain` finds a compact set of feature-bin predicates that keep
the model prediction close to `f(x)` with high precision on perturbed
neighbours (Ribeiro, Singh & Guestrin, AAAI 2018). Continuous features are
quantile-binned from the background sample.

```python
from interpretability.local import anchor_explain
from interpretability.demo_model import fit_decision_tree, make_synthetic_data

X, y, names = make_synthetic_data(n_samples=400, seed=0)
model = fit_decision_tree(X, y, max_depth=4)
result = anchor_explain(model.predict, X[0], X, n_samples=800, seed=0)
print(result["precision"], result["coverage"], result["predicates"])
```


## SAGE global importance

`sage_values` estimates SAGE values (Covert et al., 2020). It is a global,
loss-based importance with a Shapley guarantee. Each draw samples an
evaluation row and a feature ordering, then adds the features one at a time.
Each feature is credited with the drop in loss it causes. Absent features
are averaged over the whole background set.

```python
from interpretability import sage_values

res = sage_values(model.predict, X, y, loss="mse", n_permutations=512, n_background=64, seed=0)
res["values"], res["std_error"]       # loss units; positive = feature lowers the loss
res["total"]                          # == res["loss_empty"] - res["loss_full"]
```

`loss` can be `"mse"`, `"mae"`, `"log_loss"` (for a positive-class
probability), or a callable that returns per-row losses.
`sage_value_function(predict, X, y, subset)` returns the exact `v(S)` for
small-`d` checks. Compared with permutation importance, SAGE averages over
feature orderings, so correlated features and interactions share credit
instead of each being scored in isolation. A pure `x0 * x1` interaction is
split evenly between the two features.

## Counterfactual explanations (Growing Spheres)

`growing_spheres_counterfactual` answers "what is the smallest change that
would move this prediction into the target?" for any black-box `predict`.
It samples uniformly in growing spherical layers around the instance (in
standard-deviation units of the reference data) until a layer contains a
point meeting the target, takes the closest such point, then resets features
back to their original values whenever the target still holds, so the
explanation changes as few features as possible.

```python
from interpretability import growing_spheres_counterfactual

res = growing_spheres_counterfactual(
    model.predict, X[0], X,
    target_range=(None, 4.0),   # or target_class=1, or target=callable
    immutable=[4],              # never change feature 4
    bounds="reference",         # stay inside the observed feature ranges
    seed=0,
)
res["changed_feature_names"], res["delta"], res["prediction_counterfactual"]
```

`found=False` is returned when no counterfactual exists within `max_layers`
(for example when bounds make the target unreachable).

## Prototypes and criticisms (MMD-critic)

`mmd_critic` summarizes a dataset with real rows. Prototypes are picked
greedily so their distribution is close to the data's in maximum mean
discrepancy under an RBF kernel (the objective is monotone submodular, so
greedy is near-optimal). Criticisms are rows the prototypes explain badly:
the largest values of the witness function
`w(x) = mean_i k(x, x_i) - mean_j k(x, z_j)`, chosen with a log-determinant
term so they are spread out. Positive witness marks under-represented
regions, negative marks over-represented ones.

```python
from interpretability import mmd_critic

res = mmd_critic(X, n_prototypes=10, n_criticisms=5)   # median-heuristic gamma
X[res["prototypes"]], X[res["criticisms"]], res["mmd2"], res["coverage"]
res = mmd_critic(X, n_prototypes=10, labels=y_class)   # prototypes per class
```

Columns are standardized before computing kernels by default.

## Conditional permutation importance

Marginal permutation importance shuffles a column independently of the
others. With correlated features this asks the model about rows that never
occur, and a feature that is redundant given the others can still look
important. `conditional_permutation_importance` keeps each feature's
dependence on the rest, so the score drop measures what `x_j` adds beyond
the other features:

- `method="residual"` (default) regresses `x_j` on the other columns and
  permutes only the residuals, `x_j* = x̂_j + perm(r_j)`. This is exact for
  jointly Gaussian features. For `f = Σx` under MSE it recovers the
  closed-form importance `2·Var(x_j | x_{-j})`.
- `method="grid"` permutes `x_j` within joint quantile strata (`n_bins` per
  feature) of its `n_conditioning` most-correlated columns, or of explicit
  `conditioning` columns.

With one feature, or an empty conditioning set, both methods reduce to
`permutation_importance`; with the same seed they make the same draws.

```python
import numpy as np
from interpretability import conditional_permutation_importance
from interpretability.importance import permutation_importance

rng = np.random.default_rng(0)
x0 = rng.normal(size=2000)
X = np.column_stack([x0, x0 + 0.1 * rng.normal(size=2000), rng.normal(size=2000)])
y = X.sum(axis=1)
model = lambda Z: Z.sum(axis=1)

print(permutation_importance(model, X, y, seed=0)["mean"])              # x0, x1 look big
print(conditional_permutation_importance(model, X, y, seed=0)["mean"])  # x0, x1 near 0
```

## Exact Shapley values and Shapley interaction values

With a handful of features you don't need to sample. `exact_shapley`
evaluates the interventional value function `v(S) = mean_z f(x_S, z_~S)` on
**every** coalition, which takes `2^M * n_background` model rows in batched
`predict` calls. That gives the exact values that `kernel_shap` and
`sampling_shapley` estimate. It works for any black-box model and matches
`tree_shap_values` for trees when the background is a single reference row.

```python
from interpretability import banzhaf_values, exact_shapley, shapley_interaction_values

res = exact_shapley(model.predict, X[0], X[:50])
res["values"].sum() == res["prediction"] - res["baseline"]   # exact efficiency

inter = shapley_interaction_values(model.predict, X[0], X[:50], feature_names=names)
inter["interactions"]        # M x M; rows sum to the Shapley values
inter["main_effects"]        # diagonal: effect of each feature without its interactions
inter["top_interactions"]    # [(a, b, Phi_ab + Phi_ba), ...] by magnitude

banzhaf_values(model.predict, X[0], X[:50])["efficiency_gap"]
```

Off-diagonal entries split each pairwise interaction index equally between
the two features, as in SHAP interaction values. For `f = x0 * x1` with one
reference row `b`, the interaction is exactly `(x0 - b0)(x1 - b1)`. Banzhaf
values weight every coalition equally rather than by size. They equal the
Shapley values when the model has at most pairwise interactions, and stop
summing to `f(x) - baseline` once higher-order interactions are present.
Enumeration is capped at 12 features by default (`max_features`, hard cap 16).

```bash
python -m interpretability.cli exact-shapley --rows 0,1 --interactions --banzhaf
```
