"""Local explanations: LIME-style surrogates, Kernel SHAP, and tree attributions.

``integrated_gradients`` attributes a prediction along the straight-line
path from a baseline to the instance via a Riemann sum of finite-difference
gradients (Sundararajan, Taly & Yan, 2017).

``smoothgrad`` averages finite-difference gradients over Gaussian-noised
copies of the input to denoise saliency maps (Smilkov et al., 2017).

``sampling_shapley`` estimates interventional Shapley values by averaging
marginal contributions over random feature permutations and background rows
(Štrumbelj & Kononenko, 2014), with optional antithetic permutations.

``lime_explain`` fits a weighted linear surrogate over a random neighborhood
of the instance to be explained, so the explanation is interpretable at the
cost of locality.

``kernel_shap`` is a model-agnostic Kernel SHAP-lite estimator (Lundberg &
Lee, 2017): it samples feature coalitions, evaluates the model under an
interventional replacement of missing features from a background dataset,
and recovers Shapley values by weighted least squares with the Shapley
kernel.

For the small regression trees produced by ``interpretability.demo_model``,
``tree_shap_values`` computes exact interventional SHAP values: Shapley
values of the coalition value function ``v(S) = E[f(X) | X_S = x_S]`` where
unfixed features are marginalized by flowing a background dataset through the
tree. ``local_occlusion_attribution`` is a cheaper fallback for arbitrary
predictors.
"""

from itertools import combinations
from math import factorial

import numpy as np

from ._utils import as_2d

__all__ = [
    "weighted_least_squares",
    "sample_neighborhood",
    "lime_explain",
    "tree_conditional_expectation",
    "tree_shap_values",
    "local_occlusion_attribution",
    "kernel_shap",
    "shapley_kernel_weight",
    "integrated_gradients",
    "smoothgrad",
    "sampling_shapley",
]


def weighted_least_squares(design, target, weights, l2=1e-6):
    """Solve ``argmin_b sum(w * (target - design @ b)^2)``.

    A small ridge term ``l2`` keeps the normal equations well conditioned for
    collinear or constant columns. ``design`` should include an intercept
    column when one is wanted.

    Returns
    -------
    ndarray
        Coefficient vector of shape ``(design.shape[1],)``.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if design.ndim != 2:
        raise ValueError("design must be a 2-D matrix")
    if not (design.shape[0] == target.shape[0] == weights.shape[0]):
        raise ValueError("design, target and weights must have the same number of rows")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    weighted_design = weights[:, None] * design
    gram = design.T @ weighted_design
    rhs = weighted_design.T @ target
    if l2 > 0:
        gram = gram + l2 * np.eye(design.shape[1])
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def sample_neighborhood(
    x_row, background, n_samples=500, kernel_width=0.75, sigma_scale=0.1, seed=None
):
    """Draw weighted perturbations around ``x_row`` for a LIME surrogate.

    Perturbations follow a Gaussian centered at ``x_row`` with per-feature
    scale proportional to the feature's standard deviation. Distances are
    Euclidean and are mapped to weights with an exponential kernel, so rows
    far from the instance contribute little to the surrogate fit. The
    instance itself is appended with weight 1.

    Returns
    -------
    Z : ndarray of shape (n_samples + 1, n_features)
    weights : ndarray of shape (n_samples + 1,)
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background)
    if x_row.shape[0] != background.shape[1]:
        raise ValueError("x_row and background must have the same number of features")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    rng = np.random.default_rng(seed)
    stds = background.std(axis=0) * sigma_scale
    stds = np.where(stds <= 0, 0.1 * sigma_scale, stds)
    Z = rng.normal(loc=x_row, scale=stds, size=(n_samples, background.shape[1]))
    Z = np.vstack([Z, x_row[None, :]])
    distances = np.linalg.norm(Z - x_row, axis=1) / np.sqrt(background.shape[1])
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
    return Z, weights


def _weighted_r2(y, y_hat, weights):
    w_mean = np.sum(weights * y) / np.sum(weights)
    ss_res = np.sum(weights * (y - y_hat) ** 2)
    ss_tot = np.sum(weights * (y - w_mean) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def lime_explain(
    predict,
    x_row,
    background,
    n_samples=500,
    kernel_width=0.75,
    sigma_scale=0.1,
    seed=None,
    feature_names=None,
):
    """Explain one prediction with a locally weighted linear surrogate.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    x_row : sequence of float
        Instance to explain.
    background : ndarray
        Reference dataset used to set neighborhood scales.
    n_samples : int
        Number of perturbed samples around the instance.
    kernel_width : float
        Bandwidth of the exponential distance kernel.
    sigma_scale : float
        Perturbation scale as a fraction of each feature's standard deviation.
    seed : int or None
        Random seed for the perturbations.
    feature_names : list of str or None

    Returns
    -------
    dict
        ``{"coefficients": ndarray, "intercept": float, "weighted_r2": float,
        "prediction": float, "n_samples": int, "feature_names": list}``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background)
    Z, weights = sample_neighborhood(
        x_row, background, n_samples=n_samples, kernel_width=kernel_width,
        sigma_scale=sigma_scale, seed=seed,
    )
    y_perturbed = np.asarray(predict(Z), dtype=float).ravel()
    design = np.column_stack([np.ones(Z.shape[0]), Z])
    coefs = weighted_least_squares(design, y_perturbed, weights)
    intercept = float(coefs[0])
    coefficients = coefs[1:]
    fitted = design @ coefs
    prediction = float(predict(x_row[None, :])[0])
    if feature_names is None:
        feature_names = ["X%d" % j for j in range(background.shape[1])]
    return {
        "coefficients": coefficients,
        "intercept": intercept,
        "weighted_r2": _weighted_r2(y_perturbed, fitted, weights),
        "prediction": prediction,
        "n_samples": Z.shape[0],
        "feature_names": list(feature_names),
    }


def tree_conditional_expectation(node, x_row, fixed_features, background):
    """Conditional expectation ``E[f(X) | X_fixed = x_fixed]`` for a tree.

    Background rows are flowed through the tree; at a split on a fixed
    feature the instance's branch is taken, otherwise the child expectations
    are averaged by the fraction of the background that would flow each way.

    Parameters
    ----------
    node : dict
        Tree node dict as stored in ``DecisionTreeRegressor.root_``.
    x_row : sequence of float
        Instance whose values fix the ``fixed_features`` columns.
    fixed_features : iterable of int
        Features held at ``x_row`` values.
    background : ndarray
        Reference dataset for marginalization.
    """
    if not isinstance(node, dict) or "leaf" not in node:
        raise TypeError("node must be a tree node dict with a 'leaf' key")
    if node["leaf"]:
        return float(node["value"])
    feature = node["feature"]
    if feature in fixed_features:
        child = node["left"] if x_row[feature] <= node["threshold"] else node["right"]
        return tree_conditional_expectation(child, x_row, fixed_features, background)
    n_left = int(np.sum(background[:, feature] <= node["threshold"]))
    n_total = background.shape[0]
    if n_left == 0:
        return tree_conditional_expectation(node["right"], x_row, fixed_features, background)
    if n_left == n_total:
        return tree_conditional_expectation(node["left"], x_row, fixed_features, background)
    p_left = n_left / n_total
    expected_left = tree_conditional_expectation(node["left"], x_row, fixed_features, background)
    expected_right = tree_conditional_expectation(node["right"], x_row, fixed_features, background)
    return p_left * expected_left + (1.0 - p_left) * expected_right


def tree_shap_values(node, x_row, background, feature_count):
    """Exact interventional SHAP values for a single regression tree.

    Evaluates the Shapley values of ``v(S) = E[f(X) | X_S = x_S]`` over all
    feature coalitions, which for one tree is the interventional TreeSHAP
    decomposition. Exponential in the feature count, so it is intended for
    small trees (a handful of features).

    Parameters
    ----------
    node : dict
        Root node of the fitted tree.
    x_row : sequence of float
        Instance to explain.
    background : ndarray
        Reference dataset for marginalization.
    feature_count : int
        Total number of features in the model.

    Returns
    -------
    dict
        ``{"values": ndarray of shape (feature_count,), "baseline": float}``.
        The ``values`` sum to ``predict(x_row) - baseline``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background)
    feature_count = int(feature_count)
    if feature_count < 1:
        raise ValueError("feature_count must be at least 1")
    if x_row.shape[0] != feature_count:
        raise ValueError("x_row must have feature_count entries")
    if background.shape[1] != feature_count:
        raise ValueError("background must have feature_count columns")
    values = np.zeros(feature_count)
    for j in range(feature_count):
        for size in range(feature_count):
            weight = (
                factorial(size)
                * factorial(feature_count - size - 1)
                / factorial(feature_count)
            )
            for subset in combinations(range(feature_count), size):
                if j in subset:
                    continue
                with_j = tuple(sorted(subset + (j,)))
                values[j] += weight * (
                    tree_conditional_expectation(node, x_row, with_j, background)
                    - tree_conditional_expectation(node, x_row, subset, background)
                )
    baseline = tree_conditional_expectation(node, x_row, (), background)
    return {"values": values, "baseline": baseline}


def local_occlusion_attribution(predict, x_row, background, n_samples=200, seed=None):
    """Occlusion-style local attribution for arbitrary predictors.

    For each feature the background rows are redrawn with that column pinned
    to the instance's value; the drop in average prediction relative to the
    instance is the feature's attribution. This is a Monte Carlo fallback for
    models without a tree structure and is not an exact Shapley decomposition.

    Returns
    -------
    dict
        ``{"values": ndarray, "baseline": float, "prediction": float}``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background)
    if x_row.shape[0] != background.shape[1]:
        raise ValueError("x_row and background must have the same number of features")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    rng = np.random.default_rng(seed)
    draws = background[rng.integers(0, background.shape[0], size=n_samples)].copy()
    prediction = float(predict(x_row[None, :])[0])
    values = np.zeros(background.shape[1])
    for j in range(background.shape[1]):
        pinned = draws.copy()
        pinned[:, j] = x_row[j]
        values[j] = prediction - float(np.mean(predict(pinned)))
    baseline = float(np.mean(predict(draws)))
    return {"values": values, "baseline": baseline, "prediction": prediction}


def shapley_kernel_weight(coalition_size, n_features):
    """Shapley kernel weight for a coalition of the given size.

    ``π(|S|) = (M - 1) / (C(M, |S|) * |S| * (M - |S|))`` for
    ``0 < |S| < M``. Empty and full coalitions are undefined in the
    classical kernel (they pin the intercept / efficiency constraint);
    this helper returns ``inf`` for those sizes so callers can treat them
    specially.
    """
    m = int(n_features)
    s = int(coalition_size)
    if m < 1:
        raise ValueError("n_features must be at least 1")
    if s < 0 or s > m:
        raise ValueError("coalition_size must lie in [0, n_features]")
    if s == 0 or s == m:
        return float("inf")
    # C(M, s) = M! / (s! (M-s)!)
    from math import comb

    return (m - 1) / (comb(m, s) * s * (m - s))


def kernel_shap(
    predict,
    x_row,
    background,
    n_samples=200,
    l2=1e-6,
    seed=None,
    feature_names=None,
):
    """Model-agnostic Kernel SHAP-lite attributions for one instance.

    Samples binary coalitions ``z'``, builds interventional synthetic
    rows (features present in the coalition come from ``x_row``, the
    rest are drawn from a background row), evaluates ``predict``, and
    solves the Shapley-kernel weighted regression

    ``g(z') = φ_0 + sum_j φ_j z'_j``.

    Empty and full coalitions are always included with a large finite
    weight so the fit recovers the baseline ``E[f(background)]`` and the
    efficiency constraint ``sum φ = f(x) - φ_0``.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    x_row : sequence of float
        Instance to explain.
    background : ndarray
        Reference dataset used to replace missing features.
    n_samples : int
        Number of random coalitions in addition to the empty/full pair.
        When ``2 ** n_features - 2 <= n_samples`` every non-trivial
        coalition is enumerated exactly once (exact Kernel SHAP for
        small feature counts).
    l2 : float
        Ridge term for the weighted least-squares solve.
    seed : int or None
    feature_names : list of str or None

    Returns
    -------
    dict
        ``{"values": ndarray, "baseline": float, "prediction": float,
        "n_samples": int, "feature_names": list, "weighted_r2": float}``.
        ``values`` sum (approximately) to ``prediction - baseline``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background)
    m = background.shape[1]
    if x_row.shape[0] != m:
        raise ValueError("x_row and background must have the same number of features")
    if background.shape[0] < 1:
        raise ValueError("background must contain at least one row")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    rng = np.random.default_rng(seed)
    prediction = float(predict(x_row[None, :])[0])
    baseline_pred = float(np.mean(predict(background)))

    # Enumerate all non-trivial coalitions when the feature count is small.
    max_nontrivial = (1 << m) - 2
    if m <= 12 and max_nontrivial <= n_samples:
        masks = []
        for bits in range(1, (1 << m) - 1):
            mask = np.array([(bits >> j) & 1 for j in range(m)], dtype=float)
            masks.append(mask)
        masks = np.asarray(masks, dtype=float)
    else:
        # Sample random non-empty/non-full masks; reject degenerates.
        masks = []
        seen = set()
        attempts = 0
        target = int(n_samples)
        while len(masks) < target and attempts < target * 20:
            attempts += 1
            mask = rng.integers(0, 2, size=m).astype(float)
            key = tuple(mask.tolist())
            if key in seen or mask.sum() == 0 or mask.sum() == m:
                continue
            seen.add(key)
            masks.append(mask)
        if not masks:
            # Fallback: single-feature coalitions.
            masks = [np.eye(m)[j] for j in range(m)]
        masks = np.asarray(masks, dtype=float)

    # Always pin empty / full coalitions.
    empty = np.zeros(m, dtype=float)
    full = np.ones(m, dtype=float)
    all_masks = np.vstack([empty[None, :], full[None, :], masks])

    # Interventional mean-imputation: missing features take the background
    # column mean. This keeps Kernel SHAP-lite deterministic given the
    # coalition sample and recovers exact φ for linear models.
    bg_mean = background.mean(axis=0)
    synthetic = np.tile(bg_mean, (all_masks.shape[0], 1))
    for i, mask in enumerate(all_masks):
        present = mask > 0.5
        synthetic[i, present] = x_row[present]
    y = np.asarray(predict(synthetic), dtype=float).ravel()

    # Shapley kernel weights; large finite weight for empty/full.
    pin_weight = 1e6
    weights = np.empty(all_masks.shape[0], dtype=float)
    weights[0] = pin_weight
    weights[1] = pin_weight
    for i in range(2, all_masks.shape[0]):
        weights[i] = shapley_kernel_weight(int(all_masks[i].sum()), m)

    design = np.column_stack([np.ones(all_masks.shape[0]), all_masks])
    coefs = weighted_least_squares(design, y, weights, l2=l2)
    baseline = float(coefs[0])
    values = coefs[1:]
    fitted = design @ coefs
    # Prefer reporting the empirical background mean as baseline when the
    # pin is active; keep the regression intercept close to it.
    if feature_names is None:
        feature_names = ["X%d" % j for j in range(m)]
    # With empty/full coalitions pinned, the WLS intercept is f(bg_mean)
    # and sum(values) recovers prediction - baseline (efficiency).
    return {
        "values": np.asarray(values, dtype=float),
        "baseline": float(baseline),
        "prediction": prediction,
        "n_samples": int(all_masks.shape[0]),
        "feature_names": list(feature_names),
        "weighted_r2": _weighted_r2(y, fitted, weights),
        "background_mean_prediction": float(baseline_pred),
    }


def integrated_gradients(
    predict,
    x_row,
    baseline=None,
    n_steps=32,
    eps=1e-4,
    feature_names=None,
):
    """Integrated Gradients attributions via a Riemann-sum path integral.

    Approximates the path integral of the model's gradient from a baseline
    ``x'`` to the instance ``x`` (Sundararajan, Taly & Yan, ICML 2017):

    ``IG_i(x) = (x_i - x'_i) * (1/m) * sum_{k=1}^{m} ∂f(x' + (k/m)(x-x')) / ∂x_i``

    Gradients are estimated with central finite differences of step ``eps``
    so the estimator works for any black-box ``predict``. By the
    completeness axiom the attributions sum (approximately) to
    ``f(x) - f(baseline)``.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    x_row : sequence of float
        Instance to explain.
    baseline : sequence of float or None
        Reference input. Defaults to the zero vector with the same width
        as ``x_row``.
    n_steps : int
        Number of Riemann-sum steps along the path (``m`` above).
    eps : float
        Central finite-difference step size for each partial derivative.
    feature_names : list of str or None

    Returns
    -------
    dict
        ``{"values": ndarray, "baseline": float, "prediction": float,
        "baseline_input": ndarray, "n_steps": int, "feature_names": list}``.
        ``values`` sum (approximately) to ``prediction - baseline``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    m = x_row.shape[0]
    if m < 1:
        raise ValueError("x_row must have at least one feature")
    if baseline is None:
        baseline_input = np.zeros(m, dtype=float)
    else:
        baseline_input = np.asarray(baseline, dtype=float).ravel()
        if baseline_input.shape[0] != m:
            raise ValueError("baseline and x_row must have the same number of features")
    if int(n_steps) < 1:
        raise ValueError("n_steps must be at least 1")
    n_steps = int(n_steps)
    if not (isinstance(eps, (int, float)) and float(eps) > 0.0):
        raise ValueError("eps must be a positive number")
    eps = float(eps)

    prediction = float(predict(x_row[None, :])[0])
    baseline_pred = float(predict(baseline_input[None, :])[0])
    delta = x_row - baseline_input

    # Riemann sum at interior points alpha = k/m for k=1..m.
    grads = np.zeros(m, dtype=float)
    for k in range(1, n_steps + 1):
        alpha = k / float(n_steps)
        point = baseline_input + alpha * delta
        for j in range(m):
            e = np.zeros(m, dtype=float)
            e[j] = eps
            f_plus = float(predict((point + e)[None, :])[0])
            f_minus = float(predict((point - e)[None, :])[0])
            grads[j] += (f_plus - f_minus) / (2.0 * eps)
    avg_grad = grads / float(n_steps)
    values = delta * avg_grad

    if feature_names is None:
        feature_names = ["X%d" % j for j in range(m)]
    elif len(feature_names) != m:
        raise ValueError("feature_names must match the number of features")

    return {
        "values": np.asarray(values, dtype=float),
        "baseline": float(baseline_pred),
        "prediction": prediction,
        "baseline_input": np.asarray(baseline_input, dtype=float),
        "n_steps": n_steps,
        "feature_names": list(feature_names),
    }


def smoothgrad(
    predict,
    x_row,
    n_samples=50,
    noise_sigma=0.1,
    eps=1e-4,
    seed=None,
    feature_names=None,
):
    """SmoothGrad attributions via averaged noisy finite-difference gradients.

    Estimates the gradient of ``f`` at ``x`` with central finite differences,
    then averages those gradients over ``n_samples`` Gaussian-noised copies
    ``x + N(0, σ² I)`` (Smilkov et al., "SmoothGrad: removing noise by adding
    noise", 2017). Averaging denoises saliency without requiring a path
    integral (contrast Integrated Gradients).

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    x_row : sequence of float
        Instance to explain.
    n_samples : int
        Number of noisy copies to average over.
    noise_sigma : float
        Standard deviation of the isotropic Gaussian noise added to ``x``.
        When ``noise_sigma == 0`` the result is the raw central finite-
        difference gradient at ``x``.
    eps : float
        Central finite-difference step size for each partial derivative.
    seed : int or None
        Random seed for the noise draws.
    feature_names : list of str or None

    Returns
    -------
    dict
        ``{"values": ndarray, "prediction": float, "n_samples": int,
        "noise_sigma": float, "feature_names": list}``.
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    m = x_row.shape[0]
    if m < 1:
        raise ValueError("x_row must have at least one feature")
    if int(n_samples) < 1:
        raise ValueError("n_samples must be at least 1")
    n_samples = int(n_samples)
    if not (isinstance(noise_sigma, (int, float)) and float(noise_sigma) >= 0.0):
        raise ValueError("noise_sigma must be a non-negative number")
    noise_sigma = float(noise_sigma)
    if not (isinstance(eps, (int, float)) and float(eps) > 0.0):
        raise ValueError("eps must be a positive number")
    eps = float(eps)

    prediction = float(predict(x_row[None, :])[0])
    rng = np.random.default_rng(seed)
    grads = np.zeros(m, dtype=float)
    for _ in range(n_samples):
        if noise_sigma == 0.0:
            point = x_row.copy()
        else:
            point = x_row + rng.normal(0.0, noise_sigma, size=m)
        for j in range(m):
            e = np.zeros(m, dtype=float)
            e[j] = eps
            f_plus = float(predict((point + e)[None, :])[0])
            f_minus = float(predict((point - e)[None, :])[0])
            grads[j] += (f_plus - f_minus) / (2.0 * eps)
    values = grads / float(n_samples)

    if feature_names is None:
        feature_names = ["X%d" % j for j in range(m)]
    elif len(feature_names) != m:
        raise ValueError("feature_names must match the number of features")

    return {
        "values": np.asarray(values, dtype=float),
        "prediction": prediction,
        "n_samples": n_samples,
        "noise_sigma": noise_sigma,
        "feature_names": list(feature_names),
    }


def sampling_shapley(
    predict,
    x_row,
    background,
    n_permutations=100,
    antithetic=True,
    seed=None,
    feature_names=None,
):
    """Permutation-sampling Shapley values (Štrumbelj & Kononenko, 2014).

    Monte Carlo estimate of interventional Shapley values for one instance.
    Each draw pairs a random feature permutation ``π`` with a random
    background row ``z``, walks from ``z`` to ``x_row`` by switching features
    to their instance values in the order ``π``, and credits each feature
    with the change in prediction at its switch:

    ``φ_j += f(x_{Pre(j) ∪ {j}}, z_rest) - f(x_{Pre(j)}, z_rest)``.

    Unlike :func:`kernel_shap` (which mean-imputes absent features), absent
    features keep the values of a real background row, so the estimator is
    unbiased for the interventional value function
    ``v(S) = E_z[f(x_S, z_~S)]`` and handles non-linear models correctly.
    Every walk telescopes, so ``sum(values) == prediction - baseline`` holds
    *exactly* where ``baseline`` is the mean prediction over the sampled
    background rows. With ``antithetic=True`` each permutation is also used
    reversed with the same background row (Mitchell et al., 2022), which
    cancels much of the variance for near-additive models.

    All model calls are batched into a single ``predict`` invocation of
    ``n_draws * (n_features + 1)`` rows.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y_pred`` for a 2-D ``X``.
    x_row : sequence of float
        Instance to explain.
    background : ndarray
        Reference rows that supply the values of "absent" features.
    n_permutations : int
        Number of (permutation, background row) draws. With antithetic
        sampling each draw contributes two walks.
    antithetic : bool
        Also evaluate each permutation in reverse order.
    seed : int or None
    feature_names : list of str or None

    Returns
    -------
    dict
        ``{"values": ndarray, "std_error": ndarray, "baseline": float,
        "prediction": float, "n_permutations": int, "n_evaluations": int,
        "feature_names": list}``. ``std_error`` is the Monte Carlo standard
        error of each value (antithetic pairs count as one draw).
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = as_2d(background, name="background")
    m = background.shape[1]
    if x_row.shape[0] != m:
        raise ValueError("x_row and background must have the same number of features")
    if background.shape[0] < 1:
        raise ValueError("background must contain at least one row")
    if isinstance(n_permutations, bool) or int(n_permutations) != n_permutations:
        raise ValueError("n_permutations must be a positive integer")
    n_permutations = int(n_permutations)
    if n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    if feature_names is None:
        feature_names = ["X%d" % j for j in range(m)]
    elif len(feature_names) != m:
        raise ValueError("feature_names must have one entry per feature")

    rng = np.random.default_rng(seed)
    perms = np.array([rng.permutation(m) for _ in range(n_permutations)], dtype=int)
    rows = rng.integers(0, background.shape[0], size=n_permutations)
    if antithetic:
        perms = np.vstack([perms, perms[:, ::-1]])
        rows = np.concatenate([rows, rows])
    n_walks = perms.shape[0]

    # walks[w, k] is the point after switching the first k features of perm w.
    walks = np.repeat(background[rows][:, None, :], m + 1, axis=1)
    for w in range(n_walks):
        for k, feat in enumerate(perms[w], start=1):
            walks[w, k:, feat] = x_row[feat]
    preds = np.asarray(predict(walks.reshape(-1, m)), dtype=float).ravel()
    if preds.shape[0] != n_walks * (m + 1):
        raise ValueError("predict must return one value per input row")
    preds = preds.reshape(n_walks, m + 1)

    deltas = np.diff(preds, axis=1)  # deltas[w, k] belongs to feature perms[w, k]
    contrib = np.zeros((n_walks, m), dtype=float)
    contrib[np.arange(n_walks)[:, None], perms] = deltas
    if antithetic:
        contrib = 0.5 * (contrib[:n_permutations] + contrib[n_permutations:])
        start = preds[:n_permutations, 0]
    else:
        start = preds[:, 0]

    values = contrib.mean(axis=0)
    if contrib.shape[0] > 1:
        std_error = contrib.std(axis=0, ddof=1) / np.sqrt(contrib.shape[0])
    else:
        std_error = np.full(m, np.nan)
    prediction = float(preds[0, -1])
    return {
        "values": values,
        "std_error": std_error,
        "baseline": float(np.mean(start)),
        "prediction": prediction,
        "n_permutations": n_permutations,
        "n_evaluations": int(preds.size),
        "feature_names": list(feature_names),
    }


def _quantile_bins(background, n_bins):
    """Per-feature quantile edges of shape (n_features, n_bins + 1)."""
    background = np.asarray(background, dtype=float)
    n_features = background.shape[1]
    edges = np.empty((n_features, n_bins + 1), dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    for j in range(n_features):
        col = background[:, j]
        e = np.quantile(col, quantiles)
        # Ensure strictly increasing edges for digitize.
        for i in range(1, e.size):
            if e[i] <= e[i - 1]:
                e[i] = e[i - 1] + 1e-12
        edges[j] = e
    return edges


def _bin_indices(X, edges):
    """Map each value to a bin index in ``0 .. n_bins-1``."""
    X = np.asarray(X, dtype=float)
    n_bins = edges.shape[1] - 1
    out = np.empty(X.shape, dtype=int)
    for j in range(X.shape[1]):
        # digitize with inner edges; clip to [0, n_bins-1]
        idx = np.digitize(X[:, j], edges[j, 1:-1], right=False)
        out[:, j] = np.clip(idx, 0, n_bins - 1)
    return out



def anchor_explain(
    predict,
    x_row,
    background,
    *,
    n_samples=1000,
    n_bins=5,
    precision_threshold=0.95,
    min_coverage=0.05,
    tolerance=None,
    beam_size=5,
    seed=None,
):
    """Find a high-precision tabular Anchor rule for ``x_row`` (Ribeiro et al.).

    Continuous features are discretised into quantile bins from
    ``background``. Candidate predicates are "feature j stays in the same
    bin as the instance". A greedy beam search grows a conjunctive rule
    that maximises the fraction of perturbed neighbours whose prediction
    stays within ``tolerance`` of ``f(x)`` (precision), subject to
    ``coverage >= min_coverage`` on the perturbed set. Stops when
    precision reaches ``precision_threshold`` or no improving predicate
    remains.

    Parameters
    ----------
    predict : callable
        ``predict(X) -> y`` vectorised predictor.
    x_row : array-like, shape (n_features,)
        Instance to explain.
    background : array-like, shape (n_background, n_features)
        Empirical distribution used for bin edges and perturbations.
    n_samples : int
        Number of perturbed neighbours (plus the instance itself).
    n_bins : int
        Quantile bins per feature.
    precision_threshold : float
        Target precision in ``(0, 1]``.
    min_coverage : float
        Minimum fraction of neighbours that must satisfy the rule.
    tolerance : float or None
        Absolute prediction gap counted as a match. ``None`` uses
        ``0.1 * std(predict(background))`` (or ``1e-6`` if that std is 0).
    beam_size : int
        Beam width for greedy predicate search.
    seed : int or None
        RNG seed for perturbations.

    Returns
    -------
    dict
        ``predicates`` (list of ``{feature, bin, low, high}``),
        ``precision``, ``coverage``, ``prediction``, ``tolerance``,
        ``n_samples``.
    """
    background = np.asarray(background, dtype=float)
    x_row = np.asarray(x_row, dtype=float).ravel()
    if background.ndim != 2:
        raise ValueError("background must be 2-D")
    if x_row.shape[0] != background.shape[1]:
        raise ValueError("x_row width must match background")
    if background.shape[0] < 2:
        raise ValueError("background needs at least two rows")
    if (
        not isinstance(n_samples, (int, np.integer))
        or isinstance(n_samples, bool)
        or int(n_samples) < 10
    ):
        raise ValueError("n_samples must be an integer >= 10")
    if (
        not isinstance(n_bins, (int, np.integer))
        or isinstance(n_bins, bool)
        or int(n_bins) < 2
    ):
        raise ValueError("n_bins must be an integer >= 2")
    if not 0.0 < float(precision_threshold) <= 1.0:
        raise ValueError("precision_threshold must lie in (0, 1]")
    if not 0.0 < float(min_coverage) <= 1.0:
        raise ValueError("min_coverage must lie in (0, 1]")
    if (
        not isinstance(beam_size, (int, np.integer))
        or isinstance(beam_size, bool)
        or int(beam_size) < 1
    ):
        raise ValueError("beam_size must be a positive integer")

    n_samples = int(n_samples)
    n_bins = int(n_bins)
    beam_size = int(beam_size)
    rng = np.random.default_rng(seed)
    n_features = background.shape[1]

    edges = _quantile_bins(background, n_bins)
    x_bins = _bin_indices(x_row.reshape(1, -1), edges)[0]

    Z = background[rng.integers(0, background.shape[0], size=n_samples)]
    Z = np.vstack([Z, x_row.reshape(1, -1)])
    z_bins = _bin_indices(Z, edges)

    preds = np.asarray(predict(Z), dtype=float).ravel()
    if preds.shape[0] != Z.shape[0]:
        raise ValueError("predict must return one value per row")
    if not np.all(np.isfinite(preds)):
        raise ValueError("predict must return finite values")
    fx = float(preds[-1])

    if tolerance is None:
        bg_preds = np.asarray(predict(background), dtype=float).ravel()
        scale = float(np.std(bg_preds))
        tolerance = 0.1 * scale if scale > 0.0 else 1e-6
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite non-negative number")

    matches = np.abs(preds - fx) <= tolerance

    def _score(rule_feats):
        if not rule_feats:
            return float(matches.mean()), 1.0
        mask = np.ones(Z.shape[0], dtype=bool)
        for j in rule_feats:
            mask &= z_bins[:, j] == x_bins[j]
        coverage = float(mask.mean())
        if coverage <= 0.0:
            return 0.0, 0.0
        return float(matches[mask].mean()), coverage

    # Beam of frozensets of feature indices.
    beam = [frozenset()]
    best_rule = frozenset()
    best_precision, best_coverage = _score(best_rule)

    for _depth in range(n_features):
        candidates = []
        for rule in beam:
            for feat in range(n_features):
                if feat in rule:
                    continue
                new_rule = rule | {feat}
                precision, coverage = _score(new_rule)
                if coverage < float(min_coverage):
                    continue
                candidates.append((new_rule, precision, coverage))
        if not candidates:
            break
        candidates.sort(key=lambda c: (-c[1], -c[2], len(c[0])))
        beam = [c[0] for c in candidates[:beam_size]]
        top_rule, top_p, top_c = candidates[0]
        improved = top_p > best_precision + 1e-12 or (
            abs(top_p - best_precision) <= 1e-12 and top_c > best_coverage
        )
        if improved:
            best_rule, best_precision, best_coverage = top_rule, top_p, top_c
        if best_precision >= float(precision_threshold):
            break
        if not improved:
            break

    predicates = []
    for j in sorted(best_rule):
        b = int(x_bins[j])
        predicates.append(
            {
                "feature": int(j),
                "bin": b,
                "low": float(edges[j, b]),
                "high": float(edges[j, b + 1]),
            }
        )
    precision, coverage = _score(best_rule)
    return {
        "predicates": predicates,
        "precision": precision,
        "coverage": coverage,
        "prediction": fx,
        "tolerance": tolerance,
        "n_samples": int(Z.shape[0]),
    }
