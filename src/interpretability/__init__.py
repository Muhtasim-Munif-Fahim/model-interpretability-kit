"""model-interpretability-kit: model-agnostic interpretability in pure NumPy.

This package provides global and local explanations for any model exposed as a
plain callable ``predict(X) -> y``:

* ``importance``: permutation, drop-column, leave-one-covariate-out, Sobol first-order / total-order, and Morris screening
* ``sage``: SAGE global importance (Shapley values of the model's loss)
* ``partial_dependence``: 1-D/2-D partial dependence plots and ICE curves
* ``ale``: 1-D/2-D accumulated local effects curves and interaction surfaces
* ``interaction``: Friedman H-statistic for pairwise interaction strength
* ``local``: LIME-style surrogates, Anchors, Kernel SHAP-lite, permutation-sampling Shapley, Integrated Gradients, SmoothGrad, and tree-based local attributions
* ``evaluate``: faithfulness checks for local explanations
* ``report``: markdown report rendering
* ``cli``: command-line interface over the above
"""

from .ale import (
    accumulated_local_effects,
    accumulated_local_effects_2d,
    ale,
    ale_2d,
)
from .interaction import friedman_h_statistic, h_statistic
from .importance import morris_screening, morris_elementary_effects, sobol_first_order, sobol_total_order
from .local import anchor_explain, integrated_gradients, kernel_shap, sampling_shapley, smoothgrad
from .sage import sage_values
from .partial_dependence import (
    ice,
    ice_curves,
    partial_dependence,
    partial_dependence_2d,
    pdp,
)

__version__ = "0.1.0"

__all__ = [
    "accumulated_local_effects",
    "accumulated_local_effects_2d",
    "ale",
    "ale_2d",
    "ice",
    "ice_curves",
    "partial_dependence",
    "partial_dependence_2d",
    "pdp",
    "friedman_h_statistic",
    "h_statistic",
    "anchor_explain",
    "kernel_shap",
    "sampling_shapley",
    "integrated_gradients",
    "smoothgrad",
    "sage_values",
    "sobol_first_order",
    "sobol_total_order",
    "morris_screening",
    "morris_elementary_effects",
]
