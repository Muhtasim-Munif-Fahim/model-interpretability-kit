"""model-interpretability-kit: model-agnostic interpretability in pure NumPy.

This package provides global and local explanations for any model exposed as a
plain callable ``predict(X) -> y``:

* ``importance``: permutation and drop-column feature importance
* ``partial_dependence``: 1-D/2-D partial dependence plots and ICE curves
* ``ale``: 1-D/2-D accumulated local effects curves and interaction surfaces
* ``local``: LIME-style weighted linear surrogates and tree-based local attributions
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

__version__ = "0.1.0"

__all__ = [
    "accumulated_local_effects",
    "accumulated_local_effects_2d",
    "ale",
    "ale_2d",
]
