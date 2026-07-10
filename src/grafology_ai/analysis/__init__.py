"""Classical image-processing feature analyzer for handwriting samples.

This subpackage measures observable graphological indicators (slant,
stroke-width/pressure proxy, spacing, baseline, margins, ink density)
directly from pixels, as a heuristic stand-in for the not-yet-trainable
machine learning model (no real, labeled client dataset exists yet -- see
:mod:`grafology_ai.dataset.fixtures`). It does not interpret or judge
those measurements -- see :mod:`grafology_ai.analysis.features` for the
full pipeline description and its documented assumptions/limitations.
"""

from grafology_ai.analysis.features import (
    FEATURE_CONFIDENCE_KEYS,
    Features,
    analyze,
)

__all__ = [
    "FEATURE_CONFIDENCE_KEYS",
    "Features",
    "analyze",
]
