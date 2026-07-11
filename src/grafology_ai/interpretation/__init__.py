"""Interpretation layer: cautious, human-readable findings from `Features`.

This subpackage maps the raw numeric measurements produced by
`grafology_ai.analysis.analyze` onto the indicator vocabulary documented
in `docs/labeling_rubric.md`, attaching short, hedged, non-diagnostic
narrative text to each one. It does not decide anything about a person
-- see `grafology_ai.interpretation.interpret` for the full pipeline
description, its language-discipline guarantees
(`DISALLOWED_TERMS`), and its documented thresholds.
"""

from grafology_ai.interpretation.interpret import (
    ATTENTION_LOW_CONFIDENCE_MAX,
    CONCISE_INDICATOR_COUNT,
    DISALLOWED_TERMS,
    INDICATOR_CONFIDENCE_KEY,
    INDICATOR_LABELS,
    INDICATOR_ORDER,
    STRENGTH_MIN_CONFIDENCE,
    Depth,
    Finding,
    StructuredFindings,
    interpret,
)

__all__ = [
    "ATTENTION_LOW_CONFIDENCE_MAX",
    "CONCISE_INDICATOR_COUNT",
    "DISALLOWED_TERMS",
    "INDICATOR_CONFIDENCE_KEY",
    "INDICATOR_LABELS",
    "INDICATOR_ORDER",
    "STRENGTH_MIN_CONFIDENCE",
    "Depth",
    "Finding",
    "StructuredFindings",
    "interpret",
]
