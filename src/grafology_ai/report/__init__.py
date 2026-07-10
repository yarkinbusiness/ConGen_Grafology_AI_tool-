"""Report generation: render `StructuredFindings` into a markdown report.

This subpackage is the final "Minimal but useful output" stage of the
pipeline: it takes the cautious, structured, non-diagnostic content
`grafology_ai.interpretation.interpret` produces and renders it into a
fixed-section markdown report suitable for graphologist review -- see
`grafology_ai.report.generator` for the full report shape and its
language-discipline guarantees.
"""

from grafology_ai.report.generator import (
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    confidence_label,
    generate_report,
    save_report,
)

__all__ = [
    "HIGH_CONFIDENCE_THRESHOLD",
    "MEDIUM_CONFIDENCE_THRESHOLD",
    "confidence_label",
    "generate_report",
    "save_report",
]
