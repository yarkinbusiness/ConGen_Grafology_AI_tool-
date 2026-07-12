"""Report generation: render `StructuredFindings` into a markdown or PDF report.

This subpackage is the final "Minimal but useful output" stage of the
pipeline: it takes the cautious, structured, non-diagnostic content
`grafology_ai.interpretation.interpret` produces and renders it into a
fixed-section report suitable for graphologist review -- see
`grafology_ai.report.generator` for the markdown renderer and its
language-discipline guarantees, and `grafology_ai.report.pdf` for the PDF
sibling that renders the same fixed sections directly from the same
`StructuredFindings`.
"""

from grafology_ai.report.generator import (
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    confidence_label,
    generate_report,
    save_report,
)
from grafology_ai.report.pdf import render_report_pdf, save_report_pdf

__all__ = [
    "HIGH_CONFIDENCE_THRESHOLD",
    "MEDIUM_CONFIDENCE_THRESHOLD",
    "confidence_label",
    "generate_report",
    "render_report_pdf",
    "save_report",
    "save_report_pdf",
]
