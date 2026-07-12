"""Report generation: render a :class:`~grafology_ai.interpretation.StructuredFindings`
into a fixed-section markdown report.

This is the "Minimal but useful output" step of the client's technical
proposal ("Structured report template (fixed sections + disclaimer +
confidence indication)" and "Text/markdown output for graphologist
review"), and the report-generation stage the interpretation layer's own
module docstring (``grafology_ai.interpretation.interpret``) says is
expected downstream.

This module is deliberately a thin *renderer*. :func:`interpret` (see
``grafology_ai.interpretation``) already does all the work of producing
cautious, non-diagnostic, hedged narrative text -- observation and
interpretation strings, strengths, areas of attention, an overall
summary -- and already enforces its own language discipline (see
``DISALLOWED_TERMS`` and ``_assert_language_is_disciplined`` in
``grafology_ai.interpretation.interpret``). :func:`generate_report`
presents that text as-is: it never rewrites, embellishes, summarizes, or
adds evaluative language to a :class:`~grafology_ai.interpretation.Finding`'s
``observation``/``interpretation`` or to ``strengths``/
``areas_of_attention``/``overall_summary``. The only text this module
originates itself is structural boilerplate (headings, the disclaimer,
the confidence-label wording, empty-section notes, the closing note),
which is written to the same cautious standard.

Report shape
------------

:func:`generate_report` returns a complete markdown document as a string,
with fixed sections in a stable order (see its docstring for the exact
list). The disclaimer is placed directly after the title/header -- near
the top of the document, not buried at the bottom -- and a brief closing
note repeats the same reminder at the end. Every section renders
correctly (no empty/broken markdown, no dangling bullet lists, no ``None``
leaking into the text) for both ``depth="concise"`` and ``depth="indepth"``
findings, and for the near-blank/low-confidence case where ``strengths``
and/or ``areas_of_attention`` are empty.

:func:`save_report` writes the same markdown to disk. PDF output is
handled by a sibling module, `grafology_ai.report.pdf`
(:func:`~grafology_ai.report.pdf.render_report_pdf` /
:func:`~grafology_ai.report.pdf.save_report_pdf`), which renders the same
fixed sections directly from a `StructuredFindings` -- not from the
markdown string this module produces -- so the two outputs can never
structurally diverge from one another, both always being derived
straight from the same structured data. Nothing in this module needed to
change to support that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from grafology_ai.interpretation import Depth, Finding, INDICATOR_LABELS, StructuredFindings

# --- confidence label mapping -------------------------------------------------
#
# A small, documented set of thresholds mapping a Finding's numeric
# `confidence` (in [0, 1], carried over verbatim from
# `Features.confidence` -- see `grafology_ai.interpretation.interpret`)
# onto a qualitative "high"/"medium"/"low" label. These are deliberately
# chosen, documented constants rather than inline magic numbers, matching
# the convention used for thresholds throughout
# `grafology_ai.interpretation.interpret` and
# `grafology_ai.analysis.features`.

#: A finding at or above this confidence is labeled "high".
HIGH_CONFIDENCE_THRESHOLD = 0.7

#: A finding at or above this confidence (but below `HIGH_CONFIDENCE_THRESHOLD`)
#: is labeled "medium"; below this, "low".
MEDIUM_CONFIDENCE_THRESHOLD = 0.4


def confidence_label(confidence: float) -> str:
    """Map a numeric `confidence` in [0, 1] to a qualitative label.

    Returns ``"high"`` at or above `HIGH_CONFIDENCE_THRESHOLD`, ``"medium"``
    at or above `MEDIUM_CONFIDENCE_THRESHOLD`, and ``"low"`` otherwise.
    Used consistently for every :class:`~grafology_ai.interpretation.Finding`
    rendered by :func:`generate_report`.
    """
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


# --- fixed boilerplate text ---------------------------------------------------
#
# This module's own text, kept short and consistent with the framing
# `interpret()` already writes into `StructuredFindings.overall_summary`
# (a "support tool" that is "not ... a determination about the writer",
# meant to "support, not replace, a professional graphologist's own
# judgment") -- see `grafology_ai.interpretation.interpret._build_overall_summary`.
# Unlike the interpretation layer's generated findings text, this
# boilerplate is allowed to use words like "diagnosis" because it uses
# them only to explicitly *deny* diagnostic framing, which is exactly what
# the client's technical proposal calls for here ("disclaimer" section
# stating this is "not a diagnosis, not a clinical or forensic
# assessment").

_DEPTH_LABELS: dict[Depth, str] = {
    "concise": "Concise",
    "indepth": "In-depth",
}

_DISCLAIMER_HEADING = "## Important: Please Read Before Using This Report"

_DISCLAIMER_TEXT = (
    "This report is the output of an automated support tool, generated from "
    "a handful of measurable handwriting features (such as slant, stroke "
    "weight, spacing, and layout). **It is not a diagnosis, and it is not a "
    "clinical or forensic assessment.** It does not determine anything "
    "about the writer's character, health, or psychological state. It is "
    "intended to assist -- not replace -- a professional graphologist's own "
    "judgment, and every finding below should be reviewed alongside the "
    "original sample by a qualified graphologist before being relied upon."
)

_EMPTY_STRENGTHS_NOTE = (
    "No specific strengths were confidently identified from this sample."
)

_EMPTY_ATTENTION_NOTE = (
    "No specific areas of attention were identified from this sample."
)

_CLOSING_NOTE_TEXT = (
    "*Reminder: this is a support tool output, not a diagnosis. It is "
    "intended to assist -- not replace -- a professional graphologist's own "
    "review and judgment.*"
)


def _indicator_label(indicator: str) -> str:
    """Human-readable label for a `Finding.indicator` slug.

    Looks up `grafology_ai.interpretation.INDICATOR_LABELS` first (covers
    every indicator `interpret()` actually produces, e.g. `"slant"` ->
    `"Slant"`); falls back to a title-cased, underscore-stripped rendering
    of the raw slug for any indicator string not in that table, so this
    never renders a raw machine-readable name like `"slant_angle_degrees"`.
    """
    return INDICATOR_LABELS.get(indicator, indicator.replace("_", " ").strip().title())


# --- section renderers ---------------------------------------------------------


def _render_header(depth: Depth, sample_id: str | None) -> str:
    lines = ["# Handwriting Analysis Support Report"]
    if sample_id:
        lines.append(f"**Sample ID:** {sample_id}")
    lines.append(f"**Analysis depth:** {_DEPTH_LABELS.get(depth, str(depth).title())}")
    return "\n".join(lines)


def _render_disclaimer() -> str:
    return f"{_DISCLAIMER_HEADING}\n\n{_DISCLAIMER_TEXT}"


def _render_overall_summary(overall_summary: str) -> str:
    return f"## Overall Summary\n\n{overall_summary}"


def _render_finding(finding: Finding) -> str:
    label = _indicator_label(finding.indicator)
    label_word = confidence_label(finding.confidence)
    percent = round(finding.confidence * 100)
    return (
        f"### {label}\n\n"
        f"**Observation:** {finding.observation}\n\n"
        f"**Interpretation:** {finding.interpretation}\n\n"
        f"**Confidence:** {label_word} confidence ({percent}%)"
    )


def _render_findings_section(findings: Sequence[Finding]) -> str:
    if not findings:
        # Defensive: `interpret()` never returns an empty `findings` list
        # (see its docstring), but a `StructuredFindings` can in principle
        # be constructed directly with one -- render a well-formed section
        # rather than a bare, content-less heading.
        return (
            "## Findings\n\n"
            "No individual indicator findings were generated for this sample."
        )
    blocks = ["## Findings", *(_render_finding(f) for f in findings)]
    return "\n\n".join(blocks)


def _render_list_section(heading: str, items: Sequence[str], empty_note: str) -> str:
    """Render a bulleted list section, or `empty_note` if `items` is empty.

    Never renders a heading with no items under it, and never leaks a
    literal ``"None"``/empty bullet into the output.
    """
    if items:
        body = "\n".join(f"- {item}" for item in items)
    else:
        body = empty_note
    return f"## {heading}\n\n{body}"


def _render_closing_note() -> str:
    return f"---\n\n{_CLOSING_NOTE_TEXT}"


# --- public entry points --------------------------------------------------------


def generate_report(findings: StructuredFindings, sample_id: str | None = None) -> str:
    """Render `findings` into a complete markdown report string.

    Fixed sections, always in this order:

    1. A title/header (`"# Handwriting Analysis Support Report"`, plus the
       sample ID if `sample_id` is given, and the analysis depth used).
    2. A prominent disclaimer, immediately after the header -- near the
       top of the document, not buried at the bottom -- stating this is a
       support tool output, not a diagnosis or a clinical/forensic
       assessment, intended to assist rather than replace a professional
       graphologist's judgment.
    3. `"## Overall Summary"`: `findings.overall_summary`, verbatim.
    4. `"## Findings"`: one subsection per `findings.findings` entry,
       showing the indicator's human-readable label (e.g. `"Slant"`, via
       `grafology_ai.interpretation.INDICATOR_LABELS`, never the raw
       machine-readable field name), its observation and interpretation
       text verbatim, and a confidence indication combining a qualitative
       label (`confidence_label`) with a percentage.
    5. `"## Strengths"`: a bullet per `findings.strengths` entry, or a
       clear note if the list is empty.
    6. `"## Areas of Attention"`: a bullet per `findings.areas_of_attention`
       entry, or a clear note if the list is empty.
    7. A closing note briefly repeating that this is a support tool
       output, not a diagnosis.

    `findings.overall_summary`, every `Finding.observation`/
    `Finding.interpretation`, and every `strengths`/`areas_of_attention`
    entry are rendered exactly as given -- this function never rewrites,
    embellishes, or paraphrases that text (see the module docstring).

    Renders correctly (no empty/broken markdown, no dangling bullet
    lists, no literal `"None"`) for both `depth="concise"` and
    `depth="indepth"` findings, and for the near-blank/low-confidence
    case where `strengths` and/or `areas_of_attention` are empty.
    """
    sections = [
        _render_header(findings.depth, sample_id),
        _render_disclaimer(),
        _render_overall_summary(findings.overall_summary),
        _render_findings_section(findings.findings),
        _render_list_section("Strengths", findings.strengths, _EMPTY_STRENGTHS_NOTE),
        _render_list_section(
            "Areas of Attention", findings.areas_of_attention, _EMPTY_ATTENTION_NOTE
        ),
        _render_closing_note(),
    ]
    return "\n\n".join(sections).rstrip("\n") + "\n"


def save_report(
    findings: StructuredFindings, output_path: Path, sample_id: str | None = None
) -> Path:
    """Write `generate_report(findings, sample_id)` to `output_path` as markdown.

    Creates `output_path`'s parent directories if they do not already
    exist. Returns `output_path` (as a `Path`, even if a string-like was
    passed in) for convenient chaining.

    PDF output is out of scope here -- see the module docstring -- this
    only ever writes the markdown text itself; for PDF, see
    `grafology_ai.report.pdf.save_report_pdf`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_report(findings, sample_id=sample_id), encoding="utf-8")
    return output_path
