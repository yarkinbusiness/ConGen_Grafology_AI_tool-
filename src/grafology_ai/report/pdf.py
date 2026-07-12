"""PDF report generation: render a :class:`~grafology_ai.interpretation.StructuredFindings`
into a fixed-section PDF report -- the PDF sibling of
:mod:`grafology_ai.report.generator`'s markdown renderer.

This is the "Markdown-to-PDF conversion" future step
:mod:`grafology_ai.report.generator`'s module docstring named as
out-of-scope for that stage (see ``docs/roadmap.md``, phase C1): a
graphologist-facing deliverable that does not require a markdown viewer
to read.

Like :func:`~grafology_ai.report.generator.generate_report`, this module
is deliberately a thin *renderer*, built directly from a
:class:`~grafology_ai.interpretation.StructuredFindings` -- not from the
markdown string :func:`~grafology_ai.report.generator.generate_report`
produces. :func:`render_report_pdf` never constructs or manipulates a
markdown string anywhere internally; it reuses only the same *boilerplate
text constants* :mod:`grafology_ai.report.generator` defines (the
disclaimer heading/text, the empty-strengths/empty-attention notes, the
closing note text, and :func:`~grafology_ai.report.generator.confidence_label`)
so that the two renderers' wording can never drift apart, while each
independently derives its own document structure (markdown headings for
one, PDF fonts/positions for the other) from the exact same
:class:`~grafology_ai.interpretation.StructuredFindings` input. This is a
deliberate design constraint, not just a style preference: markdown and
PDF output must never structurally diverge, because both are always
derived directly from the same structured data, never from each other.

Every piece of narrative text -- ``overall_summary``, each
``Finding.observation``/``Finding.interpretation``, and every
``strengths``/``areas_of_attention`` entry -- is written to the page
exactly as given, with :meth:`fpdf.FPDF.write` (never the markdown-aware
``multi_cell(..., markdown=True)`` path -- see the note on
:class:`_ReportPDF` below). This module never rewrites, embellishes,
summarizes, or reformats that text. The only text this module originates
itself is structural/layout boilerplate (headings, the disclaimer, the
confidence-label wording, bullet markers, empty-section notes, the
closing note) -- the same split :mod:`grafology_ai.report.generator`
documents for its own output.

Report shape
------------

:func:`render_report_pdf` renders the same 7 fixed sections
:func:`~grafology_ai.report.generator.generate_report` renders, in the
same order: (1) a title/header (title, sample ID if given, analysis
depth), (2) a prominent disclaimer immediately after the header, (3) an
"Overall Summary" section, (4) a "Findings" section (one subsection per
indicator, with its human-readable label, observation, interpretation,
and confidence), (5) a "Strengths" section, (6) an "Areas of Attention"
section, (7) a closing note repeating the disclaimer. Styling is
deliberately neutral and professional only -- a plain heading/body font
hierarchy (the built-in Helvetica core font), standard page margins, and
automatic page breaks as content overflows a page -- no color scheme or
branding, since the actual visual design of this report is being handled
separately via an external design tool; this module's job is to be
correct, complete, and readable, not decorative.

Font/markup note
-----------------

fpdf2's built-in ``markdown=True`` cell rendering treats a literal ``"--"``
in text as an *underline toggle*
(``FPDF.MARKDOWN_UNDERLINE_MARKER``). This codebase's generated text uses
``" -- "`` throughout as plain em-dash-style punctuation (see e.g.
:data:`~grafology_ai.report.generator._DISCLAIMER_TEXT` and most of
:mod:`grafology_ai.interpretation.interpret`'s per-indicator narrative
text) -- never as an underline instruction. Left at fpdf2's default,
every such ``"--"`` pair would be silently consumed and reinterpreted as
an underline toggle, corrupting the rendered (and extracted) text -- for
example ``"assist -- not replace --"`` would lose both dashes entirely.
:class:`_ReportPDF` repoints that marker at a sentinel string that never
appears in generated text, which neutralizes the collision. As a second,
independent layer of protection, this module still never passes
``markdown=True`` for any narrative text carried in a
:class:`~grafology_ai.interpretation.StructuredFindings` (``write()``
does no markdown interpretation at all); ``markdown=True`` is used only
for the two boilerplate constants that actually contain fpdf2's
``**bold**`` markup (the disclaimer and closing note text), so that text
renders with real bold emphasis instead of showing literal ``"**"``
characters.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

from grafology_ai.interpretation import Depth, Finding, StructuredFindings
from grafology_ai.report.generator import (
    _CLOSING_NOTE_TEXT,
    _DEPTH_LABELS,
    _DISCLAIMER_HEADING,
    _DISCLAIMER_TEXT,
    _EMPTY_ATTENTION_NOTE,
    _EMPTY_STRENGTHS_NOTE,
    _indicator_label,
    confidence_label,
)

# --- layout constants ---------------------------------------------------------
#
# Deliberately plain and neutral (see the module docstring): a small,
# documented font-size/line-height hierarchy and standard page margins,
# not a designed visual identity -- that is separate, later work.

_FONT_FAMILY = "Helvetica"
_PAGE_MARGIN_MM = 20.0
_AUTO_PAGE_BREAK_MARGIN_MM = 20.0

_TITLE_SIZE_PT = 18.0
_TITLE_LINE_HEIGHT_MM = 9.0

_META_SIZE_PT = 10.5
_META_LINE_HEIGHT_MM = 6.0

_SECTION_HEADING_SIZE_PT = 13.0
_SECTION_HEADING_LINE_HEIGHT_MM = 7.5

_SUBSECTION_HEADING_SIZE_PT = 11.5
_SUBSECTION_HEADING_LINE_HEIGHT_MM = 6.5

_BODY_SIZE_PT = 10.5
_BODY_LINE_HEIGHT_MM = 5.5

#: Vertical gap after a section heading, before its body content.
_HEADING_GAP_MM = 3.0
#: Vertical gap between paragraphs/blocks within a section.
_PARAGRAPH_GAP_MM = 3.0
#: Extra vertical gap between one Finding subsection and the next.
_FINDING_GAP_MM = 5.0

#: Fallback text for a `StructuredFindings` constructed directly with an
#: empty `findings` list -- mirrors
#: `grafology_ai.report.generator._render_findings_section`'s equivalent
#: defensive fallback (not exported as a constant there, so restated here
#: verbatim; both renderers cover this edge case for the same reason:
#: `interpret()` itself never returns an empty `findings` list, but the
#: dataclass can in principle be built directly with one).
_EMPTY_FINDINGS_NOTE = "No individual indicator findings were generated for this sample."


class _ReportPDF(FPDF):
    """`FPDF` subclass with the `"--"`-as-underline markdown marker disabled.

    See the module docstring's "Font/markup note" for why: this
    codebase's generated text uses ``"--"`` as plain punctuation, never
    as an underline instruction, and fpdf2's default marker would corrupt
    it under ``markdown=True`` rendering. Repointing the marker at a
    sentinel that can never appear in generated text disables that
    interpretation without affecting `MARKDOWN_BOLD_MARKER` (`"**"`,
    still used for :data:`~grafology_ai.report.generator._DISCLAIMER_TEXT`
    and :data:`~grafology_ai.report.generator._CLOSING_NOTE_TEXT`).
    """

    MARKDOWN_UNDERLINE_MARKER = "\x00__grafology_ai_underline_disabled__\x00"


def _new_pdf() -> _ReportPDF:
    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=_PAGE_MARGIN_MM, top=_PAGE_MARGIN_MM, right=_PAGE_MARGIN_MM)
    pdf.set_auto_page_break(auto=True, margin=_AUTO_PAGE_BREAK_MARGIN_MM)
    pdf.add_page()
    return pdf


# --- small rendering helpers ---------------------------------------------------


def _write_heading(pdf: _ReportPDF, text: str, *, size: float, line_height: float) -> None:
    """A bold, structural heading -- never narrative text (see module docstring)."""
    pdf.set_font(_FONT_FAMILY, "B", size)
    pdf.multi_cell(0, line_height, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(_FONT_FAMILY, "", _BODY_SIZE_PT)


def _write_body(pdf: _ReportPDF, text: str, *, markdown: bool = False) -> None:
    """A body paragraph. `markdown=False` (the default) renders `text` exactly
    as given, with no markup interpretation whatsoever -- the only safe mode
    for narrative text carried in a `StructuredFindings` (see module
    docstring). `markdown=True` is used only for this module's own
    boilerplate constants that contain literal `"**bold**"` markup.
    """
    pdf.set_font(_FONT_FAMILY, "", _BODY_SIZE_PT)
    pdf.multi_cell(0, _BODY_LINE_HEIGHT_MM, text, new_x="LMARGIN", new_y="NEXT", markdown=markdown)


def _write_bulleted_list(pdf: _ReportPDF, items: list[str]) -> None:
    """A bullet per `items` entry, rendered verbatim (only the leading
    ``"- "`` bullet marker is this module's own, mirroring
    `grafology_ai.report.generator._render_list_section`'s markdown bullets).
    """
    pdf.set_font(_FONT_FAMILY, "", _BODY_SIZE_PT)
    for item in items:
        pdf.set_font(_FONT_FAMILY, "B", _BODY_SIZE_PT)
        pdf.write(_BODY_LINE_HEIGHT_MM, "- ")
        pdf.set_font(_FONT_FAMILY, "", _BODY_SIZE_PT)
        pdf.write(_BODY_LINE_HEIGHT_MM, item)
        pdf.ln(_BODY_LINE_HEIGHT_MM)


def _write_labeled_line(pdf: _ReportPDF, label: str, text: str) -> None:
    """A bold structural label (e.g. `"Observation:"`) followed by `text`,
    rendered verbatim via `write()` -- which never interprets markdown
    markup at all, so this is always a safe path for narrative text.
    """
    pdf.set_font(_FONT_FAMILY, "B", _BODY_SIZE_PT)
    pdf.write(_BODY_LINE_HEIGHT_MM, f"{label} ")
    pdf.set_font(_FONT_FAMILY, "", _BODY_SIZE_PT)
    pdf.write(_BODY_LINE_HEIGHT_MM, text)
    pdf.ln(_BODY_LINE_HEIGHT_MM)


# --- section renderers -----------------------------------------------------------


def _render_header(pdf: _ReportPDF, depth: Depth, sample_id: str | None) -> None:
    pdf.set_font(_FONT_FAMILY, "B", _TITLE_SIZE_PT)
    pdf.multi_cell(
        0, _TITLE_LINE_HEIGHT_MM, "Handwriting Analysis Support Report", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.set_font(_FONT_FAMILY, "", _META_SIZE_PT)
    if sample_id:
        _write_labeled_line(pdf, "Sample ID:", sample_id)
    depth_label = _DEPTH_LABELS.get(depth, str(depth).title())
    _write_labeled_line(pdf, "Analysis depth:", depth_label)
    pdf.ln(_PARAGRAPH_GAP_MM)


def _render_disclaimer(pdf: _ReportPDF) -> None:
    _write_heading(
        pdf,
        _DISCLAIMER_HEADING.lstrip("#").strip(),
        size=_SECTION_HEADING_SIZE_PT,
        line_height=_SECTION_HEADING_LINE_HEIGHT_MM,
    )
    pdf.ln(_HEADING_GAP_MM)
    # `_DISCLAIMER_TEXT` is this module's own boilerplate (not narrative
    # data from `StructuredFindings`) and contains fpdf2 `**bold**`
    # markup -- `markdown=True` is safe here (and only here / the closing
    # note) because `_ReportPDF` disables the colliding `"--"` marker.
    _write_body(pdf, _DISCLAIMER_TEXT, markdown=True)
    pdf.ln(_PARAGRAPH_GAP_MM)


def _render_overall_summary(pdf: _ReportPDF, overall_summary: str) -> None:
    _write_heading(
        pdf, "Overall Summary", size=_SECTION_HEADING_SIZE_PT, line_height=_SECTION_HEADING_LINE_HEIGHT_MM
    )
    pdf.ln(_HEADING_GAP_MM)
    _write_body(pdf, overall_summary)  # narrative text: verbatim, markdown=False
    pdf.ln(_PARAGRAPH_GAP_MM)


def _render_finding(pdf: _ReportPDF, finding: Finding) -> None:
    label = _indicator_label(finding.indicator)
    _write_heading(
        pdf, label, size=_SUBSECTION_HEADING_SIZE_PT, line_height=_SUBSECTION_HEADING_LINE_HEIGHT_MM
    )
    pdf.ln(1.5)
    _write_labeled_line(pdf, "Observation:", finding.observation)
    _write_labeled_line(pdf, "Interpretation:", finding.interpretation)
    label_word = confidence_label(finding.confidence)
    percent = round(finding.confidence * 100)
    _write_labeled_line(pdf, "Confidence:", f"{label_word} confidence ({percent}%)")
    pdf.ln(_FINDING_GAP_MM)


def _render_findings_section(pdf: _ReportPDF, findings: list[Finding]) -> None:
    _write_heading(
        pdf, "Findings", size=_SECTION_HEADING_SIZE_PT, line_height=_SECTION_HEADING_LINE_HEIGHT_MM
    )
    pdf.ln(_HEADING_GAP_MM)
    if not findings:
        # Defensive, mirrors `generator._render_findings_section`'s
        # equivalent fallback -- see `_EMPTY_FINDINGS_NOTE`.
        _write_body(pdf, _EMPTY_FINDINGS_NOTE)
        pdf.ln(_PARAGRAPH_GAP_MM)
        return
    for finding in findings:
        _render_finding(pdf, finding)


def _render_list_section(pdf: _ReportPDF, heading: str, items: list[str], empty_note: str) -> None:
    _write_heading(pdf, heading, size=_SECTION_HEADING_SIZE_PT, line_height=_SECTION_HEADING_LINE_HEIGHT_MM)
    pdf.ln(_HEADING_GAP_MM)
    if items:
        _write_bulleted_list(pdf, items)
    else:
        _write_body(pdf, empty_note)
    pdf.ln(_PARAGRAPH_GAP_MM)


def _render_closing_note(pdf: _ReportPDF) -> None:
    # A thin horizontal rule, echoing `generator._render_closing_note`'s
    # markdown `"---"` divider.
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(_PARAGRAPH_GAP_MM)
    # `_CLOSING_NOTE_TEXT` is this module's own boilerplate and contains
    # fpdf2 `**bold**`/single-`*` markup -- see `_render_disclaimer`.
    _write_body(pdf, _CLOSING_NOTE_TEXT, markdown=True)


# --- public entry points --------------------------------------------------------


def render_report_pdf(findings: StructuredFindings, sample_id: str | None = None) -> bytes:
    """Render `findings` into a complete PDF report and return its bytes.

    Fixed sections, in the same order as
    `grafology_ai.report.generator.generate_report`:

    1. A title/header (title, plus the sample ID if `sample_id` is
       given, and the analysis depth used).
    2. A prominent disclaimer, immediately after the header, stating this
       is a support tool output, not a diagnosis or a clinical/forensic
       assessment.
    3. "Overall Summary": `findings.overall_summary`, verbatim.
    4. "Findings": one subsection per `findings.findings` entry, showing
       the indicator's human-readable label, its observation and
       interpretation text verbatim, and a confidence indication
       (`grafology_ai.report.generator.confidence_label` plus a
       percentage).
    5. "Strengths": a bullet per `findings.strengths` entry, or
       `grafology_ai.report.generator._EMPTY_STRENGTHS_NOTE` if empty.
    6. "Areas of Attention": a bullet per `findings.areas_of_attention`
       entry, or `grafology_ai.report.generator._EMPTY_ATTENTION_NOTE` if
       empty.
    7. A closing note repeating the disclaimer.

    Built directly from `findings` -- this function never constructs or
    manipulates a markdown string (see the module docstring). Every
    narrative string (`findings.overall_summary`, every
    `Finding.observation`/`Finding.interpretation`, and every
    `strengths`/`areas_of_attention` entry) is written to the page
    exactly as given.

    Styling is a plain, neutral heading/body hierarchy on the built-in
    Helvetica core font, with standard page margins and automatic page
    breaks as content overflows a page -- see the module docstring for
    why this stays deliberately undecorated.
    """
    pdf = _new_pdf()
    _render_header(pdf, findings.depth, sample_id)
    _render_disclaimer(pdf)
    _render_overall_summary(pdf, findings.overall_summary)
    _render_findings_section(pdf, findings.findings)
    _render_list_section(pdf, "Strengths", findings.strengths, _EMPTY_STRENGTHS_NOTE)
    _render_list_section(pdf, "Areas of Attention", findings.areas_of_attention, _EMPTY_ATTENTION_NOTE)
    _render_closing_note(pdf)
    return bytes(pdf.output())


def save_report_pdf(
    findings: StructuredFindings, output_path: Path, sample_id: str | None = None
) -> Path:
    """Write `render_report_pdf(findings, sample_id)` to `output_path`.

    Mirrors `grafology_ai.report.generator.save_report`'s exact contract:
    creates `output_path`'s parent directories if they do not already
    exist, writes the PDF bytes, and returns `output_path` (as a `Path`,
    even if a string-like was passed in) for convenient chaining.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(render_report_pdf(findings, sample_id=sample_id))
    return output_path
