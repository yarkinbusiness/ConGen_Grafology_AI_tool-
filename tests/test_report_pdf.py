"""Tests for grafology_ai.report.pdf.

Fixture images reuse the same deterministic-drawing approach as
``tests/test_report.py`` / ``tests/test_interpretation.py`` (a grid of
short strokes with controllable slant/width/height/spacing) rather than
importing that file's private helpers directly, so this file's fixtures
stay decoupled from their internals -- matching the convention already
used across this test suite (see e.g. ``tests/test_pdf_input.py``'s own
docstring).

Text is extracted from the rendered PDF bytes with ``pypdfium2`` (already
a project dependency from Phase B, and already used this way in spirit
by ``tests/test_pdf_input.py``, though that file only rasterizes pages --
here we pull text via ``PdfPage.get_textpage()``).
"""

from __future__ import annotations

import math
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from grafology_ai.analysis import analyze
from grafology_ai.interpretation import INDICATOR_LABELS, Finding, StructuredFindings, interpret
from grafology_ai.report import render_report_pdf, save_report_pdf
from grafology_ai.report.generator import _EMPTY_ATTENTION_NOTE, _EMPTY_STRENGTHS_NOTE

BACKGROUND = 255
INK = 0

DISCLAIMER_CORE_PHRASE = "not a diagnosis"


def _draw_stroke_grid(
    size: tuple[int, int],
    *,
    slant_deg: float = 12.0,
    stroke_width: int = 5,
    stroke_height: int = 16,
    stroke_spacing: int = 14,
    line_spacing: int = 40,
    margin: int = 30,
) -> Image.Image:
    """Mirrors ``tests/test_report.py``'s ``_draw_stroke_grid``."""
    image = Image.new("L", size, color=BACKGROUND)
    draw = ImageDraw.Draw(image)
    dx = stroke_height * math.tan(math.radians(slant_deg))

    y = margin + stroke_height
    while y <= size[1] - margin:
        x = margin
        while x <= size[0] - margin:
            draw.line([(x, y), (x + dx, y - stroke_height)], fill=INK, width=stroke_width)
            x += stroke_spacing
        y += line_spacing
    return image.convert("RGB")


def _blank_image(size: tuple[int, int] = (300, 300)) -> Image.Image:
    return Image.new("RGB", size, color=(255, 255, 255))


def _sample_findings(depth: str) -> StructuredFindings:
    features = analyze(_draw_stroke_grid((700, 900)))
    return interpret(features, depth=depth)


def _blank_findings(depth: str = "indepth") -> StructuredFindings:
    features = analyze(_blank_image())
    return interpret(features, depth=depth)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Full text of every page of `pdf_bytes`, concatenated in page order.

    fpdf2's `multi_cell` word-wraps long paragraphs across several visual
    lines, and `pypdfium2` extraction reports each wrapped line separated
    by a line break (`"\\r\\n"`) rather than the single space that
    originally separated those words in the source text -- a text-layout
    artifact of PDF (a fixed-position page format with no native concept
    of "paragraph reflow") that has no equivalent in plain text/markdown
    output. Collapsing all whitespace runs to a single space here
    recovers the original word sequence for substring comparisons against
    `StructuredFindings` text, without discarding or reordering any
    actual content.
    """
    document = pdfium.PdfDocument(pdf_bytes)
    try:
        parts = []
        for index in range(len(document)):
            page = document[index]
            try:
                textpage = page.get_textpage()
                try:
                    parts.append(textpage.get_text_range())
                finally:
                    textpage.close()
            finally:
                page.close()
        raw_text = "\n".join(parts)
        return " ".join(raw_text.split())
    finally:
        document.close()


# --- 1: %PDF magic bytes -----------------------------------------------------


def test_render_report_pdf_starts_with_pdf_magic_bytes() -> None:
    findings = _sample_findings("indepth")
    pdf_bytes = render_report_pdf(findings)
    assert pdf_bytes.startswith(b"%PDF")


# --- 2: all required content appears, verbatim --------------------------------


def test_extracted_text_contains_all_required_content_indepth() -> None:
    findings = _sample_findings("indepth")
    pdf_bytes = render_report_pdf(findings, sample_id="sample-indepth-001")
    text = _extract_pdf_text(pdf_bytes)

    assert DISCLAIMER_CORE_PHRASE in text
    assert findings.overall_summary in text

    for finding in findings.findings:
        label = INDICATOR_LABELS[finding.indicator]
        assert label in text
        assert finding.observation in text
        assert finding.interpretation in text

    for strength in findings.strengths:
        assert strength in text
    for area in findings.areas_of_attention:
        assert area in text


# --- 3: seven fixed sections appear in the correct order -----------------------


def test_section_headings_appear_in_correct_fixed_order() -> None:
    findings = _sample_findings("indepth")
    pdf_bytes = render_report_pdf(findings, sample_id="sample-order-001")
    text = _extract_pdf_text(pdf_bytes)

    header_idx = text.index("Handwriting Analysis Support Report")
    disclaimer_idx = text.index("Please Read Before Using This Report")
    summary_idx = text.index("Overall Summary")
    findings_idx = text.index("Findings")
    strengths_idx = text.index("Strengths")
    attention_idx = text.index("Areas of Attention")
    closing_idx = text.index("Reminder: this is a support tool output")

    indices = [
        header_idx,
        disclaimer_idx,
        summary_idx,
        findings_idx,
        strengths_idx,
        attention_idx,
        closing_idx,
    ]
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))


# --- 4: empty strengths / areas_of_attention render the shared notes -----------


def test_blank_image_findings_render_empty_strengths_note() -> None:
    """A near-blank sample has no strengths (see `interpret()`'s docstring:
    a low-confidence indicator is reported as an area of attention instead,
    regardless of its bucket -- so this fixture is *not* expected to also
    have an empty `areas_of_attention`; see
    `test_directly_constructed_empty_lists_render_shared_notes` below for
    that case), matching `tests/test_report.py`'s equivalent markdown test.
    """
    findings = _blank_findings("indepth")
    assert findings.strengths == []  # sanity check on the fixture itself

    pdf_bytes = render_report_pdf(findings)
    text = _extract_pdf_text(pdf_bytes)

    assert _EMPTY_STRENGTHS_NOTE in text
    assert "None" not in text


def test_directly_constructed_empty_lists_render_shared_notes() -> None:
    findings = StructuredFindings(
        depth="concise",
        findings=[
            Finding(
                indicator="baseline",
                observation="Baseline trend is measured at +0.0 degrees, indicating a fairly steady baseline.",
                interpretation="A steady baseline is often associated with a fairly consistent mood and energy level while writing.",
                confidence=0.6,
            ),
        ],
        strengths=[],
        areas_of_attention=[],
        overall_summary="A short, cautious summary of this sample.",
    )
    pdf_bytes = render_report_pdf(findings)
    text = _extract_pdf_text(pdf_bytes)

    assert _EMPTY_STRENGTHS_NOTE in text
    assert _EMPTY_ATTENTION_NOTE in text


# --- 5: both depths work; concise renders only the concise subset --------------


def test_concise_depth_renders_only_the_concise_findings_subset() -> None:
    concise = _sample_findings("concise")
    indepth = _sample_findings("indepth")

    concise_indicators = {f.indicator for f in concise.findings}
    indepth_only_indicators = {f.indicator for f in indepth.findings} - concise_indicators
    assert indepth_only_indicators, "fixture should exercise a genuine depth difference"

    pdf_bytes = render_report_pdf(concise)
    text = _extract_pdf_text(pdf_bytes)

    findings_start = text.index("Findings")
    strengths_start = text.index("Strengths")
    findings_body = text[findings_start:strengths_start]

    for indicator in concise_indicators:
        assert INDICATOR_LABELS[indicator] in findings_body

    for indicator in indepth_only_indicators:
        assert INDICATOR_LABELS[indicator] not in findings_body


def test_indepth_depth_renders_every_indicator() -> None:
    findings = _sample_findings("indepth")
    from grafology_ai.interpretation import INDICATOR_ORDER

    assert {f.indicator for f in findings.findings} == set(INDICATOR_ORDER)

    pdf_bytes = render_report_pdf(findings)
    text = _extract_pdf_text(pdf_bytes)
    findings_start = text.index("Findings")
    strengths_start = text.index("Strengths")
    findings_body = text[findings_start:strengths_start]

    for indicator in INDICATOR_ORDER:
        assert INDICATOR_LABELS[indicator] in findings_body


# --- 6: sample_id handling; no literal "None" -----------------------------------


def test_sample_id_appears_when_given() -> None:
    findings = _sample_findings("indepth")
    pdf_bytes = render_report_pdf(findings, sample_id="sample-xyz-42")
    text = _extract_pdf_text(pdf_bytes)
    assert "sample-xyz-42" in text


def test_no_literal_none_when_sample_id_omitted() -> None:
    findings = _sample_findings("indepth")
    pdf_bytes = render_report_pdf(findings)
    text = _extract_pdf_text(pdf_bytes)
    assert "None" not in text
    assert "Sample ID" not in text


# --- 7: special-character round-trip (regression) -------------------------------


def test_degree_value_survives_pdf_round_trip() -> None:
    """A rendered numeric degree value (e.g. from the slant finding's
    observation text, `"+18.0 degrees"`-style) must survive extraction intact.
    """
    findings = StructuredFindings(
        depth="indepth",
        findings=[
            Finding(
                indicator="slant",
                observation="Slant is measured at +18.0 degrees from vertical, indicating a moderate rightward lean.",
                interpretation="This pattern is often associated with a forward-oriented, sociable communication style.",
                confidence=0.9,
            ),
        ],
        strengths=[],
        areas_of_attention=[],
        overall_summary="A short, cautious summary of this sample.",
    )
    pdf_bytes = render_report_pdf(findings)
    text = _extract_pdf_text(pdf_bytes)
    assert "+18.0 degrees" in text


def test_em_dash_style_double_hyphen_survives_pdf_round_trip() -> None:
    """Regression test: fpdf2's built-in ``markdown=True`` cell rendering
    treats a literal ``"--"`` as an *underline toggle*
    (``FPDF.MARKDOWN_UNDERLINE_MARKER``). This codebase's generated text
    uses ``" -- "`` throughout as plain em-dash-style punctuation (see
    ``grafology_ai.report.generator._DISCLAIMER_TEXT`` and most of
    ``grafology_ai.interpretation.interpret``'s narrative text) -- never
    as an underline instruction. Confirmed directly: without disabling
    that marker (see ``grafology_ai.report.pdf._ReportPDF``), a string
    like ``"assist -- not replace --"`` gets silently mangled to
    ``"assist not replace"`` by fpdf2's default markdown parser. This
    test locks in that the fix holds for both the module's own
    boilerplate text (the disclaimer, which is rendered with
    ``markdown=True``) and for a `StructuredFindings` interpretation
    string built the same way real narrative text is (containing a
    ``" -- "`` pair).
    """
    findings = StructuredFindings(
        depth="indepth",
        findings=[
            Finding(
                indicator="rhythm",
                observation="Rhythm regularity is measured at 0.50.",
                interpretation=(
                    "A moderate mix of joined and broken strokes -- neither markedly "
                    "continuous nor markedly segmented -- often points to a fairly "
                    "adaptable style of expression."
                ),
                confidence=0.5,
            ),
        ],
        strengths=[],
        areas_of_attention=[],
        overall_summary="A short, cautious summary of this sample.",
    )
    pdf_bytes = render_report_pdf(findings)
    text = _extract_pdf_text(pdf_bytes)

    # The disclaimer/closing-note boilerplate (rendered with markdown=True,
    # via `_ReportPDF`'s disabled underline marker).
    assert "assist -- not replace -- a professional graphologist" in text

    # A `Finding.interpretation` string (narrative text, rendered with
    # `write()`, which never interprets markdown at all).
    assert (
        "A moderate mix of joined and broken strokes -- neither markedly "
        "continuous nor markedly segmented -- often points to a fairly "
        "adaptable style of expression."
    ) in text


# --- 8: save_report_pdf mirrors save_report's contract --------------------------


def test_save_report_pdf_creates_parent_directories_and_returns_path(tmp_path: Path) -> None:
    findings = _sample_findings("concise")
    output_path = tmp_path / "nested" / "dirs" / "report.pdf"

    assert not output_path.parent.exists()
    returned = save_report_pdf(findings, output_path)

    assert isinstance(returned, Path)
    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"%PDF")


def test_save_report_pdf_writes_same_content_as_render_report_pdf(tmp_path: Path) -> None:
    """`save_report_pdf` writes the same *content* `render_report_pdf` would
    produce from the same `findings`/`sample_id` -- checked via extracted
    text (not raw byte equality: fpdf2 embeds a wall-clock creation
    timestamp in every PDF's metadata by default, so two separate render
    calls for identical input are not guaranteed to be byte-identical,
    even though their visible/extractable content is).
    """
    findings = _sample_findings("indepth")
    output_path = tmp_path / "report.pdf"

    returned = save_report_pdf(findings, output_path, sample_id="sample-save-001")

    assert returned == output_path
    written_text = _extract_pdf_text(output_path.read_bytes())
    rendered_text = _extract_pdf_text(render_report_pdf(findings, sample_id="sample-save-001"))
    assert written_text == rendered_text


def test_save_report_pdf_accepts_string_path(tmp_path: Path) -> None:
    findings = _sample_findings("concise")
    output_path = tmp_path / "report.pdf"

    returned = save_report_pdf(findings, str(output_path))
    assert isinstance(returned, Path)
    assert returned == output_path
    assert output_path.exists()


# --- structural independence: never builds a markdown string internally --------


def test_pdf_module_does_not_import_generate_report() -> None:
    """`render_report_pdf` must derive its output directly from
    `StructuredFindings`, never from `generate_report`'s markdown string
    (see `grafology_ai.report.pdf`'s module docstring) -- confirmed here by
    checking the module never even imports `generate_report`.
    """
    import grafology_ai.report.pdf as pdf_module

    assert not hasattr(pdf_module, "generate_report")


def test_render_report_pdf_unaffected_by_generate_report_being_broken(monkeypatch) -> None:
    """Further confirmation of structural independence: even if
    `generate_report` were broken, `render_report_pdf` must still work,
    since it never calls it.
    """
    import grafology_ai.report.generator as generator_module

    def _boom(*args, **kwargs):
        raise AssertionError("generate_report must not be called by render_report_pdf")

    monkeypatch.setattr(generator_module, "generate_report", _boom)

    findings = _sample_findings("indepth")
    pdf_bytes = render_report_pdf(findings)
    assert pdf_bytes.startswith(b"%PDF")
