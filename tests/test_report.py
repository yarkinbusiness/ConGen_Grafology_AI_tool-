"""Tests for grafology_ai.report.generator.

Fixture images reuse the same deterministic-drawing approach as
``tests/test_interpretation.py`` / ``tests/test_analysis.py`` (a grid of
short strokes with controllable slant/width/height/spacing) rather than
importing those modules' private helpers directly, so this file's
fixtures stay decoupled from their internals.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

from grafology_ai.analysis import analyze
from grafology_ai.interpretation import Finding, StructuredFindings, interpret
from grafology_ai.report import (
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    confidence_label,
    generate_report,
    save_report,
)

BACKGROUND = 255
INK = 0

# How many leading characters of a generated report the disclaimer must
# appear within -- "near the top", not just "appears somewhere". Chosen
# generously above the header block's length (title + sample ID +
# depth line + one blank line) while still ruling out a disclaimer stuck
# near the bottom of a multi-thousand-character report.
DISCLAIMER_PROXIMITY_CHARS = 600


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
    """A deterministic grid of short strokes arranged in lines.

    Mirrors ``tests/test_interpretation.py``'s ``_draw_stroke_grid``.
    """
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


REQUIRED_HEADINGS = (
    "## Overall Summary",
    "## Findings",
    "## Strengths",
    "## Areas of Attention",
)

# A phrase common to both the top disclaimer and the closing note, used to
# check the closing note is genuinely present (not just the top one).
DISCLAIMER_CORE_PHRASE = "not a diagnosis"


# --- end-to-end: generate_report(interpret(analyze(...))) at both depths ------


def test_report_structure_indepth() -> None:
    findings = _sample_findings("indepth")
    report = generate_report(findings, sample_id="sample-indepth-001")

    # Disclaimer appears near the top, not just "somewhere".
    assert DISCLAIMER_CORE_PHRASE in report[:DISCLAIMER_PROXIMITY_CHARS]

    for heading in REQUIRED_HEADINGS:
        assert heading in report

    # Every Finding's text appears verbatim.
    for finding in findings.findings:
        assert finding.observation in report
        assert finding.interpretation in report

    # Closing disclaimer note is present.
    assert report.count(DISCLAIMER_CORE_PHRASE) >= 2

    # Header content.
    assert "# Handwriting Analysis Support Report" in report
    assert "sample-indepth-001" in report
    assert "In-depth" in report


def test_report_structure_concise() -> None:
    findings = _sample_findings("concise")
    report = generate_report(findings, sample_id="sample-concise-001")

    assert DISCLAIMER_CORE_PHRASE in report[:DISCLAIMER_PROXIMITY_CHARS]

    for heading in REQUIRED_HEADINGS:
        assert heading in report

    for finding in findings.findings:
        assert finding.observation in report
        assert finding.interpretation in report

    assert report.count(DISCLAIMER_CORE_PHRASE) >= 2
    assert "Concise" in report


def test_report_without_sample_id_omits_it_cleanly() -> None:
    findings = _sample_findings("indepth")
    report = generate_report(findings)

    assert "Sample ID" not in report
    assert "None" not in report


def test_overall_summary_text_appears_verbatim() -> None:
    findings = _sample_findings("indepth")
    report = generate_report(findings)
    assert findings.overall_summary in report


# --- golden-file-style test: fixed, directly-constructed input ----------------


def _golden_findings() -> StructuredFindings:
    return StructuredFindings(
        depth="indepth",
        findings=[
            Finding(
                indicator="slant",
                observation="Slant is measured at +18.0 degrees from vertical, indicating a moderate rightward lean.",
                interpretation="This pattern is often associated with a forward-oriented, sociable communication style.",
                confidence=0.9,
            ),
            Finding(
                indicator="pressure",
                observation="Average stroke width is measured at 5.0px, indicating a moderate touch.",
                interpretation="A middle-of-the-range touch is often associated with a fairly balanced energy level.",
                confidence=0.5,
            ),
            Finding(
                indicator="rhythm",
                observation="Ink density within the writing area is measured at 0.10, indicating a sparser flow of ink.",
                interpretation="A sparser ink flow sometimes points to a more segmented, deliberate communication style.",
                confidence=0.1,
            ),
        ],
        strengths=["Slant: a moderate rightward lean sits within a commonly observed range."],
        areas_of_attention=[
            "Rhythm: only limited data could be measured for this sample, so this is an area that may benefit from closer review."
        ],
        overall_summary=(
            "This in-depth overview draws on a handful of measurable handwriting "
            "features to offer a cautious, descriptive first read of the sample. "
            "This summary is meant to support, not replace, a professional "
            "graphologist's own judgment."
        ),
    )


EXPECTED_GOLDEN_REPORT = """# Handwriting Analysis Support Report
**Sample ID:** golden-sample-001
**Analysis depth:** In-depth

## Important: Please Read Before Using This Report

This report is the output of an automated support tool, generated from a handful of measurable handwriting features (such as slant, stroke weight, spacing, and layout). **It is not a diagnosis, and it is not a clinical or forensic assessment.** It does not determine anything about the writer's character, health, or psychological state. It is intended to assist -- not replace -- a professional graphologist's own judgment, and every finding below should be reviewed alongside the original sample by a qualified graphologist before being relied upon.

## Overall Summary

This in-depth overview draws on a handful of measurable handwriting features to offer a cautious, descriptive first read of the sample. This summary is meant to support, not replace, a professional graphologist's own judgment.

## Findings

### Slant

**Observation:** Slant is measured at +18.0 degrees from vertical, indicating a moderate rightward lean.

**Interpretation:** This pattern is often associated with a forward-oriented, sociable communication style.

**Confidence:** high confidence (90%)

### Pressure (stroke weight)

**Observation:** Average stroke width is measured at 5.0px, indicating a moderate touch.

**Interpretation:** A middle-of-the-range touch is often associated with a fairly balanced energy level.

**Confidence:** medium confidence (50%)

### Rhythm

**Observation:** Ink density within the writing area is measured at 0.10, indicating a sparser flow of ink.

**Interpretation:** A sparser ink flow sometimes points to a more segmented, deliberate communication style.

**Confidence:** low confidence (10%)

## Strengths

- Slant: a moderate rightward lean sits within a commonly observed range.

## Areas of Attention

- Rhythm: only limited data could be measured for this sample, so this is an area that may benefit from closer review.

---

*Reminder: this is a support tool output, not a diagnosis. It is intended to assist -- not replace -- a professional graphologist's own review and judgment.*
"""


def test_golden_report_matches_expected_output_exactly() -> None:
    report = generate_report(_golden_findings(), sample_id="golden-sample-001")
    assert report == EXPECTED_GOLDEN_REPORT


# --- empty strengths / areas_of_attention render cleanly -----------------------


def _findings_with_empty_lists() -> StructuredFindings:
    return StructuredFindings(
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


def test_empty_strengths_and_areas_render_without_broken_markdown() -> None:
    findings = _findings_with_empty_lists()
    report = generate_report(findings)

    assert "## Strengths" in report
    assert "## Areas of Attention" in report

    strengths_idx = report.index("## Strengths")
    areas_idx = report.index("## Areas of Attention")
    strengths_body = report[strengths_idx:areas_idx]

    # No dangling bullet list under a heading with no items.
    assert "- " not in strengths_body
    # No literal "None" leaking into the text.
    assert "None" not in report
    assert "No specific strengths were confidently identified" in strengths_body

    areas_body = report[areas_idx:]
    assert "- " not in areas_body.split("---", 1)[0]
    assert "No specific areas of attention were identified" in areas_body


def test_blank_image_findings_render_without_broken_markdown() -> None:
    """A near-blank sample (few/no strengths, all areas of attention)."""
    findings = _blank_findings("indepth")
    assert findings.strengths == []  # sanity check on the fixture itself

    report = generate_report(findings)
    assert "No specific strengths were confidently identified" in report
    assert "None" not in report
    for heading in REQUIRED_HEADINGS:
        assert heading in report


# --- confidence-label mapping ---------------------------------------------------


def test_confidence_label_thresholds() -> None:
    assert confidence_label(0.9) == "high"
    assert confidence_label(HIGH_CONFIDENCE_THRESHOLD) == "high"
    assert confidence_label(0.5) == "medium"
    assert confidence_label(MEDIUM_CONFIDENCE_THRESHOLD) == "medium"
    assert confidence_label(0.1) == "low"
    assert confidence_label(0.0) == "low"
    assert confidence_label(1.0) == "high"


def test_confidence_labels_appear_near_corresponding_findings_in_report() -> None:
    findings = StructuredFindings(
        depth="indepth",
        findings=[
            Finding(indicator="slant", observation="Obs A.", interpretation="Interp A.", confidence=0.9),
            Finding(indicator="pressure", observation="Obs B.", interpretation="Interp B.", confidence=0.5),
            Finding(indicator="rhythm", observation="Obs C.", interpretation="Interp C.", confidence=0.1),
        ],
        strengths=[],
        areas_of_attention=[],
        overall_summary="Summary.",
    )
    report = generate_report(findings)

    findings_idx = report.index("## Findings")
    strengths_idx = report.index("## Strengths")
    findings_body = report[findings_idx:strengths_idx]

    slant_block = findings_body[findings_body.index("Obs A.") : findings_body.index("Obs B.")]
    pressure_block = findings_body[findings_body.index("Obs B.") : findings_body.index("Obs C.")]
    rhythm_block = findings_body[findings_body.index("Obs C.") :]

    assert "high confidence (90%)" in slant_block
    assert "medium confidence (50%)" in pressure_block
    assert "low confidence (10%)" in rhythm_block


# --- save_report ------------------------------------------------------------------


def test_save_report_writes_same_content_as_generate_report(tmp_path: Path) -> None:
    findings = _sample_findings("indepth")
    output_path = tmp_path / "report.md"

    returned = save_report(findings, output_path, sample_id="sample-xyz")

    assert returned == output_path
    assert output_path.read_text(encoding="utf-8") == generate_report(
        findings, sample_id="sample-xyz"
    )


def test_save_report_creates_parent_directories(tmp_path: Path) -> None:
    findings = _sample_findings("concise")
    output_path = tmp_path / "nested" / "dirs" / "report.md"

    assert not output_path.parent.exists()
    returned = save_report(findings, output_path)

    assert output_path.exists()
    assert returned == output_path
    assert output_path.read_text(encoding="utf-8") == generate_report(findings)
