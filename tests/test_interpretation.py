"""Tests for grafology_ai.interpretation.interpret.

Fixture images reuse the same deterministic-drawing approach as
``tests/test_analysis.py`` (a grid of short strokes with controllable
slant/width/height/spacing) rather than importing that module's
private helpers directly, so this file's fixtures stay decoupled from
``test_analysis.py``'s internals.
"""

from __future__ import annotations

import math
import re

from PIL import Image, ImageDraw

from grafology_ai.analysis import Features, analyze
from grafology_ai.interpretation import (
    DISALLOWED_TERMS,
    INDICATOR_ORDER,
    Finding,
    StructuredFindings,
    interpret,
)

BACKGROUND = 255
INK = 0


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

    Mirrors ``tests/test_analysis.py``'s ``_draw_stroke_grid``: each
    stroke runs from ``(x, y)`` (bottom) to ``(x + dx, y -
    stroke_height)`` (top), with ``dx = stroke_height * tan(slant_deg)``.
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


def _sample_features() -> Features:
    return analyze(_draw_stroke_grid((700, 900)))


def _blank_features() -> Features:
    return analyze(_blank_image())


def _all_text(result: StructuredFindings) -> list[str]:
    """Every generated text string in a StructuredFindings, for scanning."""
    texts = [result.overall_summary, *result.strengths, *result.areas_of_attention]
    for finding in result.findings:
        texts.append(finding.observation)
        texts.append(finding.interpretation)
    return texts


# --- depth: concise vs indepth -------------------------------------------------


def test_indepth_produces_more_findings_than_concise() -> None:
    features = _sample_features()
    concise = interpret(features, depth="concise")
    indepth = interpret(features, depth="indepth")

    assert len(indepth.findings) > len(concise.findings)


def test_indepth_produces_more_total_text_than_concise() -> None:
    features = _sample_features()
    concise = interpret(features, depth="concise")
    indepth = interpret(features, depth="indepth")

    concise_len = sum(len(t) for t in _all_text(concise))
    indepth_len = sum(len(t) for t in _all_text(indepth))

    assert indepth_len > concise_len


def test_concise_is_not_a_truncated_copy_of_indepth() -> None:
    """Concise findings use genuinely different (shorter) wording, not `text[:N]`."""
    features = _sample_features()
    concise = interpret(features, depth="concise")
    indepth = interpret(features, depth="indepth")

    indepth_by_indicator = {f.indicator: f for f in indepth.findings}
    for finding in concise.findings:
        full_interpretation = indepth_by_indicator[finding.indicator].interpretation
        assert finding.interpretation != full_interpretation
        assert not full_interpretation.startswith(finding.interpretation)


def test_depth_field_matches_requested_depth() -> None:
    features = _sample_features()
    assert interpret(features, depth="concise").depth == "concise"
    assert interpret(features, depth="indepth").depth == "indepth"
    assert interpret(features).depth == "indepth"  # documented default


# --- coverage: one Finding per major Features field group ----------------------


def test_indepth_covers_every_indicator_group() -> None:
    features = _sample_features()
    result = interpret(features, depth="indepth")

    found_indicators = {f.indicator for f in result.findings}
    assert found_indicators == set(INDICATOR_ORDER)

    # Explicit coverage check against the field groups named in the task
    # brief: slant, stroke width/pressure, letter size, line spacing, word
    # spacing, baseline, margins, ink density/rhythm.
    required_groups = {
        "slant": {"slant"},
        "stroke_width_pressure": {"pressure", "pressure_consistency"},
        "letter_size": {"letter_size"},
        "line_spacing": {"line_spacing"},
        "word_spacing": {"word_spacing"},
        "baseline": {"baseline"},
        "margins": {"margins_horizontal", "margins_vertical"},
        "ink_density_rhythm": {"rhythm"},
    }
    for group_name, indicators in required_groups.items():
        assert indicators & found_indicators, f"no Finding for group {group_name!r}"


# --- language discipline -------------------------------------------------------


def _direct_assertion_hits(text: str) -> list[str]:
    """Find "you are"/"you're" as a direct-assertion pattern (case-insensitive)."""
    hits = []
    for pattern in ("you are", "you're"):
        if re.search(re.escape(pattern), text, re.IGNORECASE):
            hits.append(pattern)
    return hits


def test_language_discipline_denylist_scan_indepth() -> None:
    features = _sample_features()
    result = interpret(features, depth="indepth")

    violations = []
    for text in _all_text(result):
        lowered = text.lower()
        for term in DISALLOWED_TERMS:
            if term in lowered:
                violations.append((term, text))
        violations.extend((pattern, text) for pattern in _direct_assertion_hits(text))

    assert violations == [], f"disallowed language found: {violations}"


def test_language_discipline_denylist_scan_concise() -> None:
    features = _sample_features()
    result = interpret(features, depth="concise")

    violations = []
    for text in _all_text(result):
        lowered = text.lower()
        for term in DISALLOWED_TERMS:
            if term in lowered:
                violations.append((term, text))
        violations.extend((pattern, text) for pattern in _direct_assertion_hits(text))

    assert violations == [], f"disallowed language found: {violations}"


def test_language_discipline_denylist_scan_blank_image_both_depths() -> None:
    features = _blank_features()

    for depth in ("concise", "indepth"):
        result = interpret(features, depth=depth)
        violations = []
        for text in _all_text(result):
            lowered = text.lower()
            for term in DISALLOWED_TERMS:
                if term in lowered:
                    violations.append((term, text))
            violations.extend((pattern, text) for pattern in _direct_assertion_hits(text))
        assert violations == [], f"disallowed language found at depth={depth!r}: {violations}"


def test_denylist_covers_documented_examples() -> None:
    """The denylist actually includes the examples the task brief calls out."""
    lowered_denylist = [t.lower() for t in DISALLOWED_TERMS]
    for example in ("disorder", "diagnos", "pathology", "always", "never", "definitely"):
        assert any(example in term or term in example for term in lowered_denylist), example


# --- confidence carries over, never fabricated ----------------------------------


def test_finding_confidence_matches_features_confidence_indepth() -> None:
    features = _sample_features()
    result = interpret(features, depth="indepth")

    from grafology_ai.interpretation.interpret import INDICATOR_CONFIDENCE_KEY

    for finding in result.findings:
        expected_key = INDICATOR_CONFIDENCE_KEY[finding.indicator]
        assert finding.confidence == features.confidence[expected_key]


def test_finding_confidence_matches_features_confidence_concise() -> None:
    features = _sample_features()
    result = interpret(features, depth="concise")

    from grafology_ai.interpretation.interpret import INDICATOR_CONFIDENCE_KEY

    for finding in result.findings:
        expected_key = INDICATOR_CONFIDENCE_KEY[finding.indicator]
        assert finding.confidence == features.confidence[expected_key]


def test_finding_confidence_matches_features_confidence_blank_image() -> None:
    features = _blank_features()
    result = interpret(features, depth="indepth")

    from grafology_ai.interpretation.interpret import INDICATOR_CONFIDENCE_KEY

    for finding in result.findings:
        expected_key = INDICATOR_CONFIDENCE_KEY[finding.indicator]
        assert finding.confidence == features.confidence[expected_key]


# --- blank / low-confidence input does not crash --------------------------------


def test_interpret_blank_image_does_not_crash() -> None:
    features = _blank_features()
    for depth in ("concise", "indepth"):
        result = interpret(features, depth=depth)
        assert isinstance(result, StructuredFindings)


def test_interpret_blank_image_produces_minimal_but_sensible_output() -> None:
    features = _blank_features()
    result = interpret(features, depth="indepth")

    assert len(result.findings) > 0
    for finding in result.findings:
        assert finding.observation
        assert finding.interpretation
        assert finding.confidence == 0.0
    assert result.overall_summary
    # Every indicator is at zero confidence, so nothing should be reported
    # as a confident strength.
    assert result.strengths == []
    # But it should still say *something* useful about the low reliability.
    assert len(result.areas_of_attention) > 0


# --- general structure / API -----------------------------------------------------


def test_structured_findings_are_frozen() -> None:
    import pytest

    features = _sample_features()
    result = interpret(features)
    with pytest.raises(Exception):
        result.depth = "concise"  # type: ignore[misc]


def test_finding_is_frozen() -> None:
    import pytest

    features = _sample_features()
    result = interpret(features)
    with pytest.raises(Exception):
        result.findings[0].confidence = 1.0  # type: ignore[misc]


def test_findings_are_finding_instances() -> None:
    features = _sample_features()
    result = interpret(features)
    for finding in result.findings:
        assert isinstance(finding, Finding)


def test_strengths_and_areas_of_attention_are_string_lists() -> None:
    features = _sample_features()
    result = interpret(features)
    assert all(isinstance(s, str) for s in result.strengths)
    assert all(isinstance(s, str) for s in result.areas_of_attention)


def test_overall_summary_is_a_short_paragraph() -> None:
    features = _sample_features()
    result = interpret(features)
    # Roughly 2-4 sentences: a crude sentence-count proxy via terminal
    # punctuation, generous enough not to be brittle to minor wording
    # changes.
    sentence_count = result.overall_summary.count(". ")
    assert 1 <= sentence_count <= 6
    assert len(result.overall_summary) < 1200
