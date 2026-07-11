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
from grafology_ai.interpretation.interpret import (
    CONCISE_INDICATOR_COUNT,
    INDICATOR_CONFIDENCE_KEY,
    ORGANIZATION_LOOSE_MAX,
    ORGANIZATION_PLANNED_MIN,
    RHYTHM_IRREGULAR_MAX,
    RHYTHM_REGULAR_MIN,
    STROKE_CONNECTEDNESS_BROKEN_MAX,
    STROKE_CONNECTEDNESS_CONNECTED_MIN,
    _candidate_organization,
    _candidate_rhythm,
    _candidate_stroke_continuity,
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


def _make_features(*, confidence: dict[str, float] | None = None, **field_overrides: float) -> Features:
    """Directly construct a `Features` record landing in a chosen indicator bucket.

    Mirrors ``tests/test_evaluation.py::_make_features``'s direct-construction
    pattern: every field gets a plausible filler default (confidence fixed
    at 1.0 for every field unless overridden via ``confidence``), so a test
    can override just the one field it cares about (e.g.
    ``rhythm_regularity=0.1``) to land a candidate builder in a specific
    bucket without needing to render and analyze an actual image.
    """
    base_confidence = {key: 1.0 for key in Features.__dataclass_fields__ if key != "confidence"}
    if confidence:
        base_confidence.update(confidence)
    defaults: dict[str, float] = dict(
        slant_angle_degrees=0.0,
        stroke_width_mean=5.0,
        stroke_width_std=1.0,
        letter_size_estimate=15.0,
        line_spacing_mean=30.0,
        word_spacing_mean=20.0,
        baseline_slope_degrees=0.0,
        margin_left_px=40.0,
        margin_right_px=40.0,
        margin_top_px=40.0,
        margin_bottom_px=40.0,
        ink_density=0.2,
        rhythm_regularity=0.5,
        stroke_connectedness=0.5,
        organization_score=0.5,
    )
    defaults.update(field_overrides)
    return Features(confidence=base_confidence, **defaults)


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
        "rhythm": {"rhythm"},
        "stroke_continuity": {"stroke_continuity"},
        "organization": {"organization"},
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


# --- Phase A4: 12-indicator shape (rhythm re-keyed, stroke_continuity/organization new) -


def test_indepth_returns_exactly_12_findings_one_per_indicator_order() -> None:
    features = _sample_features()
    result = interpret(features, depth="indepth")

    assert len(INDICATOR_ORDER) == 12
    assert len(result.findings) == 12
    assert [f.indicator for f in result.findings] == list(INDICATOR_ORDER)
    assert "stroke_continuity" in {f.indicator for f in result.findings}
    assert "organization" in {f.indicator for f in result.findings}


def test_concise_still_returns_exactly_4_findings() -> None:
    features = _sample_features()
    result = interpret(features, depth="concise")

    assert CONCISE_INDICATOR_COUNT == 4
    assert len(result.findings) == 4


# --- Phase A4: _candidate_rhythm rewritten around rhythm_regularity ------------------


def test_candidate_rhythm_irregular_bucket() -> None:
    features = _make_features(rhythm_regularity=RHYTHM_IRREGULAR_MAX, confidence={"rhythm_regularity": 0.42})
    candidate = _candidate_rhythm(features)

    assert candidate.indicator == "rhythm"
    assert candidate.tag == "notable"
    assert "irregular" in candidate.descriptor
    assert f"{RHYTHM_IRREGULAR_MAX:.2f}" in candidate.observation
    assert "variable, shifting pace" in candidate.interpretation_full
    assert "variable, shifting pace" in candidate.interpretation_brief
    assert candidate.interpretation_brief != candidate.interpretation_full
    assert candidate.confidence == 0.42


def test_candidate_rhythm_moderate_bucket() -> None:
    midpoint = (RHYTHM_IRREGULAR_MAX + RHYTHM_REGULAR_MIN) / 2.0
    features = _make_features(rhythm_regularity=midpoint, confidence={"rhythm_regularity": 0.55})
    candidate = _candidate_rhythm(features)

    assert candidate.tag == "typical"
    assert "moderately regular" in candidate.descriptor
    assert "fairly typical, adaptable pace" in candidate.interpretation_full
    assert candidate.confidence == 0.55


def test_candidate_rhythm_regular_bucket() -> None:
    features = _make_features(rhythm_regularity=RHYTHM_REGULAR_MIN, confidence={"rhythm_regularity": 0.9})
    candidate = _candidate_rhythm(features)

    assert candidate.tag == "typical"
    assert "regular, evenly repeating rhythm" in candidate.descriptor
    assert "steady, practiced pace" in candidate.interpretation_full
    assert candidate.confidence == 0.9


# --- Phase A4: new "stroke_continuity" indicator, keyed to stroke_connectedness -----


def test_candidate_stroke_continuity_broken_bucket() -> None:
    features = _make_features(
        stroke_connectedness=STROKE_CONNECTEDNESS_BROKEN_MAX, confidence={"stroke_connectedness": 0.2}
    )
    candidate = _candidate_stroke_continuity(features)

    assert candidate.indicator == "stroke_continuity"
    assert candidate.tag == "notable"
    assert "broken or segmented" in candidate.descriptor
    assert f"{STROKE_CONNECTEDNESS_BROKEN_MAX:.2f}" in candidate.observation
    assert "deliberate, step-by-step style" in candidate.interpretation_full
    assert "deliberate, step-by-step style" in candidate.interpretation_brief
    assert candidate.interpretation_brief != candidate.interpretation_full
    assert candidate.confidence == 0.2


def test_candidate_stroke_continuity_moderate_bucket() -> None:
    midpoint = (STROKE_CONNECTEDNESS_BROKEN_MAX + STROKE_CONNECTEDNESS_CONNECTED_MIN) / 2.0
    features = _make_features(stroke_connectedness=midpoint, confidence={"stroke_connectedness": 0.6})
    candidate = _candidate_stroke_continuity(features)

    assert candidate.tag == "typical"
    assert "moderate mix of joined and broken" in candidate.descriptor
    assert "fairly adaptable style of expression" in candidate.interpretation_full
    assert candidate.confidence == 0.6


def test_candidate_stroke_continuity_connected_bucket() -> None:
    features = _make_features(
        stroke_connectedness=STROKE_CONNECTEDNESS_CONNECTED_MIN, confidence={"stroke_connectedness": 0.95}
    )
    candidate = _candidate_stroke_continuity(features)

    assert candidate.tag == "typical"
    assert "mostly joined and continuous" in candidate.descriptor
    assert "fluid flow from one thought to the next" in candidate.interpretation_full
    assert candidate.confidence == 0.95


# --- Phase A4: new "organization" indicator, keyed to organization_score ------------


def test_candidate_organization_loose_bucket() -> None:
    features = _make_features(
        organization_score=ORGANIZATION_LOOSE_MAX, confidence={"organization_score": 0.15}
    )
    candidate = _candidate_organization(features)

    assert candidate.indicator == "organization"
    assert candidate.tag == "notable"
    assert "loosely organized" in candidate.descriptor
    assert f"{ORGANIZATION_LOOSE_MAX:.2f}" in candidate.observation
    assert "spontaneous, in-the-moment approach" in candidate.interpretation_full
    assert "spontaneous approach" in candidate.interpretation_brief
    assert candidate.interpretation_brief != candidate.interpretation_full
    assert candidate.confidence == 0.15


def test_candidate_organization_moderate_bucket() -> None:
    midpoint = (ORGANIZATION_LOOSE_MAX + ORGANIZATION_PLANNED_MIN) / 2.0
    features = _make_features(organization_score=midpoint, confidence={"organization_score": 0.5})
    candidate = _candidate_organization(features)

    assert candidate.tag == "typical"
    assert "moderately organized" in candidate.descriptor
    assert "fairly typical, adaptable approach" in candidate.interpretation_full
    assert candidate.confidence == 0.5


def test_candidate_organization_planned_bucket() -> None:
    features = _make_features(
        organization_score=ORGANIZATION_PLANNED_MIN, confidence={"organization_score": 0.8}
    )
    candidate = _candidate_organization(features)

    assert candidate.tag == "typical"
    assert "consistently organized layout" in candidate.descriptor
    assert "planned, methodical approach" in candidate.interpretation_full
    assert candidate.confidence == 0.8


# --- Phase A4: Finding.confidence is a verbatim carry-over for the new indicators ----


def test_new_indicator_confidence_is_verbatim_features_confidence() -> None:
    features = _make_features(
        rhythm_regularity=0.1,
        stroke_connectedness=0.9,
        organization_score=0.5,
        confidence={
            "rhythm_regularity": 0.33,
            "stroke_connectedness": 0.77,
            "organization_score": 0.11,
        },
    )
    result = interpret(features, depth="indepth")
    findings_by_indicator = {f.indicator: f for f in result.findings}

    assert findings_by_indicator["rhythm"].confidence == features.confidence["rhythm_regularity"] == 0.33
    assert (
        findings_by_indicator["stroke_continuity"].confidence
        == features.confidence["stroke_connectedness"]
        == 0.77
    )
    assert (
        findings_by_indicator["organization"].confidence
        == features.confidence["organization_score"]
        == 0.11
    )
    assert INDICATOR_CONFIDENCE_KEY["rhythm"] == "rhythm_regularity"
    assert INDICATOR_CONFIDENCE_KEY["stroke_continuity"] == "stroke_connectedness"
    assert INDICATOR_CONFIDENCE_KEY["organization"] == "organization_score"


# --- Phase A4: language discipline scan extended to hit every new/changed bucket ----


def test_language_discipline_scan_covers_new_indicator_buckets() -> None:
    """Extend the denylist scan with inputs that hit every new/changed bucket.

    The default `_sample_features()`/`_blank_features()` fixtures used by
    the existing scan tests don't exercise most of these buckets (a drawn
    stroke grid lands in one bucket per indicator, and a blank image
    always reads as the lowest bucket), so this test builds a matrix of
    directly-constructed `Features` records that each land in a different
    bucket of the three new/rewritten builders.
    """
    bucket_values = [
        RHYTHM_IRREGULAR_MAX,
        (RHYTHM_IRREGULAR_MAX + RHYTHM_REGULAR_MIN) / 2.0,
        RHYTHM_REGULAR_MIN,
    ]
    candidate_matrix = [
        _make_features(rhythm_regularity=v, stroke_connectedness=v, organization_score=v)
        for v in bucket_values
    ]

    violations: list[tuple[str, str]] = []
    for features in candidate_matrix:
        for depth in ("concise", "indepth"):
            result = interpret(features, depth=depth)
            for text in _all_text(result):
                lowered = text.lower()
                for term in DISALLOWED_TERMS:
                    if term in lowered:
                        violations.append((term, text))
                violations.extend((pattern, text) for pattern in _direct_assertion_hits(text))

    assert violations == [], f"disallowed language found: {violations}"
