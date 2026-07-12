"""Interpretation layer: turn raw :class:`~grafology_ai.analysis.Features`
measurements into cautious, human-readable findings.

:mod:`grafology_ai.analysis.features` deliberately stops at *measurement*
-- it reports numbers, not judgments (see its module docstring). This
module is the next stage: it maps those numbers onto the indicator
vocabulary documented in ``docs/labeling_rubric.md`` (slant, pressure,
letter size, spacing, baseline movement, margins, rhythm, stroke
continuity, overall organization) and attaches short, hedged, descriptive
narrative text to each one.

This is explicitly *not* a diagnostic layer. Every piece of generated
text is written to the same standard ``docs/labeling_rubric.md`` sets for
its own labels: cautious and descriptive rather than diagnostic, phrased
about the *writing* rather than as a direct claim about the person, and
never treated as conclusive on its own. :data:`DISALLOWED_TERMS`
documents the concrete denylist this module's output is written to avoid
(diagnostic/clinical labels, absolute claims, evaluative "weakness"/
"problem" framing, and direct "you are ..." assertions) -- see
``tests/test_interpretation.py`` for the automated scan that enforces it.

Output shape
------------

:func:`interpret` returns a :class:`StructuredFindings`: a list of
per-indicator :class:`Finding` records, plus a ``strengths`` /
``areas_of_attention`` breakdown and an ``overall_summary`` paragraph
synthesizing the sample -- the shape the project brief calls for (see
``README.md``: "structured markdown reports intended to assist expert
graphologists ... a support tool, not a diagnostic tool"). Two depths are
supported (:data:`Depth`): ``"indepth"`` covers every measured indicator;
``"concise"`` surfaces a short, higher-confidence-first subset with more
condensed narrative text, for callers that want a quick read rather than
a full breakdown.

Downstream, a later report-generation stage is expected to render a
:class:`StructuredFindings` into the markdown report described in
``README.md``; this module only builds the structured, cautious content
that stage will format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from grafology_ai.analysis import Features

Depth = Literal["concise", "indepth"]


# --- language discipline ----------------------------------------------------

#: Denylist of diagnostic/clinical labels, absolute-claim words, and
#: evaluative "weakness"/"problem" framing that must never appear in any
#: generated finding, strength, area-of-attention, or summary text (see
#: the module docstring and ``docs/labeling_rubric.md``'s "cautious and
#: descriptive rather than diagnostic" framing). Matched as
#: case-insensitive substrings, so e.g. ``"diagnos"`` also catches
#: "diagnosis"/"diagnostic"/"diagnose", and ``"patholog"`` catches
#: "pathology"/"pathological". This module's own generated text is
#: written to avoid every one of these terms; ``tests/test_interpretation.py``
#: scans the full generated output against this exact list.
DISALLOWED_TERMS: tuple[str, ...] = (
    # Diagnostic / clinical labels -- this is a descriptive support tool,
    # not a diagnostic or medical one (see README.md).
    "disorder",
    "diagnos",
    "patholog",
    "syndrome",
    "illness",
    "disease",
    "abnormal",
    # Absolute claims -- every finding here is a hedged, probabilistic
    # reading, never a certainty.
    "always",
    "never",
    "definitely",
    "certainly",
    "guarantee",
    "prove",
    # Evaluative "weakness"/"problem" framing -- areas of attention are
    # framed as "may benefit from closer review", never as a deficiency.
    "weakness",
    "problem",
    "flaw",
    "deficient",
    # Direct-assertion-about-the-person phrasing -- findings are phrased
    # about the writing/sample, never asserted directly at the reader.
    "you are",
    "you're",
)


def _assert_language_is_disciplined(text: str) -> None:
    """Raise ``ValueError`` if ``text`` contains a :data:`DISALLOWED_TERMS` hit.

    Defense in depth alongside the test-suite scan: every piece of text
    this module builds is a fixed template, so this should never fire in
    practice, but it makes a future accidental template edit fail loudly
    rather than silently shipping disallowed language.
    """
    lowered = text.lower()
    for term in DISALLOWED_TERMS:
        if term in lowered:
            raise ValueError(
                f"Generated interpretation text contains disallowed term {term!r}: {text!r}"
            )


# --- public data model --------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One interpreted observation about a single measured indicator.

    Attributes:
        indicator: Short slug for which measured feature this finding is
            about (e.g. ``"slant"``, ``"pressure"``) -- see
            :data:`INDICATOR_LABELS` for the human-readable form and
            :data:`INDICATOR_CONFIDENCE_KEY` for the
            :class:`~grafology_ai.analysis.Features` field it is derived
            from.
        observation: A plain, factual description of what was measured
            (e.g. "Slant is measured at +18.0 degrees from vertical,
            indicating a moderate rightward lean."). No interpretation or
            judgment language.
        interpretation: The cautious, hedged narrative text connecting
            the observation to a possible personality/relational/
            communication/emotional/energy/organizational theme (see the
            module docstring). Always non-diagnostic and non-absolute --
            see :data:`DISALLOWED_TERMS`.
        confidence: Carried over verbatim from the corresponding
            :attr:`~grafology_ai.analysis.Features.confidence` entry --
            never fabricated or derived independently.
    """

    indicator: str
    observation: str
    interpretation: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-safe ``dict`` of every field, by name."""
        return {
            "indicator": self.indicator,
            "observation": self.observation,
            "interpretation": self.interpretation,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class StructuredFindings:
    """The full interpreted output for one handwriting sample.

    Attributes:
        depth: Which :data:`Depth` this was built at.
        findings: One :class:`Finding` per covered indicator. At
            ``depth="indepth"`` this covers every indicator in
            :data:`INDICATOR_ORDER`; at ``depth="concise"`` it is a
            shorter, higher-confidence-first subset
            (:data:`CONCISE_INDICATOR_COUNT` of them) with more condensed
            narrative text.
        strengths: Short, hedged descriptions of indicators whose
            reading fell in a commonly observed range with reasonable
            confidence -- phrased descriptively, never as praise or a
            score.
        areas_of_attention: Short, hedged descriptions of indicators
            that were either measured with limited confidence or whose
            reading was comparatively pronounced/atypical -- phrased as
            "may benefit from closer review", never as a deficiency (see
            :data:`DISALLOWED_TERMS`).
        overall_summary: A short (2-4 sentence) synthesis of the sample.
            Always includes an explicit reminder that this is a
            descriptive support reading, not a determination about the
            writer, intended to support -- not replace -- a professional
            graphologist's own judgment.
    """

    depth: Depth
    findings: list[Finding]
    strengths: list[str]
    areas_of_attention: list[str]
    overall_summary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-safe ``dict`` of every field, by name.

        ``depth`` is a ``Literal["concise", "indepth"]``, already a plain
        ``str`` at runtime. ``findings`` is converted to a list of plain
        dicts via :meth:`Finding.to_dict`, not left as a list of
        :class:`Finding` instances.
        """
        return {
            "depth": self.depth,
            "findings": [finding.to_dict() for finding in self.findings],
            "strengths": list(self.strengths),
            "areas_of_attention": list(self.areas_of_attention),
            "overall_summary": self.overall_summary,
        }


# --- indicator identity: names, labels, and Features confidence mapping -----

#: Fixed, canonical order indicators are built and (at ``depth="indepth"``)
#: reported in. Also used as the tie-break order when :func:`interpret`
#: selects the top :data:`CONCISE_INDICATOR_COUNT` indicators by
#: confidence for ``depth="concise"`` (Python's sort is stable, so equal
#: confidences -- e.g. an all-zero-confidence blank image -- fall back to
#: this order), which keeps that selection deterministic.
INDICATOR_ORDER: tuple[str, ...] = (
    "slant",
    "pressure",
    "pressure_consistency",
    "letter_size",
    "line_spacing",
    "word_spacing",
    "baseline",
    "margins_horizontal",
    "margins_vertical",
    "rhythm",
    "stroke_continuity",
    "organization",
)

#: Human-readable label for each indicator, used when rendering
#: ``strengths``/``areas_of_attention`` text.
INDICATOR_LABELS: dict[str, str] = {
    "slant": "Slant",
    "pressure": "Pressure (stroke weight)",
    "pressure_consistency": "Pressure consistency",
    "letter_size": "Letter size",
    "line_spacing": "Line spacing",
    "word_spacing": "Word spacing",
    "baseline": "Baseline steadiness",
    "margins_horizontal": "Left/right margin balance",
    "margins_vertical": "Top/bottom margin balance",
    "rhythm": "Rhythm",
    "stroke_continuity": "Stroke continuity",
    "organization": "Overall organization",
}

#: Which single :class:`~grafology_ai.analysis.Features` confidence-dict
#: key each indicator's :attr:`Finding.confidence` is read from. Margins
#: are split into a horizontal indicator (left/right balance, keyed off
#: ``margin_left_px``'s confidence) and a vertical one (top/bottom
#: balance, keyed off ``margin_top_px``'s confidence) rather than one
#: combined "margins" indicator, so every indicator maps to exactly one
#: real, unaveraged :class:`~grafology_ai.analysis.Features` confidence
#: value -- this is what lets :attr:`Finding.confidence` be a verbatim
#: carry-over rather than a fabricated combination (in practice all four
#: margin confidences are equal anyway, see
#: ``Features._build_confidence``'s geometric-confidence flat split).
INDICATOR_CONFIDENCE_KEY: dict[str, str] = {
    "slant": "slant_angle_degrees",
    "pressure": "stroke_width_mean",
    "pressure_consistency": "stroke_width_std",
    "letter_size": "letter_size_estimate",
    "line_spacing": "line_spacing_mean",
    "word_spacing": "word_spacing_mean",
    "baseline": "baseline_slope_degrees",
    "margins_horizontal": "margin_left_px",
    "margins_vertical": "margin_top_px",
    "rhythm": "rhythm_regularity",
    "stroke_continuity": "stroke_connectedness",
    "organization": "organization_score",
}


# --- thresholds ---------------------------------------------------------------
#
# Every cutoff below is a deliberately chosen, documented constant rather
# than an inline magic number (matching the convention in
# `grafology_ai.analysis.features`). These are reasonable illustrative
# cutoffs, not values derived from a labeled dataset -- no such dataset
# exists yet (see `grafology_ai.dataset.fixtures` and this module's
# docstring) -- and are the first place to revisit once real labeled data
# is available to calibrate against.

#: Slant: |angle| at or below this reads as upright/vertical ("neutral").
#: Above this and up to `SLANT_MODERATE_MAX_DEGREES` reads as a moderate
#: lean; beyond that, a pronounced lean. Mirrors the rubric's plain
#: "right slant" / "left slant or vertical" distinction
#: (docs/labeling_rubric.md).
SLANT_NEUTRAL_MAX_DEGREES = 5.0
SLANT_MODERATE_MAX_DEGREES = 20.0

#: Pressure (stroke-width proxy, pixels): mean stroke width at or below
#: this reads as "light"; at or above `PRESSURE_FIRM_MIN_PX` reads as
#: "firm"; in between is "medium". Pixel-absolute, so implicitly assumes
#: a roughly consistent intake scan resolution across samples -- a
#: DPI-normalized version of this measurement is future work.
PRESSURE_LIGHT_MAX_PX = 3.0
PRESSURE_FIRM_MIN_PX = 7.0

#: Pressure consistency (stroke-width standard deviation, pixels): at or
#: above this, pressure is read as "variable" rather than "consistent".
PRESSURE_STD_VARIABLE_MIN_PX = 2.5

#: Letter size (pixels): at or below this reads as "small"; at or above
#: `LETTER_SIZE_LARGE_MIN_PX` reads as "large"; in between is "medium".
LETTER_SIZE_SMALL_MAX_PX = 10.0
LETTER_SIZE_LARGE_MIN_PX = 20.0

#: Line/word spacing are read as a *ratio to letter size* rather than an
#: absolute pixel count, so the classification is resolution-independent
#: (unlike the pressure/letter-size cutoffs above, which are absolute
#: pixel values). Below this ratio, spacing reads as "tight"; at or above
#: the generous threshold, "generous"; in between, "moderate".
LINE_SPACING_TIGHT_RATIO_MAX = 1.5
LINE_SPACING_GENEROUS_RATIO_MIN = 3.0
WORD_SPACING_TIGHT_RATIO_MAX = 1.0
WORD_SPACING_GENEROUS_RATIO_MIN = 3.0

#: Below this estimated letter size (pixels), the line/word spacing
#: ratios above are not meaningful (dividing by a near-zero or
#: unreliable letter-size estimate), so spacing is instead reported as
#: "indeterminate" rather than computing a misleading ratio.
MIN_LETTER_SIZE_FOR_RATIO_PX = 1.0

#: Baseline slope (degrees): |slope| at or below this reads as "steady";
#: at or above `BASELINE_WANDERING_MIN_DEGREES`, "wandering"; in between,
#: a "slight drift".
BASELINE_STEADY_MAX_DEGREES = 2.0
BASELINE_WANDERING_MIN_DEGREES = 6.0

#: Margins (pixels): a margin at or below this is read as the writing
#: running close to that edge of the page, regardless of the opposite
#: margin's size.
MARGIN_NEAR_EDGE_MAX_PX = 5.0

#: Margins: if the larger of a margin pair is more than this many times
#: the smaller (floored at 1px to avoid a division blow-up on a
#: near-zero smaller margin), the pair reads as "imbalanced" rather than
#: "balanced".
MARGIN_IMBALANCE_RATIO = 2.0

#: Rhythm regularity (`Features.rhythm_regularity`, an inverse-CV
#: composite in [0, 1] where 1.0 means highly regular/evenly-repeating
#: stroke shape, size, and spacing -- see that field's docstring): at or
#: below this reads as "irregular" (halting, unevenly repeating); at or
#: above `RHYTHM_REGULAR_MIN`, "regular"; in between, a moderate middle
#: band. Distinct from the old `INK_DENSITY_LOW_MAX`/`INK_DENSITY_HIGH_MIN`
#: cutoffs this indicator used before A4 -- those described ink coverage
#: (a density fraction), this describes regularity of repetition, a
#: different [0, 1] scale entirely.
RHYTHM_IRREGULAR_MAX = 0.35
RHYTHM_REGULAR_MIN = 0.65

#: Stroke continuity (`Features.stroke_connectedness`, in [0, 1] where 1.0
#: means strokes read as fully joined/continuous and 0.0 means fully
#: segmented/printed -- see that field's docstring): at or below this
#: reads as "broken/segmented"; at or above
#: `STROKE_CONNECTEDNESS_CONNECTED_MIN`, "connected/joined"; in between, a
#: "partially joined" middle band.
STROKE_CONNECTEDNESS_BROKEN_MAX = 0.35
STROKE_CONNECTEDNESS_CONNECTED_MIN = 0.65

#: Overall organization (`Features.organization_score`, in [0, 1] where
#: 1.0 means a highly organized/planned layout -- see that field's
#: docstring): at or below this reads as "loosely organized"; at or above
#: `ORGANIZATION_PLANNED_MIN`, "planned, consistent"; in between, a
#: moderately organized middle band.
ORGANIZATION_LOOSE_MAX = 0.35
ORGANIZATION_PLANNED_MIN = 0.65

#: How many indicators `depth="concise"` reports (out of the
#: `len(INDICATOR_ORDER)` available at `depth="indepth"`). Picked to keep
#: the concise output meaningfully shorter (fewer than half the
#: indicators) while still surfacing enough indicators to touch several
#: of the personality/relational/communication/energy/organizational
#: themes the project brief calls for.
CONCISE_INDICATOR_COUNT = 4

#: An indicator with confidence at or above this, whose reading falls in
#: a "typical"/middle-of-the-range bucket, contributes a `strengths`
#: entry.
STRENGTH_MIN_CONFIDENCE = 0.4

#: An indicator with confidence *below* this is reported as an
#: `areas_of_attention` entry regardless of its reading's bucket (there
#: is not enough underlying signal to characterize it confidently either
#: way) -- see `Features.confidence`'s docstring for what drives low
#: confidence (e.g. a near-empty image with few detected strokes/lines).
ATTENTION_LOW_CONFIDENCE_MAX = 0.3


# --- internal candidate representation ----------------------------------------


class _Candidate(NamedTuple):
    """One indicator's fully-built finding content, before depth selection."""

    indicator: str
    observation: str
    interpretation_full: str
    interpretation_brief: str
    confidence: float
    descriptor: str
    tag: Literal["typical", "notable"]


def _confidence_for(features: Features, indicator: str) -> float:
    """Look up an indicator's confidence verbatim from `Features.confidence`."""
    return features.confidence[INDICATOR_CONFIDENCE_KEY[indicator]]


# --- per-indicator classification ----------------------------------------------


def _candidate_slant(features: Features) -> _Candidate:
    angle = features.slant_angle_degrees
    abs_angle = abs(angle)

    if abs_angle <= SLANT_NEUTRAL_MAX_DEGREES:
        descriptor = "an upright, roughly vertical stroke angle"
        full = (
            "An upright slant is often associated with a measured, self-contained "
            "communication style and a tendency to weigh emotional expression before "
            "showing it outwardly, though many careful writers also simply favor an "
            "upright hand for legibility."
        )
        brief = "An upright slant often points toward a measured, self-contained communication style."
        tag: Literal["typical", "notable"] = "typical"
    elif angle > 0:
        if abs_angle <= SLANT_MODERATE_MAX_DEGREES:
            descriptor = "a moderate rightward lean"
            full = (
                "This pattern is often associated with a forward-oriented, sociable "
                "communication style and a degree of comfort engaging with others, "
                "though this varies by individual and by writing context."
            )
            brief = "A moderate right lean often points toward a sociable, forward-oriented communication style."
            tag = "typical"
        else:
            descriptor = "a pronounced rightward lean"
            full = (
                "A marked rightward lean like this is often associated, in a general "
                "sense, with an outward-oriented, expressive communication style and a "
                "readiness to engage with others -- though very pronounced slants are "
                "also worth reading alongside the rest of the sample, since the same "
                "pattern can also reflect momentary factors like writing speed or a "
                "hurried pace."
            )
            brief = (
                "A pronounced right lean often points toward an outward, expressive "
                "communication style, though it is worth reading alongside other "
                "patterns in the sample."
            )
            tag = "notable"
    else:
        if abs_angle <= SLANT_MODERATE_MAX_DEGREES:
            descriptor = "a moderate leftward lean"
            full = (
                "This pattern is often associated with a more reserved or "
                "self-contained communication style, and sometimes with a tendency to "
                "process feelings internally before expressing them, though left "
                "slant is also common simply among left-handed writers adjusting pen "
                "angle."
            )
            brief = "A moderate left lean often points toward a more reserved, self-contained communication style."
            tag = "typical"
        else:
            descriptor = "a pronounced leftward lean"
            full = (
                "A marked leftward lean like this is often associated, in a general "
                "sense, with a notably reserved or guarded communication style -- "
                "though, as with any pronounced pattern, it is worth reading alongside "
                "the rest of the sample rather than in isolation."
            )
            brief = (
                "A pronounced left lean often points toward a notably reserved "
                "communication style, best read alongside other patterns."
            )
            tag = "notable"

    observation = f"Slant is measured at {angle:+.1f} degrees from vertical, indicating {descriptor}."
    return _Candidate(
        "slant", observation, full, brief, _confidence_for(features, "slant"), descriptor, tag
    )


def _candidate_pressure(features: Features) -> _Candidate:
    value = features.stroke_width_mean

    if value <= PRESSURE_LIGHT_MAX_PX:
        descriptor = "a comparatively light touch"
        full = (
            "A lighter touch like this is sometimes associated with a lower-key or "
            "more conserving energy expenditure while writing, and with a gentler, "
            "less assertive communication style -- though stroke weight is also "
            "affected by pen type, paper, and writing speed, so this reading is best "
            "treated as one input among several."
        )
        brief = "A lighter touch is sometimes associated with a gentler, lower-key energy and communication style."
        tag: Literal["typical", "notable"] = "notable"
    elif value >= PRESSURE_FIRM_MIN_PX:
        descriptor = "a comparatively firm, heavier touch"
        full = (
            "A firmer touch like this is sometimes associated with a more energetic "
            "or assertive engagement with the task at hand, and with a more direct "
            "communication style -- though, as with light pressure, stroke weight is "
            "also shaped by pen and writing conditions, so this reading is best "
            "treated as one input among several."
        )
        brief = "A firmer touch is sometimes associated with a more energetic, direct engagement with the task."
        tag = "notable"
    else:
        descriptor = "a moderate, middle-of-the-range touch"
        full = (
            "A middle-of-the-range touch is often associated with a fairly balanced, "
            "adaptable energy level and communication style, neither notably forceful "
            "nor notably light-handed."
        )
        brief = "A moderate touch is often associated with a fairly balanced energy level."
        tag = "typical"

    observation = (
        f"Average stroke width is measured at {value:.1f}px, read here as a pressure "
        "proxy (true pen pressure cannot be recovered from a static image), "
        f"indicating {descriptor}."
    )
    return _Candidate(
        "pressure", observation, full, brief, _confidence_for(features, "pressure"), descriptor, tag
    )


def _candidate_pressure_consistency(features: Features) -> _Candidate:
    std = features.stroke_width_std

    if std >= PRESSURE_STD_VARIABLE_MIN_PX:
        descriptor = "a noticeably variable pressure across the sample"
        full = (
            "Noticeably variable pressure across a sample is sometimes associated "
            "with shifting energy, mood, or focus over the course of writing, though "
            "everyday factors such as pen angle, fatigue, or writing surface also "
            "commonly affect pressure consistency."
        )
        brief = "Variable pressure sometimes points to shifting energy or focus, though everyday factors also affect it."
        tag: Literal["typical", "notable"] = "notable"
    else:
        descriptor = "a fairly steady, consistent pressure throughout the sample"
        full = (
            "Steady pressure across a sample is often associated with a fairly "
            "stable emotional and energetic register while writing, though a short "
            "or simple sample can also read as steady just by having little room to "
            "vary."
        )
        brief = "Steady pressure is often associated with a fairly stable energy and emotional register while writing."
        tag = "typical"

    observation = f"Stroke-width variation is measured at a standard deviation of {std:.1f}px, indicating {descriptor}."
    return _Candidate(
        "pressure_consistency",
        observation,
        full,
        brief,
        _confidence_for(features, "pressure_consistency"),
        descriptor,
        tag,
    )


def _candidate_letter_size(features: Features) -> _Candidate:
    value = features.letter_size_estimate

    if value <= LETTER_SIZE_SMALL_MAX_PX:
        descriptor = "comparatively small, compact lettering"
        full = (
            "Smaller, more compact lettering is sometimes associated with a more "
            "inward or detail-focused attention style and a preference for economy "
            "of expression, though small writing can also simply reflect available "
            "page space or a fine-tipped pen."
        )
        brief = "Compact lettering sometimes points to a more inward, detail-focused attention style."
        tag: Literal["typical", "notable"] = "notable"
    elif value >= LETTER_SIZE_LARGE_MIN_PX:
        descriptor = "comparatively large lettering"
        full = (
            "Larger lettering is sometimes associated with a more outward, "
            "expressive self-presentation and a wish to occupy visible space, though "
            "letter size is also shaped by pen width, page size, and the writer's "
            "habitual scale."
        )
        brief = "Larger lettering sometimes points to a more outward, expressive self-presentation."
        tag = "notable"
    else:
        descriptor = "letter sizing in a moderate, middle-of-the-range band"
        full = (
            "Middle-of-the-range letter sizing is often associated with a fairly "
            "balanced, adaptable self-presentation in writing, neither notably "
            "expansive nor notably compact."
        )
        brief = "Moderate letter sizing often points to a fairly balanced self-presentation."
        tag = "typical"

    observation = f"Letter size is estimated at {value:.1f}px, indicating {descriptor}."
    return _Candidate(
        "letter_size",
        observation,
        full,
        brief,
        _confidence_for(features, "letter_size"),
        descriptor,
        tag,
    )


def _spacing_ratio_bucket(
    spacing_value: float,
    letter_size: float,
    *,
    tight_max: float,
    generous_min: float,
) -> tuple[str, float | None, Literal["typical", "notable"]]:
    """Shared tight/moderate/generous-vs-letter-size bucketing.

    Returns ``(bucket_name, ratio_or_None, tag)``; ``ratio`` is ``None``
    for the "indeterminate" bucket (letter size too small to divide by
    meaningfully -- see `MIN_LETTER_SIZE_FOR_RATIO_PX`).
    """
    if letter_size < MIN_LETTER_SIZE_FOR_RATIO_PX:
        return "indeterminate", None, "notable"
    ratio = spacing_value / letter_size
    if ratio < tight_max:
        return "tight", ratio, "notable"
    if ratio >= generous_min:
        return "generous", ratio, "notable"
    return "moderate", ratio, "typical"


def _candidate_line_spacing(features: Features) -> _Candidate:
    bucket, ratio, tag = _spacing_ratio_bucket(
        features.line_spacing_mean,
        features.letter_size_estimate,
        tight_max=LINE_SPACING_TIGHT_RATIO_MAX,
        generous_min=LINE_SPACING_GENEROUS_RATIO_MIN,
    )

    texts = {
        "tight": (
            "a fairly tight, close arrangement of lines",
            (
                "Tightly spaced lines are sometimes associated with an economical "
                "use of page space and a fast-paced or efficiency-minded working "
                "style, though tight spacing can also simply reflect limited page "
                "space or a wish to fit more onto one sheet."
            ),
            "Tight line spacing sometimes points to an economical, efficiency-minded working style.",
        ),
        "moderate": (
            "a moderate, evenly balanced arrangement of lines",
            (
                "A moderate, balanced line spacing is often associated with a "
                "fairly organized, clear approach to structuring written work."
            ),
            "Moderate line spacing often points to a fairly organized approach to structuring work.",
        ),
        "generous": (
            "a fairly open, generously spaced arrangement of lines",
            (
                "More generously spaced lines are sometimes associated with a "
                "preference for clarity and breathing room in how work is "
                "organized, though this can also simply reflect a larger page or a "
                "wish to leave room for notes."
            ),
            "Generous line spacing sometimes points to a preference for clarity and breathing room in how work is organized.",
        ),
        "indeterminate": (
            "an arrangement that could not be reliably compared to letter scale",
            (
                "There is not enough reliably measured content in this sample to "
                "characterize line spacing relative to letter scale; this reading "
                "should be treated as minimal."
            ),
            "Not enough content was measured here to characterize line spacing meaningfully.",
        ),
    }
    descriptor, full, brief = texts[bucket]

    if ratio is None:
        observation = (
            f"Line spacing measures {features.line_spacing_mean:.1f}px, but letter "
            "size could not be reliably estimated in this sample, so spacing cannot "
            "be meaningfully compared to letter scale here."
        )
    else:
        observation = (
            f"Line spacing measures {features.line_spacing_mean:.1f}px, about "
            f"{ratio:.1f}x the estimated letter size, indicating {descriptor}."
        )

    return _Candidate(
        "line_spacing",
        observation,
        full,
        brief,
        _confidence_for(features, "line_spacing"),
        descriptor,
        tag,
    )


def _candidate_word_spacing(features: Features) -> _Candidate:
    bucket, ratio, tag = _spacing_ratio_bucket(
        features.word_spacing_mean,
        features.letter_size_estimate,
        tight_max=WORD_SPACING_TIGHT_RATIO_MAX,
        generous_min=WORD_SPACING_GENEROUS_RATIO_MIN,
    )

    texts = {
        "tight": (
            "words positioned fairly close together",
            (
                "Words positioned close together are sometimes associated with a "
                "preference for closeness or connection in relational style, though "
                "tight word spacing can also simply reflect an effort to conserve "
                "page space."
            ),
            "Close word spacing sometimes points to a preference for closeness in relational style.",
        ),
        "moderate": (
            "a moderate, evenly balanced gap between words",
            (
                "A moderate, balanced gap between words is often associated with a "
                "fairly comfortable, adaptable relational style -- neither notably "
                "distant nor notably close."
            ),
            "Moderate word spacing often points to a fairly comfortable, adaptable relational style.",
        ),
        "generous": (
            "words positioned with generous separation",
            (
                "Generously spaced words are sometimes associated with a preference "
                "for independence or personal space in relational style, though this "
                "can also reflect a larger, more open page layout."
            ),
            "Generous word spacing sometimes points to a preference for independence or personal space.",
        ),
        "indeterminate": (
            "a spacing that could not be reliably compared to letter scale",
            (
                "There is not enough reliably measured content in this sample to "
                "characterize word spacing relative to letter scale; this reading "
                "should be treated as minimal."
            ),
            "Not enough content was measured here to characterize word spacing meaningfully.",
        ),
    }
    descriptor, full, brief = texts[bucket]

    if ratio is None:
        observation = (
            f"Word spacing measures {features.word_spacing_mean:.1f}px, but letter "
            "size could not be reliably estimated in this sample, so spacing cannot "
            "be meaningfully compared to letter scale here."
        )
    else:
        observation = (
            f"Word spacing measures {features.word_spacing_mean:.1f}px, about "
            f"{ratio:.1f}x the estimated letter size, indicating {descriptor}."
        )

    return _Candidate(
        "word_spacing",
        observation,
        full,
        brief,
        _confidence_for(features, "word_spacing"),
        descriptor,
        tag,
    )


def _candidate_baseline(features: Features) -> _Candidate:
    slope = features.baseline_slope_degrees
    abs_slope = abs(slope)
    direction = "rising" if slope > 0 else "falling" if slope < 0 else "level"

    if abs_slope <= BASELINE_STEADY_MAX_DEGREES:
        descriptor = "a fairly steady baseline"
        full = (
            "A steady baseline is often associated with a fairly consistent mood "
            "and energy level while writing, and with a settled, even-paced working "
            "style."
        )
        brief = "A steady baseline often points to a fairly consistent mood and energy level while writing."
        tag: Literal["typical", "notable"] = "typical"
    elif abs_slope <= BASELINE_WANDERING_MIN_DEGREES:
        drift_word = "upward" if slope > 0 else "downward"
        descriptor = f"a slight {drift_word} drift across the line"
        full = (
            "A slight baseline drift is a common, everyday pattern and is often "
            "associated with normal shifts in energy, posture, or focus over the "
            "course of writing, without necessarily implying anything beyond that."
        )
        brief = "A slight baseline drift is a common pattern, often reflecting normal shifts in energy or focus."
        tag = "typical"
    else:
        drift_word = "upward" if slope > 0 else "downward"
        descriptor = f"a noticeable {drift_word} trend across the line"
        full = (
            "A more noticeably wandering baseline is sometimes associated with "
            "fluctuating energy, mood, or focus over the course of writing, though "
            "page angle, writing surface, and physical posture also commonly "
            "influence this measurement."
        )
        brief = "A more wandering baseline sometimes points to fluctuating energy or focus, though writing conditions also play a role."
        tag = "notable"

    observation = (
        f"Baseline trend is measured at {slope:+.1f} degrees ({direction} "
        f"left-to-right), indicating {descriptor}."
    )
    return _Candidate(
        "baseline", observation, full, brief, _confidence_for(features, "baseline"), descriptor, tag
    )


def _margin_bucket(smaller: float, larger: float) -> tuple[str, Literal["typical", "notable"]]:
    if smaller <= MARGIN_NEAR_EDGE_MAX_PX or larger <= MARGIN_NEAR_EDGE_MAX_PX:
        return "near_edge", "notable"
    ratio = larger / max(smaller, 1.0)
    if ratio > MARGIN_IMBALANCE_RATIO:
        return "imbalanced", "notable"
    return "balanced", "typical"


def _candidate_margins_horizontal(features: Features) -> _Candidate:
    left, right = features.margin_left_px, features.margin_right_px
    bucket, tag = _margin_bucket(min(left, right), max(left, right))

    texts = {
        "balanced": (
            "a fairly balanced left/right margin",
            (
                "Fairly balanced left and right margins are often associated with a "
                "planned, organized approach to using the page and, more broadly, "
                "with a considered approach to structuring one's work."
            ),
            "Balanced margins often point to a planned, organized approach to using the page.",
        ),
        "imbalanced": (
            "a noticeably imbalanced left/right margin",
            (
                "A noticeably uneven left/right margin is sometimes associated with "
                "a less planned or more spontaneous approach to using available "
                "space, though this can also simply reflect the writer running out "
                "of room or working from an unusual page layout."
            ),
            "Uneven left/right margins sometimes point to a more spontaneous approach to using space.",
        ),
        "near_edge": (
            "writing running close to the left or right edge of the page",
            (
                "Writing that runs close to the edge of the page is sometimes "
                "associated with a wish to make full use of available space, or "
                "with writing under some time or space pressure, though this also "
                "commonly reflects a small page or a densely worded sample."
            ),
            "Writing near the page edge sometimes points to a wish to use available space fully, or to working under some space pressure.",
        ),
    }
    descriptor, full, brief = texts[bucket]
    observation = (
        f"Left margin measures {left:.0f}px and right margin measures {right:.0f}px, "
        f"indicating {descriptor}."
    )
    return _Candidate(
        "margins_horizontal",
        observation,
        full,
        brief,
        _confidence_for(features, "margins_horizontal"),
        descriptor,
        tag,
    )


def _candidate_margins_vertical(features: Features) -> _Candidate:
    top, bottom = features.margin_top_px, features.margin_bottom_px
    bucket, tag = _margin_bucket(min(top, bottom), max(top, bottom))

    texts = {
        "balanced": (
            "a fairly balanced top/bottom margin",
            (
                "Fairly balanced top and bottom margins are often associated with a "
                "planned, organized approach to starting and finishing a piece of "
                "written work."
            ),
            "Balanced top/bottom margins often point to a planned approach to starting and finishing written work.",
        ),
        "imbalanced": (
            "a noticeably imbalanced top/bottom margin",
            (
                "A noticeably uneven top/bottom margin is sometimes associated with "
                "a less planned approach to pacing a piece of written work from "
                "start to finish, though this can also simply reflect the writer "
                "running out of page space."
            ),
            "Uneven top/bottom margins sometimes point to a less planned approach to pacing written work.",
        ),
        "near_edge": (
            "writing running close to the top or bottom edge of the page",
            (
                "Writing that runs close to the top or bottom edge of the page is "
                "sometimes associated with a wish to make full use of available "
                "space, or with writing under some time or space pressure, though "
                "this also commonly reflects a small page or a densely worded "
                "sample."
            ),
            "Writing near the top or bottom page edge sometimes points to a wish to use available space fully, or to working under some space pressure.",
        ),
    }
    descriptor, full, brief = texts[bucket]
    observation = (
        f"Top margin measures {top:.0f}px and bottom margin measures {bottom:.0f}px, "
        f"indicating {descriptor}."
    )
    return _Candidate(
        "margins_vertical",
        observation,
        full,
        brief,
        _confidence_for(features, "margins_vertical"),
        descriptor,
        tag,
    )


def _candidate_rhythm(features: Features) -> _Candidate:
    regularity = features.rhythm_regularity

    if regularity <= RHYTHM_IRREGULAR_MAX:
        descriptor = "an irregular, unevenly repeating rhythm of stroke shape, size, and spacing"
        full = (
            "An irregular rhythm like this is sometimes associated with a more "
            "variable, shifting pace of expression from word to word or line to "
            "line, though this also commonly reflects a short or hurried sample, "
            "an unfamiliar writing surface, or simply natural variation across a "
            "small amount of writing."
        )
        brief = "An irregular rhythm sometimes points to a more variable, shifting pace of expression."
        tag: Literal["typical", "notable"] = "notable"
    elif regularity >= RHYTHM_REGULAR_MIN:
        descriptor = "a regular, evenly repeating rhythm of stroke shape, size, and spacing"
        full = (
            "A regular, evenly repeating rhythm is often associated with a fairly "
            "steady, practiced pace of expression and a consistent working tempo "
            "across the sample, though a short or simple sample can also read as "
            "regular just by having little room to vary."
        )
        brief = "A regular, evenly repeating rhythm often points to a steady, practiced pace of expression."
        tag = "typical"
    else:
        descriptor = "a moderately regular rhythm, neither markedly even nor markedly uneven"
        full = (
            "A moderate rhythm reading is often associated with a fairly typical, "
            "adaptable pace of expression, neither notably steady nor notably "
            "variable."
        )
        brief = "A moderate rhythm reading often points to a fairly typical, adaptable pace of expression."
        tag = "typical"

    observation = (
        f"Rhythm regularity is measured at {regularity:.2f} (a 0-1 composite of "
        "how evenly stroke shape, size, and spacing repeat across the sample), "
        f"indicating {descriptor}."
    )
    return _Candidate(
        "rhythm", observation, full, brief, _confidence_for(features, "rhythm"), descriptor, tag
    )


def _candidate_stroke_continuity(features: Features) -> _Candidate:
    connectedness = features.stroke_connectedness

    if connectedness <= STROKE_CONNECTEDNESS_BROKEN_MAX:
        descriptor = "strokes that read as noticeably broken or segmented, with frequent apparent pen lifts"
        full = (
            "Frequently broken or lifted strokes are sometimes associated with a "
            "more deliberate, step-by-step style of thought or expression -- "
            "pausing between pieces of a word -- though this also commonly reflects "
            "a printed (rather than cursive) writing style, a fine pen, or simply "
            "an unhurried pace."
        )
        brief = "Frequently broken strokes sometimes point to a more deliberate, step-by-step style of expression."
        tag: Literal["typical", "notable"] = "notable"
    elif connectedness >= STROKE_CONNECTEDNESS_CONNECTED_MIN:
        descriptor = "strokes that read as mostly joined and continuous, with few apparent pen lifts"
        full = (
            "Mostly joined, continuous strokes are sometimes associated with a "
            "more fluid flow from one thought to the next and an easier transition "
            "between ideas while writing, though this also commonly reflects a "
            "cursive-leaning writing style learned early on or simply a faster "
            "writing pace."
        )
        brief = "Mostly joined strokes sometimes point to a more fluid flow from one thought to the next."
        tag = "typical"
    else:
        descriptor = "strokes with a moderate mix of joined and broken sections"
        full = (
            "A moderate mix of joined and broken strokes is often associated with "
            "a fairly adaptable style of expression, neither markedly continuous "
            "nor markedly segmented."
        )
        brief = "A moderate mix of joined and broken strokes often points to a fairly adaptable style of expression."
        tag = "typical"

    observation = (
        f"Stroke connectedness is measured at {connectedness:.2f} (a 0-1 score "
        "comparing detected stroke segments to estimated word count within each "
        f"line), indicating {descriptor}."
    )
    return _Candidate(
        "stroke_continuity",
        observation,
        full,
        brief,
        _confidence_for(features, "stroke_continuity"),
        descriptor,
        tag,
    )


def _candidate_organization(features: Features) -> _Candidate:
    score = features.organization_score

    if score <= ORGANIZATION_LOOSE_MAX:
        descriptor = "a loosely organized layout, with uneven line spacing, ragged line starts, or wandering baselines"
        full = (
            "A loosely organized layout is sometimes associated with a more "
            "spontaneous, in-the-moment approach to arranging written work, though "
            "this also commonly reflects an unruled page, a rushed sample, or "
            "unfamiliar writing conditions rather than anything about planning "
            "style generally."
        )
        brief = "A loosely organized layout sometimes points to a more spontaneous approach to arranging written work."
        tag: Literal["typical", "notable"] = "notable"
    elif score >= ORGANIZATION_PLANNED_MIN:
        descriptor = "a consistently organized layout, with evenly spaced lines, aligned line starts, and level baselines"
        full = (
            "A consistently organized layout is often associated with a planned, "
            "methodical approach to arranging written work, though a short or "
            "simple sample can also read as organized just by having little room "
            "to drift."
        )
        brief = "A consistently organized layout often points to a planned, methodical approach to arranging work."
        tag = "typical"
    else:
        descriptor = "a moderately organized layout, neither notably consistent nor notably uneven"
        full = (
            "A moderately organized layout is often associated with a fairly "
            "typical, adaptable approach to arranging written work on the page."
        )
        brief = "A moderately organized layout often points to a fairly typical, adaptable approach to arranging work."
        tag = "typical"

    observation = (
        f"Overall organization is measured at {score:.2f} (a 0-1 composite of "
        "inter-line spacing, line-start alignment, and baseline consistency "
        f"across the sample), indicating {descriptor}."
    )
    return _Candidate(
        "organization",
        observation,
        full,
        brief,
        _confidence_for(features, "organization"),
        descriptor,
        tag,
    )


#: Builder functions in the same fixed order as `INDICATOR_ORDER`.
_CANDIDATE_BUILDERS = (
    _candidate_slant,
    _candidate_pressure,
    _candidate_pressure_consistency,
    _candidate_letter_size,
    _candidate_line_spacing,
    _candidate_word_spacing,
    _candidate_baseline,
    _candidate_margins_horizontal,
    _candidate_margins_vertical,
    _candidate_rhythm,
    _candidate_stroke_continuity,
    _candidate_organization,
)


def _build_candidates(features: Features) -> list[_Candidate]:
    """Build all `INDICATOR_ORDER` candidates for `features`, in order."""
    candidates = [builder(features) for builder in _CANDIDATE_BUILDERS]
    assert tuple(c.indicator for c in candidates) == INDICATOR_ORDER
    return candidates


# --- strengths / areas of attention / overall summary --------------------------


def _derive_strengths(candidates: list[_Candidate]) -> list[str]:
    """Hedged strengths text for confidently-"typical" candidates.

    See `STRENGTH_MIN_CONFIDENCE` / `ATTENTION_LOW_CONFIDENCE_MAX`: a
    candidate below the low-confidence floor is *not* eligible here even
    if its bucket is "typical" -- low confidence is reported as an area
    of attention instead (see `_derive_areas_of_attention`), since there
    is not enough signal to call it anything with confidence.
    """
    strengths = []
    for c in candidates:
        if c.confidence < ATTENTION_LOW_CONFIDENCE_MAX:
            continue
        if c.tag == "typical" and c.confidence >= STRENGTH_MIN_CONFIDENCE:
            label = INDICATOR_LABELS[c.indicator]
            strengths.append(
                f"{label}: {c.descriptor} sits within a commonly observed range, "
                "which supports a reasonably confident, easy-to-characterize "
                "reading here."
            )
    return strengths


def _derive_areas_of_attention(candidates: list[_Candidate]) -> list[str]:
    """Hedged areas-of-attention text: low confidence first, then "notable" bucket.

    Phrased throughout as "may benefit from closer review", never as a
    deficiency, weakness, or problem (see `DISALLOWED_TERMS`).
    """
    areas = []
    for c in candidates:
        label = INDICATOR_LABELS[c.indicator]
        if c.confidence < ATTENTION_LOW_CONFIDENCE_MAX:
            areas.append(
                f"{label}: only limited data could be measured for this sample "
                "(e.g. little visible ink, or too few detected strokes or lines), "
                "so this is an area that may benefit from closer review or a "
                "clearer sample."
            )
        elif c.tag == "notable":
            areas.append(
                f"{label}: {c.descriptor} is a less typical, more pronounced "
                "pattern -- an area that may benefit from closer review alongside "
                "the rest of the sample rather than being read on its own."
            )
    return areas


def _build_overall_summary(candidates: list[_Candidate], depth: Depth) -> str:
    """A short (2-4 sentence), hedged synthesis of `candidates`.

    Always ends with an explicit reminder that this is a descriptive
    support reading, not a determination about the writer -- see the
    module docstring and `StructuredFindings.overall_summary`.
    """
    if candidates:
        avg_confidence = sum(c.confidence for c in candidates) / len(candidates)
    else:
        avg_confidence = 0.0
    notable = [c for c in candidates if c.tag == "notable" and c.confidence >= ATTENTION_LOW_CONFIDENCE_MAX]

    if not candidates or avg_confidence < ATTENTION_LOW_CONFIDENCE_MAX:
        middle_sentence = (
            "Very little could be measured with confidence in this sample -- "
            "likely because there is little visible writing to work from -- so "
            "the reading here should be treated as minimal and provisional rather "
            "than informative."
        )
    elif notable:
        labels = ", ".join(INDICATOR_LABELS[c.indicator].lower() for c in notable[:3])
        middle_sentence = (
            f"The most notable patterns in this sample relate to {labels}, though "
            "every handwriting sample reflects many overlapping, everyday factors "
            "(mood, writing surface, time pressure, and more), so no single "
            "measurement should be read in isolation."
        )
    else:
        middle_sentence = (
            "Overall, the measured patterns in this sample fall within commonly "
            "observed ranges, though every handwriting sample reflects many "
            "overlapping, everyday factors, so no single measurement should be "
            "read in isolation."
        )

    depth_note = "in-depth" if depth == "indepth" else "concise"
    return (
        f"This {depth_note} overview draws on a handful of measurable handwriting "
        "features -- such as slant, stroke weight, spacing, and layout -- to offer "
        f"a cautious, descriptive first read of the sample. {middle_sentence} This "
        "summary is meant to support, not replace, a professional graphologist's "
        "own judgment: it offers a descriptive first read rather than a "
        "determination about the writer's character, health, or psychological "
        "state, and no single finding here should be treated as a settled fact on "
        "its own."
    )


# --- public entry point ---------------------------------------------------------


def interpret(features: Features, depth: Depth = "indepth") -> StructuredFindings:
    """Map measured `features` onto a cautious, structured interpretation.

    At `depth="indepth"`, `findings` covers every indicator in
    `INDICATOR_ORDER` (one `Finding` per measured `Features` field
    group -- slant, pressure/stroke-width [mean and consistency], letter
    size, line spacing, word spacing, baseline, margins [horizontal and
    vertical], rhythm, stroke continuity, and overall organization), with
    the fuller narrative text for each.

    At `depth="concise"`, `findings` is restricted to the
    `CONCISE_INDICATOR_COUNT` indicators with the *highest* confidence
    for this sample (ties broken by `INDICATOR_ORDER`, so the selection
    is deterministic), each with shorter, more condensed narrative text
    -- meaningfully less total output than `depth="indepth"`, not a
    truncated copy of it.

    `strengths` / `areas_of_attention` / `overall_summary` are always
    derived only from the indicators actually included in `findings` for
    the requested depth (see `_derive_strengths` /
    `_derive_areas_of_attention` / `_build_overall_summary`).

    Never raises on a low-confidence/near-blank `features` (e.g. from
    `analyze()` on a blank image, see
    `grafology_ai.analysis.features._empty_features`) -- every
    classification branch has a defined default, and `overall_summary`
    explicitly notes when a sample carried too little signal to read
    confidently.
    """
    candidates = _build_candidates(features)

    if depth == "indepth":
        selected = candidates
    elif depth == "concise":
        ranked = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        top_indicators = {c.indicator for c in ranked[:CONCISE_INDICATOR_COUNT]}
        selected = [c for c in candidates if c.indicator in top_indicators]
    else:
        raise ValueError(f"Unknown depth: {depth!r}; expected 'concise' or 'indepth'")

    findings = []
    for c in selected:
        interpretation_text = c.interpretation_full if depth == "indepth" else c.interpretation_brief
        _assert_language_is_disciplined(c.observation)
        _assert_language_is_disciplined(interpretation_text)
        findings.append(
            Finding(
                indicator=c.indicator,
                observation=c.observation,
                interpretation=interpretation_text,
                confidence=c.confidence,
            )
        )

    strengths = _derive_strengths(selected)
    areas_of_attention = _derive_areas_of_attention(selected)
    for text in (*strengths, *areas_of_attention):
        _assert_language_is_disciplined(text)

    overall_summary = _build_overall_summary(selected, depth)
    _assert_language_is_disciplined(overall_summary)

    return StructuredFindings(
        depth=depth,
        findings=findings,
        strengths=strengths,
        areas_of_attention=areas_of_attention,
        overall_summary=overall_summary,
    )
