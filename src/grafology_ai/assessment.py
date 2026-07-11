"""Day-zero dataset assessment: "how good is this dataset overall?"

This is the dataset-level counterpart to :mod:`grafology_ai.intake`
(``run_intake`` answers "is this dataset structurally/technically usable?"
one sample at a time) and to :mod:`grafology_ai.report.generator`
(``generate_report`` renders one *sample's* analysis). Neither of those
tools answers the question the client's technical proposal asks for up
front: "provide a first technical assessment of the available dataset,
highlighting strengths, limitations and areas for improvement" -- a
*dataset-wide* read covering composition, label coverage against the
9-indicator rubric in ``docs/labeling_rubric.md``, and how the dataset's
size compares against the proposal's own stated thresholds ("operational
minimum (prototype): 50-100 labeled samples; recommended to start
stabilizing: 300+ labeled samples").

:func:`assess_dataset` is that check. It deliberately reuses
:func:`grafology_ai.intake.run_intake` for all quality/validation
judgment (accepted/flagged/rejected counts, manifest errors, the fix-it
checklist) rather than reimplementing any of that logic -- this module
only adds the composition/label-coverage/size-threshold analysis that
``run_intake`` does not attempt. :func:`render_assessment_report` renders
the result into a fixed-section markdown report, following the same
"never fabricate, never leak `None`, never render a dangling empty bullet
list" conventions :mod:`grafology_ai.report.generator` establishes for the
per-sample report. :func:`save_assessment_report` writes that markdown to
disk, mirroring :func:`grafology_ai.report.generator.save_report`.

This module never labels a dataset and never collects more of one -- both
are explicitly the client's responsibility, per the technical proposal's
"What Is NOT Included" section (see the ``grafology_ai.intake`` module
docstring for the same point made about intake). It only assesses what is
already there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from grafology_ai.intake import IntakeReport, run_intake
from grafology_ai.pipeline.ingest import ingest_dataset

#: The 9 rubric indicators from ``docs/labeling_rubric.md``, in the same
#: order that document lists them, keyed by the snake_case name a
#: graphologist records label values under in a
#: :class:`~grafology_ai.dataset.schema.ManifestEntry`'s ``labels`` dict.
RUBRIC_INDICATORS: tuple[str, ...] = (
    "pressure",
    "slant",
    "letter_size",
    "spacing",
    "baseline_movement",
    "margins",
    "rhythm",
    "stroke_continuity",
    "overall_organization",
)

#: Human-readable label for each rubric indicator, used only when
#: rendering the label-coverage table -- never affects the computed
#: coverage numbers themselves.
RUBRIC_INDICATOR_LABELS: dict[str, str] = {
    "pressure": "Pressure",
    "slant": "Slant",
    "letter_size": "Letter Size",
    "spacing": "Spacing",
    "baseline_movement": "Baseline Movement",
    "margins": "Margins",
    "rhythm": "Rhythm",
    "stroke_continuity": "Stroke Continuity",
    "overall_organization": "Overall Organization",
}

#: The client's technical proposal's two dataset-size thresholds:
#: "operational minimum (prototype): 50-100 labeled samples; recommended
#: to start stabilizing: 300+ labeled samples". `PROTOTYPE_MINIMUM` uses
#: the low end of the 50-100 prototype range (the point below which the
#: dataset is not even at the prototype floor); `STABILIZING_THRESHOLD` is
#: the 300+ figure.
PROTOTYPE_MINIMUM = 50
STABILIZING_THRESHOLD = 300

#: The three possible size-bracket verdicts `assess_dataset` computes,
#: worded to match the client-facing language a reviewer should see
#: verbatim in the rendered report.
SIZE_BELOW_PROTOTYPE_MINIMUM = "below prototype minimum"
SIZE_MEETS_PROTOTYPE_BELOW_STABILIZING = "meets prototype minimum, below stabilizing threshold"
SIZE_MEETS_STABILIZING_THRESHOLD = "meets stabilizing threshold"

#: A rubric indicator with coverage below this fraction is called out by
#: name in `recommendations` as needing more labeling. Chosen as a
#: deliberately low bar (fewer than half the dataset labeled for that
#: indicator) so this only fires for genuine, actionable gaps -- not for
#: an indicator that is merely short of 100%.
LOW_LABEL_COVERAGE_THRESHOLD = 0.5


def _count_by(entries: Sequence[Any], key_fn: Any) -> dict[str, int]:
    """Tally `entries` into a `{key: count}` dict using `key_fn(entry)`."""
    counts: dict[str, int] = {}
    for entry in entries:
        key = key_fn(entry)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _normalized_language(language: str | None) -> str:
    """`entry.language`, or `"unknown"` if unset/blank."""
    stripped = (language or "").strip()
    return stripped or "unknown"


def _has_label(entry: Any, indicator: str) -> bool:
    """True if `entry.labels[indicator]` is present and non-blank."""
    return bool((entry.labels.get(indicator) or "").strip())


def _classify_size_bracket(total_samples: int) -> str:
    """Which of the proposal's two size thresholds `total_samples` falls under."""
    if total_samples < PROTOTYPE_MINIMUM:
        return SIZE_BELOW_PROTOTYPE_MINIMUM
    if total_samples < STABILIZING_THRESHOLD:
        return SIZE_MEETS_PROTOTYPE_BELOW_STABILIZING
    return SIZE_MEETS_STABILIZING_THRESHOLD


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _build_recommendations(
    *,
    total_samples: int,
    labeled_sample_count: int,
    label_coverage: dict[str, float],
    intake_report: IntakeReport,
    size_bracket: str,
) -> list[str]:
    """Cautious, evidence-based recommendations derived from the computed numbers.

    Only ever emits a recommendation for a category with a real, measured
    gap -- no generic boilerplate, and nothing emitted for a category with
    no problem (e.g. a dataset that already meets the stabilizing
    threshold gets no size recommendation at all).
    """
    recommendations: list[str] = []

    if size_bracket == SIZE_BELOW_PROTOTYPE_MINIMUM:
        gap = PROTOTYPE_MINIMUM - total_samples
        recommendations.append(
            f"Dataset size ({total_samples} samples) is below the prototype "
            f"minimum of {PROTOTYPE_MINIMUM} -- collect at least {gap} more "
            f"{_plural(gap, 'sample', 'samples')} to reach the prototype minimum."
        )
    elif size_bracket == SIZE_MEETS_PROTOTYPE_BELOW_STABILIZING:
        gap = STABILIZING_THRESHOLD - total_samples
        recommendations.append(
            f"Dataset size ({total_samples} samples) meets the prototype "
            f"minimum but is below the recommended stabilizing threshold of "
            f"{STABILIZING_THRESHOLD} -- collect {gap} more "
            f"{_plural(gap, 'sample', 'samples')} to reach it."
        )
    # SIZE_MEETS_STABILIZING_THRESHOLD: size is not a gap -- no recommendation.

    if total_samples > 0 and labeled_sample_count == 0:
        recommendations.append(
            "No samples in this dataset have any ground-truth labels "
            "recorded for any of the 9 rubric indicators. A ground truth "
            "(predefined indicator labels, or 'gold' reports written by "
            "graphologists) must exist before this dataset can support "
            "training or evaluation -- prioritize labeling a subset of "
            "samples across all indicators."
        )
    elif total_samples > 0:
        low_coverage_indicators = [
            indicator
            for indicator in RUBRIC_INDICATORS
            if label_coverage.get(indicator, 0.0) < LOW_LABEL_COVERAGE_THRESHOLD
        ]
        if low_coverage_indicators:
            names = ", ".join(
                RUBRIC_INDICATOR_LABELS[indicator] for indicator in low_coverage_indicators
            )
            recommendations.append(
                "Label coverage is below "
                f"{round(LOW_LABEL_COVERAGE_THRESHOLD * 100)}% for: {names} -- "
                "prioritize labeling more samples for these indicators."
            )

    if intake_report.rejected > 0:
        recommendations.append(
            f"{intake_report.rejected} "
            f"{_plural(intake_report.rejected, 'sample was', 'samples were')} "
            "rejected during intake validation -- see the quality/validation "
            "summary above for which check(s) failed, and recapture or "
            "rescan those samples accordingly."
        )

    if intake_report.manifest_errors:
        recommendations.append(
            f"{len(intake_report.manifest_errors)} manifest-level "
            f"{_plural(len(intake_report.manifest_errors), 'error was', 'errors were')} "
            "found -- resolve these (see the quality/validation summary "
            "above) before the dataset can be fully assessed."
        )

    return recommendations


@dataclass(frozen=True)
class DatasetAssessment:
    """A first technical assessment of one dataset: composition, label
    coverage, size vs. the proposal's thresholds, and recommendations.

    Attributes:
        dataset_dir: The dataset directory this assessment is about.
        total_samples: Total samples listed in the manifest (0 if the
            manifest itself could not be read -- see
            ``intake_report.manifest_errors``).
        intake_report: The underlying :class:`~grafology_ai.intake.IntakeReport`
            this assessment reuses verbatim for all quality/validation
            judgment -- accepted/flagged/rejected counts, manifest errors,
            and the fix-it checklist.
        acquisition_method_counts: Sample count by
            :attr:`~grafology_ai.dataset.schema.ManifestEntry.acquisition_method`
            (``"photo"``/``"scan"``).
        quality_counts: Sample count by
            :attr:`~grafology_ai.dataset.schema.ManifestEntry.quality`
            (``"high"``/``"medium"``/``"low"``).
        language_counts: Sample count by
            :attr:`~grafology_ai.dataset.schema.ManifestEntry.language`,
            with unset/blank language grouped under ``"unknown"``.
        label_coverage: For each of the 9 rubric indicators (see
            `RUBRIC_INDICATORS`), the fraction (in ``[0, 1]``) of samples
            with a non-empty label for that indicator. ``0.0`` for every
            indicator when `total_samples` is 0 -- never raises or
            produces a NaN/undefined value.
        labeled_sample_count: Count of samples with a non-empty label for
            at least one rubric indicator.
        fully_labeled_sample_count: Count of samples with a non-empty
            label for *every* rubric indicator.
        size_bracket: Which of the proposal's two size thresholds the
            dataset currently falls under -- one of
            `SIZE_BELOW_PROTOTYPE_MINIMUM`,
            `SIZE_MEETS_PROTOTYPE_BELOW_STABILIZING`, or
            `SIZE_MEETS_STABILIZING_THRESHOLD`.
        recommendations: Cautious, evidence-based recommendation strings
            derived from the fields above -- see `_build_recommendations`.
            Empty only if every category above is problem-free.
    """

    dataset_dir: Path
    total_samples: int
    intake_report: IntakeReport
    acquisition_method_counts: dict[str, int] = field(default_factory=dict)
    quality_counts: dict[str, int] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    label_coverage: dict[str, float] = field(default_factory=dict)
    labeled_sample_count: int = 0
    fully_labeled_sample_count: int = 0
    size_bracket: str = SIZE_BELOW_PROTOTYPE_MINIMUM
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this assessment to a plain JSON-compatible dict."""
        return {
            "dataset_dir": str(self.dataset_dir),
            "total_samples": self.total_samples,
            "intake_report": self.intake_report.to_dict(),
            "acquisition_method_counts": dict(self.acquisition_method_counts),
            "quality_counts": dict(self.quality_counts),
            "language_counts": dict(self.language_counts),
            "label_coverage": dict(self.label_coverage),
            "labeled_sample_count": self.labeled_sample_count,
            "fully_labeled_sample_count": self.fully_labeled_sample_count,
            "size_bracket": self.size_bracket,
            "recommendations": list(self.recommendations),
        }


def assess_dataset(dataset_dir: Path) -> DatasetAssessment:
    """Assess a raw dataset directory: composition, label coverage, size, gaps.

    Reuses :func:`grafology_ai.intake.run_intake` for all quality/validation
    judgment (accepted/flagged/rejected counts, manifest errors, the
    fix-it checklist) -- none of that logic is reimplemented here. This
    function only adds composition breakdowns (acquisition method,
    quality, language), per-indicator label coverage against the 9-item
    rubric in ``docs/labeling_rubric.md``, and a comparison of the
    dataset's size against the client technical proposal's two stated
    thresholds (50 = prototype minimum, 300 = recommended stabilizing
    threshold).

    Never raises for a malformed dataset: if the manifest cannot be read
    (missing ``manifest.json``, invalid JSON, a broken entry, a missing
    image file), `intake_report.manifest_errors` carries the diagnostic
    detail (exactly as `run_intake` reports it) and every composition/
    label-coverage field on the returned assessment is empty/zero rather
    than raising.

    Args:
        dataset_dir: Directory containing ``manifest.json`` and
            ``images/`` (the same layout `generate_fixture_dataset`
            produces, and `run_intake` expects).

    Returns:
        A `DatasetAssessment` summarizing the dataset.
    """
    dataset_dir = Path(dataset_dir)

    intake_report = run_intake(dataset_dir)

    try:
        entries = [sample.entry for sample in ingest_dataset(dataset_dir)]
    except Exception:  # noqa: BLE001 - deliberately broad, mirroring
        # `run_intake`'s own graceful-degradation contract: a broken
        # manifest already produced diagnostics via
        # `intake_report.manifest_errors` above, so composition/label
        # stats here simply come back empty rather than raising a second,
        # redundant error.
        entries = []

    total_samples = len(entries)

    acquisition_method_counts = _count_by(entries, lambda e: e.acquisition_method)
    quality_counts = _count_by(entries, lambda e: e.quality)
    language_counts = _count_by(entries, lambda e: _normalized_language(e.language))

    label_coverage: dict[str, float] = {}
    for indicator in RUBRIC_INDICATORS:
        if total_samples == 0:
            label_coverage[indicator] = 0.0
        else:
            labeled = sum(1 for entry in entries if _has_label(entry, indicator))
            label_coverage[indicator] = labeled / total_samples

    labeled_sample_count = sum(
        1
        for entry in entries
        if any(_has_label(entry, indicator) for indicator in RUBRIC_INDICATORS)
    )
    fully_labeled_sample_count = sum(
        1
        for entry in entries
        if all(_has_label(entry, indicator) for indicator in RUBRIC_INDICATORS)
    )

    size_bracket = _classify_size_bracket(total_samples)

    recommendations = _build_recommendations(
        total_samples=total_samples,
        labeled_sample_count=labeled_sample_count,
        label_coverage=label_coverage,
        intake_report=intake_report,
        size_bracket=size_bracket,
    )

    return DatasetAssessment(
        dataset_dir=dataset_dir,
        total_samples=total_samples,
        intake_report=intake_report,
        acquisition_method_counts=acquisition_method_counts,
        quality_counts=quality_counts,
        language_counts=language_counts,
        label_coverage=label_coverage,
        labeled_sample_count=labeled_sample_count,
        fully_labeled_sample_count=fully_labeled_sample_count,
        size_bracket=size_bracket,
        recommendations=recommendations,
    )


# --- markdown rendering -------------------------------------------------------
#
# Follows the same conventions `grafology_ai.report.generator` establishes
# for the per-sample report: fixed sections in a stable order, no dangling
# empty bullet lists, no literal "None" leaking into the text, and text
# reused verbatim (not reworded) from `IntakeReport` where relevant instead
# of recomputing it.

_EMPTY_BREAKDOWN_NOTE = "No samples available to break down."
_NO_QUALITY_ISSUES_NOTE = (
    "No manifest-level errors or fix-it items were found during intake validation."
)
_EMPTY_STRENGTHS_NOTE = "No specific strengths were identified for this dataset."
_EMPTY_LIMITATIONS_NOTE = "No specific limitations were identified for this dataset."
_EMPTY_RECOMMENDATIONS_NOTE = "No recommendations -- no gaps were identified."

_SIZE_BRACKET_TEXT: dict[str, str] = {
    SIZE_BELOW_PROTOTYPE_MINIMUM: (
        f"This dataset is **below the prototype minimum** of "
        f"{PROTOTYPE_MINIMUM} labeled samples the technical proposal "
        "calls for as an operational minimum."
    ),
    SIZE_MEETS_PROTOTYPE_BELOW_STABILIZING: (
        f"This dataset **meets the prototype minimum** of "
        f"{PROTOTYPE_MINIMUM} samples, but is **below the recommended "
        f"stabilizing threshold** of {STABILIZING_THRESHOLD}+ samples."
    ),
    SIZE_MEETS_STABILIZING_THRESHOLD: (
        f"This dataset **meets the recommended stabilizing threshold** of "
        f"{STABILIZING_THRESHOLD}+ samples."
    ),
}


def _render_header(assessment: DatasetAssessment) -> str:
    return (
        "# Dataset Assessment Report\n"
        f"**Dataset directory:** {assessment.dataset_dir}"
    )


def _render_breakdown(label: str, counts: dict[str, int]) -> str:
    if not counts:
        return f"**By {label}:** {_EMPTY_BREAKDOWN_NOTE}"
    total = sum(counts.values())
    lines = [f"**By {label}:**"]
    for key in sorted(counts):
        count = counts[key]
        percent = round(100 * count / total) if total else 0
        lines.append(f"- {key}: {count} ({percent}%)")
    return "\n".join(lines)


def _render_composition_section(assessment: DatasetAssessment) -> str:
    blocks = [
        "## Composition",
        f"**Total samples:** {assessment.total_samples}",
        _render_breakdown("acquisition method", assessment.acquisition_method_counts),
        _render_breakdown("quality", assessment.quality_counts),
        _render_breakdown("language", assessment.language_counts),
    ]
    return "\n\n".join(blocks)


def _render_quality_section(assessment: DatasetAssessment) -> str:
    intake = assessment.intake_report
    lines = [
        "## Quality & Validation Summary",
        (
            f"**Accepted:** {intake.accepted}  "
            f"**Flagged:** {intake.flagged}  "
            f"**Rejected:** {intake.rejected}"
        ),
    ]

    if intake.manifest_errors:
        error_lines = "\n".join(
            f"- [{error.error_type}] {error.message}" for error in intake.manifest_errors
        )
        lines.append(f"**Manifest-level errors:**\n{error_lines}")

    if intake.fix_it_checklist:
        checklist_lines = "\n".join(f"- {item}" for item in intake.fix_it_checklist)
        lines.append(f"**Fix-it checklist:**\n{checklist_lines}")

    if not intake.manifest_errors and not intake.fix_it_checklist:
        lines.append(_NO_QUALITY_ISSUES_NOTE)

    return "\n\n".join(lines)


def _render_label_coverage_section(assessment: DatasetAssessment) -> str:
    header = "## Label Coverage"
    intro = (
        f"{assessment.labeled_sample_count} of {assessment.total_samples} sample(s) "
        "have at least one labeled indicator; "
        f"{assessment.fully_labeled_sample_count} of {assessment.total_samples} "
        "sample(s) are fully labeled across all 9 rubric indicators."
    )
    table_lines = ["| Indicator | Coverage |", "|---|---|"]
    for indicator in RUBRIC_INDICATORS:
        coverage = assessment.label_coverage.get(indicator, 0.0)
        percent = round(coverage * 100)
        table_lines.append(f"| {RUBRIC_INDICATOR_LABELS[indicator]} | {percent}% |")
    table = "\n".join(table_lines)
    return "\n\n".join([header, intro, table])


def _render_size_analysis_section(assessment: DatasetAssessment) -> str:
    header = "## Size vs. Thresholds"
    body = _SIZE_BRACKET_TEXT.get(
        assessment.size_bracket,
        f"Size bracket: {assessment.size_bracket}.",
    )
    detail = (
        f"Total samples: {assessment.total_samples}. Labeled samples "
        f"(at least one indicator): {assessment.labeled_sample_count}. "
        f"Fully labeled samples (all 9 indicators): "
        f"{assessment.fully_labeled_sample_count}. Thresholds: "
        f"{PROTOTYPE_MINIMUM} (prototype minimum), "
        f"{STABILIZING_THRESHOLD}+ (recommended stabilizing threshold)."
    )
    return "\n\n".join([header, body, detail])


def _derive_strengths(assessment: DatasetAssessment) -> list[str]:
    """Strengths derived strictly from measured facts -- never fabricated.

    Each candidate strength is only added when the underlying condition it
    describes is actually true for `assessment`; a dataset with no such
    conditions produces an empty list, not generic filler.
    """
    strengths: list[str] = []

    if assessment.intake_report.is_clean:
        strengths.append(
            "No manifest-level errors and no rejected samples were found "
            "during intake validation."
        )

    if (
        len(assessment.acquisition_method_counts) > 1
        and all(count > 0 for count in assessment.acquisition_method_counts.values())
    ):
        strengths.append(
            "Samples were acquired via more than one method (e.g. both "
            "photo and scan), giving some diversity in acquisition method."
        )

    if assessment.size_bracket == SIZE_MEETS_STABILIZING_THRESHOLD:
        strengths.append(
            "Dataset size meets the technical proposal's recommended "
            "stabilizing threshold."
        )

    fully_covered = [
        indicator
        for indicator in RUBRIC_INDICATORS
        if assessment.label_coverage.get(indicator, 0.0) >= 1.0 and assessment.total_samples > 0
    ]
    if fully_covered:
        names = ", ".join(RUBRIC_INDICATOR_LABELS[indicator] for indicator in fully_covered)
        strengths.append(f"Full (100%) label coverage for: {names}.")

    return strengths


def _derive_limitations(assessment: DatasetAssessment) -> list[str]:
    """Limitations derived strictly from measured gaps -- never fabricated."""
    limitations: list[str] = []

    if assessment.size_bracket != SIZE_MEETS_STABILIZING_THRESHOLD:
        limitations.append(
            f"Dataset size ({assessment.total_samples} samples) does not yet "
            f"meet the recommended stabilizing threshold of "
            f"{STABILIZING_THRESHOLD}+ samples."
        )

    if assessment.total_samples > 0 and assessment.labeled_sample_count == 0:
        limitations.append(
            "No samples have any ground-truth labels recorded for any "
            "rubric indicator."
        )
    else:
        zero_coverage = [
            indicator
            for indicator in RUBRIC_INDICATORS
            if assessment.label_coverage.get(indicator, 0.0) == 0.0
        ]
        if zero_coverage and assessment.total_samples > 0:
            names = ", ".join(
                RUBRIC_INDICATOR_LABELS[indicator] for indicator in zero_coverage
            )
            limitations.append(f"Zero label coverage for: {names}.")

    if assessment.intake_report.rejected > 0:
        limitations.append(
            f"{assessment.intake_report.rejected} sample(s) were rejected "
            "during intake validation and are not usable as-is."
        )

    if assessment.intake_report.manifest_errors:
        limitations.append(
            f"{len(assessment.intake_report.manifest_errors)} manifest-level "
            "error(s) prevented full validation of this dataset."
        )

    return limitations


def _render_list_section(heading: str, items: Sequence[str], empty_note: str) -> str:
    """Render a bulleted list section, or `empty_note` if `items` is empty.

    Never renders a heading with no items under it, and never leaks a
    literal ``"None"``/empty bullet into the output -- mirrors
    `grafology_ai.report.generator._render_list_section`.
    """
    body = "\n".join(f"- {item}" for item in items) if items else empty_note
    return f"## {heading}\n\n{body}"


def render_assessment_report(assessment: DatasetAssessment) -> str:
    """Render `assessment` into a complete markdown report string.

    Fixed sections, always in this order:

    1. A title/header (`"# Dataset Assessment Report"`, plus the dataset
       directory).
    2. `"## Composition"`: total sample count, plus breakdowns by
       acquisition method, quality, and language.
    3. `"## Quality & Validation Summary"`: accepted/flagged/rejected
       counts and (when present) manifest errors and the fix-it
       checklist, reusing `assessment.intake_report`'s own text verbatim
       rather than recomputing it.
    4. `"## Label Coverage"`: a markdown table of each of the 9 rubric
       indicators (`RUBRIC_INDICATORS`) and its coverage percentage.
    5. `"## Size vs. Thresholds"`: which of the proposal's two size
       thresholds (50 prototype minimum, 300+ stabilizing) the dataset
       currently falls under.
    6. `"## Strengths"`: derived, evidence-based strengths, or a clear
       note if none were identified.
    7. `"## Limitations"`: derived, evidence-based limitations, or a
       clear note if none were identified.
    8. `"## Recommendations"`: `assessment.recommendations`, or a clear
       note if the list is empty.

    Renders correctly (no empty/broken markdown, no dangling bullet
    lists, no literal `"None"`) for a fully-labeled, partially-labeled,
    or entirely-unlabeled dataset, and for a dataset with zero samples.
    """
    sections = [
        _render_header(assessment),
        _render_composition_section(assessment),
        _render_quality_section(assessment),
        _render_label_coverage_section(assessment),
        _render_size_analysis_section(assessment),
        _render_list_section("Strengths", _derive_strengths(assessment), _EMPTY_STRENGTHS_NOTE),
        _render_list_section(
            "Limitations", _derive_limitations(assessment), _EMPTY_LIMITATIONS_NOTE
        ),
        _render_list_section(
            "Recommendations", assessment.recommendations, _EMPTY_RECOMMENDATIONS_NOTE
        ),
    ]
    return "\n\n".join(sections).rstrip("\n") + "\n"


def save_assessment_report(assessment: DatasetAssessment, output_path: Path) -> Path:
    """Write `render_assessment_report(assessment)` to `output_path` as markdown.

    Creates `output_path`'s parent directories if they do not already
    exist. Returns `output_path` (as a `Path`, even if a string-like was
    passed in) for convenient chaining -- mirrors
    `grafology_ai.report.generator.save_report`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_assessment_report(assessment), encoding="utf-8")
    return output_path
