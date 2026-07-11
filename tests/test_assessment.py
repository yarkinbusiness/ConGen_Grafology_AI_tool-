"""Tests for grafology_ai.assessment (assess_dataset / render_assessment_report /
save_assessment_report).

Two dataset fixtures are used throughout:

- The synthetic fixture generator (`generate_fixture_dataset`), for a
  realistic-shaped, always-unlabeled dataset with a known composition and
  a known number of rejected (blur) samples.
- A small, hand-constructed 3-sample dataset (manifest written directly by
  this test file, not via the fixture generator) with deliberately chosen
  label coverage, for exact-percentage golden-style assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from grafology_ai.assessment import (
    PROTOTYPE_MINIMUM,
    RUBRIC_INDICATORS,
    SIZE_BELOW_PROTOTYPE_MINIMUM,
    STABILIZING_THRESHOLD,
    DatasetAssessment,
    assess_dataset,
    render_assessment_report,
    save_assessment_report,
)
from grafology_ai.dataset.fixtures import generate_fixture_dataset
from grafology_ai.intake import run_intake

FIXTURE_COUNT = 20
FIXTURE_SEED = 0


# --- helpers for the hand-constructed dataset ----------------------------------


def _make_valid_image(path: Path, size: tuple[int, int] = (900, 700)) -> None:
    """A simple, high-contrast, unblurred image that clears the automated
    resolution/blur/contrast checks -- mirrors the drawing approach used in
    ``tests/test_report.py`` / ``tests/test_validation.py``."""
    image = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = 60
    while y < size[1] - 60:
        draw.line([(60, y), (size[0] - 60, y - 25)], fill=(0, 0, 0), width=5)
        y += 45
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _write_hand_built_dataset(root: Path) -> None:
    """3 samples: one fully labeled (all 9 rubric indicators), one
    partially labeled (3 of 9: pressure, slant, rhythm), one unlabeled.
    The manifest is written directly as JSON, not via
    ``generate_fixture_dataset``.
    """
    images_dir = root / "images"
    for sample_id in ("s-full", "s-partial", "s-none"):
        _make_valid_image(images_dir / f"{sample_id}.png")

    full_labels = {indicator: "value" for indicator in RUBRIC_INDICATORS}
    partial_labels = {"pressure": "heavy", "slant": "right", "rhythm": "regular"}

    manifest = [
        {
            "sample_id": "s-full",
            "acquisition_method": "scan",
            "quality": "high",
            "language": "en",
            "labels": full_labels,
        },
        {
            "sample_id": "s-partial",
            "acquisition_method": "photo",
            "quality": "high",
            "language": "en",
            "labels": partial_labels,
        },
        {
            "sample_id": "s-none",
            "acquisition_method": "photo",
            "quality": "medium",
            "language": None,
            "labels": {},
        },
    ]
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# --- assess_dataset: fixture dataset (count=20, seed=0) -------------------------


def test_assess_dataset_fixture_composition_and_size_and_labels(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)

    assessment = assess_dataset(raw_dir)

    assert assessment.total_samples == FIXTURE_COUNT

    # Composition breakdowns sum correctly to the total.
    assert sum(assessment.acquisition_method_counts.values()) == FIXTURE_COUNT
    assert sum(assessment.quality_counts.values()) == FIXTURE_COUNT
    assert sum(assessment.language_counts.values()) == FIXTURE_COUNT
    assert set(assessment.acquisition_method_counts) <= {"photo", "scan"}
    assert set(assessment.quality_counts) <= {"high", "medium", "low"}

    # Fixtures are never labeled: every one of the 9 indicators must show
    # 0% coverage, handled gracefully (no crash, no NaN).
    assert len(assessment.label_coverage) == len(RUBRIC_INDICATORS)
    for indicator in RUBRIC_INDICATORS:
        assert assessment.label_coverage[indicator] == 0.0
    assert assessment.labeled_sample_count == 0
    assert assessment.fully_labeled_sample_count == 0

    # 20 < 50 -> below prototype minimum.
    assert assessment.size_bracket == SIZE_BELOW_PROTOTYPE_MINIMUM

    # Known distribution for count=20, seed=0 (see tests/test_intake.py):
    # 3 samples rejected on blur.
    assert assessment.intake_report.rejected == 3

    # Recommendations mention both the size gap and the complete lack of
    # labels -- this is exactly the case (tiny + fully unlabeled dataset)
    # the report needs to flag clearly.
    recommendation_text = " ".join(assessment.recommendations)
    assert str(PROTOTYPE_MINIMUM) in recommendation_text
    assert "prototype minimum" in recommendation_text.lower()
    gap = PROTOTYPE_MINIMUM - FIXTURE_COUNT
    assert str(gap) in recommendation_text
    assert "no ground-truth labels" in recommendation_text.lower() or "ground truth" in (
        recommendation_text.lower()
    )
    assert "rejected" in recommendation_text.lower()


def test_assess_dataset_matches_run_intake_directly(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)

    assessment = assess_dataset(raw_dir)
    direct_intake = run_intake(raw_dir)

    # assess_dataset reuses run_intake rather than reimplementing it.
    assert assessment.intake_report.total_samples == direct_intake.total_samples
    assert assessment.intake_report.accepted == direct_intake.accepted
    assert assessment.intake_report.flagged == direct_intake.flagged
    assert assessment.intake_report.rejected == direct_intake.rejected
    assert assessment.intake_report.fix_it_checklist == direct_intake.fix_it_checklist


# --- assess_dataset: hand-built dataset (golden-style exact percentages) --------


def test_assess_dataset_label_coverage_exact_percentages(tmp_path: Path) -> None:
    root = tmp_path / "hand_built"
    _write_hand_built_dataset(root)

    assessment = assess_dataset(root)

    assert assessment.total_samples == 3
    assert assessment.labeled_sample_count == 2  # s-full, s-partial
    assert assessment.fully_labeled_sample_count == 1  # s-full only

    # Indicators labeled in both s-full and s-partial: 2/3.
    for indicator in ("pressure", "slant", "rhythm"):
        assert assessment.label_coverage[indicator] == pytest.approx(2 / 3)

    # Indicators labeled only in s-full: 1/3 (~33%).
    for indicator in ("letter_size", "spacing", "baseline_movement", "margins",
                       "stroke_continuity", "overall_organization"):
        assert assessment.label_coverage[indicator] == pytest.approx(1 / 3)

    # Every rubric indicator is present in the coverage dict.
    assert set(assessment.label_coverage) == set(RUBRIC_INDICATORS)


# --- strengths derivation: does not fabricate ------------------------------------


def test_strengths_do_not_fabricate_when_conditions_are_false(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)

    assessment = assess_dataset(raw_dir)
    # Sanity-check the premises this test relies on: this dataset has
    # rejected samples (not clean) and is below the stabilizing threshold.
    assert assessment.intake_report.rejected > 0
    assert not assessment.intake_report.is_clean
    assert assessment.size_bracket != "meets stabilizing threshold"
    assert assessment.labeled_sample_count == 0  # no full label coverage possible

    report = render_assessment_report(assessment)
    strengths_idx = report.index("## Strengths")
    limitations_idx = report.index("## Limitations")
    strengths_body = report[strengths_idx:limitations_idx]

    # None of these true-only-if-the-underlying-fact-holds strengths may
    # appear, because none of their underlying conditions are true here.
    assert "No manifest-level errors and no rejected samples" not in strengths_body
    assert "recommended stabilizing threshold" not in strengths_body
    assert "Full (100%) label coverage" not in strengths_body


def test_strengths_appear_when_conditions_are_true(tmp_path: Path) -> None:
    root = tmp_path / "hand_built"
    _write_hand_built_dataset(root)
    assessment = assess_dataset(root)

    report = render_assessment_report(assessment)
    strengths_idx = report.index("## Strengths")
    limitations_idx = report.index("## Limitations")
    strengths_body = report[strengths_idx:limitations_idx]

    # This dataset genuinely has diverse acquisition methods (scan + photo)
    # and full coverage for the 3 partially-labeled indicators is NOT 100%,
    # but the fully-labeled sample does give some indicators 100% coverage
    # only if every sample has that indicator labeled -- here no indicator
    # reaches 100% (max is 2/3), so that specific strength must be absent,
    # while the acquisition-method diversity strength must be present.
    assert "Full (100%) label coverage" not in strengths_body
    assert "diversity in acquisition method" in strengths_body


# --- render_assessment_report: required sections + cross-checked values --------


def test_render_assessment_report_has_all_required_sections(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)
    assessment = assess_dataset(raw_dir)

    report = render_assessment_report(assessment)

    for heading in (
        "# Dataset Assessment Report",
        "## Composition",
        "## Quality & Validation Summary",
        "## Label Coverage",
        "## Size vs. Thresholds",
        "## Strengths",
        "## Limitations",
        "## Recommendations",
    ):
        assert heading in report

    assert "None" not in report
    assert str(FIXTURE_COUNT) in report
    # Fix-it checklist text is reused verbatim, not recomputed.
    for item in assessment.intake_report.fix_it_checklist:
        assert item in report


def test_render_assessment_report_label_coverage_table_matches_dataclass(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hand_built"
    _write_hand_built_dataset(root)
    assessment = assess_dataset(root)

    report = render_assessment_report(assessment)

    # Cross-check at least 2 values from the dataclass against the
    # rendered markdown table.
    assert "| Pressure | 67% |" in report  # 2/3 -> round(66.67) == 67
    assert "| Letter Size | 33% |" in report  # 1/3 -> round(33.33) == 33


def test_render_assessment_report_handles_zero_samples_without_crashing(
    tmp_path: Path,
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assessment = assess_dataset(empty_dir)
    assert assessment.total_samples == 0
    assert assessment.intake_report.manifest_errors

    report = render_assessment_report(assessment)
    for heading in (
        "# Dataset Assessment Report",
        "## Composition",
        "## Quality & Validation Summary",
        "## Label Coverage",
        "## Size vs. Thresholds",
        "## Strengths",
        "## Limitations",
        "## Recommendations",
    ):
        assert heading in report
    assert "None" not in report


# --- size-threshold bracket wording ----------------------------------------------


def test_size_bracket_wording_for_each_regime(tmp_path: Path) -> None:
    below_dir = tmp_path / "below"
    generate_fixture_dataset(below_dir, count=20, seed=1)
    assert assess_dataset(below_dir).size_bracket == "below prototype minimum"

    mid_dir = tmp_path / "mid"
    generate_fixture_dataset(mid_dir, count=60, seed=2)
    assessment_mid = assess_dataset(mid_dir)
    assert assessment_mid.total_samples == 60
    assert assessment_mid.size_bracket == "meets prototype minimum, below stabilizing threshold"

    above_dir = tmp_path / "above"
    generate_fixture_dataset(above_dir, count=STABILIZING_THRESHOLD, seed=3)
    assessment_above = assess_dataset(above_dir)
    assert assessment_above.size_bracket == "meets stabilizing threshold"
    # No size recommendation once the dataset meets the stabilizing threshold.
    recommendation_text = " ".join(assessment_above.recommendations)
    assert "prototype minimum" not in recommendation_text
    assert "stabilizing threshold" not in recommendation_text


# --- save_assessment_report -------------------------------------------------------


def test_save_assessment_report_writes_same_content_as_render(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)
    assessment = assess_dataset(raw_dir)
    output_path = tmp_path / "assessment_report.md"

    returned = save_assessment_report(assessment, output_path)

    assert returned == output_path
    assert output_path.read_text(encoding="utf-8") == render_assessment_report(assessment)


def test_save_assessment_report_creates_parent_directories(tmp_path: Path) -> None:
    root = tmp_path / "hand_built"
    _write_hand_built_dataset(root)
    assessment = assess_dataset(root)
    output_path = tmp_path / "nested" / "dirs" / "assessment_report.md"

    assert not output_path.parent.exists()
    returned = save_assessment_report(assessment, output_path)

    assert output_path.exists()
    assert returned == output_path
    assert output_path.read_text(encoding="utf-8") == render_assessment_report(assessment)


# --- DatasetAssessment.to_dict is JSON-serializable ------------------------------


def test_dataset_assessment_to_dict_is_json_serializable(tmp_path: Path) -> None:
    root = tmp_path / "hand_built"
    _write_hand_built_dataset(root)
    assessment = assess_dataset(root)

    data = assessment.to_dict()
    serialized = json.dumps(data)
    assert isinstance(serialized, str)
    assert data["total_samples"] == 3
    assert data["size_bracket"] == SIZE_BELOW_PROTOTYPE_MINIMUM
