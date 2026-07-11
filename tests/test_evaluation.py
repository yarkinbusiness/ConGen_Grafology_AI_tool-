"""Tests for grafology_ai.evaluation -- a comparison HARNESS, not a real evaluation.

Every test in this file exercises the metrics-computation and
baseline-vs-heuristic comparison *machinery* end to end against synthetic
fixture images (:func:`generate_fixture_dataset`), the M2.3 plumbing-proof
:class:`~grafology_ai.training.baseline.BaselineModel`, and fabricated
synthetic labels (:mod:`grafology_ai.training.synthetic_labels`). None of
this proves anything about real graphological accuracy -- see
``src/grafology_ai/evaluation/__init__.py``'s package docstring. The goal
here is to prove the metrics math is exactly right (the golden-file-style
tests below), that missing ground truth is handled correctly, and that the
comparison/report wiring works end to end without crashing on edge cases.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from grafology_ai.analysis.features import Features
from grafology_ai.dataset.fixtures import generate_fixture_dataset
from grafology_ai.analysis import analyze
from grafology_ai.evaluation import (
    ComparisonReport,
    EvaluationMetrics,
    compare_baseline_to_heuristic,
    evaluate_predictions,
    extract_heuristic_pressure_reading,
    render_comparison_report,
)
from grafology_ai.interpretation.interpret import PRESSURE_FIRM_MIN_PX, PRESSURE_LIGHT_MAX_PX
from grafology_ai.training import (
    PRESSURE_LABELS,
    generate_synthetic_pressure_labels,
    train_baseline,
)

FIXTURE_COUNT = 24
FIXTURE_SEED = 0


def _make_features(stroke_width_mean: float, stroke_width_std: float = 0.5) -> Features:
    """Build a minimal, valid Features record with a chosen stroke_width_mean.

    Only stroke_width_mean/stroke_width_std are exercised by this
    package's pressure logic; every other field is a plausible filler
    value, and confidence is set to full (1.0) for every field so nothing
    downstream treats this as a low-confidence/near-blank sample.
    """
    confidence = {key: 1.0 for key in Features.__dataclass_fields__ if key != "confidence"}
    return Features(
        slant_angle_degrees=0.0,
        stroke_width_mean=stroke_width_mean,
        stroke_width_std=stroke_width_std,
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
        confidence=confidence,
    )


def _fixture_features(tmp_path: Path, *, count: int = FIXTURE_COUNT, seed: int = FIXTURE_SEED) -> list[Features]:
    """Generate a fixture dataset and run analyze() on every image (test helper)."""
    raw_dir = tmp_path / f"raw-{seed}-{count}"
    entries = generate_fixture_dataset(raw_dir, count=count, seed=seed)
    images_dir = raw_dir / "images"
    return [analyze(images_dir / f"{entry.sample_id}.png") for entry in entries]


# --- evaluate_predictions: golden-file-style metrics math -------------------


def test_evaluate_predictions_golden_confusion_matrix_and_per_class_metrics() -> None:
    """Hand-constructed predictions/ground-truth with hand-calculated expected metrics.

    ground_truth = ["cat", "cat", "dog", "dog", "dog", "bird"]
    predictions  = ["cat", "dog", "dog", "dog", "cat", "bird"]

    By hand:
      confusion_matrix (rows=true, cols=predicted):
        cat:  {cat: 1, dog: 1, bird: 0}   (2 true "cat": one predicted cat, one predicted dog)
        dog:  {cat: 1, dog: 2, bird: 0}   (3 true "dog": one predicted cat, two predicted dog)
        bird: {cat: 0, dog: 0, bird: 1}   (1 true "bird": predicted bird)

      accuracy = (1 + 2 + 1) / 6 = 4/6 = 0.6666...

      cat:  TP=1, FP=1 (dog->cat), FN=1 (cat->dog)
            precision=1/2=0.5, recall=1/2=0.5, f1=0.5
      dog:  TP=2, FP=1 (cat->dog), FN=1 (cat->dog... the one true dog predicted cat)
            precision=2/3, recall=2/3, f1=2/3
      bird: TP=1, FP=0, FN=0
            precision=1.0, recall=1.0, f1=1.0
    """
    ground_truth = ["cat", "cat", "dog", "dog", "dog", "bird"]
    predictions = ["cat", "dog", "dog", "dog", "cat", "bird"]

    metrics = evaluate_predictions(predictions, ground_truth)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.labels == ("bird", "cat", "dog")
    assert metrics.n_total == 6
    assert metrics.n_excluded == 0
    assert metrics.n_evaluated == 6

    assert metrics.confusion_matrix == {
        "cat": {"cat": 1, "dog": 1, "bird": 0},
        "dog": {"cat": 1, "dog": 2, "bird": 0},
        "bird": {"cat": 0, "dog": 0, "bird": 1},
    }

    assert metrics.accuracy == pytest.approx(4 / 6)

    assert metrics.precision["cat"] == pytest.approx(0.5)
    assert metrics.recall["cat"] == pytest.approx(0.5)
    assert metrics.f1["cat"] == pytest.approx(0.5)

    assert metrics.precision["dog"] == pytest.approx(2 / 3)
    assert metrics.recall["dog"] == pytest.approx(2 / 3)
    assert metrics.f1["dog"] == pytest.approx(2 / 3)

    assert metrics.precision["bird"] == pytest.approx(1.0)
    assert metrics.recall["bird"] == pytest.approx(1.0)
    assert metrics.f1["bird"] == pytest.approx(1.0)


def test_evaluate_predictions_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        evaluate_predictions(["a", "b"], ["a"])


def test_evaluate_predictions_rejects_unknown_label_in_explicit_labels() -> None:
    with pytest.raises(ValueError):
        evaluate_predictions(["a"], ["b"], labels=["a"])


# --- evaluate_predictions: missing ground truth handling --------------------


def test_evaluate_predictions_excludes_missing_ground_truth() -> None:
    predictions = ["light", "moderate", "firm", "light", "moderate"]
    ground_truth = ["light", None, "firm", "", "moderate"]

    metrics = evaluate_predictions(predictions, ground_truth, labels=PRESSURE_LABELS)

    # Two missing entries (None and "") excluded; three evaluated.
    assert metrics.n_total == 5
    assert metrics.n_excluded == 2
    assert metrics.n_evaluated == 3

    # All three evaluated pairs are correct (light/light, firm/firm, moderate/moderate).
    assert metrics.accuracy == pytest.approx(1.0)

    # Missing entries must not be counted anywhere in the confusion matrix.
    total_matrix_count = sum(
        count for row in metrics.confusion_matrix.values() for count in row.values()
    )
    assert total_matrix_count == 3


def test_evaluate_predictions_all_missing_ground_truth_does_not_crash() -> None:
    predictions = ["light", "firm"]
    ground_truth = [None, None]

    metrics = evaluate_predictions(predictions, ground_truth)

    assert metrics.n_total == 2
    assert metrics.n_excluded == 2
    assert metrics.n_evaluated == 0
    assert metrics.accuracy == 0.0
    # Every precision/recall/F1 falls back to 0.0 rather than raising.
    assert all(v == 0.0 for v in metrics.precision.values())
    assert all(v == 0.0 for v in metrics.recall.values())
    assert all(v == 0.0 for v in metrics.f1.values())


# --- extract_heuristic_pressure_reading: reflects interpret.py's real logic --


def test_extract_heuristic_pressure_reading_matches_interpret_thresholds() -> None:
    # At or below PRESSURE_LIGHT_MAX_PX -> "light" (interpret._candidate_pressure's
    # `value <= PRESSURE_LIGHT_MAX_PX` branch).
    assert extract_heuristic_pressure_reading(_make_features(PRESSURE_LIGHT_MAX_PX)) == "light"
    assert extract_heuristic_pressure_reading(_make_features(0.5)) == "light"

    # At or above PRESSURE_FIRM_MIN_PX -> "firm".
    assert extract_heuristic_pressure_reading(_make_features(PRESSURE_FIRM_MIN_PX)) == "firm"
    assert extract_heuristic_pressure_reading(_make_features(20.0)) == "firm"

    # Strictly between the two thresholds -> "moderate".
    midpoint = (PRESSURE_LIGHT_MAX_PX + PRESSURE_FIRM_MIN_PX) / 2
    assert extract_heuristic_pressure_reading(_make_features(midpoint)) == "moderate"

    # Every possible reading must be a real PRESSURE_LABELS entry.
    for value in (0.0, PRESSURE_LIGHT_MAX_PX, midpoint, PRESSURE_FIRM_MIN_PX, 100.0):
        assert extract_heuristic_pressure_reading(_make_features(value)) in PRESSURE_LABELS


# --- compare_baseline_to_heuristic: end-to-end -------------------------------


def test_compare_baseline_to_heuristic_end_to_end(tmp_path: Path) -> None:
    features_list = _fixture_features(tmp_path, count=FIXTURE_COUNT, seed=FIXTURE_SEED)
    labels = generate_synthetic_pressure_labels(features_list)

    # A meaningful mix of labels, otherwise this wouldn't exercise a real
    # multi-class comparison (mirrors the same guard in test_training.py).
    assert len(set(labels)) >= 2

    train_pairs = list(zip(features_list, labels))[:-6]
    held_out_features = features_list[-6:]
    held_out_labels = labels[-6:]

    model = train_baseline(train_pairs, seed=0)

    features_and_labels = list(zip(held_out_features, held_out_labels))
    report = compare_baseline_to_heuristic(model, features_and_labels)

    assert isinstance(report, ComparisonReport)
    assert report.n_samples == 6
    assert report.n_ground_truth_available == 6

    assert 0.0 <= report.agreement_rate <= 1.0
    assert report.n_agreements == round(report.agreement_rate * report.n_samples)

    for metrics in (report.baseline_metrics, report.heuristic_metrics):
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.n_evaluated == 6
        assert metrics.n_excluded == 0

        # Confusion matrix row sums equal the number of true instances of
        # that label; column sums equal the number of predicted instances.
        # Every row/column sum must be non-negative and the grand total
        # must equal n_evaluated.
        grand_total = 0
        for true_label in metrics.labels:
            row_sum = sum(metrics.confusion_matrix[true_label].values())
            assert row_sum >= 0
            grand_total += row_sum
        assert grand_total == metrics.n_evaluated

        for label in metrics.labels:
            assert 0.0 <= metrics.precision[label] <= 1.0
            assert 0.0 <= metrics.recall[label] <= 1.0
            assert 0.0 <= metrics.f1[label] <= 1.0


def test_compare_baseline_to_heuristic_handles_missing_ground_truth(tmp_path: Path) -> None:
    features_list = _fixture_features(tmp_path, count=FIXTURE_COUNT, seed=FIXTURE_SEED)
    labels = generate_synthetic_pressure_labels(features_list)

    model = train_baseline(list(zip(features_list, labels))[:-6], seed=0)

    held_out_features = features_list[-6:]
    # Mix of present and missing ground truth.
    partial_labels: list[str | None] = [labels[-6], None, labels[-4], "", labels[-2], labels[-1]]

    report = compare_baseline_to_heuristic(model, list(zip(held_out_features, partial_labels)))

    assert report.n_samples == 6
    assert report.n_ground_truth_available == 4
    assert report.baseline_metrics.n_evaluated == 4
    assert report.baseline_metrics.n_excluded == 2
    assert report.heuristic_metrics.n_evaluated == 4
    assert report.heuristic_metrics.n_excluded == 2

    # Agreement rate is still computed over all 6 samples, ground truth or not.
    assert 0.0 <= report.agreement_rate <= 1.0
    assert report.n_agreements <= report.n_samples


# --- render_comparison_report ------------------------------------------------


def _required_sections() -> tuple[str, ...]:
    return (
        "Sample Summary",
        "Agreement: Baseline vs Heuristic",
        "Baseline vs Ground Truth",
        "Heuristic vs Ground Truth",
    )


def test_render_comparison_report_contains_required_sections_and_disclaimer(tmp_path: Path) -> None:
    features_list = _fixture_features(tmp_path, count=FIXTURE_COUNT, seed=FIXTURE_SEED)
    labels = generate_synthetic_pressure_labels(features_list)
    model = train_baseline(list(zip(features_list, labels))[:-6], seed=0)

    report = compare_baseline_to_heuristic(
        model, list(zip(features_list[-6:], labels[-6:]))
    )
    text = render_comparison_report(report)

    for section in _required_sections():
        assert section in text

    # Explicit plumbing-proof / synthetic-data disclaimer.
    assert "plumbing-proof" in text.lower()
    assert "synthetic" in text.lower()
    assert "not a real accuracy" in text.lower()

    # Never leaks a bare "None" from a missing value into rendered text.
    assert "\nNone\n" not in text


def test_render_comparison_report_does_not_crash_on_tiny_sample() -> None:
    features = [_make_features(1.0), _make_features(8.0), _make_features(5.0)]
    labels = generate_synthetic_pressure_labels(features)
    model = train_baseline(list(zip(features, labels)), seed=0, n_iterations=10)

    report = compare_baseline_to_heuristic(model, list(zip(features, labels)))
    text = render_comparison_report(report)

    assert text.strip() != ""
    for section in _required_sections():
        assert section in text


def test_render_comparison_report_does_not_crash_with_no_ground_truth() -> None:
    features = [_make_features(1.0), _make_features(8.0), _make_features(5.0)]
    labels = generate_synthetic_pressure_labels(features)
    model = train_baseline(list(zip(features, labels)), seed=0, n_iterations=10)

    all_missing: list[tuple[Features, str | None]] = [(f, None) for f in features]
    report = compare_baseline_to_heuristic(model, all_missing)

    assert report.n_ground_truth_available == 0
    assert report.baseline_metrics.n_evaluated == 0
    assert report.heuristic_metrics.n_evaluated == 0
    # Agreement is still well-defined without any ground truth.
    assert 0.0 <= report.agreement_rate <= 1.0

    text = render_comparison_report(report)
    assert text.strip() != ""
    for section in _required_sections():
        assert section in text
    assert "no ground-truth labels were available" in text.lower()


def test_render_comparison_report_handles_empty_sample_set() -> None:
    # Degenerate but should not crash: zero samples.
    empty_features: list[tuple[Features, str | None]] = []
    labels = ["light", "moderate", "firm"]
    model = train_baseline(
        [(_make_features(1.0), "light"), (_make_features(8.0), "firm")], seed=0, n_iterations=10
    )
    del labels  # unused, kept for readability of intent above

    report = compare_baseline_to_heuristic(model, empty_features)
    assert report.n_samples == 0
    assert report.agreement_rate == 0.0

    text = render_comparison_report(report)
    assert text.strip() != ""
    for section in _required_sections():
        assert section in text
