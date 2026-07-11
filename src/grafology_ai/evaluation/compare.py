"""Compare the M2.3 plumbing-proof baseline model to the heuristic interpretation layer.

PLUMBING PROOF -- see the package docstring
(``grafology_ai/evaluation/__init__.py``). For a set of
(:class:`~grafology_ai.analysis.features.Features`, ground-truth) samples,
:func:`compare_baseline_to_heuristic` runs both "predictors" --
:func:`grafology_ai.training.baseline.predict` (the trained baseline
model) and :func:`grafology_ai.evaluation.heuristic.extract_heuristic_pressure_reading`
(the shipped heuristic interpretation layer, read as a pressure bucket) --
over the same samples, scores each against ground truth where available
(:func:`grafology_ai.evaluation.metrics.evaluate_predictions`), and also
reports how often the two predictors agree *with each other*, independent
of ground truth -- which stays meaningful even when there is no ground
truth at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from grafology_ai.analysis.features import Features
from grafology_ai.evaluation.heuristic import extract_heuristic_pressure_reading
from grafology_ai.evaluation.metrics import EvaluationMetrics, evaluate_predictions
from grafology_ai.training.baseline import BaselineModel, predict
from grafology_ai.training.synthetic_labels import PRESSURE_LABELS


@dataclass(frozen=True)
class ComparisonReport:
    """The outcome of comparing a baseline model to the heuristic reading.

    PLUMBING PROOF, not a real accuracy claim -- see the package docstring.

    Attributes:
        n_samples: Total number of (Features, ground_truth) pairs compared.
        n_ground_truth_available: Number of those samples with a
            non-missing ground-truth label (see
            :mod:`grafology_ai.evaluation.metrics`'s "Missing ground
            truth" handling).
        baseline_predictions: The baseline model's prediction for each
            sample, in input order.
        heuristic_predictions: The heuristic reading for each sample, in
            input order.
        ground_truth: The ground-truth label for each sample (``None``
            where missing), in input order.
        baseline_metrics: :class:`~grafology_ai.evaluation.metrics.EvaluationMetrics`
            for ``baseline_predictions`` vs ``ground_truth``.
        heuristic_metrics: :class:`~grafology_ai.evaluation.metrics.EvaluationMetrics`
            for ``heuristic_predictions`` vs ``ground_truth``.
        agreement_rate: Fraction of samples where
            ``baseline_predictions[i] == heuristic_predictions[i]``, in
            ``[0, 1]``. Computed independent of ``ground_truth`` -- unlike
            :attr:`baseline_metrics`/:attr:`heuristic_metrics`, this stays
            meaningful even with no ground truth available at all.
            ``0.0`` when ``n_samples == 0``.
        n_agreements: Raw count of samples where the baseline and
            heuristic predictions matched each other (the numerator of
            :attr:`agreement_rate`).
    """

    n_samples: int
    n_ground_truth_available: int
    baseline_predictions: tuple[str, ...]
    heuristic_predictions: tuple[str, ...]
    ground_truth: tuple[str | None, ...]
    baseline_metrics: EvaluationMetrics
    heuristic_metrics: EvaluationMetrics
    agreement_rate: float
    n_agreements: int


def compare_baseline_to_heuristic(
    model: BaselineModel,
    features_and_labels: list[tuple[Features, str | None]],
) -> ComparisonReport:
    """Compare ``model``'s predictions to the heuristic reading, over ``features_and_labels``.

    PLUMBING PROOF -- see the package docstring. For each ``(features,
    ground_truth)`` pair, computes:

    - ``model``'s prediction (:func:`grafology_ai.training.baseline.predict`).
    - The heuristic's reading
      (:func:`grafology_ai.evaluation.heuristic.extract_heuristic_pressure_reading`).
    - Whether each matches ``ground_truth``, where ``ground_truth`` is
      present (rolled up into :attr:`ComparisonReport.baseline_metrics`/
      :attr:`ComparisonReport.heuristic_metrics`).

    Both sets of metrics, and the raw predictions, are computed over a
    shared label vocabulary -- the sorted union of
    :data:`grafology_ai.training.synthetic_labels.PRESSURE_LABELS`,
    ``model.classes``, every baseline/heuristic prediction actually
    produced, and every non-missing ground-truth value -- so
    ``baseline_metrics`` and ``heuristic_metrics`` are directly comparable
    (same confusion-matrix shape, same class list) even if one predictor
    happens not to produce every label on this particular sample set.

    Args:
        model: A trained :class:`~grafology_ai.training.baseline.BaselineModel`
            (see :func:`grafology_ai.training.baseline.train_baseline`).
        features_and_labels: ``(Features, ground_truth)`` pairs.
            ``ground_truth`` may be ``None`` (or ``""``) for samples with
            no known label -- see
            :mod:`grafology_ai.evaluation.metrics`'s "Missing ground
            truth" handling. May be empty.

    Returns:
        A :class:`ComparisonReport`.
    """
    n_samples = len(features_and_labels)

    baseline_predictions = [predict(model, features) for features, _label in features_and_labels]
    heuristic_predictions = [
        extract_heuristic_pressure_reading(features) for features, _label in features_and_labels
    ]
    ground_truth: list[str | None] = [label for _features, label in features_and_labels]

    n_ground_truth_available = sum(
        1 for label in ground_truth if label is not None and label != ""
    )

    shared_labels = sorted(
        set(PRESSURE_LABELS)
        | set(model.classes)
        | set(baseline_predictions)
        | set(heuristic_predictions)
        | {label for label in ground_truth if label is not None and label != ""}
    )

    baseline_metrics = evaluate_predictions(baseline_predictions, ground_truth, labels=shared_labels)
    heuristic_metrics = evaluate_predictions(heuristic_predictions, ground_truth, labels=shared_labels)

    n_agreements = sum(
        1
        for baseline_pred, heuristic_pred in zip(baseline_predictions, heuristic_predictions)
        if baseline_pred == heuristic_pred
    )
    agreement_rate = (n_agreements / n_samples) if n_samples > 0 else 0.0

    return ComparisonReport(
        n_samples=n_samples,
        n_ground_truth_available=n_ground_truth_available,
        baseline_predictions=tuple(baseline_predictions),
        heuristic_predictions=tuple(heuristic_predictions),
        ground_truth=tuple(ground_truth),
        baseline_metrics=baseline_metrics,
        heuristic_metrics=heuristic_metrics,
        agreement_rate=agreement_rate,
        n_agreements=n_agreements,
    )
