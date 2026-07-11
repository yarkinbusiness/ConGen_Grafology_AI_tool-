"""Classification metrics -- accuracy, confusion matrix, per-class precision/recall/F1.

PLUMBING PROOF -- see the package docstring
(``grafology_ai/evaluation/__init__.py``). Implemented from scratch in
plain Python (no numpy/scikit-learn needed for this bookkeeping-scale
math), matching this repository's convention of hand-rolling small,
well-understood algorithms rather than adding a dependency for them (see
:mod:`grafology_ai.training.baseline`, :mod:`grafology_ai.analysis.features`).

This module is deliberately generic: it takes plain ``predictions``/
``ground_truth`` label lists and knows nothing about pressure, features,
or models -- :mod:`grafology_ai.evaluation.compare` is the module that
wires this up to :mod:`grafology_ai.training` and
:mod:`grafology_ai.evaluation.heuristic`.

Missing ground truth
---------------------

A ``ground_truth`` entry of ``None`` or ``""`` (empty string) is treated
as "no label available for this sample" and is **excluded** from every
computed metric: it contributes to neither the confusion matrix, nor
accuracy's numerator/denominator, nor any per-class count. It is not
counted as a wrong prediction, and it does not silently vanish either --
:attr:`EvaluationMetrics.n_excluded` reports exactly how many entries were
excluded this way, alongside :attr:`EvaluationMetrics.n_total` (the input
length) and :attr:`EvaluationMetrics.n_evaluated` (``n_total - n_excluded``,
i.e. how many pairs the metrics below were actually computed from).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _is_missing(label: str | None) -> bool:
    """True if a ground-truth entry counts as "missing" (see module docstring)."""
    return label is None or label == ""


@dataclass(frozen=True)
class EvaluationMetrics:
    """Accuracy, confusion matrix, and per-class precision/recall/F1.

    Attributes:
        labels: The label vocabulary these metrics were computed over, in
            a fixed, sorted order (either inferred from the data or the
            caller-supplied ``labels``, see :func:`evaluate_predictions`).
            Every key in :attr:`confusion_matrix`, :attr:`precision`,
            :attr:`recall`, and :attr:`f1` is one of these labels.
        n_total: Number of ``(prediction, ground_truth)`` pairs passed in,
            before excluding missing ground truth.
        n_excluded: Number of pairs excluded because their ``ground_truth``
            entry was missing (``None`` or ``""``) -- see the module
            docstring's "Missing ground truth" section.
        n_evaluated: ``n_total - n_excluded`` -- how many pairs every
            metric below was actually computed from.
        accuracy: Fraction of evaluated pairs where the prediction exactly
            matches ground truth, in ``[0, 1]``. Defined as ``0.0`` when
            ``n_evaluated == 0`` (nothing to score), rather than raising
            or producing NaN.
        confusion_matrix: ``confusion_matrix[true_label][predicted_label]``
            is the count of evaluated pairs with that true/predicted
            combination. Every ``(true_label, predicted_label)`` cell for
            every pair of :attr:`labels` is present (defaulting to ``0``),
            so the structure is always fully-formed/rectangular, never
            sparse.
        precision: Per-class precision, ``TP / (TP + FP)``, keyed by
            label. ``0.0`` for a class with no predicted instances
            (``TP + FP == 0``), documented here rather than raising a
            division-by-zero error.
        recall: Per-class recall, ``TP / (TP + FN)``, keyed by label.
            ``0.0`` for a class with no true instances (``TP + FN == 0``).
        f1: Per-class F1, the harmonic mean of :attr:`precision` and
            :attr:`recall` for that class, keyed by label. ``0.0`` when
            precision and recall are both ``0.0``.
    """

    labels: tuple[str, ...]
    n_total: int
    n_excluded: int
    n_evaluated: int
    accuracy: float
    confusion_matrix: dict[str, dict[str, int]]
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]


def evaluate_predictions(
    predictions: list[str],
    ground_truth: list[str | None],
    labels: Sequence[str] | None = None,
) -> EvaluationMetrics:
    """Compute accuracy, a confusion matrix, and per-class precision/recall/F1.

    Args:
        predictions: Predicted labels, one per sample.
        ground_truth: Ground-truth labels, one per sample, in the same
            order as ``predictions``. Entries that are ``None`` or ``""``
            are treated as missing and excluded -- see the module
            docstring's "Missing ground truth" section.
        labels: The label vocabulary to report metrics over. If ``None``
            (the default), it is inferred as the sorted union of every
            prediction and every non-missing ground-truth value. If
            explicitly given, every prediction and non-missing
            ground-truth value must be one of ``labels`` (raises
            ``ValueError`` otherwise), so a caller-supplied vocabulary
            never silently drops observed classes.

    Returns:
        An :class:`EvaluationMetrics`.

    Raises:
        ValueError: If ``predictions`` and ``ground_truth`` have different
            lengths, or if an explicitly-supplied ``labels`` does not
            cover every prediction/non-missing-ground-truth value present.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions and ground_truth must have the same length, got "
            f"{len(predictions)} and {len(ground_truth)}"
        )

    n_total = len(predictions)
    pairs = [
        (pred, truth)
        for pred, truth in zip(predictions, ground_truth)
        if not _is_missing(truth)
    ]
    n_excluded = n_total - len(pairs)
    n_evaluated = len(pairs)

    if labels is None:
        inferred = {pred for pred, _truth in pairs} | {truth for _pred, truth in pairs}
        resolved_labels = tuple(sorted(inferred))
    else:
        resolved_labels = tuple(labels)
        label_set = set(resolved_labels)
        unknown = {pred for pred, _truth in pairs if pred not in label_set} | {
            truth for _pred, truth in pairs if truth not in label_set
        }
        if unknown:
            raise ValueError(
                f"predictions/ground_truth contain labels not in the supplied "
                f"labels={resolved_labels!r}: {sorted(unknown)!r}"
            )

    confusion_matrix: dict[str, dict[str, int]] = {
        true_label: {pred_label: 0 for pred_label in resolved_labels}
        for true_label in resolved_labels
    }
    correct = 0
    for pred, truth in pairs:
        confusion_matrix[truth][pred] += 1
        if pred == truth:
            correct += 1

    accuracy = (correct / n_evaluated) if n_evaluated > 0 else 0.0

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    for label in resolved_labels:
        tp = confusion_matrix[label][label]
        fp = sum(
            confusion_matrix[true_label][label]
            for true_label in resolved_labels
            if true_label != label
        )
        fn = sum(
            confusion_matrix[label][pred_label]
            for pred_label in resolved_labels
            if pred_label != label
        )
        label_precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        label_recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        label_f1 = (
            (2 * label_precision * label_recall / (label_precision + label_recall))
            if (label_precision + label_recall) > 0
            else 0.0
        )
        precision[label] = label_precision
        recall[label] = label_recall
        f1[label] = label_f1

    return EvaluationMetrics(
        labels=resolved_labels,
        n_total=n_total,
        n_excluded=n_excluded,
        n_evaluated=n_evaluated,
        accuracy=accuracy,
        confusion_matrix=confusion_matrix,
        precision=precision,
        recall=recall,
        f1=f1,
    )
