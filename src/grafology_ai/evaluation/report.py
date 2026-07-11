"""Render a :class:`~grafology_ai.evaluation.compare.ComparisonReport` to markdown.

PLUMBING PROOF -- see the package docstring
(``grafology_ai/evaluation/__init__.py``). Follows the style conventions
of this repository's other report renderer,
:mod:`grafology_ai.report.generator`: a fixed section order, no dangling
empty sections (an empty/degenerate case renders a clear note instead of a
blank heading), and a prominent disclaimer placed near the top of the
document rather than buried at the bottom. Unlike
:mod:`grafology_ai.report.generator` (which renders already-cautious,
non-diagnostic narrative text as-is), every string this module originates
is its own structural/boilerplate text, written to make unmistakably
explicit that this is a plumbing-proof artifact built from synthetic
fixture data -- not a real accuracy evaluation of either the baseline
model or the heuristic interpretation layer.
"""

from __future__ import annotations

from grafology_ai.evaluation.compare import ComparisonReport
from grafology_ai.evaluation.metrics import EvaluationMetrics

_DISCLAIMER_HEADING = "## Important: Please Read Before Using This Report"

_DISCLAIMER_TEXT = (
    "**This is a plumbing-proof comparison harness output, not a real "
    "accuracy evaluation.** No real, graphologist-labeled handwriting "
    "dataset exists yet. The \"baseline\" predictions come from the "
    "M2.3 plumbing-proof `BaselineModel` "
    "(`grafology_ai.training.baseline`) -- a hand-rolled logistic "
    "regression over two numbers, never trained on real graphological "
    "data. The \"heuristic\" predictions come from the existing "
    "descriptive interpretation layer (`grafology_ai.interpretation.interpret`), "
    "read as a pressure bucket. Any ground truth used below is "
    "**fabricated, synthetic labeling** "
    "(`grafology_ai.training.synthetic_labels`), not real graphologist "
    "ground truth. Every number in this report exists to prove the "
    "comparison *machinery* works end to end, not to make any claim about "
    "real-world graphological accuracy."
)

_CLOSING_NOTE_TEXT = (
    "*Reminder: this comparison was computed entirely against synthetic "
    "fixture data and/or fabricated synthetic labels. It demonstrates "
    "working comparison machinery, not a real accuracy result. Re-run "
    "this same harness once a real fine-tuned model and real "
    "graphologist-labeled ground truth exist.*"
)

_NO_GROUND_TRUTH_NOTE = (
    "No ground-truth labels were available for any sample in this set "
    "(`n_ground_truth_available=0`). The accuracy/precision/recall/F1 "
    "figures below are computed from zero evaluated pairs and are not "
    "meaningful -- they are included only to show the metrics machinery "
    "runs without crashing when no ground truth is present."
)


def _render_header(report: ComparisonReport) -> str:
    return "# Baseline vs Heuristic Comparison Report (Plumbing Proof)"


def _render_disclaimer() -> str:
    return f"{_DISCLAIMER_HEADING}\n\n{_DISCLAIMER_TEXT}"


def _render_sample_summary(report: ComparisonReport) -> str:
    lines = [
        "## Sample Summary",
        "",
        f"- Total samples compared: {report.n_samples}",
        f"- Samples with ground truth available: {report.n_ground_truth_available}",
        f"- Samples with no ground truth (excluded from accuracy metrics): "
        f"{report.n_samples - report.n_ground_truth_available}",
    ]
    return "\n".join(lines)


def _render_agreement(report: ComparisonReport) -> str:
    lines = [
        "## Agreement: Baseline vs Heuristic",
        "",
        (
            "Fraction of samples where the baseline model and the heuristic "
            "interpretation layer predict the *same* pressure label as each "
            "other -- independent of ground truth, so this figure remains "
            "meaningful even with no ground truth available at all."
        ),
        "",
        f"- Agreement rate: {report.agreement_rate:.1%} "
        f"({report.n_agreements}/{report.n_samples} samples)",
    ]
    return "\n".join(lines)


def _render_confusion_matrix_table(metrics: EvaluationMetrics) -> str:
    if not metrics.labels:
        return "*(No label vocabulary was observed; confusion matrix is empty.)*"

    header = "| true \\ predicted | " + " | ".join(metrics.labels) + " |"
    separator = "| --- | " + " | ".join("---" for _ in metrics.labels) + " |"
    rows = [header, separator]
    for true_label in metrics.labels:
        row_cells = [str(metrics.confusion_matrix[true_label][pred_label]) for pred_label in metrics.labels]
        rows.append(f"| {true_label} | " + " | ".join(row_cells) + " |")
    return "\n".join(rows)


def _render_per_class_table(metrics: EvaluationMetrics) -> str:
    if not metrics.labels:
        return "*(No label vocabulary was observed; no per-class metrics to show.)*"

    header = "| label | precision | recall | F1 |"
    separator = "| --- | --- | --- | --- |"
    rows = [header, separator]
    for label in metrics.labels:
        rows.append(
            f"| {label} | {metrics.precision[label]:.2f} | "
            f"{metrics.recall[label]:.2f} | {metrics.f1[label]:.2f} |"
        )
    return "\n".join(rows)


def _render_metrics_section(title: str, metrics: EvaluationMetrics) -> str:
    parts = [
        f"## {title}",
        "",
        f"- Evaluated pairs: {metrics.n_evaluated} "
        f"(of {metrics.n_total} total, {metrics.n_excluded} excluded for missing ground truth)",
        f"- Accuracy: {metrics.accuracy:.1%}",
    ]
    if metrics.n_evaluated == 0:
        parts.append("")
        parts.append(_NO_GROUND_TRUTH_NOTE)
    parts.append("")
    parts.append("**Confusion matrix** (rows = true label, columns = predicted label):")
    parts.append("")
    parts.append(_render_confusion_matrix_table(metrics))
    parts.append("")
    parts.append("**Per-class precision / recall / F1:**")
    parts.append("")
    parts.append(_render_per_class_table(metrics))
    return "\n".join(parts)


def _render_closing_note() -> str:
    return f"---\n\n{_CLOSING_NOTE_TEXT}"


def render_comparison_report(report: ComparisonReport) -> str:
    """Render `report` into a complete markdown document string.

    Fixed sections, always in this order:

    1. A title/header.
    2. A prominent disclaimer, near the top of the document, explicitly
       stating this is a plumbing-proof / synthetic-data comparison, not a
       real accuracy evaluation (see the module docstring).
    3. `"## Sample Summary"`: sample counts, and how many had ground truth
       available.
    4. `"## Agreement: Baseline vs Heuristic"`: `report.agreement_rate`,
       which is meaningful even when there is no ground truth at all.
    5. `"## Baseline vs Ground Truth"`: `report.baseline_metrics` --
       accuracy, a confusion-matrix table, and a per-class
       precision/recall/F1 table. If no ground truth was available at all
       (`n_evaluated == 0`), an explicit note says so instead of
       presenting a misleading `0%` accuracy without context.
    6. `"## Heuristic vs Ground Truth"`: the same, for
       `report.heuristic_metrics`.
    7. A closing note repeating the plumbing-proof/synthetic-data caveat.

    Never crashes: renders a well-formed report for a `ComparisonReport`
    built from as few as one sample, and for one where every ground-truth
    entry was `None` (see `_render_metrics_section`'s
    `n_evaluated == 0` handling and `_render_confusion_matrix_table`'s/
    `_render_per_class_table`'s handling of an empty label vocabulary).
    """
    sections = [
        _render_header(report),
        _render_disclaimer(),
        _render_sample_summary(report),
        _render_agreement(report),
        _render_metrics_section("Baseline vs Ground Truth", report.baseline_metrics),
        _render_metrics_section("Heuristic vs Ground Truth", report.heuristic_metrics),
        _render_closing_note(),
    ]
    return "\n\n".join(sections).rstrip("\n") + "\n"
