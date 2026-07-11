"""``grafology_ai.evaluation`` -- a comparison HARNESS, not a real evaluation result.

**What this package is.** The client's technical proposal's evaluation
deliverable eventually needs a "baseline vs fine-tuned" comparison. No real
fine-tuned model and no real, graphologist-labeled dataset exist yet (see
:mod:`grafology_ai.training` and :mod:`grafology_ai.dataset.fixtures`), so
there is nothing real to evaluate. This package proves the *machinery*
works instead: comparing the M2.3 plumbing-proof
:class:`~grafology_ai.training.baseline.BaselineModel`'s predictions
against (a) the existing heuristic interpretation layer's
(:mod:`grafology_ai.interpretation.interpret`) qualitative "pressure"
reading, and (b) synthetic ground-truth labels
(:mod:`grafology_ai.training.synthetic_labels`) where available --
entirely against synthetic fixture data. When a real fine-tuned model and
real labels arrive, this same machinery (metrics computation, comparison
report, rendering) plugs in unchanged; only the inputs stop being
synthetic.

**What this package is NOT:**

- NOT a real accuracy evaluation. Every number this package produces when
  run against fixture/synthetic data is a plumbing-proof artifact, not a
  claim about real-world graphological accuracy.
- NOT wired into :func:`grafology_ai.run_analysis.run_analysis`, the CLI,
  or the API. It is a standalone evaluation/comparison harness.
- NOT dependent on scikit-learn or any other new runtime dependency --
  precision/recall/F1/confusion-matrix math is implemented from scratch in
  plain Python, matching the hand-rolled-numpy convention used throughout
  this repository (see :mod:`grafology_ai.training.baseline`,
  :mod:`grafology_ai.analysis.features`).

Modules
-------

- :mod:`grafology_ai.evaluation.heuristic` -- extracts "the heuristic's
  pressure reading" from a :class:`~grafology_ai.analysis.features.Features`
  record, in the same ``light``/``moderate``/``firm`` vocabulary the
  baseline model's labels use (see
  :func:`extract_heuristic_pressure_reading` for exactly how this stays
  tied to the real, shipped :mod:`grafology_ai.interpretation.interpret`
  logic rather than reinventing a disconnected bucketing scheme).
- :mod:`grafology_ai.evaluation.metrics` -- :func:`evaluate_predictions`:
  accuracy, confusion matrix, and per-class precision/recall/F1 for any
  predictions/ground-truth pair, with documented handling of missing
  ground truth.
- :mod:`grafology_ai.evaluation.compare` -- :func:`compare_baseline_to_heuristic`:
  runs the baseline model and the heuristic reading over the same samples
  and builds a :class:`~grafology_ai.evaluation.compare.ComparisonReport`.
- :mod:`grafology_ai.evaluation.report` -- :func:`render_comparison_report`:
  renders a :class:`~grafology_ai.evaluation.compare.ComparisonReport` into
  a fixed-section markdown report, explicitly labeled as a plumbing-proof/
  synthetic-data artifact throughout.
"""

from __future__ import annotations

from grafology_ai.evaluation.compare import ComparisonReport, compare_baseline_to_heuristic
from grafology_ai.evaluation.heuristic import extract_heuristic_pressure_reading
from grafology_ai.evaluation.metrics import EvaluationMetrics, evaluate_predictions
from grafology_ai.evaluation.report import render_comparison_report

__all__ = [
    "extract_heuristic_pressure_reading",
    "EvaluationMetrics",
    "evaluate_predictions",
    "ComparisonReport",
    "compare_baseline_to_heuristic",
    "render_comparison_report",
]
