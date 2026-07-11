"""``grafology_ai.training`` -- a PLUMBING PROOF, not a real trained model.

**What this package is.** The client's technical proposal has a line item
for "Experimentation and training (baseline + fine-tuning + light tuning)".
No real, labeled client handwriting dataset exists yet (see
:mod:`grafology_ai.dataset.fixtures`), so there is nothing real to train
on. This package exists to prove that the *machinery* -- train a baseline
model on (:class:`~grafology_ai.analysis.features.Features`, label) pairs,
run inference, save/load it with content-hash versioning consistent with
:mod:`grafology_ai.pipeline.export` -- works end to end, entirely against
synthetic fixture data and synthetic (fabricated) labels, so that when a
real labeled dataset arrives the 22-hour training budget goes toward real
experimentation instead of wiring this code for the first time.

**What this package is NOT:**

- NOT a real graphological model. The one baseline model implemented here
  (:mod:`grafology_ai.training.baseline`) is a hand-rolled, from-scratch
  multinomial logistic regression over two numbers (stroke-width mean/std)
  predicting a "pressure" bucket -- deliberately minimal, chosen only to
  exercise the train -> predict -> save -> load -> version pipeline.
- NOT trained on real data. Every test and every example in this package
  trains against :func:`grafology_ai.dataset.fixtures.generate_fixture_dataset`
  output and the synthetic labels in
  :mod:`grafology_ai.training.synthetic_labels` -- fabricated,
  documented-arbitrary bucketing of a heuristic feature, not graphologist
  ground truth.
- NOT to be used for actual handwriting analysis. Nothing in this package
  is wired into :func:`grafology_ai.run_analysis.run_analysis`, the CLI, or
  the API, and it must not be until a real labeled dataset and real
  evaluation exist.

See ``docs/labeling_rubric.md`` for the real indicator vocabulary this
stands in for one entry of ("Pressure"), and
:mod:`grafology_ai.pipeline.export` for the content-hash versioning pattern
this package's :mod:`grafology_ai.training.export` module extends.
"""

from __future__ import annotations

from grafology_ai.training.baseline import (
    BaselineModel,
    predict,
    train_baseline,
)
from grafology_ai.training.export import (
    SavedModelResult,
    load_model,
    save_model,
)
from grafology_ai.training.synthetic_labels import (
    PRESSURE_LABELS,
    generate_synthetic_pressure_label,
    generate_synthetic_pressure_labels,
)

__all__ = [
    "BaselineModel",
    "train_baseline",
    "predict",
    "SavedModelResult",
    "save_model",
    "load_model",
    "PRESSURE_LABELS",
    "generate_synthetic_pressure_label",
    "generate_synthetic_pressure_labels",
]
